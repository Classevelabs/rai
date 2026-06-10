//! rai-chat: Web-based chat UI with pondering strategies.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use rand::SeedableRng;
use serde::Deserialize;
use tiny_http::{Header, Method, Response, Server, StatusCode};

use rai_infer::chat_template::ChatTemplate;
use rai_infer::model::{InferenceWork, RaiModel};
use rai_infer::ponder::{pondered_forward, PonderConfig};
use rai_infer::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "rai-chat",
    about = "Chat with any .raimodel — edge inference with pondering"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value = "8090")]
    port: u16,
    #[arg(long, default_value = "512")]
    max_context: usize,
    /// Chat template: auto, none, few-shot, mistral, llama3
    #[arg(long, default_value = "auto")]
    chat_template: String,
}

struct AppState {
    model: RaiModel,
    tokenizer: tokenizers::Tokenizer,
    max_context: usize,
    template: ChatTemplate,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    ponder_strategy: Option<String>,
    guidance_scale: Option<f32>,
    ensemble_n: Option<usize>,
    noise_sigma: Option<f32>,
    entropy_threshold: Option<f32>,
}

fn build_ponder(req: &ChatRequest) -> PonderConfig {
    let strat = req.ponder_strategy.as_deref().unwrap_or("none");
    let gs = req.guidance_scale.unwrap_or(1.5);
    let en = req.ensemble_n.unwrap_or(3);
    let ns = req.noise_sigma.unwrap_or(0.05);
    let et = req.entropy_threshold.unwrap_or(3.0);

    match strat {
        "none" => PonderConfig::none(),
        "cfg" => PonderConfig::cfg(gs),
        "ensemble" => PonderConfig::ensemble(en, ns),
        "cfg-ensemble" => PonderConfig::cfg_ensemble(gs, en, ns),
        "adaptive" => PonderConfig::adaptive(gs, et),
        _ => PonderConfig::none(),
    }
}

fn handle_generate(state: &AppState, req_body: &str) -> Result<String> {
    let chat_req: ChatRequest = serde_json::from_str(req_body)?;
    let temperature = chat_req.temperature.unwrap_or(0.7);
    let max_tokens = chat_req.max_tokens.unwrap_or(200);
    let ponder_config = build_ponder(&chat_req);

    let sampler_config = SamplerConfig {
        temperature,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    // Format prompt using the configured chat template
    let prompt = state.template.format_prompt(&chat_req.message);
    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    let max_ctx = state
        .max_context
        .min(state.model.config.max_context as usize);
    let mut kv_cache = state.model.create_kv_cache(max_ctx);
    let mut work = InferenceWork::new();
    let mut work2 = InferenceWork::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut all_tokens = prompt_tokens.clone();
    let mut pos = 0;

    // Prefill
    let t_prefill = Instant::now();
    for &token_id in &prompt_tokens {
        if pos >= max_ctx {
            break;
        }
        let _ = pondered_forward(
            &state.model,
            token_id,
            pos,
            &mut kv_cache,
            &PonderConfig::none(),
            &mut work,
            &mut work2,
            &mut rng,
        )?;
        pos += 1;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    // Decode
    let t_decode = Instant::now();
    let mut generated_text = String::new();
    let mut tokens_generated = 0;
    let mut total_passes = 0;
    let mut hard_tokens = 0;

    for _ in 0..max_tokens {
        if pos >= max_ctx {
            break;
        }
        let last_token = *all_tokens.last().unwrap();
        let (mut logits, metrics) = pondered_forward(
            &state.model,
            last_token,
            pos,
            &mut kv_cache,
            &ponder_config,
            &mut work,
            &mut work2,
            &mut rng,
        )?;
        total_passes += metrics.forward_passes;
        if metrics.was_hard_token {
            hard_tokens += 1;
        }

        apply_repetition_penalty(&mut logits, &all_tokens, sampler_config.repetition_penalty);
        let next_token = sample_token(&mut logits, &sampler_config, &mut rng);

        // Check all common EOS tokens
        let is_eos = ["</s>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"]
            .iter()
            .any(|tok| {
                state
                    .tokenizer
                    .token_to_id(tok)
                    .map_or(false, |id| next_token == id as usize)
            });
        if is_eos {
            break;
        }

        all_tokens.push(next_token);
        pos += 1;
        tokens_generated += 1;

        // Decode entire generated suffix for correct SentencePiece spacing
        let gen_ids: Vec<u32> = all_tokens[prompt_tokens.len()..]
            .iter()
            .map(|&t| t as u32)
            .collect();
        generated_text = state.tokenizer.decode(&gen_ids, false).unwrap_or_default();

        // Stop if model generates a template-specific stop sequence
        let mut should_stop = false;
        for stop in state.template.stop_sequences() {
            if generated_text.contains(stop) {
                generated_text = generated_text.split(stop).next().unwrap_or("").to_string();
                should_stop = true;
                break;
            }
        }
        if should_stop {
            break;
        }
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = if decode_ms > 0.0 {
        tokens_generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let avg_passes = if tokens_generated > 0 {
        total_passes as f64 / tokens_generated as f64
    } else {
        0.0
    };
    let hard_pct = if tokens_generated > 0 {
        100.0 * hard_tokens as f64 / tokens_generated as f64
    } else {
        0.0
    };

    let response = serde_json::json!({
        "text": generated_text.trim(),
        "tokens": tokens_generated,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tok_per_sec": decode_tps,
        "avg_passes": avg_passes,
        "hard_tokens_pct": hard_pct,
        "strategy": format!("{:?}", ponder_config.strategy),
    });

    Ok(response.to_string())
}

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();
    let args = Args::parse();

    eprintln!("Loading model: {}", args.model.display());
    let model = RaiModel::load(&args.model)?;
    let cfg = &model.config;
    eprintln!(
        "Model loaded (hidden={}, layers={}, heads={}/{}kv, inter={}, vocab={})",
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.intermediate_size,
        cfg.vocab_size
    );
    eprintln!(
        "Weights: {:.1} MB, KV cache: {:.1} MB (max_ctx={})",
        model.file_size() as f64 / (1024.0 * 1024.0),
        model.kv_cache_bytes(args.max_context) as f64 / (1024.0 * 1024.0),
        args.max_context
    );

    eprintln!("Loading tokenizer: {}", args.tokenizer.display());
    let tokenizer =
        tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;

    let template = ChatTemplate::from_str_arg(&args.chat_template, &tokenizer);
    eprintln!("Chat template: {}", template.display_name());

    let state = Arc::new(AppState {
        model,
        tokenizer,
        max_context: args.max_context,
        template,
    });

    let infer_lock = Arc::new(Mutex::new(()));

    let addr = format!("0.0.0.0:{}", args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("bind: {e}"))?;
    eprintln!("\n  Chat UI: http://localhost:{}\n", args.port);

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        match (method, url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => {
                let resp = Response::from_data(CHAT_HTML.as_bytes().to_vec()).with_header(
                    Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                );
                let _ = request.respond(resp);
            }
            (Method::Post, "/api/chat") => {
                let mut body = String::new();
                let _ = std::io::Read::read_to_string(&mut request.as_reader(), &mut body);
                let _lock = infer_lock.lock().unwrap();
                let state = Arc::clone(&state);

                match handle_generate(&state, &body) {
                    Ok(json) => {
                        let resp = Response::from_data(json.into_bytes()).with_header(
                            Header::from_bytes("Content-Type", "application/json").unwrap(),
                        );
                        let _ = request.respond(resp);
                    }
                    Err(e) => {
                        let err = serde_json::json!({"error": e.to_string()}).to_string();
                        let resp = Response::from_data(err.into_bytes())
                            .with_status_code(StatusCode(500))
                            .with_header(
                                Header::from_bytes("Content-Type", "application/json").unwrap(),
                            );
                        let _ = request.respond(resp);
                    }
                }
            }
            (Method::Get, "/api/info") => {
                let cfg = &state.model.config;
                let info = serde_json::json!({
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "num_kv_heads": cfg.num_kv_heads,
                    "intermediate_size": cfg.intermediate_size,
                    "vocab_size": cfg.vocab_size,
                    "chat_template": state.template.display_name(),
                    "weights_mb": state.model.file_size() as f64 / (1024.0 * 1024.0),
                });
                let resp = Response::from_data(info.to_string().into_bytes())
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
                let _ = request.respond(resp);
            }
            (Method::Options, _) => {
                let resp = Response::from_data(Vec::new())
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap())
                    .with_header(
                        Header::from_bytes("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                            .unwrap(),
                    )
                    .with_header(
                        Header::from_bytes("Access-Control-Allow-Headers", "Content-Type").unwrap(),
                    );
                let _ = request.respond(resp);
            }
            _ => {
                let _ = request.respond(
                    Response::from_data(b"404".to_vec()).with_status_code(StatusCode(404)),
                );
            }
        }
    }
    Ok(())
}

const CHAT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rai-chat — Edge Inference Engine</title>
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --surface2: #1a1a28; --border: #2a2a3a;
    --text: #e0e0e8; --text-dim: #888898; --accent: #6c5ce7; --accent-glow: #6c5ce740;
    --user-bg: #1a1a3a; --bot-bg: #1a2a1a;
    --green: #00d26a; --orange: #ffa726; --red: #ff5252;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 16px; flex-shrink: 0;
  }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  header .tag {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    border: 1px solid var(--border); color: var(--text-dim);
  }
  header .tag.live { border-color: var(--green); color: var(--green); }
  .controls {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 8px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; flex-shrink: 0;
  }
  .controls label {
    font-size: 11px; color: var(--text-dim); display: flex; align-items: center; gap: 6px;
  }
  .controls select, .controls input[type="number"], .controls input[type="range"] {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 11px;
  }
  .controls input[type="range"] { width: 80px; }
  #chat {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 80%; padding: 10px 14px; border-radius: 12px;
    font-size: 13px; line-height: 1.5; animation: fadeIn 0.2s ease;
    white-space: pre-wrap; word-wrap: break-word;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; } }
  .msg.user {
    align-self: flex-end; background: var(--user-bg);
    border: 1px solid #2a2a5a; border-bottom-right-radius: 4px;
  }
  .msg.bot {
    align-self: flex-start; background: var(--bot-bg);
    border: 1px solid #2a4a2a; border-bottom-left-radius: 4px;
  }
  .msg .meta {
    font-size: 10px; color: var(--text-dim); margin-top: 6px;
    display: flex; gap: 10px; flex-wrap: wrap;
  }
  .msg .meta .fast { color: var(--green); }
  .msg .meta .med { color: var(--orange); }
  .msg .meta .slow { color: var(--red); }
  .thinking {
    align-self: flex-start; padding: 10px 14px; color: var(--text-dim);
    font-size: 12px; display: none; align-items: center; gap: 8px;
  }
  .thinking.active { display: flex; }
  .thinking .dots span {
    display: inline-block; width: 6px; height: 6px;
    background: var(--accent); border-radius: 50%; animation: bounce 1.4s infinite both;
  }
  .thinking .dots span:nth-child(2) { animation-delay: 0.16s; }
  .thinking .dots span:nth-child(3) { animation-delay: 0.32s; }
  @keyframes bounce { 0%,80%,100% { transform: scale(0); } 40% { transform: scale(1); } }
  #input-area {
    background: var(--surface); border-top: 1px solid var(--border);
    padding: 12px 20px; display: flex; gap: 10px; flex-shrink: 0;
  }
  #input-area input {
    flex: 1; background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); padding: 10px 14px; border-radius: 8px;
    font-family: inherit; font-size: 13px; outline: none; transition: border-color 0.2s;
  }
  #input-area input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
  #input-area button {
    background: var(--accent); color: white; border: none; padding: 10px 20px;
    border-radius: 8px; font-family: inherit; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: transform 0.1s;
  }
  #input-area button:hover { transform: scale(1.02); }
  #input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .welcome { text-align: center; padding: 40px 20px; color: var(--text-dim); }
  .welcome h2 { color: var(--accent); font-size: 20px; margin-bottom: 8px; }
  .welcome p { font-size: 12px; line-height: 1.6; max-width: 520px; margin: 0 auto; }
  .welcome .specs { margin-top: 16px; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
  .welcome .spec {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; font-size: 11px;
  }
  .welcome .spec b { color: var(--text); display: block; font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>rai-chat</h1>
  <span class="tag" id="model-tag">Loading...</span>
  <span class="tag" id="size-tag">...</span>
  <span class="tag live" id="status-tag">Ready</span>
</header>
<div class="controls">
  <label>Strategy
    <select id="ctl-strategy">
      <option value="none" selected>None (1 pass)</option>
      <option value="cfg">CFG (2 pass)</option>
      <option value="ensemble">Ensemble (N pass)</option>
      <option value="cfg-ensemble">CFG+Ensemble</option>
      <option value="adaptive">Adaptive</option>
    </select>
  </label>
  <label>Guidance
    <input type="range" id="ctl-guidance" min="1" max="3" step="0.1" value="1.5">
    <span id="ctl-guidance-val">1.5</span>
  </label>
  <label>Temp
    <input type="range" id="ctl-temp" min="0" max="1.5" step="0.05" value="0.5">
    <span id="ctl-temp-val">0.5</span>
  </label>
  <label>Max Tokens
    <input type="number" id="ctl-tokens" value="200" min="1" max="512" style="width:60px">
  </label>
</div>
<div id="chat">
  <div class="welcome">
    <h2>rai-infer Engine</h2>
    <p>GPTQ-4bit, native Rust inference with AVX2 SIMD.<br>
       Pondering v2: Classifier-Free Guidance amplifies contextual signal.<br>
       Supports SmolLM-135M, Mistral-7B, and LLaMA-family models.</p>
    <div class="specs">
      <div class="spec"><b>CFG</b>amplifies context</div>
      <div class="spec"><b>Ensemble</b>noise averaging</div>
      <div class="spec"><b>Adaptive</b>smart compute</div>
      <div class="spec"><b>&lt;100MB</b>total RAM</div>
    </div>
  </div>
</div>
<div class="thinking" id="thinking">
  <div class="dots"><span></span><span></span><span></span></div>
  <span id="thinking-text">Thinking...</span>
</div>
<div id="input-area">
  <input type="text" id="msg-input" placeholder="Say something..." autocomplete="off">
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'), input=document.getElementById('msg-input'),
  btn=document.getElementById('send-btn'), thinking=document.getElementById('thinking'),
  thinkingText=document.getElementById('thinking-text'), statusTag=document.getElementById('status-tag');
document.getElementById('ctl-temp').addEventListener('input',function(){document.getElementById('ctl-temp-val').textContent=this.value});
document.getElementById('ctl-guidance').addEventListener('input',function(){document.getElementById('ctl-guidance-val').textContent=this.value});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!btn.disabled)sendMessage()});

// Fetch model info on load
fetch('/api/info').then(r=>r.json()).then(d=>{
  const h=d.hidden_size, l=d.num_layers;
  document.getElementById('model-tag').textContent=`${h}h/${l}L/${d.vocab_size}v`;
  document.getElementById('size-tag').textContent=`${d.weights_mb.toFixed(0)}MB / ${d.chat_template}`;
}).catch(()=>{});

async function sendMessage(){
  const text=input.value.trim(); if(!text) return;
  addMessage(text,'user'); input.value=''; btn.disabled=true;
  statusTag.textContent='Thinking...'; statusTag.classList.remove('live');
  thinking.classList.add('active');
  const s=document.getElementById('ctl-strategy').value;
  thinkingText.textContent=s==='none'?'Generating...':
    s==='cfg'?'CFG guidance...':s==='adaptive'?'Adaptive pondering...':'Ensemble thinking...';
  try{
    const t0=performance.now();
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        message:text,
        temperature:parseFloat(document.getElementById('ctl-temp').value),
        max_tokens:parseInt(document.getElementById('ctl-tokens').value),
        ponder_strategy:s,
        guidance_scale:parseFloat(document.getElementById('ctl-guidance').value),
      })});
    const d=await res.json(), wall=performance.now()-t0;
    if(d.error){addMessage('Error: '+d.error,'bot')}
    else{
      const tps=d.tok_per_sec||0;
      let sc=tps>5?'fast':tps>2?'med':'slow';
      let m=[];
      m.push(`<span class="${sc}">${tps.toFixed(1)} tok/s</span>`);
      m.push(`<span>${d.tokens} tokens</span>`);
      m.push(`<span>${(d.avg_passes||1).toFixed(1)}x passes</span>`);
      m.push(`<span>${d.strategy||'?'}</span>`);
      if(d.hard_tokens_pct>0) m.push(`<span style="color:var(--accent)">${d.hard_tokens_pct.toFixed(0)}% hard</span>`);
      m.push(`<span>wall ${(wall/1000).toFixed(1)}s</span>`);
      addMessage(d.text||'(empty)','bot',m.join(''));
    }
  }catch(e){addMessage('Error: '+e.message,'bot')}
  btn.disabled=false; statusTag.textContent='Ready'; statusTag.classList.add('live');
  thinking.classList.remove('active'); input.focus();
}
function addMessage(text,role,meta){
  const w=chat.querySelector('.welcome'); if(w) w.remove();
  const div=document.createElement('div'); div.className='msg '+role; div.textContent=text;
  if(meta){const m=document.createElement('div');m.className='meta';m.innerHTML=meta;div.appendChild(m);}
  chat.appendChild(div); chat.scrollTop=chat.scrollHeight;
}
</script>
</body>
</html>
"##;
