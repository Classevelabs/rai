//! rai-chat: Web-based chat UI with pondering strategies.

use std::fmt;
use std::io::Read;
use std::path::PathBuf;
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

const MAX_CHAT_REQUEST_BYTES: usize = 64 * 1024;
const MAX_CHAT_GENERATION_TOKENS: usize = 512;
const MAX_ENSEMBLE_SIZE: usize = 8;
const MAX_ENTROPY_THRESHOLD: f32 = 32.0;

#[derive(Debug)]
struct ChatHttpError {
    status: u16,
    message: String,
}

impl ChatHttpError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    fn payload_too_large() -> Self {
        Self {
            status: 413,
            message: format!("request body exceeds the {MAX_CHAT_REQUEST_BYTES}-byte limit"),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: 403,
            message: message.into(),
        }
    }

    fn unsupported_media_type() -> Self {
        Self {
            status: 415,
            message: "Content-Type must be application/json".to_string(),
        }
    }

    fn internal(error: impl fmt::Display) -> Self {
        eprintln!("chat request failed: {error}");
        Self {
            status: 500,
            message: "internal server error".to_string(),
        }
    }
}

impl fmt::Display for ChatHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

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

#[derive(Debug, Deserialize)]
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

fn parse_chat_request(req_body: &str) -> Result<ChatRequest, ChatHttpError> {
    let request: ChatRequest = serde_json::from_str(req_body)
        .map_err(|error| ChatHttpError::bad_request(format!("invalid JSON request: {error}")))?;

    if request.message.trim().is_empty() {
        return Err(ChatHttpError::bad_request(
            "message must contain at least one non-whitespace character",
        ));
    }

    validate_chat_options(&request)?;

    Ok(request)
}

fn read_request_body(reader: &mut dyn Read) -> Result<String, ChatHttpError> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_CHAT_REQUEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(ChatHttpError::internal)?;

    if bytes.len() > MAX_CHAT_REQUEST_BYTES {
        return Err(ChatHttpError::payload_too_large());
    }

    String::from_utf8(bytes)
        .map_err(|_| ChatHttpError::bad_request("request body must be valid UTF-8"))
}

fn build_ponder(req: &ChatRequest) -> Result<PonderConfig, ChatHttpError> {
    validate_chat_options(req)?;
    let strat = req.ponder_strategy.as_deref().unwrap_or("none");
    let gs = req.guidance_scale.unwrap_or(1.5);
    let en = req.ensemble_n.unwrap_or(3);
    let ns = req.noise_sigma.unwrap_or(0.05);
    let et = req.entropy_threshold.unwrap_or(3.0);

    let config = match strat {
        "none" => PonderConfig::none(),
        "cfg" => PonderConfig::cfg(gs),
        "ensemble" => PonderConfig::ensemble(en, ns),
        "cfg-ensemble" => PonderConfig::cfg_ensemble(gs, en, ns),
        "adaptive" => PonderConfig::adaptive(gs, et),
        _ => unreachable!("strategy was validated"),
    };
    Ok(config)
}

fn validate_chat_options(req: &ChatRequest) -> Result<(), ChatHttpError> {
    validate_optional_f32("temperature", req.temperature, 0.01, 2.0)?;
    if req
        .max_tokens
        .is_some_and(|value| !(1..=MAX_CHAT_GENERATION_TOKENS).contains(&value))
    {
        return Err(ChatHttpError::bad_request(format!(
            "max_tokens must be between 1 and {MAX_CHAT_GENERATION_TOKENS}"
        )));
    }
    if req.ponder_strategy.as_deref().is_some_and(|strategy| {
        !matches!(
            strategy,
            "none" | "cfg" | "ensemble" | "cfg-ensemble" | "adaptive"
        )
    }) {
        return Err(ChatHttpError::bad_request(
            "ponder_strategy must be one of: none, cfg, ensemble, cfg-ensemble, adaptive",
        ));
    }
    validate_optional_f32("guidance_scale", req.guidance_scale, 0.0, 4.0)?;
    if req
        .ensemble_n
        .is_some_and(|value| !(1..=MAX_ENSEMBLE_SIZE).contains(&value))
    {
        return Err(ChatHttpError::bad_request(format!(
            "ensemble_n must be between 1 and {MAX_ENSEMBLE_SIZE}"
        )));
    }
    // Ensemble strategies need at least two passes; reject explicitly instead
    // of silently running a different configuration than requested.
    if matches!(
        req.ponder_strategy.as_deref(),
        Some("ensemble") | Some("cfg-ensemble")
    ) && req.ensemble_n.is_some_and(|value| value < 2)
    {
        return Err(ChatHttpError::bad_request(
            "ensemble strategies require ensemble_n >= 2",
        ));
    }
    validate_optional_f32("noise_sigma", req.noise_sigma, 0.0, 1.0)?;
    validate_optional_f32(
        "entropy_threshold",
        req.entropy_threshold,
        0.0,
        MAX_ENTROPY_THRESHOLD,
    )?;
    Ok(())
}

fn validate_optional_f32(
    name: &str,
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
) -> Result<(), ChatHttpError> {
    if value.is_some_and(|value| !value.is_finite() || value < minimum || value > maximum) {
        return Err(ChatHttpError::bad_request(format!(
            "{name} must be a finite number between {minimum} and {maximum}"
        )));
    }
    Ok(())
}

fn validate_prompt_tokens(
    prompt_tokens: &[usize],
    max_context: usize,
    vocab_size: usize,
) -> Result<(), ChatHttpError> {
    if prompt_tokens.is_empty() {
        return Err(ChatHttpError::bad_request(
            "message produced no tokens with the configured tokenizer",
        ));
    }
    if max_context == 0 {
        return Err(ChatHttpError::internal(
            "model has no usable context window",
        ));
    }
    if prompt_tokens.len() > max_context {
        return Err(ChatHttpError::bad_request(format!(
            "message is too long: {} tokens exceeds the {max_context}-token context window",
            prompt_tokens.len()
        )));
    }
    if prompt_tokens.iter().any(|&token| token >= vocab_size) {
        return Err(ChatHttpError::bad_request(
            "configured tokenizer produced a token outside the model vocabulary",
        ));
    }
    Ok(())
}

fn handle_generate(state: &AppState, chat_req: &ChatRequest) -> Result<String, ChatHttpError> {
    validate_chat_options(chat_req)?;
    let temperature = chat_req.temperature.unwrap_or(0.7);
    let max_tokens = chat_req.max_tokens.unwrap_or(200);
    let ponder_config = build_ponder(chat_req)?;

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
        .map_err(|_| ChatHttpError::bad_request("message could not be tokenized"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    let max_ctx = state
        .max_context
        .min(state.model.config.max_context as usize);
    validate_prompt_tokens(
        &prompt_tokens,
        max_ctx,
        state.model.config.vocab_size as usize,
    )?;
    let mut kv_cache = state
        .model
        .create_kv_cache(max_ctx)
        .map_err(ChatHttpError::internal)?;
    let mut work = InferenceWork::new();
    let mut work2 = InferenceWork::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut all_tokens = prompt_tokens.clone();
    let mut pos = 0;

    // Prefill
    let t_prefill = Instant::now();
    // Leave the final prompt token for the first decode pass. Processing every prompt token
    // here and then processing the last one again at the next position duplicates it.
    for &token_id in &prompt_tokens[..prompt_tokens.len() - 1] {
        let _ = pondered_forward(
            &state.model,
            token_id,
            pos,
            &mut kv_cache,
            &PonderConfig::none(),
            &mut work,
            &mut work2,
            &mut rng,
        )
        .map_err(ChatHttpError::internal)?;
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
        let last_token = all_tokens.last().copied().ok_or_else(|| {
            ChatHttpError::bad_request("message produced no usable prompt tokens")
        })?;
        let (mut logits, metrics) = pondered_forward(
            &state.model,
            last_token,
            pos,
            &mut kv_cache,
            &ponder_config,
            &mut work,
            &mut work2,
            &mut rng,
        )
        .map_err(ChatHttpError::internal)?;
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
                    .is_some_and(|id| next_token == id as usize)
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
    if args.max_context == 0 {
        anyhow::bail!("--max-context must be greater than zero");
    }
    let max_context = args.max_context.min(model.config.max_context as usize);
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
        model.kv_cache_bytes(max_context) as f64 / (1024.0 * 1024.0),
        max_context
    );

    eprintln!("Loading tokenizer: {}", args.tokenizer.display());
    let tokenizer =
        tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;

    let template = ChatTemplate::from_str_arg(&args.chat_template, &tokenizer);
    eprintln!("Chat template: {}", template.display_name());

    let state = AppState {
        model,
        tokenizer,
        max_context,
        template,
    };

    let addr = format!("127.0.0.1:{}", args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("bind: {e}"))?;
    let bound_port = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| anyhow::anyhow!("server did not bind an IP socket"))?
        .port();
    eprintln!("\n  Chat UI: http://localhost:{bound_port}\n");

    // The server is deliberately single-threaded: requests are handled one at
    // a time on this loop, so one generation saturates the CPU without a
    // second request competing for cores, and no synchronization is needed
    // around the model state.
    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        let host = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("Host"))
            .map(|header| header.value.as_str());
        if !host.is_some_and(|host| is_allowed_host(host, bound_port)) {
            respond_json_error(request, &ChatHttpError::forbidden("invalid Host header"));
            continue;
        }

        if method == Method::Post && url == "/api/chat" {
            let origin = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("Origin"))
                .map(|header| header.value.as_str());
            if origin.is_some_and(|origin| !is_allowed_origin(origin, bound_port)) {
                respond_json_error(
                    request,
                    &ChatHttpError::forbidden("cross-origin requests are not allowed"),
                );
                continue;
            }

            let is_json = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("Content-Type"))
                .map(|header| header.value.as_str())
                .is_some_and(|value| {
                    value
                        .split(';')
                        .next()
                        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
                });
            if !is_json {
                respond_json_error(request, &ChatHttpError::unsupported_media_type());
                continue;
            }
        }

        match (method, url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => {
                let resp = Response::from_data(CHAT_HTML.as_bytes().to_vec())
                    .with_header(
                        Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                    )
                    .with_header(
                        Header::from_bytes(
                            "Content-Security-Policy",
                            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                        )
                        .unwrap(),
                    )
                    .with_header(
                        Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap(),
                    );
                let _ = request.respond(resp);
            }
            (Method::Post, "/api/chat") => {
                let body = match read_request_body(request.as_reader()) {
                    Ok(body) => body,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };
                let chat_request = match parse_chat_request(&body) {
                    Ok(chat_request) => chat_request,
                    Err(error) => {
                        respond_json_error(request, &error);
                        continue;
                    }
                };

                match handle_generate(&state, &chat_request) {
                    Ok(json) => {
                        let resp = Response::from_data(json.into_bytes()).with_header(
                            Header::from_bytes("Content-Type", "application/json").unwrap(),
                        );
                        let _ = request.respond(resp);
                    }
                    Err(error) => respond_json_error(request, &error),
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
            (Method::Options, _) => respond_json_error(
                request,
                &ChatHttpError::forbidden("cross-origin preflight is not allowed"),
            ),
            _ => {
                let _ = request.respond(
                    Response::from_data(b"404".to_vec()).with_status_code(StatusCode(404)),
                );
            }
        }
    }
    Ok(())
}

fn respond_json_error(request: tiny_http::Request, error: &ChatHttpError) {
    let body = serde_json::json!({"error": error.to_string()}).to_string();
    let response = Response::from_data(body.into_bytes())
        .with_status_code(StatusCode(error.status))
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
    let _ = request.respond(response);
}

fn is_allowed_host(host: &str, port: u16) -> bool {
    host.eq_ignore_ascii_case(&format!("localhost:{port}")) || host == format!("127.0.0.1:{port}")
}

fn is_allowed_origin(origin: &str, port: u16) -> bool {
    origin.eq_ignore_ascii_case(&format!("http://localhost:{port}"))
        || origin == format!("http://127.0.0.1:{port}")
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
    <input type="range" id="ctl-temp" min="0.05" max="2" step="0.05" value="0.5">
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn empty_prompt_returns_bad_request_instead_of_panicking() {
        let error = parse_chat_request(r#"{"message":"   "}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("non-whitespace"));
    }

    #[test]
    fn malformed_json_returns_bad_request() {
        let error = parse_chat_request("not-json").unwrap_err();
        assert_eq!(error.status, 400);
    }

    #[test]
    fn oversized_body_returns_payload_too_large() {
        let mut body = Cursor::new(vec![b'x'; MAX_CHAT_REQUEST_BYTES + 1]);
        let error = read_request_body(&mut body).unwrap_err();
        assert_eq!(error.status, 413);
    }

    #[test]
    fn host_and_origin_checks_reject_dns_rebinding() {
        assert!(is_allowed_host("localhost:8090", 8090));
        assert!(is_allowed_origin("http://127.0.0.1:8090", 8090));
        assert!(!is_allowed_host("attacker.example:8090", 8090));
        assert!(!is_allowed_origin("https://attacker.example", 8090));
    }

    #[test]
    fn expensive_request_values_are_clamped() {
        let request = ChatRequest {
            message: "hello".to_string(),
            temperature: Some(100.0),
            max_tokens: Some(usize::MAX),
            ponder_strategy: Some("ensemble".to_string()),
            guidance_scale: Some(100.0),
            ensemble_n: Some(usize::MAX),
            noise_sigma: Some(100.0),
            entropy_threshold: None,
        };
        let error = build_ponder(&request).unwrap_err();
        assert_eq!(error.status, 400);
    }

    #[test]
    fn unknown_ponder_strategy_is_rejected() {
        let error =
            parse_chat_request(r#"{"message":"hello","ponder_strategy":"typo"}"#).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("ponder_strategy"));
    }

    #[test]
    fn invalid_numeric_options_are_rejected() {
        for body in [
            r#"{"message":"hello","temperature":0}"#,
            r#"{"message":"hello","max_tokens":0}"#,
            r#"{"message":"hello","guidance_scale":5}"#,
            r#"{"message":"hello","ensemble_n":9}"#,
            r#"{"message":"hello","noise_sigma":-1}"#,
            r#"{"message":"hello","entropy_threshold":33}"#,
        ] {
            assert_eq!(parse_chat_request(body).unwrap_err().status, 400, "{body}");
        }
    }

    #[test]
    fn over_context_and_out_of_vocabulary_prompts_are_rejected() {
        let error = validate_prompt_tokens(&[1, 2, 3], 2, 10).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("context window"));

        let error = validate_prompt_tokens(&[1, 10], 2, 10).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("vocabulary"));
    }

    #[test]
    fn internal_failures_are_sanitized() {
        let error = ChatHttpError::internal(r#"failed to open C:\\secret\\model.raimodel"#);
        assert_eq!(error.status, 500);
        assert_eq!(error.message, "internal server error");
    }
}
