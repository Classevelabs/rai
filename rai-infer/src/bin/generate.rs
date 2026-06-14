//! rai-generate: CLI for edge inference with pondering strategies.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use rai_infer::model::{BatchScratch, InferenceWork, RaiModel};
use rai_infer::ponder::{pondered_forward, PonderConfig, PonderStrategy};
use rai_infer::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};
use rai_infer::self_speculative::{SelfSpecConfig, SelfSpecDecoder};
use rai_infer::speculative::{SpeculativeConfig, SpeculativeDecoder};

#[derive(Parser, Debug)]
#[command(name = "rai-generate", about = "Edge inference with pondering")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value = "64")]
    max_tokens: usize,
    #[arg(long, default_value = "0.7")]
    temperature: f32,
    #[arg(long, default_value = "40")]
    top_k: usize,
    #[arg(long, default_value = "0.9")]
    top_p: f32,
    #[arg(long, default_value = "1.1")]
    repetition_penalty: f32,
    /// Pondering strategy: none, cfg, ensemble, cfg-ensemble, adaptive
    #[arg(long, default_value = "none")]
    ponder_strategy: String,
    /// CFG guidance scale (1.0 = no effect, 1.5 = recommended)
    #[arg(long, default_value = "1.5")]
    guidance_scale: f32,
    /// Number of ensemble passes
    #[arg(long, default_value = "3")]
    ensemble_n: usize,
    /// Noise sigma for ensemble
    #[arg(long, default_value = "0.05")]
    noise_sigma: f32,
    /// Entropy threshold for adaptive strategy
    #[arg(long, default_value = "3.0")]
    entropy_threshold: f32,
    #[arg(long, default_value = "512")]
    max_context: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long, default_value = "false")]
    verbose: bool,
    /// Chat template: auto, none, few-shot, mistral, llama3
    #[arg(long, default_value = "none")]
    chat_template: String,
    /// Draft model for speculative decoding (e.g. smollm-135m-q4.raimodel)
    #[arg(long)]
    draft: Option<PathBuf>,
    /// Number of draft tokens per speculative step
    #[arg(long, default_value = "6")]
    draft_k: usize,
    /// Self-speculative: use N layers as draft (0 = disabled)
    #[arg(long, default_value = "0")]
    self_spec_layers: usize,
    /// Self-speculative: number of draft tokens per step
    #[arg(long, default_value = "8")]
    self_spec_k: usize,
    /// Self-speculative: layer skip mode (covers full depth instead of first-N)
    #[arg(long)]
    self_spec_skip: bool,
}

fn build_ponder_config(args: &Args) -> PonderConfig {
    match args.ponder_strategy.as_str() {
        "none" => PonderConfig::none(),
        "cfg" => PonderConfig::cfg(args.guidance_scale),
        "ensemble" => PonderConfig::ensemble(args.ensemble_n, args.noise_sigma),
        "cfg-ensemble" => {
            PonderConfig::cfg_ensemble(args.guidance_scale, args.ensemble_n, args.noise_sigma)
        }
        "adaptive" => PonderConfig::adaptive(args.guidance_scale, args.entropy_threshold),
        other => {
            eprintln!("Warning: unknown strategy '{other}', using 'none'");
            PonderConfig::none()
        }
    }
}

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();
    let args = Args::parse();

    eprintln!("Loading model: {}", args.model.display());
    let t_load = Instant::now();
    let model = RaiModel::load(&args.model).context("loading model")?;
    let cfg = &model.config;
    eprintln!(
        "Model loaded in {:.1}ms (hidden={}, layers={}, heads={}/{}kv, inter={}, vocab={})",
        t_load.elapsed().as_secs_f64() * 1000.0,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.intermediate_size,
        cfg.vocab_size
    );
    eprintln!(
        "Weights: {:.1} MB",
        model.file_size() as f64 / (1024.0 * 1024.0)
    );

    eprintln!("Loading tokenizer: {}", args.tokenizer.display());
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let ponder_config = build_ponder_config(&args);
    let sampler_config = SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
    };

    let max_ctx = args.max_context.min(model.config.max_context as usize);
    let kv_bytes = model.kv_cache_bytes(max_ctx);
    eprintln!(
        "KV cache: {:.1} MB (max_ctx={})",
        kv_bytes as f64 / (1024.0 * 1024.0),
        max_ctx
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut all_tokens = prompt_tokens.clone();

    print!("{}", args.prompt);
    io::stdout().flush()?;

    // Helper: check if token is EOS
    let is_eos = |tok: usize| -> bool {
        ["</s>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"]
            .iter()
            .any(|s| {
                tokenizer
                    .token_to_id(s)
                    .is_some_and(|id| tok == id as usize)
            })
    };

    // Helper: print newly generated text with correct spacing
    let print_new_text = |all_tokens: &[usize],
                          prompt_len: usize,
                          prev_text_len: usize,
                          tokenizer: &Tokenizer|
     -> usize {
        let gen_ids: Vec<u32> = all_tokens[prompt_len..].iter().map(|&t| t as u32).collect();
        let full_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();
        let new_chars = full_text.len().saturating_sub(prev_text_len);
        if new_chars > 0 {
            print!("{}", &full_text[full_text.len() - new_chars..]);
            let _ = io::stdout().flush();
        }
        full_text.len()
    };

    if args.self_spec_layers > 0 {
        // === SELF-SPECULATIVE DECODING ===
        let total_layers = model.config.num_layers as usize;
        let draft_layers = args.self_spec_layers.min(total_layers);

        let spec_config = if args.self_spec_skip {
            eprintln!(
                "Self-speculative (layer-skip): {} of {} layers, K={}",
                draft_layers, total_layers, args.self_spec_k
            );
            SelfSpecConfig::layer_skip(
                total_layers,
                draft_layers,
                args.self_spec_k,
                sampler_config.clone(),
            )
        } else {
            eprintln!(
                "Self-speculative (early-exit): first {} of {} layers, K={}",
                draft_layers, total_layers, args.self_spec_k
            );
            SelfSpecConfig::early_exit(draft_layers, args.self_spec_k, sampler_config.clone())
        };
        eprintln!("Draft layers: {:?}", spec_config.draft_layer_indices);
        let mut decoder = SelfSpecDecoder::new(&model, max_ctx);

        // Prefill
        let t_prefill = Instant::now();
        let mut pos = decoder.prefill(&prompt_tokens)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\nPrefill: {} tokens in {:.1}ms ({:.1} tok/s)",
            prompt_tokens.len(),
            prefill_ms,
            prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
        );

        // Self-speculative decode
        let t_decode = Instant::now();
        let mut tokens_generated = 0;
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut prev_text_len = 0;

        while tokens_generated < args.max_tokens && pos < max_ctx {
            let last_token = *all_tokens.last().unwrap();
            let (new_tokens, metrics) =
                decoder.step(pos, last_token, &spec_config, &all_tokens, &mut rng)?;

            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;

            let mut hit_eos = false;
            for &tok in &new_tokens {
                if is_eos(tok) {
                    hit_eos = true;
                    break;
                }
                all_tokens.push(tok);
                pos += 1;
                tokens_generated += 1;
                if tokens_generated >= args.max_tokens {
                    break;
                }
            }

            prev_text_len =
                print_new_text(&all_tokens, prompt_tokens.len(), prev_text_len, &tokenizer);

            if args.verbose {
                eprint!("[self:{}d/{}a]", metrics.drafted, metrics.accepted);
            }

            if hit_eos {
                break;
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decode_tps = tokens_generated as f64 / (decode_ms / 1000.0);
        let accept_rate = if total_drafted > 0 {
            total_accepted as f64 / total_drafted as f64
        } else {
            0.0
        };

        println!();
        eprintln!("\n--- Self-Speculative Stats ---");
        eprintln!("Tokens: {tokens_generated}, {decode_tps:.2} tok/s");
        eprintln!(
            "Draft layers: {draft_layers}/{}, K={}",
            model.config.num_layers, args.self_spec_k
        );
        eprintln!(
            "Drafted: {total_drafted}, Accepted: {total_accepted}, Rate: {:.1}%",
            accept_rate * 100.0
        );
        eprintln!(
            "Avg tokens/step: {:.1}",
            if total_drafted > 0 {
                (total_accepted as f64 + (total_drafted as f64 / args.self_spec_k as f64))
                    / (total_drafted as f64 / args.self_spec_k as f64)
            } else {
                0.0
            }
        );
    } else if let Some(draft_path) = &args.draft {
        // === SPECULATIVE DECODING ===
        eprintln!("Loading draft model: {}", draft_path.display());
        let draft = RaiModel::load(draft_path).context("loading draft model")?;
        eprintln!(
            "Draft: hidden={}, layers={}, {:.1} MB",
            draft.config.hidden_size,
            draft.config.num_layers,
            draft.file_size() as f64 / (1024.0 * 1024.0)
        );

        let spec_config = SpeculativeConfig {
            draft_k: args.draft_k,
            sampler: sampler_config.clone(),
        };
        let mut decoder = SpeculativeDecoder::new(&draft, &model, max_ctx)?;

        // Prefill both models
        let t_prefill = Instant::now();
        let mut pos = decoder.prefill(&prompt_tokens)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\nPrefill: {} tokens in {:.1}ms ({:.1} tok/s)",
            prompt_tokens.len(),
            prefill_ms,
            prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
        );

        // Speculative decode
        let t_decode = Instant::now();
        let mut tokens_generated = 0;
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut prev_text_len = 0;

        while tokens_generated < args.max_tokens && pos < max_ctx {
            let last_token = *all_tokens.last().unwrap();
            let (new_tokens, metrics) =
                decoder.step(pos, last_token, &spec_config, &all_tokens, &mut rng)?;

            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;

            let mut hit_eos = false;
            for &tok in &new_tokens {
                if is_eos(tok) {
                    hit_eos = true;
                    break;
                }
                all_tokens.push(tok);
                pos += 1;
                tokens_generated += 1;
                if tokens_generated >= args.max_tokens {
                    break;
                }
            }

            prev_text_len =
                print_new_text(&all_tokens, prompt_tokens.len(), prev_text_len, &tokenizer);

            if args.verbose {
                eprint!("[spec:{}d/{}a]", metrics.drafted, metrics.accepted);
            }

            if hit_eos {
                break;
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decode_tps = tokens_generated as f64 / (decode_ms / 1000.0);
        let accept_rate = if total_drafted > 0 {
            total_accepted as f64 / total_drafted as f64
        } else {
            0.0
        };

        println!();
        eprintln!("\n--- Speculative Stats ---");
        eprintln!("Tokens: {tokens_generated}, {decode_tps:.2} tok/s");
        eprintln!(
            "Drafted: {total_drafted}, Accepted: {total_accepted}, Rate: {:.1}%",
            accept_rate * 100.0
        );
        eprintln!(
            "Avg accepted/step: {:.1}",
            if total_drafted > 0 {
                total_accepted as f64 / (total_drafted as f64 / args.draft_k as f64)
            } else {
                0.0
            }
        );
    } else {
        // === NORMAL DECODING ===
        let mut kv_cache = model.create_kv_cache(max_ctx);
        let mut work = InferenceWork::new();
        let mut work2 = InferenceWork::new();

        // Batched prefill: process prompt tokens 0..N-2 in one pass.
        // The last prompt token is left for the decode loop, avoiding
        // duplicate KV entries (same fix as speculative.rs prefill).
        let t_prefill = Instant::now();
        let hs = model.config.hidden_size as usize;
        let n_to_prefill = if prompt_tokens.len() > 1 {
            (prompt_tokens.len() - 1).min(max_ctx.saturating_sub(1))
        } else {
            0
        };
        let mut pos = n_to_prefill;

        if n_to_prefill > 1 {
            // Batched prefill: embed tokens 0..N-2, run through all layers in one pass
            let positions: Vec<usize> = (0..n_to_prefill).collect();
            let mut hiddens = vec![0.0f32; n_to_prefill * hs];
            for (i, &tok) in prompt_tokens[..n_to_prefill].iter().enumerate() {
                model.embed_token(tok, &mut hiddens[i * hs..(i + 1) * hs])?;
            }
            let mut batch_scratch = BatchScratch::new();
            model.forward_batch(&mut hiddens, &positions, &mut kv_cache, &mut batch_scratch)?;
        } else if n_to_prefill == 1 {
            // Two-token prompt: prefill first token, decode handles second
            let _ = pondered_forward(
                &model,
                prompt_tokens[0],
                0,
                &mut kv_cache,
                &PonderConfig::none(),
                &mut work,
                &mut work2,
                &mut rng,
            )?;
        }
        // Single-token prompt: n_to_prefill=0, pos=0, decode loop handles it.
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\nPrefill: {} tokens in {:.1}ms ({:.1} tok/s)",
            prompt_tokens.len(),
            prefill_ms,
            prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
        );

        // Decode
        let t_decode = Instant::now();
        let mut tokens_generated = 0;
        let mut total_passes = 0;
        let mut hard_tokens = 0;
        let mut prev_text_len = 0;

        for _ in 0..args.max_tokens {
            if pos >= max_ctx {
                break;
            }

            let last_token = *all_tokens.last().unwrap();
            let (mut logits, metrics) = pondered_forward(
                &model,
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

            if args.verbose && metrics.forward_passes > 1 {
                eprint!("[{}:{}p", metrics.strategy_used, metrics.forward_passes);
                if let Some(e) = metrics.entropy {
                    eprint!(" e={:.1}", e);
                }
                eprint!("]");
            }

            apply_repetition_penalty(&mut logits, &all_tokens, sampler_config.repetition_penalty);
            let next_token = sample_token(&mut logits, &sampler_config, &mut rng);
            work.scratch.logits = logits;

            if is_eos(next_token) {
                break;
            }

            all_tokens.push(next_token);
            pos += 1;
            tokens_generated += 1;
            prev_text_len =
                print_new_text(&all_tokens, prompt_tokens.len(), prev_text_len, &tokenizer);
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decode_tps = tokens_generated as f64 / (decode_ms / 1000.0);
        let avg_passes = total_passes as f64 / tokens_generated.max(1) as f64;

        println!();
        eprintln!("\n--- Stats ---");
        eprintln!("Tokens: {tokens_generated}, {decode_tps:.2} tok/s");
        eprintln!("Passes: {total_passes} total, {avg_passes:.1} avg/token");
        if ponder_config.strategy == PonderStrategy::Adaptive {
            eprintln!(
                "Hard tokens: {hard_tokens}/{tokens_generated} ({:.0}%)",
                100.0 * hard_tokens as f64 / tokens_generated.max(1) as f64
            );
        }
        eprintln!(
            "KV cache: {:.1} MB",
            kv_cache.memory_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(())
}
