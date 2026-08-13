//! `rai run` — generate text from a `.raimodel` on the CPU.
//!
//! This is the body of the old `rai-generate` binary, moved into the library so
//! that one `rai` binary can host it and so that argument validation is unit
//! testable.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use rand::SeedableRng;
use tokenizers::Tokenizer;

use crate::chat_template::ChatTemplate;
use crate::cli::{load_tokenizer, resolve_tokenizer};
use crate::lookup::{LookupConfig, LookupDecoder};
use crate::model::{BatchScratch, InferenceWork, RaiModel};
use crate::ponder::{pondered_forward, PonderConfig, PonderStrategy};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};
use crate::speculative::{SpeculativeConfig, SpeculativeDecoder};

pub const MAX_ENSEMBLE_SIZE: usize = 8;
pub const MAX_SPECULATIVE_TOKENS: usize = 32;
pub const MAX_LOOKUP_NGRAM: usize = 16;

/// Everything except the model and tokenizer paths, so that `rai run` (model as
/// a positional, tokenizer optional) and the deprecated `rai-generate`
/// (`--model` / `--tokenizer`, both required) can share one definition instead
/// of drifting apart.
#[derive(clap::Args, Debug, Clone)]
pub struct GenerationArgs {
    /// The prompt to complete
    #[arg(long)]
    pub prompt: String,
    /// Maximum number of tokens to generate
    #[arg(long, default_value = "64")]
    pub max_tokens: usize,
    /// Sampling temperature; 0 always takes the most likely token
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
    /// Keep only the K most likely tokens (0 disables)
    #[arg(long, default_value = "40")]
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability sums to P
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,
    /// Penalty applied to already-generated tokens (1.0 disables)
    #[arg(long, default_value = "1.1")]
    pub repetition_penalty: f32,
    /// Pondering strategy: none, cfg, ensemble, cfg-ensemble, adaptive
    #[arg(long, default_value = "none")]
    pub ponder_strategy: String,
    /// CFG guidance scale (1.0 = no effect, 1.5 = recommended)
    #[arg(long, default_value = "1.5")]
    pub guidance_scale: f32,
    /// Number of ensemble passes
    #[arg(long, default_value = "3")]
    pub ensemble_n: usize,
    /// Noise sigma for ensemble
    #[arg(long, default_value = "0.05")]
    pub noise_sigma: f32,
    /// Entropy threshold for adaptive strategy
    #[arg(long, default_value = "3.0")]
    pub entropy_threshold: f32,
    /// Context window to allocate the KV cache for, in tokens
    #[arg(long, default_value = "512")]
    pub max_context: usize,
    /// Random seed; the same seed and settings reproduce the same text
    #[arg(long, default_value = "42")]
    pub seed: u64,
    /// Print per-step decoding diagnostics to stderr
    #[arg(long, default_value = "false")]
    pub verbose: bool,
    /// Chat template: auto, none, few-shot, mistral, llama3, chatml, zephyr
    #[arg(long, default_value = "none")]
    pub chat_template: String,
    /// Draft model for speculative decoding (e.g. smollm-135m-q4.raimodel)
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Number of draft tokens per speculative step
    #[arg(long, default_value = "6")]
    pub draft_k: usize,
    /// Prompt-lookup speculation: draft up to N tokens copied from the context
    /// (0 = disabled). No draft model and no draft forward pass. NOTE: measured
    /// on TinyLlama-1.1B-q4 this is still ~0.7-0.9x baseline because batched
    /// verification barely amortises weight reads in this engine; small K (1-2)
    /// is closest to break-even and large K is much worse.
    #[arg(long, default_value = "0")]
    pub lookup_k: usize,
    /// Prompt-lookup: longest suffix n-gram to match first
    #[arg(long, default_value = "3")]
    pub lookup_ngram: usize,
    /// Prompt-lookup: shortest suffix n-gram to fall back to. Raising this to 2
    /// or 3 suppresses weak single-token matches, which is what makes the
    /// non-repetitive worst case expensive.
    #[arg(long, default_value = "1")]
    pub lookup_min_ngram: usize,
}

#[derive(clap::Args, Debug, Clone)]
pub struct RunArgs {
    /// The .raimodel file to run
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,
    /// Path to tokenizer.json; defaults to the one beside the model
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,
    #[command(flatten)]
    pub generation: GenerationArgs,
}

fn build_ponder_config(args: &GenerationArgs) -> Result<PonderConfig> {
    validate_args(args)?;
    Ok(match args.ponder_strategy.as_str() {
        "none" => PonderConfig::none(),
        "cfg" => PonderConfig::cfg(args.guidance_scale),
        "ensemble" => PonderConfig::ensemble(args.ensemble_n, args.noise_sigma),
        "cfg-ensemble" => {
            PonderConfig::cfg_ensemble(args.guidance_scale, args.ensemble_n, args.noise_sigma)
        }
        "adaptive" => PonderConfig::adaptive(args.guidance_scale, args.entropy_threshold),
        _ => unreachable!("ponder strategy was validated"),
    })
}

pub fn validate_args(args: &GenerationArgs) -> Result<()> {
    ensure!(!args.prompt.trim().is_empty(), "--prompt must not be empty");
    ensure!(
        args.max_tokens > 0,
        "--max-tokens must be greater than zero"
    );
    ensure!(
        args.max_context > 0,
        "--max-context must be greater than zero"
    );
    validate_f32("--temperature", args.temperature, 0.0, 2.0)?;
    validate_f32("--top-p", args.top_p, f32::MIN_POSITIVE, 1.0)?;
    validate_f32("--repetition-penalty", args.repetition_penalty, 0.01, 4.0)?;
    validate_f32("--guidance-scale", args.guidance_scale, 0.0, 4.0)?;
    validate_f32("--noise-sigma", args.noise_sigma, 0.0, 1.0)?;
    validate_f32("--entropy-threshold", args.entropy_threshold, 0.0, 32.0)?;
    ensure!(
        matches!(
            args.ponder_strategy.as_str(),
            "none" | "cfg" | "ensemble" | "cfg-ensemble" | "adaptive"
        ),
        "unknown --ponder-strategy '{}'; expected none, cfg, ensemble, cfg-ensemble, or adaptive",
        args.ponder_strategy
    );
    ensure!(
        matches!(
            args.chat_template.as_str(),
            "auto" | "none" | "few-shot" | "mistral" | "llama3" | "chatml" | "zephyr"
        ),
        "unknown --chat-template '{}'; expected auto, none, few-shot, mistral, llama3, chatml, or zephyr",
        args.chat_template
    );
    ensure!(
        (1..=MAX_ENSEMBLE_SIZE).contains(&args.ensemble_n),
        "--ensemble-n must be between 1 and {MAX_ENSEMBLE_SIZE}"
    );
    if matches!(args.ponder_strategy.as_str(), "ensemble" | "cfg-ensemble") {
        ensure!(
            args.ensemble_n >= 2,
            "ensemble strategies require --ensemble-n >= 2"
        );
    }
    // Each speculative path owns the KV cache and the decode loop, so at most
    // one may be selected.
    ensure!(
        !(args.draft.is_some() && args.lookup_k > 0),
        "--draft and --lookup-k are mutually exclusive"
    );
    if args.draft.is_some() {
        ensure!(
            (1..=MAX_SPECULATIVE_TOKENS).contains(&args.draft_k),
            "--draft-k must be between 1 and {MAX_SPECULATIVE_TOKENS}"
        );
    }
    if args.lookup_k > 0 {
        ensure!(
            args.lookup_k <= MAX_SPECULATIVE_TOKENS,
            "--lookup-k must be between 1 and {MAX_SPECULATIVE_TOKENS}"
        );
        ensure!(
            (1..=MAX_LOOKUP_NGRAM).contains(&args.lookup_ngram),
            "--lookup-ngram must be between 1 and {MAX_LOOKUP_NGRAM}"
        );
        ensure!(
            args.lookup_min_ngram >= 1 && args.lookup_min_ngram <= args.lookup_ngram,
            "--lookup-min-ngram must be between 1 and --lookup-ngram ({})",
            args.lookup_ngram
        );
    }
    if args.draft.is_some() || args.lookup_k > 0 {
        ensure!(
            args.temperature > 1e-6
                && args.top_k == 0
                && args.top_p == 1.0
                && (args.repetition_penalty - 1.0).abs() < f32::EPSILON,
            "speculative decoding currently requires --temperature > 0, --top-k 0, --top-p 1, and --repetition-penalty 1 so verification uses the exact sampled distribution"
        );
    }
    Ok(())
}

fn validate_f32(name: &str, value: f32, minimum: f32, maximum: f32) -> Result<()> {
    ensure!(
        value.is_finite() && value >= minimum && value <= maximum,
        "{name} must be a finite number between {minimum} and {maximum}"
    );
    Ok(())
}

fn incremental_suffix<'a>(previous: &str, current: &'a str) -> &'a str {
    let mut common_bytes = 0usize;
    for (before, after) in previous.chars().zip(current.chars()) {
        if before != after {
            break;
        }
        common_bytes += after.len_utf8();
    }
    &current[common_bytes..]
}

pub fn run(args: &RunArgs) -> Result<()> {
    crate::gemm::configure_thread_pool();
    let generation = &args.generation;
    validate_args(generation)?;
    let tokenizer_path = resolve_tokenizer(&args.model, args.tokenizer.as_deref())?;

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

    eprintln!("Loading tokenizer: {}", tokenizer_path.display());
    let tokenizer = load_tokenizer(&tokenizer_path)?;

    let template = ChatTemplate::from_str_arg(&generation.chat_template, &tokenizer);
    let formatted_prompt = template.format_prompt(&generation.prompt);
    let encoding = tokenizer
        .encode(formatted_prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let ponder_config = build_ponder_config(generation)?;
    let sampler_config = SamplerConfig {
        temperature: generation.temperature,
        top_k: generation.top_k,
        top_p: generation.top_p,
        repetition_penalty: generation.repetition_penalty,
    };

    let max_ctx = generation
        .max_context
        .min(model.config.max_context as usize);
    ensure!(
        !prompt_tokens.is_empty(),
        "prompt produced no tokenizer tokens"
    );
    ensure!(
        prompt_tokens.len() <= max_ctx,
        "prompt has {} tokens but the context window is {max_ctx}",
        prompt_tokens.len()
    );
    ensure!(
        prompt_tokens
            .iter()
            .all(|&token| token < model.config.vocab_size as usize),
        "tokenizer produced a token outside the model vocabulary"
    );
    let kv_bytes = model.kv_cache_bytes(max_ctx);
    eprintln!(
        "KV cache: {:.1} MB (max_ctx={})",
        kv_bytes as f64 / (1024.0 * 1024.0),
        max_ctx
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(generation.seed);
    let mut all_tokens = prompt_tokens.clone();

    print!("{}", generation.prompt);
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

    // Helper: print newly generated text with correct spacing. A trailing
    // U+FFFD usually means a multi-byte codepoint is split across tokens;
    // hold it back so the resolved character prints on a later call instead
    // of replacement-garbage reaching the terminal.
    let print_new_text = |all_tokens: &[usize],
                          prompt_len: usize,
                          previous_text: &mut String,
                          tokenizer: &Tokenizer| {
        let gen_ids: Vec<u32> = all_tokens[prompt_len..].iter().map(|&t| t as u32).collect();
        let full_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();
        let mut suffix = incremental_suffix(previous_text, &full_text);
        let mut held_bytes = 0usize;
        while suffix.ends_with('\u{FFFD}') {
            suffix = &suffix[..suffix.len() - '\u{FFFD}'.len_utf8()];
            held_bytes += '\u{FFFD}'.len_utf8();
        }
        if !suffix.is_empty() {
            print!("{suffix}");
            let _ = io::stdout().flush();
        }
        *previous_text = full_text[..full_text.len() - held_bytes].to_string();
    };

    // Helper: once decoding ends, print anything the U+FFFD holdback withheld
    // (a final replacement character is genuine and should be shown).
    let flush_held_text = |all_tokens: &[usize],
                           prompt_len: usize,
                           previous_text: &mut String,
                           tokenizer: &Tokenizer| {
        let gen_ids: Vec<u32> = all_tokens[prompt_len..].iter().map(|&t| t as u32).collect();
        let full_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();
        let suffix = incremental_suffix(previous_text, &full_text);
        if !suffix.is_empty() {
            print!("{suffix}");
            let _ = io::stdout().flush();
        }
        *previous_text = full_text;
    };

    if let Some(draft_path) = &generation.draft {
        // === SPECULATIVE DECODING ===
        eprintln!("Loading draft model: {}", draft_path.display());
        let draft = RaiModel::load(draft_path).context("loading draft model")?;
        ensure!(
            prompt_tokens
                .iter()
                .all(|&token| token < draft.config.vocab_size as usize),
            "tokenizer produced a token outside the draft model vocabulary"
        );
        eprintln!(
            "Draft: hidden={}, layers={}, {:.1} MB",
            draft.config.hidden_size,
            draft.config.num_layers,
            draft.file_size() as f64 / (1024.0 * 1024.0)
        );

        let spec_config = SpeculativeConfig {
            draft_k: generation.draft_k,
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
        let mut previous_text = String::new();

        while tokens_generated < generation.max_tokens && pos < max_ctx {
            let last_token = *all_tokens.last().unwrap();
            let (new_tokens, metrics) = decoder.step(pos, last_token, &spec_config, &mut rng)?;
            ensure!(
                !new_tokens.is_empty(),
                "speculative decoder made no progress"
            );

            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;

            let mut hit_eos = false;
            let remaining = (max_ctx - pos).min(generation.max_tokens - tokens_generated);
            for &tok in new_tokens.iter().take(remaining) {
                if is_eos(tok) {
                    hit_eos = true;
                    break;
                }
                all_tokens.push(tok);
                pos += 1;
                tokens_generated += 1;
                if tokens_generated >= generation.max_tokens {
                    break;
                }
            }

            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

            if generation.verbose {
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

        flush_held_text(
            &all_tokens,
            prompt_tokens.len(),
            &mut previous_text,
            &tokenizer,
        );
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
                total_accepted as f64 / (total_drafted as f64 / generation.draft_k as f64)
            } else {
                0.0
            }
        );
    } else if generation.lookup_k > 0 {
        // === PROMPT-LOOKUP (N-GRAM) SPECULATIVE DECODING ===
        // The draft is copied out of the context, so there is no draft model
        // and no draft forward pass: a miss costs about one ordinary step and
        // a hit produces several tokens for the price of one.
        let lookup_config = LookupConfig {
            max_draft: generation.lookup_k,
            max_ngram: generation.lookup_ngram,
            min_ngram: generation.lookup_min_ngram,
            sampler: sampler_config.clone(),
        };
        eprintln!(
            "Prompt-lookup: K={}, n-gram {}..{} (draft copied from context, no draft model)",
            generation.lookup_k, generation.lookup_ngram, generation.lookup_min_ngram
        );
        let mut decoder = LookupDecoder::new(&model, max_ctx)?;

        let t_prefill = Instant::now();
        let mut pos = decoder.prefill(&prompt_tokens)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\nPrefill: {} tokens in {:.1}ms ({:.1} tok/s)",
            prompt_tokens.len(),
            prefill_ms,
            prompt_tokens.len() as f64 / (prefill_ms / 1000.0)
        );

        let t_decode = Instant::now();
        let mut tokens_generated = 0;
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut steps = 0usize;
        let mut steps_with_draft = 0usize;
        let mut previous_text = String::new();

        while tokens_generated < generation.max_tokens && pos < max_ctx {
            // The decoder searches `all_tokens` for the draft, so the caller
            // owes it the invariant that the token at `pos` is the last one.
            ensure!(
                pos + 1 == all_tokens.len(),
                "prompt-lookup context and decode position went out of sync"
            );
            let (new_tokens, metrics) = decoder.step(pos, &all_tokens, &lookup_config, &mut rng)?;
            ensure!(
                !new_tokens.is_empty(),
                "prompt-lookup decoder made no progress"
            );

            steps += 1;
            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;
            if metrics.matched_ngram.is_some() {
                steps_with_draft += 1;
            }

            let mut hit_eos = false;
            let remaining = (max_ctx - pos).min(generation.max_tokens - tokens_generated);
            for &tok in new_tokens.iter().take(remaining) {
                if is_eos(tok) {
                    hit_eos = true;
                    break;
                }
                all_tokens.push(tok);
                pos += 1;
                tokens_generated += 1;
                if tokens_generated >= generation.max_tokens {
                    break;
                }
            }

            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

            if generation.verbose {
                eprint!(
                    "[lookup:{}d/{}a/n{}]",
                    metrics.drafted,
                    metrics.accepted,
                    metrics.matched_ngram.unwrap_or(0)
                );
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

        flush_held_text(
            &all_tokens,
            prompt_tokens.len(),
            &mut previous_text,
            &tokenizer,
        );
        println!();
        eprintln!("\n--- Prompt-Lookup Stats ---");
        eprintln!("Tokens: {tokens_generated}, {decode_tps:.2} tok/s");
        eprintln!(
            "Drafted: {total_drafted}, Accepted: {total_accepted}, Rate: {:.1}%",
            accept_rate * 100.0
        );
        eprintln!(
            "Steps: {steps} ({steps_with_draft} with a draft, {:.1}%), {:.2} tokens/step",
            100.0 * steps_with_draft as f64 / steps.max(1) as f64,
            tokens_generated as f64 / steps.max(1) as f64
        );
    } else {
        // === NORMAL DECODING ===
        let mut kv_cache = model.create_kv_cache(max_ctx)?;
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
        let mut previous_text = String::new();

        for _ in 0..generation.max_tokens {
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

            if generation.verbose && metrics.forward_passes > 1 {
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
            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decode_tps = tokens_generated as f64 / (decode_ms / 1000.0);
        let avg_passes = total_passes as f64 / tokens_generated.max(1) as f64;

        flush_held_text(
            &all_tokens,
            prompt_tokens.len(),
            &mut previous_text,
            &tokenizer,
        );
        println!();
        if tokens_generated == 0 {
            // Silence here reads as a broken tool. The usual cause is an
            // instruction-tuned model given a bare prompt: it ends the
            // sequence immediately because nothing marks a turn.
            eprintln!(
                "\nNo tokens generated: the model emitted end-of-sequence first. \
                 If this is an instruction-tuned model, pass the matching \
                 --chat-template (auto, mistral, llama3, chatml, zephyr, few-shot)."
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_args() -> GenerationArgs {
        GenerationArgs {
            prompt: "hello".to_string(),
            max_tokens: 16,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ponder_strategy: "none".to_string(),
            guidance_scale: 1.5,
            ensemble_n: 3,
            noise_sigma: 0.05,
            entropy_threshold: 3.0,
            max_context: 512,
            seed: 42,
            verbose: false,
            chat_template: "none".to_string(),
            draft: None,
            draft_k: 6,
            lookup_k: 0,
            lookup_ngram: 3,
            lookup_min_ngram: 1,
        }
    }

    fn exact_sampling(args: &mut GenerationArgs) {
        args.top_k = 0;
        args.top_p = 1.0;
        args.repetition_penalty = 1.0;
    }

    #[test]
    fn rejects_unknown_and_non_finite_options() {
        let mut args = valid_args();
        args.ponder_strategy = "typo".to_string();
        assert!(validate_args(&args).is_err());

        let mut args = valid_args();
        args.temperature = f32::NAN;
        assert!(validate_args(&args).is_err());

        let mut args = valid_args();
        args.repetition_penalty = 0.0;
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn speculative_mode_rejects_zero_k_and_filtered_sampling() {
        let mut args = valid_args();
        args.draft = Some(PathBuf::from("draft.raimodel"));
        args.draft_k = 0;
        assert!(validate_args(&args).is_err());

        args.draft_k = 4;
        assert!(validate_args(&args).is_err());

        args.top_k = 0;
        args.top_p = 1.0;
        args.repetition_penalty = 1.0;
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn lookup_mode_enforces_exact_sampling_and_exclusivity() {
        // Prompt-lookup uses the same exact-verification sampler restriction.
        let mut args = valid_args();
        args.lookup_k = 8;
        assert!(validate_args(&args).is_err());
        exact_sampling(&mut args);
        assert!(validate_args(&args).is_ok());

        // Mutually exclusive with the draft-model path.
        let mut args = valid_args();
        exact_sampling(&mut args);
        args.lookup_k = 8;
        args.draft = Some(PathBuf::from("draft.raimodel"));
        assert!(validate_args(&args).is_err());

        // Bounds on K and the n-gram length.
        let mut args = valid_args();
        exact_sampling(&mut args);
        args.lookup_k = MAX_SPECULATIVE_TOKENS + 1;
        assert!(validate_args(&args).is_err());

        let mut args = valid_args();
        exact_sampling(&mut args);
        args.lookup_k = 8;
        args.lookup_ngram = 0;
        assert!(validate_args(&args).is_err());
        args.lookup_ngram = MAX_LOOKUP_NGRAM + 1;
        assert!(validate_args(&args).is_err());

        // The n-gram floor must sit inside the ceiling.
        let mut args = valid_args();
        exact_sampling(&mut args);
        args.lookup_k = 8;
        args.lookup_ngram = 3;
        args.lookup_min_ngram = 4;
        assert!(validate_args(&args).is_err());
        args.lookup_min_ngram = 0;
        assert!(validate_args(&args).is_err());
        args.lookup_min_ngram = 3;
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn incremental_unicode_suffix_is_always_on_a_character_boundary() {
        assert_eq!(incremental_suffix("é", "é🙂"), "🙂");
        assert_eq!(incremental_suffix("é", "a🙂"), "a🙂");
        assert_eq!(incremental_suffix("same", "same"), "");
    }
}
