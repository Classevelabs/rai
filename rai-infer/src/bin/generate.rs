//! rai-generate: CLI for edge inference with pondering strategies.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use rai_infer::chat_template::ChatTemplate;
use rai_infer::lookup::{LookupConfig, LookupDecoder};
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
    /// Chat template: auto, none, few-shot, mistral, llama3, chatml, zephyr
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
    /// Prompt-lookup speculation: draft up to N tokens copied from the context
    /// (0 = disabled). No draft model and no draft forward pass. NOTE: measured
    /// on TinyLlama-1.1B-q4 this is still ~0.7-0.9x baseline because batched
    /// verification barely amortises weight reads in this engine; small K (1-2)
    /// is closest to break-even and large K is much worse.
    #[arg(long, default_value = "0")]
    lookup_k: usize,
    /// Prompt-lookup: longest suffix n-gram to match first
    #[arg(long, default_value = "3")]
    lookup_ngram: usize,
    /// Prompt-lookup: shortest suffix n-gram to fall back to. Raising this to 2
    /// or 3 suppresses weak single-token matches, which is what makes the
    /// non-repetitive worst case expensive.
    #[arg(long, default_value = "1")]
    lookup_min_ngram: usize,
}

const MAX_ENSEMBLE_SIZE: usize = 8;
const MAX_SPECULATIVE_TOKENS: usize = 32;
const MAX_LOOKUP_NGRAM: usize = 16;
/// Below this draft-acceptance rate, self-speculative decoding cannot recover
/// the cost of its draft forward passes and is slower than plain decoding.
const SELF_SPEC_USEFUL_ACCEPT_RATE: f64 = 0.30;

fn build_ponder_config(args: &Args) -> Result<PonderConfig> {
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

fn validate_args(args: &Args) -> Result<()> {
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
    // The three speculative paths each own the KV cache and the decode loop,
    // so at most one may be selected.
    let speculative_modes = usize::from(args.draft.is_some())
        + usize::from(args.self_spec_layers > 0)
        + usize::from(args.lookup_k > 0);
    ensure!(
        speculative_modes <= 1,
        "--draft, --self-spec-layers, and --lookup-k are mutually exclusive"
    );
    if args.draft.is_some() {
        ensure!(
            (1..=MAX_SPECULATIVE_TOKENS).contains(&args.draft_k),
            "--draft-k must be between 1 and {MAX_SPECULATIVE_TOKENS}"
        );
    }
    if args.self_spec_layers > 0 {
        ensure!(
            (1..=MAX_SPECULATIVE_TOKENS).contains(&args.self_spec_k),
            "--self-spec-k must be between 1 and {MAX_SPECULATIVE_TOKENS}"
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
    if args.draft.is_some() || args.self_spec_layers > 0 || args.lookup_k > 0 {
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

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();
    let args = Args::parse();
    validate_args(&args)?;

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

    let template = ChatTemplate::from_str_arg(&args.chat_template, &tokenizer);
    let formatted_prompt = template.format_prompt(&args.prompt);
    let encoding = tokenizer
        .encode(formatted_prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let ponder_config = build_ponder_config(&args)?;
    let sampler_config = SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
    };

    let max_ctx = args.max_context.min(model.config.max_context as usize);
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

    if args.self_spec_layers > 0 {
        // === SELF-SPECULATIVE DECODING ===
        let total_layers = model.config.num_layers as usize;
        ensure!(
            args.self_spec_layers < total_layers,
            "--self-spec-layers must be smaller than the model's {total_layers} layers"
        );
        let draft_layers = args.self_spec_layers;

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
            )?
        } else {
            eprintln!(
                "Self-speculative (early-exit): first {} of {} layers, K={}",
                draft_layers, total_layers, args.self_spec_k
            );
            SelfSpecConfig::early_exit(draft_layers, args.self_spec_k, sampler_config.clone())?
        };
        eprintln!("Draft layers: {:?}", spec_config.draft_layer_indices);
        eprintln!(
            "WARNING: --self-spec-layers is experimental and measures BADLY. On \
             TinyLlama-1.1B-q4 (i5-10300H, --self-spec-layers 11 --self-spec-k 4, \
             context-quoting QA prompt) it accepted 0.4% of its drafts and ran at \
             1.7 tok/s against a 25.1 tok/s baseline — roughly 15x SLOWER. An \
             untrained early exit is a poor predictor of the full model, so both the \
             draft passes and the verification are wasted. No K or layer count has \
             been found where this wins; plain decoding (no flag) is faster. This \
             run's own acceptance rate is printed in the stats below."
        );
        let mut decoder = SelfSpecDecoder::new(&model, max_ctx)?;

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
        let mut previous_text = String::new();

        while tokens_generated < args.max_tokens && pos < max_ctx {
            let last_token = *all_tokens.last().unwrap();
            let (new_tokens, metrics) = decoder.step(pos, last_token, &spec_config, &mut rng)?;
            ensure!(
                !new_tokens.is_empty(),
                "self-speculative decoder made no progress"
            );

            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;

            let mut hit_eos = false;
            let remaining = (max_ctx - pos).min(args.max_tokens - tokens_generated);
            for &tok in new_tokens.iter().take(remaining) {
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

            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

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

        flush_held_text(
            &all_tokens,
            prompt_tokens.len(),
            &mut previous_text,
            &tokenizer,
        );
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
        if total_drafted > 0 && accept_rate < SELF_SPEC_USEFUL_ACCEPT_RATE {
            eprintln!(
                "Measured acceptance {:.1}% is below the {:.0}% needed for this path to pay \
                 for its draft passes — this run was almost certainly slower than plain \
                 decoding. Drop the flag, or try --lookup-k on context-quoting workloads.",
                accept_rate * 100.0,
                SELF_SPEC_USEFUL_ACCEPT_RATE * 100.0
            );
        }
    } else if let Some(draft_path) = &args.draft {
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
        let mut previous_text = String::new();

        while tokens_generated < args.max_tokens && pos < max_ctx {
            let last_token = *all_tokens.last().unwrap();
            let (new_tokens, metrics) = decoder.step(pos, last_token, &spec_config, &mut rng)?;
            ensure!(
                !new_tokens.is_empty(),
                "speculative decoder made no progress"
            );

            total_drafted += metrics.drafted;
            total_accepted += metrics.accepted;

            let mut hit_eos = false;
            let remaining = (max_ctx - pos).min(args.max_tokens - tokens_generated);
            for &tok in new_tokens.iter().take(remaining) {
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

            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

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
                total_accepted as f64 / (total_drafted as f64 / args.draft_k as f64)
            } else {
                0.0
            }
        );
    } else if args.lookup_k > 0 {
        // === PROMPT-LOOKUP (N-GRAM) SPECULATIVE DECODING ===
        // The draft is copied out of the context, so there is no draft model
        // and no draft forward pass: a miss costs about one ordinary step and
        // a hit produces several tokens for the price of one.
        let lookup_config = LookupConfig {
            max_draft: args.lookup_k,
            max_ngram: args.lookup_ngram,
            min_ngram: args.lookup_min_ngram,
            sampler: sampler_config.clone(),
        };
        eprintln!(
            "Prompt-lookup: K={}, n-gram {}..{} (draft copied from context, no draft model)",
            args.lookup_k, args.lookup_ngram, args.lookup_min_ngram
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

        while tokens_generated < args.max_tokens && pos < max_ctx {
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
            let remaining = (max_ctx - pos).min(args.max_tokens - tokens_generated);
            for &tok in new_tokens.iter().take(remaining) {
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

            print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

            if args.verbose {
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

    fn valid_args() -> Args {
        Args {
            model: PathBuf::from("model.raimodel"),
            tokenizer: PathBuf::from("tokenizer.json"),
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
            self_spec_layers: 0,
            self_spec_k: 8,
            self_spec_skip: false,
            lookup_k: 0,
            lookup_ngram: 3,
            lookup_min_ngram: 1,
        }
    }

    fn exact_sampling(args: &mut Args) {
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

        // Mutually exclusive with the other two speculative paths.
        let mut args = valid_args();
        exact_sampling(&mut args);
        args.lookup_k = 8;
        args.self_spec_layers = 4;
        assert!(validate_args(&args).is_err());

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
