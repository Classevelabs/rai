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

use crate::chat_format::ChatFormat;
use crate::chat_template::{ChatTemplate, Role};
use crate::cli::{load_tokenizer_for_model, resolve_tokenizer};
use crate::lookup::{LookupConfig, LookupDecoder};
use crate::model::{BatchScratch, InferenceWork, RaiModel};
use crate::ponder::{pondered_forward, PonderConfig, PonderStrategy};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};
use crate::speculative::{SpeculativeConfig, SpeculativeDecoder};

/// Upper bound on `--ensemble-n`; each pass is a full forward pass.
pub const MAX_ENSEMBLE_SIZE: usize = 8;
/// Upper bound on `--draft-k` and `--lookup-k`. Verification batches every
/// drafted token, so an over-long draft costs more than it can win back.
pub const MAX_SPECULATIVE_TOKENS: usize = 32;
/// Upper bound on `--lookup-ngram`, the longest context suffix matched first.
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
    /// Maximum number of tokens to generate; the reply ends sooner when the
    /// model finishes it
    // Room for a full answer to an ordinary question, where 64 stopped one
    // partway; a model that ends its turn stops well before the limit.
    #[arg(long, default_value = "512")]
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
    /// Penalty on tokens this run has already generated (1.0 disables). The
    /// prompt is not penalized: a chat prompt holds the very end-of-turn
    /// marker the reply has to end with.
    #[arg(long, default_value = "1.1")]
    pub repetition_penalty: f32,
    /// Pondering strategy: none, cfg, ensemble, cfg-ensemble, adaptive
    #[arg(long, default_value = "none")]
    pub ponder_strategy: String,
    /// CFG guidance scale (1.0 = no effect; the strategies are experimental
    /// and unmeasured — see ponder.rs)
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
    /// Context window to allocate the KV cache for, in tokens. Defaults to the
    /// context the model was stored with, reduced to fit free memory when it
    /// would not (a note says so); a smaller value trades history for memory,
    /// a larger one is clamped to what the model supports.
    #[arg(long)]
    pub max_context: Option<usize>,
    /// Random seed; the same seed and settings reproduce the same text
    #[arg(long, default_value = "42")]
    pub seed: u64,
    /// Print per-step decoding diagnostics to stderr
    #[arg(long, default_value = "false")]
    pub verbose: bool,
    /// Chat template: auto (the model's own, recorded at conversion; plain
    /// text for a model that has none), model, none, few-shot, mistral, llama3,
    /// chatml, zephyr, phi3, gemma
    #[arg(long, default_value = "auto")]
    pub chat_template: String,
    /// Draft model for speculative decoding (e.g. smollm-135m-q4.raimodel)
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Number of draft tokens per speculative step
    #[arg(long, default_value = "6")]
    pub draft_k: usize,
    /// Prompt-lookup speculation: draft up to N tokens copied from the context
    /// (0 = disabled). No draft model and no draft forward pass. Measured on
    /// TinyLlama-1.1B-q4 at K=2: 1.12-1.20x when the output quotes the context
    /// (summarisation, RAG, code editing) and 0.74-0.88x on original prose.
    /// Off by default because the loss is as real as the gain.
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
    // `None` means "whatever the model was stored with", which is resolved
    // once the model is open; an explicit zero is still a mistake.
    ensure!(
        !matches!(args.max_context, Some(0)),
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
        matches!(args.chat_template.as_str(), "auto" | "model")
            || ChatTemplate::from_name(&args.chat_template).is_some(),
        "unknown --chat-template '{}'; expected auto, model, or one of {}",
        args.chat_template,
        ChatTemplate::NAMES.join(", ")
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
        // Each speculative loop samples the target directly; pondering would be
        // skipped without a word.
        ensure!(
            args.ponder_strategy == "none",
            "--ponder-strategy cannot be combined with --draft or --lookup-k"
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

/// The part of `current` that `previous` does not already cover.
///
/// Shared with `rai serve`'s streaming route, which faces the same problem:
/// the tokenizer is asked to decode the whole generated suffix every step (so
/// that SentencePiece spacing comes out right), and only the new tail of that
/// decode should reach the reader.
pub(crate) fn incremental_suffix<'a>(previous: &str, current: &'a str) -> &'a str {
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
    let tokenizer = load_tokenizer_for_model(&tokenizer_path, cfg.vocab_size as usize)?;

    let mut format = ChatFormat::load(&generation.chat_template, &args.model, &tokenizer)?;
    // `auto` on a model with no chat template of its own is a completion: the
    // prompt goes to the model as written, as `--chat-template none` sends it,
    // not inside the few-shot frame a chat window needs.
    if generation.chat_template == "auto" && format.builtin() == Some(ChatTemplate::FewShot) {
        format = ChatFormat::load("none", &args.model, &tokenizer)?;
    }
    if format.builtin() != Some(ChatTemplate::None) {
        eprintln!("Chat template: {}", format.name());
    }
    let prompt_tokens = format.encode(&tokenizer, &[(Role::User, generation.prompt.as_str())])?;
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let ponder_config = build_ponder_config(generation)?;
    let sampler_config = SamplerConfig {
        temperature: generation.temperature,
        top_k: generation.top_k,
        top_p: generation.top_p,
        repetition_penalty: generation.repetition_penalty,
    };

    // No flag means the window the model was stored with. A flag that asks for
    // more than the model has is still clamped down to it: the RoPE tables were
    // only built that far, so positions beyond it are not defined.
    let choice = crate::cli::serve::choose_context_window(
        generation.max_context,
        model.config.max_context as usize,
        |ctx| model.kv_cache_bytes(ctx),
    )?;
    if let Some(note) =
        crate::cli::serve::context_choice_note(choice, |ctx| model.kv_cache_bytes(ctx))
    {
        eprintln!("{note}");
    }
    let max_ctx = choice.context;
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
    // The reserved ceiling, not the resident cost: the cache allocates in
    // steps and only reaches this if the conversation reaches `max_ctx`.
    // Reporting the ceiling as though it were resident is what made a long
    // window look unaffordable when it is not.
    eprintln!(
        "KV cache: up to {} at {max_ctx} tokens (allocated as the conversation grows)",
        crate::cli::format_bytes(kv_bytes as u64)
    );
    // Now that the default follows the model's own window, a large model with a
    // long stored context asks for a cache measured in tens of gigabytes. Say so
    // with the numbers before anything is allocated: the allocation itself is
    // fallible, but on a machine that overcommits it is the *writing* of the
    // cache that fails, and that arrives as a kill rather than as an error.
    // A draft model allocates its own cache alongside the target's, so the
    // preflight has to weigh both or it under-checks by a factor of two.
    //
    // The draft is loaded further down, so its header is read here instead:
    // `read_summary` parses the header alone, without the weights, which is
    // exactly enough to size its cache before anything is allocated.
    let draft_kv_bytes = match &generation.draft {
        Some(path) => {
            let summary = crate::format::RaiModelFile::read_summary(path)
                .with_context(|| format!("reading the draft model header at {}", path.display()))?;
            let config = &summary.config;
            config.kv_cache_bytes(max_ctx.min(config.max_context as usize))
        }
        None => 0,
    };
    crate::cli::serve::check_kv_cache_fits(kv_bytes + draft_kv_bytes, max_ctx)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(generation.seed);
    let mut all_tokens = prompt_tokens.clone();

    print!("{}", generation.prompt);
    io::stdout().flush()?;

    // Which token ids end generation: the format's, which covers the
    // checkpoint's own end-of-sequence ids, every family's end markers among
    // the tokenizer's control tokens, and the template's turn markers.
    let end_ids = format.end_ids();
    let is_eos = |tok: usize| -> bool { end_ids.contains(&tok) };
    let stops = format.stop_sequences();

    // Helper: print newly generated text with correct spacing, and report
    // whether a stop sequence ended the reply (its text is never printed). A
    // trailing U+FFFD usually means a multi-byte codepoint is split across
    // tokens, and the start of a stop sequence may be one; both are held back
    // until the next token settles them.
    let print_new_text = |all_tokens: &[usize],
                          prompt_len: usize,
                          previous_text: &mut String,
                          tokenizer: &Tokenizer|
     -> bool {
        let gen_ids: Vec<u32> = all_tokens[prompt_len..].iter().map(|&t| t as u32).collect();
        let mut full_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();
        let cut = stops
            .iter()
            .filter_map(|stop| full_text.find(stop.as_str()))
            .min();
        if let Some(cut) = cut {
            full_text.truncate(cut);
        }
        let suffix = incremental_suffix(previous_text, &full_text);
        let length = if cut.is_some() {
            suffix.len()
        } else {
            crate::cli::serve::emittable_len(suffix, stops)
        };
        if length > 0 {
            print!("{}", &suffix[..length]);
            let _ = io::stdout().flush();
        }
        let printed = full_text.len() - (suffix.len() - length);
        *previous_text = full_text[..printed].to_string();
        cut.is_some()
    };

    // Helper: once decoding ends, print anything the hold-back withheld (a
    // final replacement character is genuine and should be shown), short of
    // any stop sequence.
    let flush_held_text = |all_tokens: &[usize],
                           prompt_len: usize,
                           previous_text: &mut String,
                           tokenizer: &Tokenizer| {
        let gen_ids: Vec<u32> = all_tokens[prompt_len..].iter().map(|&t| t as u32).collect();
        let mut full_text = tokenizer.decode(&gen_ids, false).unwrap_or_default();
        if let Some(cut) = stops
            .iter()
            .filter_map(|stop| full_text.find(stop.as_str()))
            .min()
        {
            full_text.truncate(cut);
        }
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

            let hit_stop = print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            );

            if generation.verbose {
                eprint!("[spec:{}d/{}a]", metrics.drafted, metrics.accepted);
            }

            if hit_eos || hit_stop {
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

            let hit_stop = print_new_text(
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

            if hit_eos || hit_stop {
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

        // Prefill every prompt token but the last, which the first decode step
        // consumes at its own position; prefilling it too would store its KV
        // entry twice. The prompt was checked against `max_ctx` above.
        let t_prefill = Instant::now();
        let n_to_prefill = prompt_tokens.len() - 1;
        model.prefill(
            &prompt_tokens[..n_to_prefill],
            0,
            &mut kv_cache,
            &mut BatchScratch::new(),
        )?;
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

        // Each step feeds the newest token at the next position, so the
        // positions run from the last prompt token to the end of the window,
        // and at most `max_tokens` of them.
        for pos in (n_to_prefill..max_ctx).take(generation.max_tokens) {
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

            apply_repetition_penalty(
                &mut logits,
                &all_tokens[prompt_tokens.len()..],
                sampler_config.repetition_penalty,
            );
            let next_token = sample_token(&mut logits, &sampler_config, &mut rng);
            work.scratch.logits = logits;

            if is_eos(next_token) {
                break;
            }

            all_tokens.push(next_token);
            tokens_generated += 1;
            if print_new_text(
                &all_tokens,
                prompt_tokens.len(),
                &mut previous_text,
                &tokenizer,
            ) {
                break;
            }
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
            // instruction-tuned model given plain text, which ends the sequence
            // at once because nothing marks a turn: `--chat-template none`, or
            // a model converted before its chat format was recorded.
            if format.builtin() == Some(ChatTemplate::None) {
                eprintln!(
                    "\nNo tokens generated: the model ended its reply at once. An \
                     instruction-tuned model needs its chat format: convert it again \
                     so the format is recorded beside it, or name one with \
                     --chat-template (chatml, llama3, mistral, zephyr, phi3, gemma, few-shot)."
                );
            } else {
                eprintln!("\nNo tokens generated: the model ended its reply at once.");
            }
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
            max_context: Some(512),
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

    /// The speculative loops sample the target directly, so a strategy given
    /// alongside them would be dropped without a word.
    #[test]
    fn pondering_is_refused_with_either_speculative_mode() {
        for speculative in [
            |args: &mut GenerationArgs| args.draft = Some(PathBuf::from("draft.raimodel")),
            |args: &mut GenerationArgs| args.lookup_k = 4,
        ] {
            let mut args = valid_args();
            exact_sampling(&mut args);
            speculative(&mut args);
            assert!(validate_args(&args).is_ok());
            args.ponder_strategy = "cfg".to_string();
            let error = validate_args(&args).unwrap_err().to_string();
            assert!(error.contains("--ponder-strategy"), "{error}");
        }
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

    /// Omitting `--max-context` is not "512", it is "ask the model". Only an
    /// explicit zero is an error.
    #[test]
    fn an_absent_context_window_is_allowed_and_an_explicit_zero_is_not() {
        let mut args = valid_args();
        args.max_context = None;
        assert!(validate_args(&args).is_ok());

        args.max_context = Some(0);
        assert!(validate_args(&args).is_err());

        args.max_context = Some(131_072);
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn incremental_unicode_suffix_is_always_on_a_character_boundary() {
        assert_eq!(incremental_suffix("é", "é🙂"), "🙂");
        assert_eq!(incremental_suffix("é", "a🙂"), "a🙂");
        assert_eq!(incremental_suffix("same", "same"), "");
    }
}
