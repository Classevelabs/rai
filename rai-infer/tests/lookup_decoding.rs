//! Bookkeeping and distribution checks for prompt-lookup speculative decoding.
//!
//! `tiny-tied.raimodel` is the target model (96-token vocabulary). The prompts
//! are deliberately periodic so the n-gram search actually fires — with random
//! prompts the decoder would spend every step on the no-draft fallback and the
//! interesting path would go untested.
//!
//! Two claims are checked over seeded short decodes:
//!  (a) every step makes progress with internally consistent metrics, the
//!      produced count never exceeds drafted+1, positions advance by exactly
//!      the produced count, and the KV cache watermark tracks that frontier
//!      (the cache panics on gaps, so completing the loop IS the assertion);
//!  (b) the aggregate FIRST-token frequency distribution matches direct
//!      sampling from the same model within a loose absolute tolerance —
//!      speculative sampling is exact, so copying tokens out of the context
//!      must not bend the output distribution.
//!
//! (b) is a smoke-level check like `speculative_equivalence.rs`, not a
//! chi-square proof: 60 runs give a per-token standard error near 0.06, so the
//! 0.25 tolerance catches gross corruption (a broken accept rule, a skipped
//! correction sample) while staying far from flakiness. All RNGs are seeded.

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::SeedableRng;

use rai_infer::lookup::{find_ngram_draft, LookupConfig, LookupDecoder};
use rai_infer::model::{RaiModel, Scratch};
use rai_infer::sampler::{sample_token, SamplerConfig};

const RUNS: usize = 60;
const TOKENS_PER_RUN: usize = 8;
const MAX_CTX: usize = 64;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// The only sampler configuration exact verification supports.
fn exact_sampler() -> SamplerConfig {
    SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    }
}

/// A periodic 12-token prompt: the trailing 3-gram always has an earlier
/// occurrence, so the very first step already has a draft to verify.
fn prompt_for_run(run: usize, vocab: usize) -> Vec<usize> {
    let cycle = [
        (run * 7 + 1) % vocab,
        (run * 11 + 3) % vocab,
        (run * 13 + 5) % vocab,
        (run * 5 + 9) % vocab,
    ];
    let mut prompt = Vec::with_capacity(12);
    for _ in 0..3 {
        prompt.extend_from_slice(&cycle);
    }
    prompt
}

fn run_seed(run: usize) -> u64 {
    0x10CA_1000 + run as u64
}

#[test]
fn lookup_runs_complete_and_first_token_distribution_matches_the_model() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture");
    let vocab = model.config.vocab_size as usize;

    let config = LookupConfig {
        max_draft: 4,
        max_ngram: 3,
        min_ngram: 1,
        sampler: exact_sampler(),
    };

    // --- Prompt-lookup decodes ----------------------------------------------
    let mut lookup_first_counts = vec![0usize; vocab];
    let mut steps_total = 0usize;
    let mut steps_with_draft = 0usize;
    let mut drafted_total = 0usize;

    for run in 0..RUNS {
        let prompt = prompt_for_run(run, vocab);
        let mut rng = StdRng::seed_from_u64(run_seed(run));
        let mut decoder = LookupDecoder::new(&model, MAX_CTX).expect("construct decoder");

        let mut pos = decoder.prefill(&prompt).expect("prefill");
        assert_eq!(
            pos,
            prompt.len() - 1,
            "prefill must leave the final prompt token for step() (run {run})"
        );

        let mut context = prompt.clone();
        let mut generated: Vec<usize> = Vec::new();
        while generated.len() < TOKENS_PER_RUN {
            // The decoder's search corpus is the live context; the token at
            // `pos` must be its last element.
            assert_eq!(
                pos + 1,
                context.len(),
                "position/context invariant broken (run {run})"
            );

            let (tokens, metrics) = decoder
                .step(pos, &context, &config, &mut rng)
                .expect("lookup step");

            // (a) progress + internally consistent metrics every step.
            assert!(
                metrics.produced >= 1,
                "step produced no tokens (run {run}, pos {pos})"
            );
            assert_eq!(tokens.len(), metrics.produced, "produced-count mismatch");
            assert!(
                metrics.produced <= metrics.drafted + 1,
                "produced {} exceeds drafted {} + 1",
                metrics.produced,
                metrics.drafted
            );
            assert!(metrics.accepted <= metrics.drafted);
            assert!(
                metrics.drafted <= config.max_draft,
                "drafted {} exceeds K={}",
                metrics.drafted,
                config.max_draft
            );
            assert_eq!(
                metrics.matched_ngram.is_some(),
                metrics.drafted > 0,
                "a matched n-gram must produce a draft and vice versa"
            );
            if let Some(n) = metrics.matched_ngram {
                assert!((config.min_ngram..=config.max_ngram).contains(&n));
            }
            assert!(tokens.iter().all(|&t| t < vocab), "token out of vocabulary");

            steps_total += 1;
            drafted_total += metrics.drafted;
            if metrics.drafted > 0 {
                steps_with_draft += 1;
            }

            pos += metrics.produced;
            context.extend_from_slice(&tokens);
            generated.extend_from_slice(&tokens);

            // The KV frontier must be exactly the accepted context. Reading or
            // storing past it panics inside kv_cache.rs, so an off-by-one here
            // fails loudly on the next step rather than corrupting attention.
            for layer in 0..model.config.num_layers as usize {
                assert_eq!(
                    decoder.kv_cache.filled(layer),
                    pos,
                    "KV watermark diverged from the produced-token frontier \
                     (run {run}, layer {layer})"
                );
            }
            assert!(pos < MAX_CTX, "position ran past the context window");
        }
        lookup_first_counts[generated[0]] += 1;
    }

    // The periodic prompts exist so the drafting path is genuinely exercised.
    // Every run's FIRST step is guaranteed a 3-gram match by construction; if
    // this ever fails the test has degenerated into a plain-decode test.
    // (Later steps depend on what the fixture model emits, which is noise, so
    // no stronger bound is claimed here.)
    assert!(
        steps_with_draft >= RUNS,
        "only {steps_with_draft}/{steps_total} steps found a draft — the n-gram \
         path is not being exercised"
    );
    assert!(drafted_total > 0);

    // --- Direct sampling from the same model, same prompts and seeds --------
    let mut direct_first_counts = vec![0usize; vocab];
    let hs = model.config.hidden_size as usize;
    for run in 0..RUNS {
        let prompt = prompt_for_run(run, vocab);
        let mut rng = StdRng::seed_from_u64(run_seed(run));
        let mut kv = model.create_kv_cache(MAX_CTX).expect("KV cache");
        let mut scratch = Scratch::new();
        let mut hidden = vec![0.0f32; hs];
        let mut normed = vec![0.0f32; hs];
        let mut logits = vec![0.0f32; vocab];

        for (p, &tok) in prompt.iter().enumerate() {
            model.embed_token(tok, &mut hidden).expect("embed");
            model
                .forward_from_hidden(&mut hidden, p, &mut kv, true, &mut scratch)
                .expect("forward");
        }
        model
            .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
            .expect("logits");
        let token = sample_token(&mut logits, &exact_sampler(), &mut rng);
        direct_first_counts[token] += 1;
    }

    // (b) loose aggregate agreement of the two first-token distributions.
    let n = RUNS as f32;
    for token in 0..vocab {
        let lookup_freq = lookup_first_counts[token] as f32 / n;
        let direct_freq = direct_first_counts[token] as f32 / n;
        let diff = (lookup_freq - direct_freq).abs();
        assert!(
            diff <= 0.25,
            "token {token}: lookup {}/{RUNS} vs direct {}/{RUNS} \
             (|Δfreq| = {diff:.3} exceeds the 0.25 smoke tolerance)",
            lookup_first_counts[token],
            direct_first_counts[token]
        );
    }

    assert_eq!(lookup_first_counts.iter().sum::<usize>(), RUNS);
    assert_eq!(direct_first_counts.iter().sum::<usize>(), RUNS);
}

#[test]
fn lookup_falls_back_to_plain_decoding_without_a_match() {
    // A strictly increasing prompt has no repeated n-gram, so the first steps
    // must take the no-draft path — one token per step, no verification batch.
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture");
    let vocab = model.config.vocab_size as usize;
    let prompt: Vec<usize> = (1..=9).map(|i| (i * 7) % vocab).collect();
    assert!(
        find_ngram_draft(&prompt, 1, 3, 4).is_none(),
        "the fallback prompt must not contain a repeated n-gram"
    );

    let config = LookupConfig {
        max_draft: 4,
        max_ngram: 3,
        min_ngram: 1,
        sampler: exact_sampler(),
    };
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut decoder = LookupDecoder::new(&model, MAX_CTX).expect("construct decoder");
    let mut pos = decoder.prefill(&prompt).expect("prefill");
    let mut context = prompt.clone();

    let (tokens, metrics) = decoder
        .step(pos, &context, &config, &mut rng)
        .expect("lookup step");
    assert_eq!(metrics.drafted, 0, "no n-gram match must mean no draft");
    assert_eq!(metrics.matched_ngram, None);
    assert_eq!(
        metrics.produced, 1,
        "the fallback produces exactly one token"
    );
    assert_eq!(metrics.accept_rate, 0.0);
    assert_eq!(tokens.len(), 1);

    pos += metrics.produced;
    context.extend_from_slice(&tokens);
    for layer in 0..model.config.num_layers as usize {
        assert_eq!(decoder.kv_cache.filled(layer), pos);
    }

    // Decoding continues normally from that frontier (this would panic if the
    // fallback had left the cache short or long).
    let (tokens, _) = decoder
        .step(pos, &context, &config, &mut rng)
        .expect("second step");
    assert!(!tokens.is_empty());
}

#[test]
fn lookup_rejects_unsupported_sampler_settings() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture");
    let mut decoder = LookupDecoder::new(&model, MAX_CTX).expect("construct decoder");
    let prompt = vec![1usize, 2, 3, 1, 2, 3];
    let pos = decoder.prefill(&prompt).expect("prefill");
    let mut rng = StdRng::seed_from_u64(1);

    let mut config = LookupConfig {
        max_draft: 4,
        max_ngram: 3,
        min_ngram: 1,
        sampler: exact_sampler(),
    };
    config.sampler.top_k = 40;
    assert!(decoder.step(pos, &prompt, &config, &mut rng).is_err());

    config.sampler.top_k = 0;
    config.sampler.temperature = 0.0;
    assert!(decoder.step(pos, &prompt, &config, &mut rng).is_err());
}
