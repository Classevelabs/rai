//! Bounded statistical-equivalence check for speculative decoding.
//!
//! `tiny-tied.raimodel` acts as the DRAFT and `tiny-untied.raimodel` as the
//! TARGET (independent weights, same 96-token vocabulary), so acceptance
//! genuinely exercises both the accept and the reject/correction paths.
//!
//! Two claims are checked over 40 seeded short decodes:
//!  (a) every speculative step makes progress (>= 1 produced token) with
//!      internally consistent metrics and cache positions, and
//!  (b) the aggregate FIRST-token frequency distribution matches direct
//!      sampling from the target model within a loose absolute tolerance.
//!
//! (b) is a smoke-level distribution check, not a chi-square proof: with 40
//! runs the per-token standard error is ~0.08, so the 0.25 tolerance flags
//! gross distribution corruption (e.g. drafting from the wrong model or
//! skipping the correction sample) while staying far from flakiness. All RNGs
//! are seeded, so the test is deterministic on a given build.

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::SeedableRng;

use rai_infer::model::{RaiModel, Scratch};
use rai_infer::sampler::{sample_token, SamplerConfig};
use rai_infer::speculative::{SpeculativeConfig, SpeculativeDecoder};

const RUNS: usize = 40;
const TOKENS_PER_RUN: usize = 6;
const MAX_CTX: usize = 64;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// The only sampler configuration exact speculative verification supports.
fn exact_sampler() -> SamplerConfig {
    SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    }
}

/// A varying 3-token prompt per run, all ids within the fixture vocabulary.
fn prompt_for_run(run: usize, vocab: usize) -> Vec<usize> {
    vec![
        (run * 7 + 1) % vocab,
        (run * 11 + 3) % vocab,
        (run * 13 + 5) % vocab,
    ]
}

fn run_seed(run: usize) -> u64 {
    0xC0FF_EE00 + run as u64
}

#[test]
fn speculative_runs_complete_and_first_token_distribution_matches_target() {
    let draft = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load draft fixture");
    let target = RaiModel::load(&fixture("tiny-untied.raimodel")).expect("load target fixture");
    let vocab = target.config.vocab_size as usize;
    assert_eq!(
        vocab, draft.config.vocab_size as usize,
        "fixtures must share a vocabulary for speculative decoding"
    );

    let config = SpeculativeConfig {
        draft_k: 3,
        sampler: exact_sampler(),
    };

    // --- Speculative decodes -------------------------------------------------
    let mut spec_first_counts = vec![0usize; vocab];
    for run in 0..RUNS {
        let prompt = prompt_for_run(run, vocab);
        let mut rng = StdRng::seed_from_u64(run_seed(run));
        let mut decoder =
            SpeculativeDecoder::new(&draft, &target, MAX_CTX).expect("construct decoder");

        let mut pos = decoder.prefill(&prompt).expect("prefill");
        assert_eq!(
            pos,
            prompt.len() - 1,
            "prefill leaves the final prompt token for step() (run {run})"
        );

        let mut last_token = *prompt.last().unwrap();
        let mut generated: Vec<usize> = Vec::new();
        while generated.len() < TOKENS_PER_RUN {
            let (tokens, metrics) = decoder
                .step(pos, last_token, &config, &mut rng)
                .expect("speculative step");

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
            assert!(tokens.iter().all(|&t| t < vocab), "token out of vocabulary");

            // Positions advance by exactly the produced count; the decoder's
            // KV watermark checks panic loudly if this bookkeeping ever
            // diverges from the caches, so completing the loop is itself the
            // position-consistency assertion.
            pos += metrics.produced;
            assert!(pos < MAX_CTX, "position ran past the context window");
            last_token = *tokens.last().unwrap();
            generated.extend_from_slice(&tokens);
        }
        spec_first_counts[generated[0]] += 1;
    }

    // --- Direct target-only sampling, same prompts and seeds -----------------
    let mut direct_first_counts = vec![0usize; vocab];
    let hs = target.config.hidden_size as usize;
    for run in 0..RUNS {
        let prompt = prompt_for_run(run, vocab);
        let mut rng = StdRng::seed_from_u64(run_seed(run));
        let mut kv = target.create_kv_cache(MAX_CTX).expect("target KV cache");
        let mut scratch = Scratch::new();
        let mut hidden = vec![0.0f32; hs];
        let mut normed = vec![0.0f32; hs];
        let mut logits = vec![0.0f32; vocab];

        for (p, &tok) in prompt.iter().enumerate() {
            target.embed_token(tok, &mut hidden).expect("embed");
            target
                .forward_from_hidden(&mut hidden, p, &mut kv, true, &mut scratch)
                .expect("target forward");
        }
        target
            .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
            .expect("target logits");
        let token = sample_token(&mut logits, &exact_sampler(), &mut rng);
        direct_first_counts[token] += 1;
    }

    // (b) loose aggregate agreement of the two first-token distributions.
    let n = RUNS as f32;
    for token in 0..vocab {
        let spec_freq = spec_first_counts[token] as f32 / n;
        let direct_freq = direct_first_counts[token] as f32 / n;
        let diff = (spec_freq - direct_freq).abs();
        assert!(
            diff <= 0.25,
            "token {token}: speculative {}/{RUNS} vs direct {}/{RUNS} \
             (|Δfreq| = {diff:.3} exceeds the 0.25 smoke tolerance)",
            spec_first_counts[token],
            direct_first_counts[token]
        );
    }

    // Sanity: both histograms account for every run.
    assert_eq!(spec_first_counts.iter().sum::<usize>(), RUNS);
    assert_eq!(direct_first_counts.iter().sum::<usize>(), RUNS);
}
