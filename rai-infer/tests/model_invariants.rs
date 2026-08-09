//! Model-level invariants on the committed tiny fixtures.
//!
//! The one that matters most: `forward_batch` over consecutive positions must
//! agree with sequential `forward_from_hidden` calls. Speculative decoding's
//! exactness argument rests entirely on that equivalence — the draft's tokens
//! are verified through the batched path against distributions the sequential
//! path would have produced.

use std::path::{Path, PathBuf};

use rai_infer::kv_cache::KVCache;
use rai_infer::model::{BatchScratch, RaiModel, Scratch};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// A deterministic pseudo-random token sequence within the fixture's vocab.
fn token_sequence(vocab: usize, len: usize) -> Vec<usize> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % vocab
        })
        .collect()
}

fn sequential_logits(
    model: &RaiModel,
    tokens: &[usize],
    max_ctx: usize,
) -> Vec<Vec<f32>> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let mut kv = model.create_kv_cache(max_ctx).unwrap();
    let mut scratch = Scratch::new();
    let mut all = Vec::new();
    let mut hidden = vec![0.0f32; hs];
    for (pos, &tok) in tokens.iter().enumerate() {
        hidden.resize(hs, 0.0);
        model.embed_token(tok, &mut hidden).unwrap();
        model
            .forward_from_hidden(&mut hidden, pos, &mut kv, true, &mut scratch)
            .unwrap();
        let mut normed = vec![0.0f32; hs];
        let mut logits = vec![0.0f32; vs];
        model
            .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
            .unwrap();
        all.push(logits);
    }
    all
}

fn batched_logits(model: &RaiModel, tokens: &[usize], max_ctx: usize) -> Vec<Vec<f32>> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let n = tokens.len();
    let mut kv = model.create_kv_cache(max_ctx).unwrap();
    let mut batch_scratch = BatchScratch::new();
    let mut hiddens = vec![0.0f32; n * hs];
    for (i, &tok) in tokens.iter().enumerate() {
        model
            .embed_token(tok, &mut hiddens[i * hs..(i + 1) * hs])
            .unwrap();
    }
    let positions: Vec<usize> = (0..n).collect();
    model
        .forward_batch(&mut hiddens, &positions, &mut kv, &mut batch_scratch)
        .unwrap();
    let mut normed = vec![0.0f32; n * hs];
    let mut logits = vec![0.0f32; n * vs];
    model
        .hidden_to_logits_batch(&hiddens, &mut normed, &mut logits, n)
        .unwrap();
    (0..n).map(|i| logits[i * vs..(i + 1) * vs].to_vec()).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

fn check_batch_matches_sequential(fixture_name: &str) {
    let model = RaiModel::load(&fixture(fixture_name)).expect("load fixture model");
    let vocab = model.config.vocab_size as usize;
    let tokens = token_sequence(vocab, 12);
    let max_ctx = 32;

    let sequential = sequential_logits(&model, &tokens, max_ctx);
    let batched = batched_logits(&model, &tokens, max_ctx);

    assert_eq!(sequential.len(), batched.len());
    let mut max_diff = 0.0f32;
    for (pos, (s, b)) in sequential.iter().zip(&batched).enumerate() {
        assert!(
            s.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()),
            "non-finite logits at position {pos}"
        );
        // The token-level invariant speculative verification relies on.
        assert_eq!(
            argmax(s),
            argmax(b),
            "argmax diverged at position {pos} ({fixture_name})"
        );
        for (i, (x, y)) in s.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff <= 2e-3,
                "logit mismatch at pos {pos} token {i}: sequential {x} vs batched {y} ({fixture_name})"
            );
        }
    }
    eprintln!("{fixture_name}: max |sequential - batched| logit diff = {max_diff:e}");
}

#[test]
fn batched_forward_matches_sequential_tied() {
    check_batch_matches_sequential("tiny-tied.raimodel");
}

#[test]
fn batched_forward_matches_sequential_untied() {
    check_batch_matches_sequential("tiny-untied.raimodel");
}

#[test]
fn forward_partial_runs_layer_subsets() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture model");
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let mut kv = model.create_kv_cache(16).unwrap();
    let mut scratch = Scratch::new();

    // Draft with only the first layer of two, as self-speculative early-exit does.
    let mut hidden = vec![0.0f32; hs];
    model.embed_token(1, &mut hidden).unwrap();
    model
        .forward_partial(&mut hidden, 0, &mut kv, &mut scratch, &[0])
        .unwrap();
    let mut normed = vec![0.0f32; hs];
    let mut logits = vec![0.0f32; vs];
    model
        .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
        .unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn cleared_cache_rejects_stale_position_reads() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture model");
    let hs = model.config.hidden_size as usize;
    let mut kv: KVCache = model.create_kv_cache(16).unwrap();
    let mut scratch = Scratch::new();
    let mut hidden = vec![0.0f32; hs];

    model.embed_token(1, &mut hidden).unwrap();
    model
        .forward_from_hidden(&mut hidden, 0, &mut kv, true, &mut scratch)
        .unwrap();
    kv.clear();

    // Decoding at position 1 against a cleared cache must fail loudly (the
    // watermark gate), not silently attend over zeroed memory.
    let mut hidden2 = vec![0.0f32; hs];
    model.embed_token(2, &mut hidden2).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.forward_from_hidden(&mut hidden2, 1, &mut kv, true, &mut scratch);
    }));
    assert!(
        result.is_err(),
        "forward at position 1 on a cleared cache should panic via the watermark gate"
    );
}
