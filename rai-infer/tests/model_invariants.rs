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
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % vocab
        })
        .collect()
}

fn sequential_logits(model: &RaiModel, tokens: &[usize], max_ctx: usize) -> Vec<Vec<f32>> {
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
    (0..n)
        .map(|i| logits[i * vs..(i + 1) * vs].to_vec())
        .collect()
}

fn check_batch_matches_sequential(fixture_name: &str) {
    let model = RaiModel::load(&fixture(fixture_name)).expect("load fixture model");
    let vocab = model.config.vocab_size as usize;
    let tokens = token_sequence(vocab, 12);
    let max_ctx = 32;

    let sequential = sequential_logits(&model, &tokens, max_ctx);
    let batched = batched_logits(&model, &tokens, max_ctx);

    assert_eq!(sequential.len(), batched.len());
    for (pos, (s, b)) in sequential.iter().zip(&batched).enumerate() {
        assert!(
            s.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()),
            "non-finite logits at position {pos}"
        );
        // Exact, not close: both paths quantize each activation the same way
        // and sum in the same order, on the AVX2 kernels and the scalar ones,
        // so any difference at all is a kernel that drifted.
        assert_eq!(
            s, b,
            "batched logits differ from sequential at position {pos} ({fixture_name})"
        );
    }
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

/// The logits for the last token after prefilling every earlier one through
/// `RaiModel::prefill` from `start`, with `0..start` already in the cache.
fn logits_after_prefill(model: &RaiModel, tokens: &[usize], split: usize) -> Vec<f32> {
    let hs = model.config.hidden_size as usize;
    let vs = model.config.vocab_size as usize;
    let n = tokens.len();
    let mut kv = model.create_kv_cache(n).unwrap();
    let mut batch = BatchScratch::new();
    // Two calls: the second resumes at `split`, the way the server reuses a
    // previous turn's cache.
    model
        .prefill(&tokens[..split], 0, &mut kv, &mut batch)
        .unwrap();
    model
        .prefill(&tokens[split..n - 1], split, &mut kv, &mut batch)
        .unwrap();
    assert_eq!(kv.filled_min(), n - 1);

    let mut hidden = vec![0.0f32; hs];
    model.embed_token(tokens[n - 1], &mut hidden).unwrap();
    model
        .forward_from_hidden(&mut hidden, n - 1, &mut kv, true, &mut Scratch::new())
        .unwrap();
    let mut normed = vec![0.0f32; hs];
    let mut logits = vec![0.0f32; vs];
    model
        .hidden_to_logits_into(&hidden, &mut normed, &mut logits)
        .unwrap();
    logits
}

/// Prefill runs in `PREFILL_CHUNK`-token batches. Where the chunk boundaries
/// fall, and whether the prefill resumes from a cached prefix, must not show
/// in the cache: the next token's logits agree with the fully sequential path.
#[test]
fn chunked_and_resumed_prefill_match_the_sequential_path() {
    use rai_infer::model::PREFILL_CHUNK;
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture model");
    let vocab = model.config.vocab_size as usize;
    let n = PREFILL_CHUNK + 37;
    let tokens = token_sequence(vocab, n);
    let sequential = sequential_logits(&model, &tokens, n);
    let expected = sequential.last().unwrap();

    for split in [0, 1, 100, PREFILL_CHUNK, n - 1] {
        let got = logits_after_prefill(&model, &tokens, split);
        assert_eq!(&got, expected, "logits differ (split {split})");
    }
}

/// A cache reused for a shorter sequence holds exactly that sequence: what a
/// longer, earlier one left past it no longer counts as context, and the next
/// position decodes as it does on a fresh cache.
#[test]
fn prefill_into_a_used_cache_discards_the_old_tail() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture model");
    let vocab = model.config.vocab_size as usize;
    let long = token_sequence(vocab, 20);
    let short: Vec<usize> = token_sequence(vocab, 7)
        .into_iter()
        .map(|token| (token + 3) % vocab)
        .collect();
    let hs = model.config.hidden_size as usize;
    let mut kv = model.create_kv_cache(32).unwrap();
    let mut batch = BatchScratch::new();

    model.prefill(&long, 0, &mut kv, &mut batch).unwrap();
    assert_eq!(kv.filled_min(), long.len());
    model
        .prefill(&short[..short.len() - 1], 0, &mut kv, &mut batch)
        .unwrap();
    assert_eq!(
        kv.filled_min(),
        short.len() - 1,
        "the earlier sequence's tail still counts as context"
    );

    let mut hidden = vec![0.0f32; hs];
    model
        .embed_token(short[short.len() - 1], &mut hidden)
        .unwrap();
    model
        .forward_from_hidden(
            &mut hidden,
            short.len() - 1,
            &mut kv,
            true,
            &mut Scratch::new(),
        )
        .unwrap();
    let mut normed = vec![0.0f32; hs];
    let mut reused = vec![0.0f32; vocab];
    model
        .hidden_to_logits_into(&hidden, &mut normed, &mut reused)
        .unwrap();
    assert_eq!(reused, logits_after_prefill(&model, &short, 0));

    // An empty prefill still settles the cache at its start.
    model.prefill(&[], 2, &mut kv, &mut batch).unwrap();
    assert_eq!(kv.filled_min(), 2);
}

/// A prefill that would leave a gap below its start, or run past the cache,
/// is refused with an error instead of reaching the cache's panics.
#[test]
fn prefill_refuses_a_gap_and_an_overrun() {
    let model = RaiModel::load(&fixture("tiny-tied.raimodel")).expect("load fixture model");
    let mut kv = model.create_kv_cache(8).unwrap();
    let mut batch = BatchScratch::new();
    let gap = model.prefill(&[1, 2], 3, &mut kv, &mut batch).unwrap_err();
    assert!(gap.to_string().contains("unwritten"), "{gap}");
    let overrun = model.prefill(&[1; 9], 0, &mut kv, &mut batch).unwrap_err();
    assert!(overrun.to_string().contains("exceeds"), "{overrun}");
    assert_eq!(kv.filled_min(), 0, "a refused prefill stores nothing");
}
