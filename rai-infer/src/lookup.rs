//! Prompt-lookup (n-gram) speculative decoding.
//!
//! The draft comes from the context instead of from a model: take the last
//! `n` tokens of the sequence, find the most recent EARLIER occurrence of that
//! same n-gram, and propose the tokens that followed it. Drafting therefore
//! costs a slice scan rather than a forward pass, which is the whole point —
//! the measured failure of the early-exit self-speculative path (0.4-2.2%
//! acceptance on TinyLlama-1.1B, 4-15x slower than plain decoding) is that its
//! draft is both expensive AND a poor predictor. A free draft only has to be
//! right sometimes to win.
//!
//! Verification is the existing exact algorithm, unchanged: the target model
//! scores `[last_token, draft...]` in ONE batched forward pass, each draft
//! token is accepted with probability `min(1, p_target/p_draft)`, and the
//! first rejection is replaced by a sample from the normalised residual
//! `max(0, p_target - p_draft)`. The draft distribution here is a point mass
//! on the copied token (`p_draft = 1`), so acceptance reduces to
//! `p_target(draft_token)` — but it is computed through the same
//! [`softmax_prob_of`] / [`sample_correction`] core as the model-drafted path
//! so there is exactly one acceptance rule in the crate.
//!
//! When no n-gram matches at all, no draft is produced and the step degenerates
//! to a single ordinary decode — verification is never paid for an empty draft.
//!
//! MEASURED STATUS ON THIS ENGINE (TinyLlama-1.1B-q4, i5-10300H, 2026-08-13).
//! The drafting works; the speedup does not, yet. On a 315-token
//! context-quoting QA prompt, K=2 accepts 55.8% of drafted tokens, drafts on
//! 89.6% of steps and yields 1.98 tokens/step — a genuine 2x cut in decode
//! STEPS — but throughput came out at 0.93x baseline (median of 5 interleaved
//! A/B repetitions, range 0.79-1.10x). The reason is that `forward_batch` only
//! partially amortises weight reads on this kernel: fitting cost = a + b*batch
//! to the step timings puts the marginal cost of one extra verification slot at
//! ~44% of a full single-token decode step, instead of the ~0% the technique
//! assumes. That caps ANY speculative scheme here at about (a+b)/b ~= 2.2x even
//! with 100% acceptance, and the break-even at K=2 needs ~1.89 tokens/step
//! against the 1.98 actually achieved — i.e. it straddles break-even. The
//! algorithm is the right one and is exact; the batched GEMM is what has to
//! improve for it to pay.
//!
//! On non-repetitive text (open-ended creative writing) `min_ngram = 1` is
//! actively harmful: it still drafts on 40% of steps but yields only 1.12
//! tokens/step, measuring ~0.58x baseline. Raising `min_ngram` to 2 suppresses
//! those weak single-token matches — drafting drops to 5.6% of steps at 1.00
//! tokens/step, i.e. the path genuinely degrades to plain decoding instead of
//! paying for drafts that cannot pay back.
//!
//! Exact sampling requires an unfiltered positive-temperature softmax; other
//! sampler configurations are rejected, matching the other speculative paths.

// The verification loop indexes parallel token/logit/cache sequences by hand so
// that acceptance positions stay aligned with KV-cache positions.
#![allow(clippy::needless_range_loop)]

use anyhow::{ensure, Result};
use rand::Rng;

use crate::kv_cache::KVCache;
use crate::model::{BatchScratch, RaiModel, Scratch};
use crate::sampler::{sample_token, SamplerConfig};
use crate::speculative::{sample_correction, softmax_prob_of};

/// Configuration for prompt-lookup decoding.
#[derive(Debug, Clone)]
pub struct LookupConfig {
    /// Maximum number of tokens to copy out of the context per step (K).
    pub max_draft: usize,
    /// Longest suffix n-gram to try matching first.
    pub max_ngram: usize,
    /// Shortest suffix n-gram to fall back to (>= 1).
    pub min_ngram: usize,
    /// Sampler config (must be the exact-verification configuration).
    pub sampler: SamplerConfig,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            max_draft: 10,
            max_ngram: 3,
            min_ngram: 1,
            sampler: SamplerConfig {
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
        }
    }
}

/// Where a draft was found in the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupMatch {
    /// Length of the suffix n-gram that matched.
    pub ngram: usize,
    /// Index of the first drafted token in the token sequence.
    pub start: usize,
    /// Number of drafted tokens (>= 1).
    pub len: usize,
}

impl LookupMatch {
    /// The drafted tokens themselves. `tokens` must be the same slice the
    /// match was found in.
    pub fn draft<'a>(&self, tokens: &'a [usize]) -> &'a [usize] {
        &tokens[self.start..self.start + self.len]
    }
}

/// Metrics from one prompt-lookup step.
#[derive(Debug, Clone)]
pub struct LookupMetrics {
    pub accepted: usize,
    pub drafted: usize,
    pub produced: usize,
    pub accept_rate: f32,
    /// n-gram length that produced the draft, `None` when nothing matched.
    pub matched_ngram: Option<usize>,
}

/// Find the draft for the current sequence suffix.
///
/// Tries suffix lengths from `max_ngram` down to `min_ngram` and, for each,
/// scans backwards for the LAST strictly-earlier occurrence of that suffix.
/// The tokens following that occurrence become the draft, capped at
/// `max_draft` and at the end of `tokens` (never drafts past the context end,
/// and never drafts the suffix's own trailing position).
///
/// Returns `None` when nothing matches, when `max_draft == 0`, or when the
/// sequence is too short for any earlier occurrence to exist.
pub fn find_ngram_draft(
    tokens: &[usize],
    min_ngram: usize,
    max_ngram: usize,
    max_draft: usize,
) -> Option<LookupMatch> {
    let len = tokens.len();
    if max_draft == 0 || len < 2 {
        return None;
    }
    // An n-gram match must start at index <= len - n - 1 for a following token
    // to exist, so n can never usefully exceed len - 1.
    let hi = max_ngram.min(len - 1);
    let lo = min_ngram.max(1);
    if lo > hi {
        return None;
    }

    for n in (lo..=hi).rev() {
        let suffix = &tokens[len - n..];
        // Candidate start indices are 0..=len-n-1: index len-n is the suffix
        // itself and has nothing after it. Scan downwards so the FIRST hit is
        // the most recent occurrence.
        for start in (0..=(len - n - 1)).rev() {
            if &tokens[start..start + n] != suffix {
                continue;
            }
            let draft_start = start + n;
            let draft_len = max_draft.min(len - draft_start);
            if draft_len == 0 {
                continue;
            }
            return Some(LookupMatch {
                ngram: n,
                start: draft_start,
                len: draft_len,
            });
        }
    }
    None
}

/// Prompt-lookup decoder: draft by copying from context, verify with the model.
pub struct LookupDecoder<'a> {
    pub model: &'a RaiModel,
    pub kv_cache: KVCache,
    scratch: Scratch,
    batch_scratch: BatchScratch,
    hidden: Vec<f32>,
    verify_hiddens: Vec<f32>,
    verify_normed: Vec<f32>,
    logits: Vec<f32>,
    /// One-hot draft "logits": `NEG_INFINITY` everywhere except the drafted
    /// token, which is set to `INFINITY` for the duration of one comparison.
    /// This feeds the shared acceptance core a genuine point-mass
    /// distribution without a special-cased code path.
    draft_logits: Vec<f32>,
    max_ctx: usize,
}

impl<'a> LookupDecoder<'a> {
    pub fn new(model: &'a RaiModel, max_ctx: usize) -> Result<Self> {
        let ctx = max_ctx.min(model.config.max_context as usize);
        ensure!(ctx > 0, "prompt-lookup context must be non-zero");
        let hs = model.config.hidden_size as usize;
        let vs = model.config.vocab_size as usize;
        Ok(Self {
            model,
            kv_cache: model.create_kv_cache(ctx)?,
            scratch: Scratch::new(),
            batch_scratch: BatchScratch::new(),
            hidden: vec![0.0; hs],
            verify_hiddens: Vec::new(),
            verify_normed: Vec::new(),
            logits: vec![0.0; vs],
            draft_logits: vec![f32::NEG_INFINITY; vs],
            max_ctx: ctx,
        })
    }

    /// Prefill all prompt tokens except the final one, which the first decode
    /// step consumes. Returns the position for the first [`Self::step`] call.
    ///
    /// Uses one batched pass like the plain decode path in `rai-generate`
    /// rather than the token-at-a-time loop the other speculative decoders
    /// use: measured on TinyLlama-1.1B-q4 with a 315-token prompt that was
    /// 16.2 s instead of 34.9 s, and prompt-lookup's whole reason to exist is
    /// long context-heavy prompts.
    pub fn prefill(&mut self, prompt_tokens: &[usize]) -> Result<usize> {
        let hs = self.model.config.hidden_size as usize;
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt must contain at least one token"
        );
        ensure!(
            prompt_tokens.len() <= self.max_ctx,
            "prompt exceeds the prompt-lookup context window"
        );
        let n = prompt_tokens.len() - 1;
        if n == 0 {
            return Ok(0);
        }

        self.verify_hiddens.resize(n * hs, 0.0);
        for (i, &token_id) in prompt_tokens[..n].iter().enumerate() {
            self.model
                .embed_token(token_id, &mut self.verify_hiddens[i * hs..(i + 1) * hs])?;
        }
        let positions: Vec<usize> = (0..n).collect();
        self.model.forward_batch(
            &mut self.verify_hiddens,
            &positions,
            &mut self.kv_cache,
            &mut self.batch_scratch,
        )?;
        Ok(n)
    }

    /// One prompt-lookup decoding step.
    ///
    /// `context` is the whole token sequence so far; its LAST element is the
    /// token that lives at `pos` and has not been through the model yet
    /// (i.e. `pos == context.len() - 1` when the caller starts from
    /// [`Self::prefill`] and appends every produced token). The full sequence
    /// is needed because it is also the search corpus for the draft.
    ///
    /// No generation-history argument beyond that is taken: exact verification
    /// requires `repetition_penalty == 1.0`, so sampling never consults
    /// previously generated tokens.
    pub fn step(
        &mut self,
        pos: usize,
        context: &[usize],
        config: &LookupConfig,
        rng: &mut impl Rng,
    ) -> Result<(Vec<usize>, LookupMetrics)> {
        let hs = self.model.config.hidden_size as usize;
        let vs = self.model.config.vocab_size as usize;
        validate_lookup_config(config)?;
        ensure!(
            !context.is_empty(),
            "prompt-lookup step requires a non-empty context"
        );
        ensure!(
            pos < self.max_ctx,
            "decode position exceeds the context window"
        );
        let last_token = *context.last().unwrap();

        // Keep one verification slot for the last drafted token: draft token i
        // is verified at position pos + 1 + i, so the batch touches pos..=pos+k.
        let budget = self.max_ctx.saturating_sub(pos).saturating_sub(1);
        let matched = if budget == 0 {
            None
        } else {
            find_ngram_draft(
                context,
                config.min_ngram,
                config.max_ngram,
                config.max_draft.min(budget),
            )
        };

        // No n-gram hit: take one ordinary decode step rather than paying a
        // verification batch for an empty draft.
        let Some(matched) = matched else {
            let token = self.decode_one(pos, last_token, config, rng)?;
            return Ok((
                vec![token],
                LookupMetrics {
                    accepted: 0,
                    drafted: 0,
                    produced: 1,
                    accept_rate: 0.0,
                    matched_ngram: None,
                },
            ));
        };

        let draft_tokens: Vec<usize> = matched.draft(context).to_vec();
        let n_drafted = draft_tokens.len();
        debug_assert!(n_drafted >= 1 && n_drafted <= budget);
        ensure!(
            draft_tokens.iter().all(|&t| t < vs),
            "context contains a token outside the model vocabulary"
        );

        // ================================================================
        // Verify [last_token, draft_0..draft_{k-1}] in ONE batched pass
        // ================================================================
        // Positions: [pos, pos+1, ..., pos+k].
        // Logits at batch index i predict the token after position pos+i, so
        // index i is the verdict on draft_tokens[i].
        let verify_batch = n_drafted + 1;
        self.verify_hiddens.resize(verify_batch * hs, 0.0);
        self.verify_normed.resize(verify_batch * hs, 0.0);

        let mut verify_tokens = Vec::with_capacity(verify_batch);
        verify_tokens.push(last_token);
        verify_tokens.extend_from_slice(&draft_tokens);
        let verify_positions: Vec<usize> = (0..verify_batch).map(|i| pos + i).collect();

        for (i, &tok) in verify_tokens.iter().enumerate() {
            self.model
                .embed_token(tok, &mut self.verify_hiddens[i * hs..(i + 1) * hs])?;
        }

        self.model.forward_batch(
            &mut self.verify_hiddens,
            &verify_positions,
            &mut self.kv_cache,
            &mut self.batch_scratch,
        )?;

        let mut all_target_logits = vec![0.0f32; verify_batch * vs];
        self.model.hidden_to_logits_batch(
            &self.verify_hiddens,
            &mut self.verify_normed,
            &mut all_target_logits,
            verify_batch,
        )?;

        // ================================================================
        // Accept / reject — identical rule to speculative.rs
        // ================================================================
        let mut accepted_tokens = Vec::with_capacity(verify_batch);
        let mut n_accepted = 0usize;

        for i in 0..n_drafted {
            let draft_token = draft_tokens[i];
            let target_logits_i = &mut all_target_logits[i * vs..(i + 1) * vs];

            // Point-mass draft distribution for this candidate.
            self.draft_logits[draft_token] = f32::INFINITY;
            let p_target =
                softmax_prob_of(target_logits_i, draft_token, config.sampler.temperature);
            let p_draft =
                softmax_prob_of(&self.draft_logits, draft_token, config.sampler.temperature);

            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 0.0 {
                1.0
            } else {
                0.0
            };

            let u: f32 = rng.gen();
            let accepted = u < accept_prob;
            if accepted {
                accepted_tokens.push(draft_token);
                n_accepted += 1;
            } else {
                let correction =
                    sample_correction(target_logits_i, &self.draft_logits, &config.sampler, rng);
                accepted_tokens.push(correction);
            }
            // Restore the all-`NEG_INFINITY` invariant before the next candidate.
            self.draft_logits[draft_token] = f32::NEG_INFINITY;
            if !accepted {
                break;
            }
        }

        // Every draft token accepted: the target's final verification position
        // yields a free bonus token. `verify_batch == n_drafted + 1`, so index
        // `n_drafted` is always that position.
        if n_accepted == n_drafted {
            let last_idx = n_drafted;
            let target_logits_last = &mut all_target_logits[last_idx * vs..(last_idx + 1) * vs];
            let bonus = sample_token(target_logits_last, &config.sampler, rng);
            accepted_tokens.push(bonus);
        }

        let produced = accepted_tokens.len();

        // Drop rejected-draft KV entries: verification wrote through
        // pos + n_drafted, but only pos + produced positions are real context
        // for the next step. The watermark checks in kv_cache.rs turn a
        // mistake here into a panic instead of silent garbage attention.
        self.kv_cache.truncate(pos + produced);

        // n_drafted >= 1 whenever this point is reached.
        let accept_rate = n_accepted as f32 / n_drafted as f32;
        Ok((
            accepted_tokens,
            LookupMetrics {
                accepted: n_accepted,
                drafted: n_drafted,
                produced,
                accept_rate,
                matched_ngram: Some(matched.ngram),
            },
        ))
    }

    /// One ordinary (non-speculative) decode step at `pos`.
    fn decode_one(
        &mut self,
        pos: usize,
        last_token: usize,
        config: &LookupConfig,
        rng: &mut impl Rng,
    ) -> Result<usize> {
        let hs = self.model.config.hidden_size as usize;
        let vs = self.model.config.vocab_size as usize;
        self.hidden.resize(hs, 0.0);
        self.model.embed_token(last_token, &mut self.hidden)?;
        self.model.forward_from_hidden(
            &mut self.hidden,
            pos,
            &mut self.kv_cache,
            true,
            &mut self.scratch,
        )?;
        self.scratch.normed.resize(hs, 0.0);
        self.logits.resize(vs, 0.0);
        self.model.hidden_to_logits_into(
            &self.hidden,
            &mut self.scratch.normed,
            &mut self.logits,
        )?;
        Ok(sample_token(&mut self.logits, &config.sampler, rng))
    }
}

fn validate_lookup_config(config: &LookupConfig) -> Result<()> {
    ensure!(
        config.max_draft > 0,
        "prompt-lookup max_draft must be non-zero"
    );
    ensure!(
        config.min_ngram >= 1 && config.min_ngram <= config.max_ngram,
        "prompt-lookup requires 1 <= min_ngram <= max_ngram"
    );
    ensure!(
        config.sampler.temperature.is_finite()
            && config.sampler.temperature > 1e-6
            && config.sampler.top_k == 0
            && config.sampler.top_p == 1.0
            && config.sampler.repetition_penalty == 1.0,
        "prompt-lookup decoding requires finite temperature > 0, top_k=0, top_p=1, and repetition_penalty=1"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LookupConfig {
        LookupConfig::default()
    }

    #[test]
    fn finds_the_most_recent_match() {
        // The suffix [1,2] also occurs at index 0; the search must take the
        // later occurrence at index 4 and draft what followed IT.
        let tokens = [1, 2, 9, 9, 1, 2, 7, 8, 1, 2];
        let m = find_ngram_draft(&tokens, 1, 2, 2).expect("match");
        assert_eq!(m.ngram, 2);
        assert_eq!(m.start, 6);
        assert_eq!(m.draft(&tokens), &[7, 8]);
        // (Drafting from index 0 would have produced [9, 9].)
    }

    #[test]
    fn prefers_the_longest_ngram_then_falls_back() {
        // Suffix [5,6,7]: the 3-gram occurs only at index 0.
        let tokens = [5, 6, 7, 1, 2, 3, 6, 7, 4, 5, 6, 7];
        let long = find_ngram_draft(&tokens, 1, 3, 2).expect("3-gram match");
        assert_eq!(long.ngram, 3);
        assert_eq!(long.draft(&tokens), &[1, 2]);

        // Restricting to 2-grams takes the more recent [6,7] at index 6.
        let short = find_ngram_draft(&tokens, 2, 2, 2).expect("2-gram match");
        assert_eq!(short.ngram, 2);
        assert_eq!(short.start, 8);
        assert_eq!(short.draft(&tokens), &[4, 5]);
    }

    #[test]
    fn respects_the_minimum_ngram() {
        // Only the final token repeats, so a min of 2 must find nothing.
        let tokens = [4, 1, 2, 3, 4];
        assert!(find_ngram_draft(&tokens, 2, 3, 4).is_none());
        let m = find_ngram_draft(&tokens, 1, 3, 4).expect("1-gram match");
        assert_eq!(m.ngram, 1);
        assert_eq!(m.start, 1);
        assert_eq!(m.draft(&tokens), &[1, 2, 3, 4]);
    }

    #[test]
    fn returns_nothing_when_absent_or_degenerate() {
        assert!(find_ngram_draft(&[1, 2, 3, 4, 5], 1, 3, 4).is_none());
        assert!(find_ngram_draft(&[], 1, 3, 4).is_none());
        assert!(find_ngram_draft(&[7], 1, 3, 4).is_none());
        // Zero draft budget produces no draft at all.
        assert!(find_ngram_draft(&[1, 2, 1, 2], 1, 2, 0).is_none());
    }

    #[test]
    fn never_drafts_past_the_context_end() {
        // Suffix [3] matches at index 2, leaving only two following tokens
        // even though K asks for 8: the draft must stop at the context end.
        let tokens = [1, 2, 3, 4, 3];
        let m = find_ngram_draft(&tokens, 1, 1, 8).expect("match");
        assert_eq!(m.start, 3);
        assert_eq!(m.len, 2);
        assert_eq!(m.start + m.len, tokens.len());
        assert_eq!(m.draft(&tokens), &[4, 3]);

        // The suffix's own occurrence is never used as its own match: a
        // sequence whose only occurrence of the suffix is the suffix itself
        // yields nothing.
        assert!(find_ngram_draft(&[1, 2, 3], 3, 3, 4).is_none());
    }

    #[test]
    fn caps_draft_at_k_and_at_the_matchable_ngram_length() {
        let tokens = [1, 2, 3, 4, 5, 6, 1, 2];
        let m = find_ngram_draft(&tokens, 1, 2, 3).expect("match");
        assert_eq!(m.ngram, 2);
        assert_eq!(m.draft(&tokens), &[3, 4, 5]);

        // max_ngram longer than the sequence is clamped, not an error.
        let m = find_ngram_draft(&tokens, 1, 99, 2).expect("match");
        assert_eq!(m.ngram, 2);
        assert_eq!(m.draft(&tokens), &[3, 4]);
    }

    #[test]
    fn rejects_filtered_or_zero_progress_configuration() {
        assert!(validate_lookup_config(&cfg()).is_ok());

        let mut config = cfg();
        config.max_draft = 0;
        assert!(validate_lookup_config(&config).is_err());

        let mut config = cfg();
        config.min_ngram = 0;
        assert!(validate_lookup_config(&config).is_err());

        let mut config = cfg();
        config.min_ngram = 4;
        config.max_ngram = 3;
        assert!(validate_lookup_config(&config).is_err());

        let mut config = cfg();
        config.sampler.top_k = 40;
        assert!(validate_lookup_config(&config).is_err());

        let mut config = cfg();
        config.sampler.temperature = 0.0;
        assert!(validate_lookup_config(&config).is_err());
    }

    #[test]
    fn point_mass_draft_logits_are_exactly_one_hot() {
        // The acceptance core must see p_draft == 1 for the drafted token and
        // 0 elsewhere; anything else silently changes the accept probability.
        let mut logits = vec![f32::NEG_INFINITY; 8];
        logits[3] = f32::INFINITY;
        assert_eq!(softmax_prob_of(&logits, 3, 1.0), 1.0);
        assert_eq!(softmax_prob_of(&logits, 0, 1.0), 0.0);
        assert_eq!(softmax_prob_of(&logits, 7, 0.25), 0.0);
    }
}
