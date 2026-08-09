//! Speculative decoding: use a fast draft model to generate candidates,
//! verify them in a single batched forward pass through the target model.
//!
//! Exact target-model sampling is supported only for positive-temperature softmax with
//! top-k/top-p/repetition filtering disabled; other sampler configurations are rejected.
//!
//! Key requirement: draft and target must share the exact same token-to-ID mapping. Matching
//! vocabulary sizes alone cannot prove this, so callers must supply both models with one tokenizer.
//!
//! Performance model (typical dual-channel DDR4, ~25 GB/s):
//!   Draft (50MB, K=8): 8 × 2ms = 16ms
//!   Verify (3.7GB, 1 batched pass): 150ms
//!   Accept ~4-5 tokens → 24-30 tok/s (vs 6 tok/s baseline)

// Draft/verification loops intentionally index parallel token, probability, logit, and cache
// sequences. Explicit counters keep acceptance positions aligned with KV-cache positions.
#![allow(clippy::explicit_counter_loop, clippy::needless_range_loop)]

use anyhow::{bail, Result};
use rand::Rng;

use crate::kv_cache::KVCache;
use crate::model::{BatchScratch, RaiModel, Scratch};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate before verification.
    pub draft_k: usize,
    /// Sampler config (shared between draft and target).
    pub sampler: SamplerConfig,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_k: 6,
            sampler: SamplerConfig {
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
        }
    }
}

/// Metrics from one speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeMetrics {
    pub accepted: usize,
    pub drafted: usize,
    pub produced: usize,
    pub accept_rate: f32,
}

/// Speculative decoder with batched verification.
///
/// Draft model generates K tokens autoregressively (fast, small model).
/// Target model verifies all K tokens in ONE batched forward pass (reads weights once).
pub struct SpeculativeDecoder<'a> {
    pub draft: &'a RaiModel,
    pub target: &'a RaiModel,
    pub draft_kv: KVCache,
    pub target_kv: KVCache,
    draft_scratch: Scratch,
    target_scratch: Scratch, // FIX BUG 1: separate scratch for target
    target_batch_scratch: BatchScratch,
    draft_logits_buf: Vec<f32>, // FIX BUG 4: clear naming
    vocab_size: usize,          // shared vocab size (validated at construction)
    max_ctx: usize,
}

impl<'a> SpeculativeDecoder<'a> {
    pub fn new(draft: &'a RaiModel, target: &'a RaiModel, max_ctx: usize) -> Result<Self> {
        // FIX BUG 3: validate vocab sizes match
        let dvs = draft.config.vocab_size as usize;
        let tvs = target.config.vocab_size as usize;
        if dvs != tvs {
            bail!(
                "Speculative decoding requires matching vocab sizes. \
                 Draft vocab={dvs}, target vocab={tvs}. \
                 Draft and target must use the same tokenizer."
            );
        }

        let draft_ctx = max_ctx.min(draft.config.max_context as usize);
        let target_ctx = max_ctx.min(target.config.max_context as usize);
        let shared_ctx = draft_ctx.min(target_ctx);
        if shared_ctx == 0 {
            bail!("speculative decoding context must be non-zero");
        }
        Ok(Self {
            draft,
            target,
            draft_kv: draft.create_kv_cache(draft_ctx),
            target_kv: target.create_kv_cache(target_ctx),
            draft_scratch: Scratch::new(),
            target_scratch: Scratch::new(),
            target_batch_scratch: BatchScratch::new(),
            draft_logits_buf: vec![0.0; dvs],
            vocab_size: dvs,
            max_ctx: shared_ctx,
        })
    }

    /// Prefill both models with prompt tokens.
    ///
    /// Processes tokens 0..N-2 to fill the KV cache. The last prompt token
    /// is left unprocessed so `step()` can handle it correctly at the right
    /// position (avoiding duplicate KV entries).
    ///
    /// Returns the position for the next `step()` call (= N-1 for N tokens,
    /// or 0 for a single-token prompt).
    pub fn prefill(&mut self, prompt_tokens: &[usize]) -> Result<usize> {
        let dhs = self.draft.config.hidden_size as usize;
        let ths = self.target.config.hidden_size as usize;
        if prompt_tokens.is_empty() {
            bail!("prompt must contain at least one token");
        }
        if prompt_tokens.len() > self.max_ctx {
            bail!("prompt exceeds the speculative context window");
        }
        let mut pos = 0;

        // FIX BUG 2: process tokens 0..N-2, leave last for step()
        let n_to_prefill = if prompt_tokens.len() > 1 {
            prompt_tokens.len() - 1
        } else {
            0
        };

        let mut draft_hidden = vec![0.0f32; dhs];
        let mut target_hidden = vec![0.0f32; ths];

        for &token_id in &prompt_tokens[..n_to_prefill] {
            // Prefill draft
            self.draft.embed_token(token_id, &mut draft_hidden)?;
            self.draft.forward_from_hidden(
                &mut draft_hidden,
                pos,
                &mut self.draft_kv,
                true,
                &mut self.draft_scratch,
            )?;

            // Prefill target (uses its own scratch)
            self.target.embed_token(token_id, &mut target_hidden)?;
            self.target.forward_from_hidden(
                &mut target_hidden,
                pos,
                &mut self.target_kv,
                true,
                &mut self.target_scratch,
            )?;

            pos += 1;
        }
        Ok(pos)
    }

    /// One speculative decoding step with batched verification.
    ///
    /// 1. Draft generates K tokens autoregressively (fast).
    /// 2. Target verifies all K+1 tokens in ONE batched forward pass (1× bandwidth).
    /// 3. Accept/reject via standard speculative algorithm.
    ///
    /// `pos` is the position where `last_token` will be processed.
    /// After prefill of N tokens, pos = N-1 and last_token = prompt[N-1].
    pub fn step(
        &mut self,
        pos: usize,
        last_token: usize,
        config: &SpeculativeConfig,
        all_tokens: &[usize],
        rng: &mut impl Rng,
    ) -> Result<(Vec<usize>, SpeculativeMetrics)> {
        let dhs = self.draft.config.hidden_size as usize;
        let ths = self.target.config.hidden_size as usize;
        let vs = self.vocab_size;
        validate_speculative_config(config)?;
        if pos >= self.max_ctx {
            bail!("decode position exceeds the speculative context window");
        }
        let k = config
            .draft_k
            .min(self.max_ctx.saturating_sub(pos).saturating_sub(1));

        if k == 0 {
            let mut target_hidden = vec![0.0f32; ths];
            self.target.embed_token(last_token, &mut target_hidden)?;
            self.target.forward_from_hidden(
                &mut target_hidden,
                pos,
                &mut self.target_kv,
                true,
                &mut self.target_scratch,
            )?;
            self.target_scratch.normed.resize(ths, 0.0);
            let mut logits = vec![0.0; vs];
            self.target.hidden_to_logits_into(
                &target_hidden,
                &mut self.target_scratch.normed,
                &mut logits,
            )?;
            apply_repetition_penalty(&mut logits, all_tokens, config.sampler.repetition_penalty);
            let token = sample_token(&mut logits, &config.sampler, rng);
            return Ok((
                vec![token],
                SpeculativeMetrics {
                    accepted: 0,
                    drafted: 0,
                    produced: 1,
                    accept_rate: 0.0,
                },
            ));
        }

        // ================================================================
        // Phase 1: Draft generates K candidate tokens (autoregressive)
        // ================================================================
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits_list = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut draft_pos = pos;
        let mut draft_hidden = vec![0.0f32; dhs];

        for _ in 0..k {
            // Draft forward: embed + full forward
            self.draft.embed_token(current_token, &mut draft_hidden)?;
            self.draft.forward_from_hidden(
                &mut draft_hidden,
                draft_pos,
                &mut self.draft_kv,
                true,
                &mut self.draft_scratch,
            )?;

            // Get draft logits
            self.draft_scratch.normed.resize(dhs, 0.0);
            self.draft_logits_buf.resize(vs, 0.0);
            self.draft.hidden_to_logits_into(
                &draft_hidden,
                &mut self.draft_scratch.normed,
                &mut self.draft_logits_buf,
            )?;

            // Apply repetition penalty
            let mut context = all_tokens.to_vec();
            context.extend_from_slice(&draft_tokens);
            apply_repetition_penalty(
                &mut self.draft_logits_buf,
                &context,
                config.sampler.repetition_penalty,
            );

            // Save draft logits (before sampling mutates them) and sample
            let draft_logits_snapshot = self.draft_logits_buf.clone();
            let next = sample_token(&mut self.draft_logits_buf, &config.sampler, rng);

            draft_logits_list.push(draft_logits_snapshot);
            draft_tokens.push(next);
            current_token = next;
            draft_pos += 1;
        }

        let n_drafted = draft_tokens.len();
        if n_drafted == 0 {
            return Ok((
                vec![],
                SpeculativeMetrics {
                    accepted: 0,
                    drafted: 0,
                    produced: 0,
                    accept_rate: 0.0,
                },
            ));
        }

        // ================================================================
        // Phase 2: Target verifies ALL tokens in ONE batched forward pass
        // ================================================================
        // Tokens: [last_token, draft_0, draft_1, ..., draft_{K-1}]
        // Positions: [pos, pos+1, ..., pos+K]
        // Target logits[i] predicts what comes after position pos+i.
        let verify_batch = n_drafted + 1;

        let mut verify_tokens = Vec::with_capacity(verify_batch);
        verify_tokens.push(last_token);
        verify_tokens.extend_from_slice(&draft_tokens);

        let verify_positions: Vec<usize> = (0..verify_batch).map(|i| pos + i).collect();

        // Embed all tokens into target hidden states
        let mut verify_hiddens = vec![0.0f32; verify_batch * ths];
        for (i, &tok) in verify_tokens.iter().enumerate() {
            self.target
                .embed_token(tok, &mut verify_hiddens[i * ths..(i + 1) * ths])?;
        }

        // Batched forward through ALL target layers (reads weights once!)
        self.target.forward_batch(
            &mut verify_hiddens,
            &verify_positions,
            &mut self.target_kv,
            &mut self.target_batch_scratch,
        )?;

        // Compute logits for all positions
        let mut verify_normed = vec![0.0f32; verify_batch * ths];
        let mut all_target_logits = vec![0.0f32; verify_batch * vs];
        self.target.hidden_to_logits_batch(
            &verify_hiddens,
            &mut verify_normed,
            &mut all_target_logits,
            verify_batch,
        )?;

        // ================================================================
        // Phase 3: Accept/reject
        // ================================================================
        let mut accepted_tokens = Vec::with_capacity(n_drafted + 1);
        let mut n_accepted = 0;

        for i in 0..n_drafted {
            let draft_token = draft_tokens[i];
            let target_logits_i = &mut all_target_logits[i * vs..(i + 1) * vs];

            // Apply repetition penalty to target logits
            let mut context = all_tokens.to_vec();
            for j in 0..i {
                context.push(draft_tokens[j]);
            }
            apply_repetition_penalty(target_logits_i, &context, config.sampler.repetition_penalty);

            // Compare probabilities
            let p_target =
                softmax_prob_of(target_logits_i, draft_token, config.sampler.temperature);
            let p_draft = softmax_prob_of(
                &draft_logits_list[i],
                draft_token,
                config.sampler.temperature,
            );

            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 0.0 {
                1.0
            } else {
                0.0
            };

            let u: f32 = rng.gen();
            if u < accept_prob {
                accepted_tokens.push(draft_token);
                n_accepted += 1;
            } else {
                // Reject: sample correction from max(0, p_target - p_draft)
                let correction =
                    sample_correction(target_logits_i, &draft_logits_list[i], &config.sampler, rng);
                accepted_tokens.push(correction);
                break;
            }
        }

        // If all K tokens accepted, sample bonus from target's last logits
        if n_accepted == n_drafted && n_drafted > 0 {
            let last_idx = n_drafted;
            if last_idx < verify_batch {
                let target_logits_last = &mut all_target_logits[last_idx * vs..(last_idx + 1) * vs];
                let mut context = all_tokens.to_vec();
                context.extend_from_slice(&draft_tokens);
                apply_repetition_penalty(
                    target_logits_last,
                    &context,
                    config.sampler.repetition_penalty,
                );
                let bonus = sample_token(target_logits_last, &config.sampler, rng);
                accepted_tokens.push(bonus);

                // The draft loop proposed the last draft token but did not yet process it as
                // input. Advance its KV cache so the next step has no missing context position.
                self.draft
                    .embed_token(draft_tokens[n_drafted - 1], &mut draft_hidden)?;
                self.draft.forward_from_hidden(
                    &mut draft_hidden,
                    pos + n_drafted,
                    &mut self.draft_kv,
                    true,
                    &mut self.draft_scratch,
                )?;
            }
        }

        // Drop rejected-draft KV entries in both caches: positions beyond the
        // tokens this step actually produced must not count as valid context.
        // The next step() refills from exactly this frontier, and the cache's
        // gap/watermark checks now enforce that instead of trusting it.
        let produced = accepted_tokens.len();
        self.draft_kv.truncate(pos + produced);
        self.target_kv.truncate(pos + produced);
        let accept_rate = if n_drafted > 0 {
            n_accepted as f32 / n_drafted as f32
        } else {
            0.0
        };

        Ok((
            accepted_tokens,
            SpeculativeMetrics {
                accepted: n_accepted,
                drafted: n_drafted,
                produced,
                accept_rate,
            },
        ))
    }
}

fn validate_speculative_config(config: &SpeculativeConfig) -> Result<()> {
    if config.draft_k == 0 {
        bail!("speculative draft_k must be non-zero");
    }
    if !config.sampler.temperature.is_finite() || config.sampler.temperature <= 1e-6 {
        bail!("speculative decoding requires a finite positive temperature");
    }
    if config.sampler.top_k != 0
        || config.sampler.top_p != 1.0
        || config.sampler.repetition_penalty != 1.0
    {
        bail!("speculative decoding requires top_k=0, top_p=1, and repetition_penalty=1");
    }
    Ok(())
}

/// Softmax probability of a specific token.
fn softmax_prob_of(logits: &[f32], token_id: usize, temperature: f32) -> f32 {
    if token_id >= logits.len() {
        return 0.0;
    }
    softmax_probabilities(logits, temperature)[token_id]
}

fn softmax_probabilities(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature.is_finite() && temperature > 1e-6 {
        temperature
    } else {
        1.0
    };
    let mut scaled: Vec<f32> = logits
        .iter()
        .map(|&value| {
            let value = value / temp;
            if value.is_nan() {
                f32::NEG_INFINITY
            } else {
                value
            }
        })
        .collect();
    let positive_infinities = scaled
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    if positive_infinities > 0 {
        let probability = 1.0 / positive_infinities as f32;
        for value in &mut scaled {
            *value = if *value == f32::INFINITY {
                probability
            } else {
                0.0
            };
        }
        return scaled;
    }
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return vec![1.0 / scaled.len() as f32; scaled.len()];
    }
    let mut sum = 0.0;
    for value in &mut scaled {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / scaled.len() as f32; scaled.len()];
    }
    for value in &mut scaled {
        *value /= sum;
    }
    scaled
}

/// Sample correction token from max(0, p_target - p_draft).
fn sample_correction(
    target_logits: &[f32],
    draft_logits: &[f32],
    config: &SamplerConfig,
    rng: &mut impl Rng,
) -> usize {
    let n = target_logits.len().min(draft_logits.len());
    if n == 0 {
        return 0;
    }
    let target_probs = softmax_probabilities(&target_logits[..n], config.temperature);
    let draft_probs = softmax_probabilities(&draft_logits[..n], config.temperature);

    let mut adj_sum = 0.0f64;
    let mut adjusted = vec![0.0f32; n];
    for i in 0..n {
        let diff = (target_probs[i] - draft_probs[i]).max(0.0);
        adjusted[i] = diff;
        adj_sum += f64::from(diff);
    }

    if !adj_sum.is_finite() || adj_sum <= 0.0 {
        let mut logits_copy = target_logits.to_vec();
        return sample_token(&mut logits_copy, config, rng);
    }

    let u: f64 = rng.gen::<f64>() * adj_sum;
    let mut cumsum = 0.0f64;
    for i in 0..n {
        cumsum += f64::from(adjusted[i]);
        if cumsum >= u {
            return i;
        }
    }
    n - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_mode_rejects_filtered_or_zero_progress_configuration() {
        let mut config = SpeculativeConfig::default();
        assert!(validate_speculative_config(&config).is_ok());
        config.draft_k = 0;
        assert!(validate_speculative_config(&config).is_err());

        config.draft_k = 4;
        config.sampler.top_k = 40;
        assert!(validate_speculative_config(&config).is_err());
    }

    #[test]
    fn probability_helper_handles_non_finite_logits() {
        let probabilities = softmax_probabilities(&[f32::NAN, f32::INFINITY, f32::INFINITY], 1.0);
        assert_eq!(probabilities, vec![0.0, 0.5, 0.5]);

        let probabilities = softmax_probabilities(&[f32::NAN, f32::NEG_INFINITY], 1.0);
        assert_eq!(probabilities, vec![0.5, 0.5]);
    }
}
