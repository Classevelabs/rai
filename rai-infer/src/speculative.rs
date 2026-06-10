//! Speculative decoding: use a fast draft model to generate candidates,
//! verify them in a single batched forward pass through the target model.
//!
//! The output is mathematically identical to pure target-model decoding —
//! no quality loss, only speed gain.
//!
//! Key requirement: draft and target must share the SAME TOKENIZER (same vocab_size).
//! Different tokenizers → 0% acceptance → slower than baseline.
//!
//! Performance model (typical dual-channel DDR4, ~25 GB/s):
//!   Draft (50MB, K=8): 8 × 2ms = 16ms
//!   Verify (3.7GB, 1 batched pass): 150ms
//!   Accept ~4-5 tokens → 24-30 tok/s (vs 6 tok/s baseline)

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
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
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
        let draft_max = self.draft.config.max_context as usize;
        let target_max = self.target.config.max_context as usize;
        let max_pos = draft_max.min(target_max);
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
            if pos >= max_pos {
                break;
            }

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
        let draft_max = self.draft.config.max_context as usize;
        let target_max = self.target.config.max_context as usize;
        let k = config.draft_k;

        // ================================================================
        // Phase 1: Draft generates K candidate tokens (autoregressive)
        // ================================================================
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits_list = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut draft_pos = pos;
        let mut draft_hidden = vec![0.0f32; dhs];

        for _ in 0..k {
            if draft_pos >= draft_max || draft_pos >= target_max {
                break;
            }

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

            let accept_prob = if p_draft > 1e-10 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 1e-10 {
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
            }
        }

        // No KV fixup needed on rejection. Stale entries from rejected drafts
        // sit at positions beyond the accepted range. Since attention is bounded
        // by `0..=pos` (layers.rs), stale entries at higher positions are never
        // attended to. The next step() call naturally overwrites them when the
        // draft/target process new tokens at those positions.

        let produced = accepted_tokens.len();
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

/// Softmax probability of a specific token.
fn softmax_prob_of(logits: &[f32], token_id: usize, temperature: f32) -> f32 {
    if token_id >= logits.len() {
        return 0.0;
    }
    let temp = temperature.max(1e-6);
    let scaled = logits[token_id] / temp;
    let max_l = logits
        .iter()
        .map(|&l| l / temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l / temp - max_l).exp()).sum();
    (scaled - max_l).exp() / sum_exp
}

/// Sample correction token from max(0, p_target - p_draft).
fn sample_correction(
    target_logits: &[f32],
    draft_logits: &[f32],
    config: &SamplerConfig,
    rng: &mut impl Rng,
) -> usize {
    let n = target_logits.len().min(draft_logits.len());
    let temp = config.temperature.max(1e-6);

    let max_t = target_logits
        .iter()
        .map(|&l| l / temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_d = draft_logits
        .iter()
        .map(|&l| l / temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_t: f32 = target_logits
        .iter()
        .map(|&l| (l / temp - max_t).exp())
        .sum();
    let sum_d: f32 = draft_logits.iter().map(|&l| (l / temp - max_d).exp()).sum();

    let mut adj_sum = 0.0f32;
    let mut adjusted = vec![0.0f32; n];
    for i in 0..n {
        let p_t = (target_logits[i] / temp - max_t).exp() / sum_t;
        let p_d = (draft_logits[i] / temp - max_d).exp() / sum_d;
        let diff = (p_t - p_d).max(0.0);
        adjusted[i] = diff;
        adj_sum += diff;
    }

    if adj_sum < 1e-10 {
        let mut logits_copy = target_logits.to_vec();
        return sample_token(&mut logits_copy, config, rng);
    }

    let u: f32 = rng.gen::<f32>() * adj_sum;
    let mut cumsum = 0.0f32;
    for i in 0..n {
        cumsum += adjusted[i];
        if cumsum >= u {
            return i;
        }
    }
    n - 1
}
