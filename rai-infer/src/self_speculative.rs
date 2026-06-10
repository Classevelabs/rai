//! Self-speculative decoding: use the model's own first N layers as a fast draft.
//!
//! No separate draft model needed — the same tokenizer is guaranteed.
//! Draft: first N layers (~N/32 of model bandwidth) → fast token proposals.
//! Verify: all 32 layers via batched GEMM (read weights once for K tokens) → exact quality.
//!
//! Expected speedup: 2-4× over standard decode, depending on draft quality.

use anyhow::Result;
use rand::Rng;

use crate::kv_cache::KVCache;
use crate::model::{BatchScratch, RaiModel, Scratch};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};

/// Configuration for self-speculative decoding.
#[derive(Debug, Clone)]
pub struct SelfSpecConfig {
    /// Layer indices to run for draft (e.g., [0,4,8,...,28,31] for strided skip).
    pub draft_layer_indices: Vec<usize>,
    /// Number of draft tokens to generate per step.
    pub draft_k: usize,
    /// Sampler config.
    pub sampler: SamplerConfig,
}

impl SelfSpecConfig {
    /// Create config using first N layers (simple early exit).
    pub fn early_exit(num_layers: usize, draft_k: usize, sampler: SamplerConfig) -> Self {
        Self {
            draft_layer_indices: (0..num_layers).collect(),
            draft_k,
            sampler,
        }
    }

    /// Create config using strided layer selection (covers full model depth).
    /// Always includes the last layer for better lm_head compatibility.
    pub fn layer_skip(
        total_layers: usize,
        num_draft_layers: usize,
        draft_k: usize,
        sampler: SamplerConfig,
    ) -> Self {
        let stride = total_layers / num_draft_layers.max(1);
        let mut indices: Vec<usize> = (0..total_layers).step_by(stride.max(1)).collect();
        // Always include the last layer
        let last = total_layers - 1;
        if indices.last().copied() != Some(last) {
            indices.push(last);
        }
        Self {
            draft_layer_indices: indices,
            draft_k,
            sampler,
        }
    }
}

/// Metrics from one self-speculative decoding step.
#[derive(Debug, Clone)]
pub struct SelfSpecMetrics {
    pub accepted: usize,
    pub drafted: usize,
    pub produced: usize,
    pub accept_rate: f32,
}

/// Self-speculative decoder: draft with first N layers, verify with full model.
pub struct SelfSpecDecoder<'a> {
    pub model: &'a RaiModel,
    pub kv_cache: KVCache,
    draft_scratch: Scratch,
    batch_scratch: BatchScratch,
    draft_hidden: Vec<f32>,
    verify_hiddens: Vec<f32>,
    verify_normed: Vec<f32>,
    verify_logits: Vec<f32>,
}

impl<'a> SelfSpecDecoder<'a> {
    pub fn new(model: &'a RaiModel, max_ctx: usize) -> Self {
        let ctx = max_ctx.min(model.config.max_context as usize);
        let hs = model.config.hidden_size as usize;
        let vs = model.config.vocab_size as usize;
        Self {
            model,
            kv_cache: model.create_kv_cache(ctx),
            draft_scratch: Scratch::new(),
            batch_scratch: BatchScratch::new(),
            draft_hidden: vec![0.0; hs],
            verify_hiddens: Vec::new(),
            verify_normed: Vec::new(),
            verify_logits: vec![0.0; vs],
        }
    }

    /// Prefill the full model with prompt tokens using standard forward.
    pub fn prefill(&mut self, prompt_tokens: &[usize]) -> Result<usize> {
        let hs = self.model.config.hidden_size as usize;
        let max_ctx = self.model.config.max_context as usize;
        let mut pos = 0;

        for &token_id in prompt_tokens {
            if pos >= max_ctx {
                break;
            }
            self.draft_hidden.resize(hs, 0.0);
            self.model.embed_token(token_id, &mut self.draft_hidden)?;
            self.model.forward_from_hidden(
                &mut self.draft_hidden,
                pos,
                &mut self.kv_cache,
                true,
                &mut self.draft_scratch,
            )?;
            pos += 1;
        }
        Ok(pos)
    }

    /// One self-speculative decoding step.
    ///
    /// 1. Draft K tokens using first N layers (fast).
    /// 2. Verify all tokens using full model with batched GEMM (1× bandwidth).
    /// 3. Accept/reject via standard speculative decoding.
    pub fn step(
        &mut self,
        pos: usize,
        last_token: usize,
        config: &SelfSpecConfig,
        all_tokens: &[usize],
        rng: &mut impl Rng,
    ) -> Result<(Vec<usize>, SelfSpecMetrics)> {
        let hs = self.model.config.hidden_size as usize;
        let vs = self.model.config.vocab_size as usize;
        let max_ctx = self.model.config.max_context as usize;
        let k = config.draft_k;

        // ================================================================
        // Phase 1: Draft K tokens using first N layers
        // ================================================================
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_probs = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut draft_pos = pos;

        for _ in 0..k {
            if draft_pos >= max_ctx {
                break;
            }

            // Embed + partial forward (selected draft layers)
            self.draft_hidden.resize(hs, 0.0);
            self.model
                .embed_token(current_token, &mut self.draft_hidden)?;
            self.model.forward_partial(
                &mut self.draft_hidden,
                draft_pos,
                &mut self.kv_cache,
                &mut self.draft_scratch,
                &config.draft_layer_indices,
            )?;

            // Get draft logits
            self.draft_scratch.normed.resize(hs, 0.0);
            self.verify_logits.resize(vs, 0.0);
            self.model.hidden_to_logits_into(
                &self.draft_hidden,
                &mut self.draft_scratch.normed,
                &mut self.verify_logits,
            )?;

            // Apply repetition penalty
            let mut context = all_tokens.to_vec();
            context.extend_from_slice(&draft_tokens);
            apply_repetition_penalty(
                &mut self.verify_logits,
                &context,
                config.sampler.repetition_penalty,
            );

            // Compute draft probabilities (softmax)
            let draft_logits = self.verify_logits.clone();
            let next = sample_token(&mut self.verify_logits, &config.sampler, rng);

            // Store probability of sampled token
            let prob = softmax_prob_of(&draft_logits, next, config.sampler.temperature);
            draft_probs.push((draft_logits, prob));
            draft_tokens.push(next);
            current_token = next;
            draft_pos += 1;
        }

        let n_drafted = draft_tokens.len();
        if n_drafted == 0 {
            return Ok((
                vec![],
                SelfSpecMetrics {
                    accepted: 0,
                    drafted: 0,
                    produced: 0,
                    accept_rate: 0.0,
                },
            ));
        }

        // ================================================================
        // Phase 2: Verify using full model with batched GEMM
        // ================================================================
        // We need to run all K+1 tokens (last_token + K draft tokens) through
        // the FULL model. The KV cache for layers 0..N was written during drafting,
        // but layers N..L need to be filled. So we run the full model for all tokens.
        //
        // First, we need to "undo" the draft's KV writes for layers 0..N at the
        // draft positions, because the batched full forward will rewrite them
        // (with identical values, since layers 0..N are deterministic).
        // Actually, the batched forward will overwrite them, so we don't need to undo.

        let verify_batch = n_drafted + 1; // last_token + K draft tokens
        self.verify_hiddens.resize(verify_batch * hs, 0.0);
        self.verify_normed.resize(verify_batch * hs, 0.0);

        // Embed all tokens
        let mut verify_tokens = Vec::with_capacity(verify_batch);
        verify_tokens.push(last_token);
        verify_tokens.extend_from_slice(&draft_tokens);

        let mut verify_positions = Vec::with_capacity(verify_batch);
        for i in 0..verify_batch {
            verify_positions.push(pos + i);
        }

        for (i, &tok) in verify_tokens.iter().enumerate() {
            self.model
                .embed_token(tok, &mut self.verify_hiddens[i * hs..(i + 1) * hs])?;
        }

        // Batched forward through ALL layers
        self.model.forward_batch(
            &mut self.verify_hiddens,
            &verify_positions,
            &mut self.kv_cache,
            &mut self.batch_scratch,
        )?;

        // Compute logits for all tokens
        let mut all_target_logits = vec![0.0f32; verify_batch * vs];
        self.model.hidden_to_logits_batch(
            &self.verify_hiddens,
            &mut self.verify_normed,
            &mut all_target_logits,
            verify_batch,
        )?;

        // ================================================================
        // Phase 3: Accept/reject
        // ================================================================
        // Target logits at position i predict token at position i+1.
        // target_logits[0] → what target would generate at pos+1 (should match draft_tokens[0])
        // target_logits[i] → what target would generate at pos+i+1 (should match draft_tokens[i])
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

            let p_target =
                softmax_prob_of(target_logits_i, draft_token, config.sampler.temperature);
            let p_draft = draft_probs[i].1;

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
                    sample_correction(target_logits_i, &draft_probs[i].0, &config.sampler, rng);
                accepted_tokens.push(correction);
                break;
            }
        }

        // If all K tokens accepted, sample bonus from target's last logits
        if n_accepted == n_drafted && n_drafted > 0 {
            let last_idx = n_drafted; // target_logits for the last draft token position
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

        let produced = accepted_tokens.len();
        let accept_rate = if n_drafted > 0 {
            n_accepted as f32 / n_drafted as f32
        } else {
            0.0
        };

        Ok((
            accepted_tokens,
            SelfSpecMetrics {
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
