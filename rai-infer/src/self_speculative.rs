//! Self-speculative decoding: use the model's own first N layers as a fast draft.
//!
//! No separate draft model is needed because the target model proposes its own drafts.
//! This path is experimental. Exact sampling requires unfiltered positive-temperature
//! softmax; unsupported sampler configurations are rejected.

// Draft/verification loops intentionally index parallel token, probability, logit, and cache
// sequences. Explicit counters keep acceptance positions aligned with KV-cache positions.
#![allow(clippy::explicit_counter_loop, clippy::needless_range_loop)]

use anyhow::{ensure, Result};
use rand::Rng;

use crate::kv_cache::KVCache;
use crate::model::{BatchScratch, RaiModel, Scratch};
use crate::sampler::{apply_repetition_penalty, sample_token, SamplerConfig};
use crate::speculative::{sample_correction, softmax_prob_of};

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
    pub fn early_exit(num_layers: usize, draft_k: usize, sampler: SamplerConfig) -> Result<Self> {
        ensure!(
            num_layers > 0,
            "self-speculative draft must use at least one layer"
        );
        ensure!(draft_k > 0, "self-speculative draft_k must be non-zero");
        Ok(Self {
            draft_layer_indices: (0..num_layers).collect(),
            draft_k,
            sampler,
        })
    }

    /// Create config using strided layer selection (covers full model depth).
    /// Produces exactly `num_draft_layers` indices, spread evenly and always
    /// ending at the last layer for lm_head compatibility. (The previous
    /// stride-then-append construction could silently run up to 40% more
    /// draft layers than requested.)
    pub fn layer_skip(
        total_layers: usize,
        num_draft_layers: usize,
        draft_k: usize,
        sampler: SamplerConfig,
    ) -> Result<Self> {
        ensure!(total_layers > 0, "model must contain at least one layer");
        ensure!(
            num_draft_layers > 0 && num_draft_layers <= total_layers,
            "draft layer count must be within the model"
        );
        ensure!(draft_k > 0, "self-speculative draft_k must be non-zero");
        let last = total_layers - 1;
        let indices: Vec<usize> = if num_draft_layers == 1 {
            vec![last]
        } else {
            // i * last / (n-1) is strictly increasing for n <= total_layers,
            // starts at 0, and ends exactly at the last layer.
            (0..num_draft_layers)
                .map(|i| i * last / (num_draft_layers - 1))
                .collect()
        };
        debug_assert_eq!(indices.len(), num_draft_layers);
        Ok(Self {
            draft_layer_indices: indices,
            draft_k,
            sampler,
        })
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
    max_ctx: usize,
}

impl<'a> SelfSpecDecoder<'a> {
    pub fn new(model: &'a RaiModel, max_ctx: usize) -> Result<Self> {
        let ctx = max_ctx.min(model.config.max_context as usize);
        ensure!(ctx > 0, "self-speculative context must be non-zero");
        let hs = model.config.hidden_size as usize;
        let vs = model.config.vocab_size as usize;
        Ok(Self {
            model,
            kv_cache: model.create_kv_cache(ctx)?,
            draft_scratch: Scratch::new(),
            batch_scratch: BatchScratch::new(),
            draft_hidden: vec![0.0; hs],
            verify_hiddens: Vec::new(),
            verify_normed: Vec::new(),
            verify_logits: vec![0.0; vs],
            max_ctx: ctx,
        })
    }

    /// Prefill all prompt tokens except the final token, which the first decode step consumes.
    pub fn prefill(&mut self, prompt_tokens: &[usize]) -> Result<usize> {
        let hs = self.model.config.hidden_size as usize;
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt must contain at least one token"
        );
        ensure!(
            prompt_tokens.len() <= self.max_ctx,
            "prompt exceeds the self-speculative context window"
        );
        let mut pos = 0;

        for &token_id in &prompt_tokens[..prompt_tokens.len() - 1] {
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
        ensure!(
            pos < self.max_ctx,
            "decode position exceeds the context window"
        );
        ensure!(
            config.draft_k > 0,
            "self-speculative draft_k must be non-zero"
        );
        ensure!(
            config.sampler.temperature.is_finite()
                && config.sampler.temperature > 1e-6
                && config.sampler.top_k == 0
                && config.sampler.top_p == 1.0
                && config.sampler.repetition_penalty == 1.0,
            "self-speculative decoding requires finite temperature > 0, top_k=0, top_p=1, and repetition_penalty=1"
        );
        ensure!(
            !config.draft_layer_indices.is_empty()
                && config
                    .draft_layer_indices
                    .iter()
                    .all(|&layer| layer < self.model.config.num_layers as usize),
            "self-speculative draft layers are invalid"
        );
        // Keep one verification position for the last drafted token. At the final context
        // position there is no draft budget, so fall back to one exact target-model step.
        let k = config
            .draft_k
            .min(self.max_ctx.saturating_sub(pos).saturating_sub(1));

        if k == 0 {
            self.draft_hidden.resize(hs, 0.0);
            self.model.embed_token(last_token, &mut self.draft_hidden)?;
            self.model.forward_from_hidden(
                &mut self.draft_hidden,
                pos,
                &mut self.kv_cache,
                true,
                &mut self.draft_scratch,
            )?;
            self.draft_scratch.normed.resize(hs, 0.0);
            self.verify_logits.resize(vs, 0.0);
            self.model.hidden_to_logits_into(
                &self.draft_hidden,
                &mut self.draft_scratch.normed,
                &mut self.verify_logits,
            )?;
            apply_repetition_penalty(
                &mut self.verify_logits,
                all_tokens,
                config.sampler.repetition_penalty,
            );
            let token = sample_token(&mut self.verify_logits, &config.sampler, rng);
            return Ok((
                vec![token],
                SelfSpecMetrics {
                    accepted: 0,
                    drafted: 0,
                    produced: 1,
                    accept_rate: 0.0,
                },
            ));
        }

        // ================================================================
        // Phase 1: Draft K tokens using first N layers
        // ================================================================
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_probs = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut draft_pos = pos;

        for _ in 0..k {
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

        // Drop rejected-draft KV entries: the shared cache was written through
        // pos + n_drafted by verification, but only pos + produced positions
        // are real context for the next step. The watermark checks in
        // kv_cache.rs enforce this frontier from here on.
        self.kv_cache.truncate(pos + produced);

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
