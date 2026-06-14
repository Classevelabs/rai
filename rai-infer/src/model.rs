//! RaiModel: load any .raimodel, decomposed forward pass primitives.

use anyhow::{Context, Result};
use std::path::Path;

use crate::format::{self, ModelConfig, RaiModelFile};
use crate::gemm::{embed_lookup, tied_lm_head, w4a32_matmul, w4a32_matvec};
use crate::kv_cache::KVCache;
use crate::layers::{
    compute_attention, gqa_attention_decode, rms_norm, rms_norm_with_residual, silu_mul_inplace,
    swiglu_mlp, vec_add, AttentionWork, MlpWork, RoPETable,
};

pub struct RaiModel {
    pub config: ModelConfig,
    file: RaiModelFile,
    rope: RoPETable,
    layer_input_norms: Vec<Vec<f32>>,
    layer_post_attn_norms: Vec<Vec<f32>>,
    final_norm_weights: Vec<f32>,
    /// True if the model has a separate (untied) lm_head.
    pub has_separate_lm_head: bool,
}

impl RaiModel {
    pub fn load(path: &Path) -> Result<Self> {
        let file = RaiModelFile::open(path).context("loading .raimodel")?;
        let config = file.config.clone();
        let rope = RoPETable::new(
            config.head_dim as usize,
            config.max_context as usize,
            config.rope_theta,
        );

        let num_layers = config.num_layers as usize;
        let mut layer_input_norms = Vec::with_capacity(num_layers);
        let mut layer_post_attn_norms = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = file
                .layer(i)
                .with_context(|| format!("parsing layer {i}"))?;
            layer_input_norms.push(format::read_norm_weights(&layer.input_layernorm));
            layer_post_attn_norms.push(format::read_norm_weights(&layer.post_attn_layernorm));
        }
        let final_norm = file.final_norm()?;
        let final_norm_weights = format::read_norm_weights(&final_norm);
        let has_separate_lm_head = file.has_lm_head();

        Ok(Self {
            config,
            file,
            rope,
            layer_input_norms,
            layer_post_attn_norms,
            final_norm_weights,
            has_separate_lm_head,
        })
    }

    /// Access the underlying file for direct layer access (profiling).
    pub fn file_ref(&self) -> &RaiModelFile {
        &self.file
    }

    pub fn create_kv_cache(&self, max_ctx: usize) -> KVCache {
        KVCache::new(
            self.config.num_layers as usize,
            self.config.num_kv_heads as usize,
            max_ctx,
            self.config.head_dim as usize,
        )
    }

    /// Estimate KV cache memory for a given max context length (in bytes).
    pub fn kv_cache_bytes(&self, max_ctx: usize) -> usize {
        let nl = self.config.num_layers as usize;
        let nkv = self.config.num_kv_heads as usize;
        let hd = self.config.head_dim as usize;
        // 2 (K+V) * layers * kv_heads * max_ctx * head_dim * 4 bytes
        2 * nl * nkv * max_ctx * hd * std::mem::size_of::<f32>()
    }

    /// Model file size in bytes.
    pub fn file_size(&self) -> usize {
        self.file.data_len()
    }

    /// Look up a token's embedding and dequantize to f32.
    pub fn embed_token(&self, token_id: usize, output: &mut [f32]) -> Result<()> {
        let embed = self.file.embedding()?;
        embed_lookup(
            output,
            token_id,
            embed.data,
            embed.group_params,
            embed.vocab_size,
            embed.hidden_size,
            embed.group_size,
        );
        Ok(())
    }

    /// Forward pass through all transformer layers.
    /// `hidden` is modified in-place. `scratch` is reusable workspace (separate from hidden).
    pub fn forward_from_hidden(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KVCache,
        store_kv: bool,
        scratch: &mut Scratch,
    ) -> Result<()> {
        let hs = self.config.hidden_size as usize;
        let nh = self.config.num_heads as usize;
        let nkv = self.config.num_kv_heads as usize;
        let hd = self.config.head_dim as usize;
        let nl = self.config.num_layers as usize;
        let eps = self.config.norm_eps;

        scratch.resize(hs);

        for li in 0..nl {
            let layer = self.file.layer(li)?;

            // Attention block
            rms_norm_with_residual(
                &mut scratch.normed,
                &mut scratch.residual,
                hidden,
                &self.layer_input_norms[li],
                eps,
            );
            gqa_attention_decode(
                &mut scratch.attn_out,
                &scratch.normed,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &self.rope,
                kv_cache,
                li,
                pos,
                nh,
                nkv,
                hd,
                &mut scratch.attn_work,
                store_kv,
            );
            vec_add(hidden, &scratch.residual, &scratch.attn_out);

            // MLP block
            rms_norm_with_residual(
                &mut scratch.normed,
                &mut scratch.residual,
                hidden,
                &self.layer_post_attn_norms[li],
                eps,
            );
            swiglu_mlp(
                &mut scratch.mlp_out,
                &scratch.normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                &mut scratch.mlp_work,
            );
            vec_add(hidden, &scratch.residual, &scratch.mlp_out);
        }
        Ok(())
    }

    /// Apply final RMSNorm and compute logits into a pre-allocated buffer.
    /// Uses separate lm_head if available (untied models like Mistral-7B),
    /// otherwise falls back to tied embedding weights (SmolLM-135M).
    pub fn hidden_to_logits_into(
        &self,
        hidden: &[f32],
        normed_buf: &mut [f32],
        logits: &mut [f32],
    ) -> Result<()> {
        rms_norm(
            normed_buf,
            hidden,
            &self.final_norm_weights,
            self.config.norm_eps,
        );
        if self.has_separate_lm_head {
            let lm = self.file.lm_head()?.expect("lm_head section missing");
            w4a32_matvec(
                logits,
                lm.nibble_data,
                lm.group_params,
                normed_buf,
                lm.rows,
                lm.cols,
                lm.group_size,
            );
        } else {
            let embed = self.file.embedding()?;
            tied_lm_head(
                logits,
                normed_buf,
                embed.data,
                embed.group_params,
                embed.vocab_size,
                embed.hidden_size,
                embed.group_size,
            );
        }
        Ok(())
    }

    /// Apply final RMSNorm and compute logits via tied embedding (allocating version).
    pub fn hidden_to_logits(&self, hidden: &[f32], normed_buf: &mut [f32]) -> Result<Vec<f32>> {
        let vs = self.config.vocab_size as usize;
        let mut logits = vec![0.0f32; vs];
        self.hidden_to_logits_into(hidden, normed_buf, &mut logits)?;
        Ok(logits)
    }

    /// Convenience: full forward pass (embed → layers → logits), stores KV.
    pub fn forward(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KVCache,
        hidden: &mut Vec<f32>,
        scratch: &mut Scratch,
    ) -> Result<Vec<f32>> {
        let hs = self.config.hidden_size as usize;
        hidden.resize(hs, 0.0);
        self.embed_token(token_id, hidden)?;
        self.forward_from_hidden(hidden, pos, kv_cache, true, scratch)?;
        scratch.normed.resize(hs, 0.0);
        self.hidden_to_logits(hidden, &mut scratch.normed)
    }

    /// Partial forward: process only selected layers.
    /// Used as the "draft" phase in self-speculative decoding.
    ///
    /// `layer_indices` specifies which layers to run (e.g., [0,4,8,12,...,28,31]
    /// for strided layer skipping that covers the full model depth).
    pub fn forward_partial(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KVCache,
        scratch: &mut Scratch,
        layer_indices: &[usize],
    ) -> Result<()> {
        let hs = self.config.hidden_size as usize;
        let nh = self.config.num_heads as usize;
        let nkv = self.config.num_kv_heads as usize;
        let hd = self.config.head_dim as usize;
        let eps = self.config.norm_eps;
        let nl = self.config.num_layers as usize;

        scratch.resize(hs);

        for &li in layer_indices {
            if li >= nl {
                continue;
            }
            let layer = self.file.layer(li)?;

            rms_norm_with_residual(
                &mut scratch.normed,
                &mut scratch.residual,
                hidden,
                &self.layer_input_norms[li],
                eps,
            );
            gqa_attention_decode(
                &mut scratch.attn_out,
                &scratch.normed,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &self.rope,
                kv_cache,
                li,
                pos,
                nh,
                nkv,
                hd,
                &mut scratch.attn_work,
                true,
            );
            vec_add(hidden, &scratch.residual, &scratch.attn_out);

            rms_norm_with_residual(
                &mut scratch.normed,
                &mut scratch.residual,
                hidden,
                &self.layer_post_attn_norms[li],
                eps,
            );
            swiglu_mlp(
                &mut scratch.mlp_out,
                &scratch.normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                &mut scratch.mlp_work,
            );
            vec_add(hidden, &scratch.residual, &scratch.mlp_out);
        }
        Ok(())
    }

    /// Batched forward: process multiple tokens through all layers.
    /// Linear projections use batched GEMM (read weights once for all tokens).
    /// Attention is per-token (causal). This is the verification phase
    /// of self-speculative decoding.
    ///
    /// `hiddens` layout: `[batch * hidden_size]`, token b at `[b*hs..(b+1)*hs]`.
    /// `positions[b]` is the position for token b (must be in ascending order).
    pub fn forward_batch(
        &self,
        hiddens: &mut [f32],
        positions: &[usize],
        kv_cache: &mut KVCache,
        bs: &mut BatchScratch,
    ) -> Result<()> {
        let batch = positions.len();
        let hs = self.config.hidden_size as usize;
        let nh = self.config.num_heads as usize;
        let nkv = self.config.num_kv_heads as usize;
        let hd = self.config.head_dim as usize;
        let nl = self.config.num_layers as usize;
        let inter = self.config.intermediate_size as usize;
        let eps = self.config.norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        bs.resize(batch, hs, q_dim, kv_dim, inter);

        for li in 0..nl {
            let layer = self.file.layer(li)?;

            // 1. RMSNorm + save residual for each token
            for b in 0..batch {
                let h = &hiddens[b * hs..(b + 1) * hs];
                let n = &mut bs.normed[b * hs..(b + 1) * hs];
                let r = &mut bs.residual[b * hs..(b + 1) * hs];
                rms_norm_with_residual(n, r, h, &self.layer_input_norms[li], eps);
            }

            // 2. Batched QKV projection (read weights once for all tokens)
            w4a32_matmul(
                &mut bs.q_batch,
                layer.q_proj.nibble_data,
                layer.q_proj.group_params,
                &bs.normed,
                q_dim,
                hs,
                batch,
                layer.q_proj.group_size,
            );
            w4a32_matmul(
                &mut bs.k_batch,
                layer.k_proj.nibble_data,
                layer.k_proj.group_params,
                &bs.normed,
                kv_dim,
                hs,
                batch,
                layer.k_proj.group_size,
            );
            w4a32_matmul(
                &mut bs.v_batch,
                layer.v_proj.nibble_data,
                layer.v_proj.group_params,
                &bs.normed,
                kv_dim,
                hs,
                batch,
                layer.v_proj.group_size,
            );

            // 3. Per-token: RoPE + KV store + attention (causal, sequential)
            for b in 0..batch {
                let pos = positions[b];
                compute_attention(
                    &mut bs.attn_out[b * q_dim..(b + 1) * q_dim],
                    &mut bs.q_batch[b * q_dim..(b + 1) * q_dim],
                    &mut bs.k_batch[b * kv_dim..(b + 1) * kv_dim],
                    &bs.v_batch[b * kv_dim..(b + 1) * kv_dim],
                    &self.rope,
                    kv_cache,
                    li,
                    pos,
                    nh,
                    nkv,
                    hd,
                    &mut bs.scores,
                );
            }

            // 4. Batched O projection
            w4a32_matmul(
                &mut bs.o_out,
                layer.o_proj.nibble_data,
                layer.o_proj.group_params,
                &bs.attn_out,
                hs,
                q_dim,
                batch,
                layer.o_proj.group_size,
            );

            // 5. Add residual
            for b in 0..batch {
                let h = &mut hiddens[b * hs..(b + 1) * hs];
                let r = &bs.residual[b * hs..(b + 1) * hs];
                let o = &bs.o_out[b * hs..(b + 1) * hs];
                vec_add(h, r, o);
            }

            // 6. MLP: RMSNorm
            for b in 0..batch {
                let h = &hiddens[b * hs..(b + 1) * hs];
                let n = &mut bs.normed[b * hs..(b + 1) * hs];
                let r = &mut bs.residual[b * hs..(b + 1) * hs];
                rms_norm_with_residual(n, r, h, &self.layer_post_attn_norms[li], eps);
            }

            // 7. Batched gate + up projections
            w4a32_matmul(
                &mut bs.gate_batch,
                layer.gate_proj.nibble_data,
                layer.gate_proj.group_params,
                &bs.normed,
                inter,
                hs,
                batch,
                layer.gate_proj.group_size,
            );
            w4a32_matmul(
                &mut bs.up_batch,
                layer.up_proj.nibble_data,
                layer.up_proj.group_params,
                &bs.normed,
                inter,
                hs,
                batch,
                layer.up_proj.group_size,
            );

            // 8. SiLU(gate) * up per token
            for b in 0..batch {
                silu_mul_inplace(
                    &mut bs.gate_batch[b * inter..(b + 1) * inter],
                    &bs.up_batch[b * inter..(b + 1) * inter],
                    inter,
                );
            }

            // 9. Batched down projection
            w4a32_matmul(
                &mut bs.mlp_out,
                layer.down_proj.nibble_data,
                layer.down_proj.group_params,
                &bs.gate_batch,
                hs,
                inter,
                batch,
                layer.down_proj.group_size,
            );

            // 10. Add residual
            for b in 0..batch {
                let h = &mut hiddens[b * hs..(b + 1) * hs];
                let r = &bs.residual[b * hs..(b + 1) * hs];
                let m = &bs.mlp_out[b * hs..(b + 1) * hs];
                vec_add(h, r, m);
            }
        }
        Ok(())
    }

    /// Compute logits for a batch of hidden states.
    pub fn hidden_to_logits_batch(
        &self,
        hiddens: &[f32],
        normed: &mut [f32],
        logits: &mut [f32],
        batch: usize,
    ) -> Result<()> {
        let hs = self.config.hidden_size as usize;
        let vs = self.config.vocab_size as usize;

        for b in 0..batch {
            rms_norm(
                &mut normed[b * hs..(b + 1) * hs],
                &hiddens[b * hs..(b + 1) * hs],
                &self.final_norm_weights,
                self.config.norm_eps,
            );
        }

        if self.has_separate_lm_head {
            let lm = self.file.lm_head()?.expect("lm_head section missing");
            w4a32_matmul(
                logits,
                lm.nibble_data,
                lm.group_params,
                normed,
                vs,
                hs,
                batch,
                lm.group_size,
            );
        } else {
            // Tied embedding: process per-token (no batched tied_lm_head)
            let embed = self.file.embedding()?;
            for b in 0..batch {
                tied_lm_head(
                    &mut logits[b * vs..(b + 1) * vs],
                    &normed[b * hs..(b + 1) * hs],
                    embed.data,
                    embed.group_params,
                    embed.vocab_size,
                    embed.hidden_size,
                    embed.group_size,
                );
            }
        }
        Ok(())
    }

    /// Access RoPE table (for self-speculative decoder).
    pub fn rope(&self) -> &RoPETable {
        &self.rope
    }
}

/// Scratch workspace for forward passes (does NOT contain the hidden state).
pub struct Scratch {
    pub normed: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub attn_work: AttentionWork,
    pub mlp_work: MlpWork,
    /// Pre-allocated logits buffer (vocab_size f32s). Avoids 192 KB alloc per token.
    pub logits: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            normed: Vec::new(),
            attn_out: Vec::new(),
            mlp_out: Vec::new(),
            residual: Vec::new(),
            attn_work: AttentionWork::new(),
            mlp_work: MlpWork::new(),
            logits: Vec::new(),
        }
    }
    fn resize(&mut self, hs: usize) {
        self.normed.resize(hs, 0.0);
        self.attn_out.resize(hs, 0.0);
        self.mlp_out.resize(hs, 0.0);
        self.residual.resize(hs, 0.0);
    }
    /// Ensure the logits buffer is sized for the given vocab.
    pub fn resize_logits(&mut self, vocab_size: usize) {
        self.logits.resize(vocab_size, 0.0);
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Workspace bundle: hidden state + scratch. Convenience for callers.
pub struct InferenceWork {
    pub hidden: Vec<f32>,
    pub scratch: Scratch,
}

impl InferenceWork {
    pub fn new() -> Self {
        Self {
            hidden: Vec::new(),
            scratch: Scratch::new(),
        }
    }
}

impl Default for InferenceWork {
    fn default() -> Self {
        Self::new()
    }
}

/// Workspace for batched forward passes (self-speculative verification).
pub struct BatchScratch {
    pub normed: Vec<f32>,
    pub residual: Vec<f32>,
    pub q_batch: Vec<f32>,
    pub k_batch: Vec<f32>,
    pub v_batch: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub o_out: Vec<f32>,
    pub gate_batch: Vec<f32>,
    pub up_batch: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub scores: Vec<f32>,
}

impl BatchScratch {
    pub fn new() -> Self {
        Self {
            normed: Vec::new(),
            residual: Vec::new(),
            q_batch: Vec::new(),
            k_batch: Vec::new(),
            v_batch: Vec::new(),
            attn_out: Vec::new(),
            o_out: Vec::new(),
            gate_batch: Vec::new(),
            up_batch: Vec::new(),
            mlp_out: Vec::new(),
            scores: Vec::new(),
        }
    }

    fn resize(&mut self, batch: usize, hs: usize, q_dim: usize, kv_dim: usize, inter: usize) {
        self.normed.resize(batch * hs, 0.0);
        self.residual.resize(batch * hs, 0.0);
        self.q_batch.resize(batch * q_dim, 0.0);
        self.k_batch.resize(batch * kv_dim, 0.0);
        self.v_batch.resize(batch * kv_dim, 0.0);
        self.attn_out.resize(batch * q_dim, 0.0);
        self.o_out.resize(batch * hs, 0.0);
        self.gate_batch.resize(batch * inter, 0.0);
        self.up_batch.resize(batch * inter, 0.0);
        self.mlp_out.resize(batch * hs, 0.0);
    }
}

impl Default for BatchScratch {
    fn default() -> Self {
        Self::new()
    }
}
