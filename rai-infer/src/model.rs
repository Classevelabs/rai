//! RaiModel: load any .raimodel, decomposed forward pass primitives.

// The layer loop indexes model metadata, cache slots, and multiple work buffers in lockstep;
// retaining explicit indices keeps those coupled invariants visible and avoids iterator churn.
#![allow(clippy::needless_range_loop)]

use anyhow::{ensure, Context, Result};
use std::path::Path;

use crate::format::{self, ModelConfig, RaiModelFile, NUM_PROJECTIONS};
use crate::gemm::{embed_lookup, tied_lm_head, w4a8_matmul, w4a8_matvec};
use crate::kv_cache::KVCache;
use crate::layers::{
    add_bias, attend_batch, glu_mul_inplace, gqa_attention_decode, rms_norm,
    rms_norm_with_residual, swiglu_mlp, vec_add, AttentionWork, MlpWork, ProjectionBiases,
    RoPETable,
};

/// One layer's decoded bias vectors, indexed in `format::PROJECTION_NAMES`
/// order. Decoded once at load time for the same reason the norm vectors are:
/// the file buffer is a `Vec<u8>` with no alignment guarantee at an arbitrary
/// section offset, and the hot path wants `&[f32]`.
pub type LayerBiases = [Option<Vec<f32>>; NUM_PROJECTIONS];

/// A `LayerBiases` with nothing set — the shape every v1 model has.
fn no_biases() -> LayerBiases {
    Default::default()
}

/// A loaded `.raimodel`, ready to run.
///
/// Owns the one heap buffer the file was read into and the values derived from
/// it at load time: the RoPE tables, the dequantized f32 norm weights, and any
/// projection biases. Quantized weights are *not* copied out — the forward pass
/// reads them straight from the file buffer and unpacks them inside the GEMM
/// inner loop, so no fp32 copy of the weights ever exists.
pub struct RaiModel {
    /// Architecture and quantization parameters read from the file header.
    pub config: ModelConfig,
    file: RaiModelFile,
    rope: RoPETable,
    layer_input_norms: Vec<Vec<f32>>,
    layer_post_attn_norms: Vec<Vec<f32>>,
    layer_biases: Vec<LayerBiases>,
    final_norm_weights: Vec<f32>,
    /// True if the model has a separate (untied) lm_head.
    pub has_separate_lm_head: bool,
}

impl RaiModel {
    /// Read and validate a `.raimodel` file.
    ///
    /// The whole file is read into one heap buffer, its header and section
    /// bounds are checked against each other, and every quantization scale and
    /// norm weight is checked for finiteness before this returns. A malformed
    /// or truncated file is an error here rather than a wrong answer later, so
    /// a model that loads is one the kernels can run.
    ///
    /// # Panics
    /// Every failure — including every form of malformed input and a RoPE
    /// table that does not fit in memory — returns `Err`.
    pub fn load(path: &Path) -> Result<Self> {
        let file = RaiModelFile::open(path).context("loading .raimodel")?;
        let config = file.config.clone();
        let rope = RoPETable::with_scaling(
            config.head_dim as usize,
            config.max_context as usize,
            config.rope_theta,
            config.rope_scaling,
        )
        .context("building the RoPE table")?;

        let num_layers = config.num_layers as usize;
        let mut layer_input_norms = Vec::with_capacity(num_layers);
        let mut layer_post_attn_norms = Vec::with_capacity(num_layers);
        let mut layer_biases = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = file
                .layer(i)
                .with_context(|| format!("parsing layer {i}"))?;
            layer_input_norms.push(format::read_norm_weights(&layer.input_layernorm));
            layer_post_attn_norms.push(format::read_norm_weights(&layer.post_attn_layernorm));
            let mut biases = no_biases();
            for (index, slot) in biases.iter_mut().enumerate() {
                *slot = layer.biases[index].map(format::read_f32_vector);
            }
            layer_biases.push(biases);
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
            layer_biases,
            final_norm_weights,
            has_separate_lm_head,
        })
    }

    /// Access the underlying file for direct layer access (profiling).
    pub fn file_ref(&self) -> &RaiModelFile {
        &self.file
    }

    /// Borrow layer `li`'s bias vectors in `format::PROJECTION_NAMES` order.
    /// Every entry is `None` for a model without biases, and the kernels then
    /// skip the add entirely.
    fn biases_for(&self, li: usize) -> ProjectionBiases<'_> {
        let stored = &self.layer_biases[li];
        std::array::from_fn(|index| stored[index].as_deref())
    }

    /// Allocate a KV cache for this model. Errors (rather than aborting) when
    /// the requested context cannot be allocated; `kv_cache_bytes` lets
    /// callers pre-flight the size.
    pub fn create_kv_cache(&self, max_ctx: usize) -> Result<KVCache> {
        ensure!(max_ctx > 0, "KV cache context must be non-zero");
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
        2usize
            .saturating_mul(nl)
            .saturating_mul(nkv)
            .saturating_mul(max_ctx)
            .saturating_mul(hd)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    /// Model file size in bytes.
    pub fn file_size(&self) -> usize {
        self.file.data_len()
    }

    /// Look up a token's embedding and dequantize to f32.
    pub fn embed_token(&self, token_id: usize, output: &mut [f32]) -> Result<()> {
        let embed = self.file.embedding()?;
        ensure!(
            token_id < embed.vocab_size,
            "token id is out of model vocabulary"
        );
        ensure!(
            output.len() >= embed.hidden_size,
            "embedding output buffer is too small"
        );
        embed_lookup(
            output,
            token_id,
            embed.data,
            embed.group_params,
            embed.vocab_size,
            embed.hidden_size,
            embed.group_size,
        );
        // Gemma multiplies the input embedding by sqrt(hidden_size). This is
        // applied here rather than folded into the stored table on purpose:
        // Gemma also *ties* lm_head to that same table, and `tied_lm_head`
        // must see the unscaled weights or every logit comes out ~45x too
        // large. A folded table would be correct on the way in and wrong on
        // the way out, which no test of the embedding alone would catch.
        if self.config.embed_scale != 1.0 {
            let scale = self.config.embed_scale;
            for value in output[..embed.hidden_size].iter_mut() {
                *value *= scale;
            }
        }
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
        ensure!(hidden.len() == hs, "hidden state has the wrong size");
        ensure!(
            pos < self.config.max_context as usize,
            "position exceeds model context"
        );

        scratch.resize(hs);

        for li in 0..nl {
            let layer = self.file.layer(li)?;
            let biases = self.biases_for(li);

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
                &biases,
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
                self.config.activation,
                &biases,
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
        let hidden_size = self.config.hidden_size as usize;
        let vocab_size = self.config.vocab_size as usize;
        ensure!(hidden.len() >= hidden_size, "hidden buffer is too small");
        ensure!(normed_buf.len() >= hidden_size, "norm buffer is too small");
        ensure!(logits.len() >= vocab_size, "logit buffer is too small");
        rms_norm(
            normed_buf,
            hidden,
            &self.final_norm_weights,
            self.config.norm_eps,
        );
        if self.has_separate_lm_head {
            let lm = self.file.lm_head()?.expect("lm_head section missing");
            w4a8_matvec(
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
        ensure!(
            hidden.len() == hs,
            "partial hidden buffer has the wrong size"
        );
        ensure!(
            pos < self.config.max_context as usize,
            "partial forward position exceeds model context"
        );
        ensure!(
            !layer_indices.is_empty(),
            "partial forward requires at least one layer"
        );
        ensure!(
            layer_indices.iter().all(|&layer| layer < nl),
            "partial forward layer index is out of range"
        );
        ensure!(
            layer_indices.windows(2).all(|pair| pair[0] < pair[1]),
            "partial forward layers must be strictly ascending"
        );

        scratch.resize(hs);

        for &li in layer_indices {
            let layer = self.file.layer(li)?;
            let biases = self.biases_for(li);

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
                &biases,
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
                self.config.activation,
                &biases,
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
        ensure!(batch > 0, "batch must contain at least one token");
        ensure!(
            hiddens.len() == batch.checked_mul(hs).context("batch size overflow")?,
            "batched hidden buffer has the wrong size"
        );
        ensure!(
            positions
                .iter()
                .all(|position| *position < self.config.max_context as usize),
            "batch position exceeds model context"
        );
        ensure!(
            positions
                .windows(2)
                .all(|pair| pair[1] == pair[0].saturating_add(1)),
            "batch positions must be strictly consecutive and ascending"
        );

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
            w4a8_matmul(
                &mut bs.q_batch,
                layer.q_proj.nibble_data,
                layer.q_proj.group_params,
                &bs.normed,
                q_dim,
                hs,
                batch,
                layer.q_proj.group_size,
            );
            w4a8_matmul(
                &mut bs.k_batch,
                layer.k_proj.nibble_data,
                layer.k_proj.group_params,
                &bs.normed,
                kv_dim,
                hs,
                batch,
                layer.k_proj.group_size,
            );
            w4a8_matmul(
                &mut bs.v_batch,
                layer.v_proj.nibble_data,
                layer.v_proj.group_params,
                &bs.normed,
                kv_dim,
                hs,
                batch,
                layer.v_proj.group_size,
            );

            // 2b. Projection biases, added outside the quantized inner loop.
            //     Cost is `batch * rows` adds against `batch * rows * cols`
            //     multiply-accumulates, so keeping it separate is free and
            //     leaves the W4A8 kernels untouched.
            let biases = self.biases_for(li);
            add_bias_batch(&mut bs.q_batch, biases[0], q_dim, batch);
            add_bias_batch(&mut bs.k_batch, biases[1], kv_dim, batch);
            add_bias_batch(&mut bs.v_batch, biases[2], kv_dim, batch);

            // 3a. RoPE + KV store for every token first. This is cheap and
            //     inherently sequential (the cache forbids gaps).
            for b in 0..batch {
                let pos = positions[b];
                self.rope
                    .apply(&mut bs.q_batch[b * q_dim..(b + 1) * q_dim], nh, pos);
                self.rope
                    .apply(&mut bs.k_batch[b * kv_dim..(b + 1) * kv_dim], nkv, pos);
                kv_cache.store(
                    li,
                    pos,
                    &bs.k_batch[b * kv_dim..(b + 1) * kv_dim],
                    &bs.v_batch[b * kv_dim..(b + 1) * kv_dim],
                );
            }

            // 3b. Attend for every token in parallel. Token b still reads only
            //     positions 0..=b, so this is the same causal computation the
            //     interleaved loop performed — just no longer serialized.
            attend_batch(
                &mut bs.attn_out[..batch * q_dim],
                &bs.q_batch[..batch * q_dim],
                kv_cache,
                li,
                positions,
                nh,
                nkv,
                hd,
            );

            // 4. Batched O projection
            w4a8_matmul(
                &mut bs.o_out,
                layer.o_proj.nibble_data,
                layer.o_proj.group_params,
                &bs.attn_out,
                hs,
                q_dim,
                batch,
                layer.o_proj.group_size,
            );
            add_bias_batch(&mut bs.o_out, biases[3], hs, batch);

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
            w4a8_matmul(
                &mut bs.gate_batch,
                layer.gate_proj.nibble_data,
                layer.gate_proj.group_params,
                &bs.normed,
                inter,
                hs,
                batch,
                layer.gate_proj.group_size,
            );
            w4a8_matmul(
                &mut bs.up_batch,
                layer.up_proj.nibble_data,
                layer.up_proj.group_params,
                &bs.normed,
                inter,
                hs,
                batch,
                layer.up_proj.group_size,
            );
            add_bias_batch(&mut bs.gate_batch, biases[4], inter, batch);
            add_bias_batch(&mut bs.up_batch, biases[5], inter, batch);

            // 8. act(gate) * up per token
            for b in 0..batch {
                glu_mul_inplace(
                    self.config.activation,
                    &mut bs.gate_batch[b * inter..(b + 1) * inter],
                    &bs.up_batch[b * inter..(b + 1) * inter],
                    inter,
                );
            }

            // 9. Batched down projection
            w4a8_matmul(
                &mut bs.mlp_out,
                layer.down_proj.nibble_data,
                layer.down_proj.group_params,
                &bs.gate_batch,
                hs,
                inter,
                batch,
                layer.down_proj.group_size,
            );
            add_bias_batch(&mut bs.mlp_out, biases[6], hs, batch);

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
        ensure!(batch > 0, "batch must contain at least one token");
        ensure!(
            hiddens.len() >= batch.checked_mul(hs).context("batch size overflow")?,
            "batched hidden buffer is too small"
        );
        ensure!(
            normed.len() >= batch.checked_mul(hs).context("norm size overflow")?,
            "batched norm buffer is too small"
        );
        ensure!(
            logits.len() >= batch.checked_mul(vs).context("logit size overflow")?,
            "batched logit buffer is too small"
        );

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
            w4a8_matmul(
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

/// Add a bias to every token's slice of a batched activation buffer.
///
/// Delegates to the same [`add_bias`] the single-token decode path uses, on a
/// slice of the same length starting at a multiple of `row`, so the batched and
/// sequential paths stay bit-identical (see `tests/model_invariants.rs`).
fn add_bias_batch(buffer: &mut [f32], bias: Option<&[f32]>, row: usize, batch: usize) {
    let Some(bias) = bias else {
        return;
    };
    for b in 0..batch {
        add_bias(&mut buffer[b * row..(b + 1) * row], Some(bias));
    }
}

/// Scratch workspace for forward passes (does NOT contain the hidden state).
#[derive(Default)]
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

/// Workspace bundle: hidden state + scratch. Convenience for callers.
#[derive(Default)]
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

/// Workspace for batched forward passes (self-speculative verification).
#[derive(Default)]
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
