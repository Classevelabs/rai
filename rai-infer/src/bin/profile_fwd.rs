#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "The profiler mirrors the forward-pass hot path and uses explicit indices to keep timings comparable to production kernels."
)]

//! Per-operation timing profiler for RaiModel forward pass.
//! Measures time spent in each operation across all layers.

use anyhow::{Context, Result};
use rai_infer::model::{InferenceWork, RaiModel};
use std::time::Instant;

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();

    let model_path = std::path::Path::new("rai-infer/scripts/smollm-135m-q4.raimodel");
    eprintln!("Loading model...");
    let model = RaiModel::load(model_path).context("loading model")?;

    let hs = model.config.hidden_size as usize;
    let nh = model.config.num_heads as usize;
    let nkv = model.config.num_kv_heads as usize;
    let hd = model.config.head_dim as usize;
    let nl = model.config.num_layers as usize;
    let eps = model.config.norm_eps;
    let vs = model.config.vocab_size as usize;

    let max_ctx = 512;
    let mut kv_cache = model.create_kv_cache(max_ctx);
    let mut work = InferenceWork::new();

    // Fill KV cache with some positions so attention has work to do
    let pos = 32;
    let mut hidden = vec![0.1f32; hs];
    for p in 0..pos {
        model.embed_token(1, &mut hidden)?;
        model.forward_from_hidden(&mut hidden, p, &mut kv_cache, true, &mut work.scratch)?;
    }

    // Now profile one full forward pass at position `pos`
    model.embed_token(42, &mut hidden)?;

    // Profile individual operations
    let file = model.file_ref();
    let rope = rai_infer::layers::RoPETable::new(hd, max_ctx, model.config.rope_theta);

    let mut normed = vec![0.0f32; hs];
    let mut attn_out = vec![0.0f32; hs];
    let mut mlp_out = vec![0.0f32; hs];
    let mut residual = vec![0.0f32; hs];
    let mut attn_work = rai_infer::layers::AttentionWork::new();
    let mut mlp_work = rai_infer::layers::MlpWork::new();

    // Warmup
    for _ in 0..3 {
        let layer = file.layer(0)?;
        rai_infer::layers::rms_norm(
            &mut normed,
            &hidden,
            &rai_infer::format::read_norm_weights(&layer.input_layernorm),
            eps,
        );
    }

    // Detailed per-operation timing (aggregate across all layers)
    let iters = 30;
    let mut t_norm1_total = 0.0f64;
    let mut t_qkv_total = 0.0f64;
    let mut t_rope_total = 0.0f64;
    let mut t_kv_store_total = 0.0f64;
    let mut t_attn_total = 0.0f64;
    let mut t_oproj_total = 0.0f64;
    let mut t_residual1_total = 0.0f64;
    let mut t_norm2_total = 0.0f64;
    let mut t_gate_up_total = 0.0f64;
    let mut t_residual2_total = 0.0f64;
    let mut t_final_norm_total = 0.0f64;
    let mut t_lm_head_total = 0.0f64;

    let norm_weights: Vec<_> = (0..nl)
        .map(|i| {
            let l = file.layer(i).unwrap();
            (
                rai_infer::format::read_norm_weights(&l.input_layernorm),
                rai_infer::format::read_norm_weights(&l.post_attn_layernorm),
            )
        })
        .collect();
    let final_norm_weights = rai_infer::format::read_norm_weights(&file.final_norm().unwrap());

    for _ in 0..iters {
        model.embed_token(42, &mut hidden)?;

        for li in 0..nl {
            let layer = file.layer(li)?;

            // RMSNorm 1 (fused with residual save)
            let t = Instant::now();
            rai_infer::layers::rms_norm_with_residual(
                &mut normed,
                &mut residual,
                &hidden,
                &norm_weights[li].0,
                eps,
            );
            t_norm1_total += t.elapsed().as_secs_f64();

            // Fused QKV
            attn_work.q.resize(nh * hd, 0.0);
            attn_work.k.resize(nkv * hd, 0.0);
            attn_work.v.resize(nkv * hd, 0.0);
            let t = Instant::now();
            rai_infer::gemm::w4a32_fused_qkv(
                &mut attn_work.q,
                &mut attn_work.k,
                &mut attn_work.v,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &normed,
            );
            t_qkv_total += t.elapsed().as_secs_f64();

            // RoPE
            let t = Instant::now();
            rope.apply(&mut attn_work.q, nh, pos);
            rope.apply(&mut attn_work.k, nkv, pos);
            t_rope_total += t.elapsed().as_secs_f64();

            // KV store
            let t = Instant::now();
            kv_cache.store(li, pos, &attn_work.k, &attn_work.v);
            t_kv_store_total += t.elapsed().as_secs_f64();

            // Attention scores + weighted sum
            let scale = 1.0 / (hd as f32).sqrt();
            let heads_per_kv = nh / nkv;
            attn_work.attn_out.resize(nh * hd, 0.0);
            attn_work.scores.resize(pos + 1, 0.0);
            let t = Instant::now();
            for qh in 0..nh {
                let kvh = qh / heads_per_kv;
                let q_head = &attn_work.q[qh * hd..(qh + 1) * hd];
                let out_head = &mut attn_work.attn_out[qh * hd..(qh + 1) * hd];
                // AVX2 path
                unsafe {
                    attention_head_wrapper(
                        q_head,
                        out_head,
                        &mut attn_work.scores,
                        &kv_cache,
                        li,
                        kvh,
                        pos,
                        hd,
                        scale,
                    );
                }
            }
            t_attn_total += t.elapsed().as_secs_f64();

            // O_proj
            attn_out.resize(hs, 0.0);
            let t = Instant::now();
            rai_infer::gemm::w4a32_matvec(
                &mut attn_out,
                layer.o_proj.nibble_data,
                layer.o_proj.group_params,
                &attn_work.attn_out,
                layer.o_proj.rows,
                layer.o_proj.cols,
                layer.o_proj.group_size,
            );
            t_oproj_total += t.elapsed().as_secs_f64();

            // Residual add
            let t = Instant::now();
            rai_infer::layers::vec_add(&mut hidden, &residual, &attn_out);
            t_residual1_total += t.elapsed().as_secs_f64();

            // RMSNorm 2 (fused with residual save)
            let t = Instant::now();
            rai_infer::layers::rms_norm_with_residual(
                &mut normed,
                &mut residual,
                &hidden,
                &norm_weights[li].1,
                eps,
            );
            t_norm2_total += t.elapsed().as_secs_f64();

            // Fused gate+up
            let t = Instant::now();
            rai_infer::layers::swiglu_mlp(
                &mut mlp_out,
                &normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                &mut mlp_work,
            );
            // Can't easily separate gate_up vs silu vs down here — time the whole MLP
            let mlp_elapsed = t.elapsed().as_secs_f64();
            t_gate_up_total += mlp_elapsed; // We'll label this "MLP total"

            // Residual add
            let t = Instant::now();
            rai_infer::layers::vec_add(&mut hidden, &residual, &mlp_out);
            t_residual2_total += t.elapsed().as_secs_f64();
        }

        // Final norm
        let t = Instant::now();
        rai_infer::layers::rms_norm(&mut normed, &hidden, &final_norm_weights, eps);
        t_final_norm_total += t.elapsed().as_secs_f64();

        // LM head
        let embed = file.embedding()?;
        let mut logits = vec![0.0f32; vs];
        let t = Instant::now();
        rai_infer::gemm::tied_lm_head(
            &mut logits,
            &normed,
            embed.data,
            embed.group_params,
            embed.vocab_size,
            embed.hidden_size,
            embed.group_size,
        );
        t_lm_head_total += t.elapsed().as_secs_f64();
    }

    let n = iters as f64;
    let us = |total: f64| total / n * 1e6;

    eprintln!(
        "\n=== Per-token forward pass breakdown (avg of {} iters, pos={}) ===",
        iters, pos
    );
    eprintln!("RMSNorm input  (30×):  {:7.1} μs", us(t_norm1_total));
    eprintln!("Fused QKV      (30×):  {:7.1} μs", us(t_qkv_total));
    eprintln!("RoPE           (30×):  {:7.1} μs", us(t_rope_total));
    eprintln!("KV store       (30×):  {:7.1} μs", us(t_kv_store_total));
    eprintln!("Attention      (30×):  {:7.1} μs", us(t_attn_total));
    eprintln!("O_proj         (30×):  {:7.1} μs", us(t_oproj_total));
    eprintln!("Residual add1  (30×):  {:7.1} μs", us(t_residual1_total));
    eprintln!("RMSNorm post   (30×):  {:7.1} μs", us(t_norm2_total));
    eprintln!(
        "MLP total      (30×):  {:7.1} μs  (gate_up + silu + down)",
        us(t_gate_up_total)
    );
    eprintln!("Residual add2  (30×):  {:7.1} μs", us(t_residual2_total));
    eprintln!("Final norm     (1×):   {:7.1} μs", us(t_final_norm_total));
    eprintln!("LM head        (1×):   {:7.1} μs", us(t_lm_head_total));

    let total = us(t_norm1_total)
        + us(t_qkv_total)
        + us(t_rope_total)
        + us(t_kv_store_total)
        + us(t_attn_total)
        + us(t_oproj_total)
        + us(t_residual1_total)
        + us(t_norm2_total)
        + us(t_gate_up_total)
        + us(t_residual2_total)
        + us(t_final_norm_total)
        + us(t_lm_head_total);
    eprintln!(
        "\nTotal:                 {:7.1} μs ({:.1} tok/s)",
        total,
        1e6 / total
    );

    // Group by category
    let gemm_us = us(t_qkv_total) + us(t_oproj_total) + us(t_gate_up_total) + us(t_lm_head_total);
    let serial_us = us(t_norm1_total)
        + us(t_rope_total)
        + us(t_kv_store_total)
        + us(t_attn_total)
        + us(t_residual1_total)
        + us(t_norm2_total)
        + us(t_residual2_total)
        + us(t_final_norm_total);
    eprintln!(
        "\nGEMM compute:          {:7.1} μs ({:.1}%)",
        gemm_us,
        gemm_us / total * 100.0
    );
    eprintln!(
        "Serial overhead:       {:7.1} μs ({:.1}%)",
        serial_us,
        serial_us / total * 100.0
    );

    Ok(())
}

/// Wrapper to call the AVX2 attention from layers module.
/// This mimics what gqa_attention_decode does internally.
unsafe fn attention_head_wrapper(
    q_head: &[f32],
    out_head: &mut [f32],
    scores: &mut [f32],
    kv_cache: &rai_infer::kv_cache::KVCache,
    layer_idx: usize,
    kvh: usize,
    pos: usize,
    head_dim: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let chunks8 = head_dim / 8;
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..=pos {
        let k_cached = kv_cache.get_k(layer_idx, kvh, t, head_dim);
        let mut acc = _mm256_setzero_ps();
        for c in 0..chunks8 {
            let off = c * 8;
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(q_head.as_ptr().add(off)),
                _mm256_loadu_ps(k_cached.as_ptr().add(off)),
                acc,
            );
        }
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let s2 = _mm_add_ps(s, shuf);
        let hi2 = _mm_movehl_ps(s2, s2);
        let dot = _mm_cvtss_f32(_mm_add_ss(s2, hi2));
        let sc = dot * scale;
        *scores.get_unchecked_mut(t) = sc;
        if sc > max_score {
            max_score = sc;
        }
    }
    let mut sum_exp = 0.0f32;
    for t in 0..=pos {
        let v = (*scores.get_unchecked(t) - max_score).exp();
        *scores.get_unchecked_mut(t) = v;
        sum_exp += v;
    }
    let inv_sum = 1.0 / sum_exp;
    for t in 0..=pos {
        *scores.get_unchecked_mut(t) *= inv_sum;
    }
    for c in 0..chunks8 {
        _mm256_storeu_ps(out_head.as_mut_ptr().add(c * 8), _mm256_setzero_ps());
    }
    for t in 0..=pos {
        let v_cached = kv_cache.get_v(layer_idx, kvh, t, head_dim);
        let score_v = _mm256_set1_ps(*scores.get_unchecked(t));
        for c in 0..chunks8 {
            let off = c * 8;
            let cur = _mm256_loadu_ps(out_head.as_ptr().add(off));
            let val = _mm256_loadu_ps(v_cached.as_ptr().add(off));
            _mm256_storeu_ps(
                out_head.as_mut_ptr().add(off),
                _mm256_fmadd_ps(score_v, val, cur),
            );
        }
    }
}
