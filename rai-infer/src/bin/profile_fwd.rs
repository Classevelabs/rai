//! Per-operation timing profiler for RaiModel forward passes.
//!
//! Three things this reports, in the order the analysis needs them:
//!
//! 1. The exact number of weight bytes a decode token must read. That is the
//!    denominator of the memory roofline; pass `--bandwidth-gbs` (measured
//!    with `bw-bench`) and it prints the roofline arithmetic instead of
//!    leaving it to a guess.
//! 2. A decode breakdown at several context positions, so the shift in the
//!    cost distribution as context grows is visible rather than inferred from
//!    one position.
//! 3. A prefill breakdown of the batched path, which is a different kernel mix
//!    from decode and is the worst user-facing latency.
//!
//! Every table ends with a sum-of-parts against an uninstrumented end-to-end
//! run. When those two disagree, the instrumentation is lying and the numbers
//! must not be used.

// The profiler indexes per-layer timing inputs deliberately and keeps explicit
// dimensions so the measured operation stays visible at the call site.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use anyhow::{Context, Result};
use clap::Parser;
use rai_infer::format::RaiModelFile;
use rai_infer::kv_cache::KVCache;
use rai_infer::model::{BatchScratch, RaiModel, Scratch};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "profile-fwd",
    about = "Per-operation timing profile of RaiModel forward passes"
)]
struct Args {
    /// Path to the .raimodel file to profile
    #[arg(long)]
    model: PathBuf,

    /// Comma-separated context positions to profile decode at
    #[arg(long, default_value = "8,128,512")]
    positions: String,

    /// Prompt length to profile the batched prefill path with (0 disables)
    #[arg(long, default_value = "301")]
    prefill: usize,

    /// Decode iterations averaged per position
    #[arg(long, default_value = "20")]
    iters: usize,

    /// Measured achievable read bandwidth in GB/s (from `bw-bench`).
    /// Enables the roofline arithmetic.
    #[arg(long)]
    bandwidth_gbs: Option<f64>,
}

/// Weight bytes a single decode token must read, broken down by tensor role.
struct WeightBytes {
    attn: usize,
    mlp: usize,
    embed: usize,
    lm_head: usize,
}

fn linear_bytes(linear: &rai_infer::format::QuantizedLinear<'_>) -> usize {
    linear.nibble_data.len() + linear.group_params.len()
}

fn weight_bytes(file: &RaiModelFile, num_layers: usize) -> Result<WeightBytes> {
    let mut attn = 0usize;
    let mut mlp = 0usize;
    for i in 0..num_layers {
        let l = file
            .layer(i)
            .with_context(|| format!("reading layer {i}"))?;
        attn += linear_bytes(&l.q_proj)
            + linear_bytes(&l.k_proj)
            + linear_bytes(&l.v_proj)
            + linear_bytes(&l.o_proj);
        mlp += linear_bytes(&l.gate_proj) + linear_bytes(&l.up_proj) + linear_bytes(&l.down_proj);
    }
    let embed = file.embedding()?;
    let embed_bytes = embed.data.len() + embed.group_params.len();
    // A tied model reads the embedding table a second time as the LM head; an
    // untied model reads its own lm_head section instead.
    let lm_head = match file.lm_head()? {
        Some(lm) => linear_bytes(&lm),
        None => embed_bytes,
    };
    Ok(WeightBytes {
        attn,
        mlp,
        // The embedding table is random-access — one row per token — so it is
        // reported but excluded from the streamed-weight roofline. A tied
        // model still streams it in full through `lm_head`.
        embed: embed_bytes,
        lm_head,
    })
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() -> Result<()> {
    rai_infer::gemm::configure_thread_pool();
    let args = Args::parse();
    ensure_profile_cpu_support()?;

    let positions = parse_positions(&args.positions)?;

    eprintln!("Loading model: {}", args.model.display());
    let model = RaiModel::load(&args.model)
        .with_context(|| format!("loading model {}", args.model.display()))?;

    let cfg = &model.config;
    let hs = cfg.hidden_size as usize;
    let nh = cfg.num_heads as usize;
    let nkv = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let nl = cfg.num_layers as usize;
    let inter = cfg.intermediate_size as usize;
    let vs = cfg.vocab_size as usize;
    let eps = cfg.norm_eps;

    eprintln!("\n=== Model ===");
    eprintln!("hidden {hs}  layers {nl}  q_heads {nh}  kv_heads {nkv}  head_dim {hd}");
    eprintln!(
        "intermediate {inter}  vocab {vs}  max_ctx {}  group_size {}",
        cfg.max_context, cfg.group_size
    );
    eprintln!(
        "file {:.1} MB  tied_lm_head {}",
        mb(model.file_size()),
        !model.has_separate_lm_head
    );

    // --- Roofline denominator -------------------------------------------
    let w = weight_bytes(model.file_ref(), nl)?;
    let streamed = w.attn + w.mlp + w.lm_head;
    eprintln!("\n=== Weight bytes read per decode token ===");
    eprintln!("Attention projections (q,k,v,o):  {:8.1} MB", mb(w.attn));
    eprintln!("MLP projections (gate,up,down):   {:8.1} MB", mb(w.mlp));
    eprintln!("LM head:                          {:8.1} MB", mb(w.lm_head));
    eprintln!("-------------------------------------------------");
    eprintln!("Streamed per token:               {:8.1} MB", mb(streamed));
    eprintln!(
        "(embedding table {:.1} MB is random-access, one row per token, and is\n \
         not streamed — it is counted above only via the tied LM head)",
        mb(w.embed)
    );

    if let Some(bw) = args.bandwidth_gbs {
        let ceiling = bw * 1e9 / streamed as f64;
        eprintln!("\n=== Memory roofline ===");
        eprintln!("Measured achievable read bandwidth: {bw:.1} GB/s");
        eprintln!(
            "Ceiling = {:.1} GB/s / {:.1} MB per token = {:.1} tok/s",
            bw,
            mb(streamed),
            ceiling
        );
        eprintln!("(compare a measured decode rate against this to get efficiency %)");
    }

    // --- Decode profile at each position --------------------------------
    let max_pos = positions.iter().copied().max().unwrap_or(0);
    let need_ctx = (max_pos + 2).max(args.prefill + 1);
    let max_ctx = need_ctx.min(cfg.max_context as usize);
    if need_ctx > max_ctx {
        anyhow::bail!(
            "requested positions/prefill need {need_ctx} context but the model allows {max_ctx}"
        );
    }

    for &pos in &positions {
        decode_profile(&model, pos, max_ctx, args.iters)?;
    }

    // --- Prefill profile ------------------------------------------------
    if args.prefill > 1 {
        prefill_profile(&model, args.prefill, max_ctx, eps)?;
    }

    Ok(())
}

fn parse_positions(raw: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        out.push(
            t.parse::<usize>()
                .with_context(|| format!("parsing position {t:?}"))?,
        );
    }
    anyhow::ensure!(!out.is_empty(), "--positions must list at least one value");
    Ok(out)
}

/// Fill a fresh KV cache with `pos` real entries so attention has work to do.
fn warm_cache(model: &RaiModel, pos: usize, max_ctx: usize) -> Result<(KVCache, Vec<f32>)> {
    let hs = model.config.hidden_size as usize;
    let mut kv = model.create_kv_cache(max_ctx)?;
    let mut scratch = Scratch::new();
    let mut hidden = vec![0.0f32; hs];
    for p in 0..pos {
        model.embed_token(1, &mut hidden)?;
        model.forward_from_hidden(&mut hidden, p, &mut kv, true, &mut scratch)?;
    }
    Ok((kv, hidden))
}

fn decode_profile(model: &RaiModel, pos: usize, max_ctx: usize, iters: usize) -> Result<()> {
    let cfg = &model.config;
    let hs = cfg.hidden_size as usize;
    let nh = cfg.num_heads as usize;
    let nkv = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let nl = cfg.num_layers as usize;
    let vs = cfg.vocab_size as usize;
    let eps = cfg.norm_eps;

    let (mut kv_cache, _) = warm_cache(model, pos, max_ctx)?;
    let file = model.file_ref();
    let rope =
        rai_infer::layers::RoPETable::with_scaling(hd, max_ctx, cfg.rope_theta, cfg.rope_scaling)
            .unwrap();

    let mut hidden = vec![0.0f32; hs];
    let mut normed = vec![0.0f32; hs];
    let mut attn_out = vec![0.0f32; hs];
    let mut mlp_out = vec![0.0f32; hs];
    let mut residual = vec![0.0f32; hs];
    let mut attn_work = rai_infer::layers::AttentionWork::new();
    let mut mlp_work = rai_infer::layers::MlpWork::new();

    let mut norm_weights = Vec::with_capacity(nl);
    for i in 0..nl {
        let l = file.layer(i)?;
        norm_weights.push((
            rai_infer::format::read_norm_weights(&l.input_layernorm),
            rai_infer::format::read_norm_weights(&l.post_attn_layernorm),
        ));
    }
    let final_norm_weights =
        rai_infer::format::read_norm_weights(&file.final_norm().context("reading final norm")?);

    // Warmup: fault in the weight pages so the first timed iteration is not
    // measuring page faults.
    let mut warm_scratch = Scratch::new();
    for _ in 0..2 {
        model.embed_token(42, &mut hidden)?;
        model.forward_from_hidden(&mut hidden, pos, &mut kv_cache, true, &mut warm_scratch)?;
    }

    let mut t_norm1 = 0.0f64;
    let mut t_qkv = 0.0f64;
    let mut t_rope = 0.0f64;
    let mut t_kv_store = 0.0f64;
    let mut t_attn = 0.0f64;
    let mut t_oproj = 0.0f64;
    let mut t_res1 = 0.0f64;
    let mut t_norm2 = 0.0f64;
    let mut t_mlp = 0.0f64;
    let mut t_res2 = 0.0f64;
    let mut t_final_norm = 0.0f64;
    let mut t_lm_head = 0.0f64;

    let mut logits = vec![0.0f32; vs];

    for _ in 0..iters {
        model.embed_token(42, &mut hidden)?;

        for li in 0..nl {
            let layer = file.layer(li)?;

            let t = Instant::now();
            rai_infer::layers::rms_norm_with_residual(
                &mut normed,
                &mut residual,
                &hidden,
                &norm_weights[li].0,
                eps,
            );
            t_norm1 += t.elapsed().as_secs_f64();

            attn_work.q.resize(nh * hd, 0.0);
            attn_work.k.resize(nkv * hd, 0.0);
            attn_work.v.resize(nkv * hd, 0.0);
            let t = Instant::now();
            rai_infer::gemm::w4a8_fused_qkv(
                &mut attn_work.q,
                &mut attn_work.k,
                &mut attn_work.v,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &normed,
            );
            t_qkv += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rope.apply(&mut attn_work.q, nh, pos);
            rope.apply(&mut attn_work.k, nkv, pos);
            t_rope += t.elapsed().as_secs_f64();

            let t = Instant::now();
            kv_cache.store(li, pos, &attn_work.k, &attn_work.v);
            t_kv_store += t.elapsed().as_secs_f64();

            attn_work.attn_out.resize(nh * hd, 0.0);
            attn_work.scores.resize(nh * (pos + 1), 0.0);
            let t = Instant::now();
            rai_infer::layers::attention_all_heads_for_profiling(
                &mut attn_work.attn_out,
                &mut attn_work.scores,
                &attn_work.q,
                &kv_cache,
                li,
                pos,
                nh,
                nkv,
                hd,
            );
            t_attn += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rai_infer::gemm::w4a8_matvec(
                &mut attn_out,
                layer.o_proj.nibble_data,
                layer.o_proj.group_params,
                &attn_work.attn_out,
                layer.o_proj.rows,
                layer.o_proj.cols,
                layer.o_proj.group_size,
            );
            t_oproj += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rai_infer::layers::vec_add(&mut hidden, &residual, &attn_out);
            t_res1 += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rai_infer::layers::rms_norm_with_residual(
                &mut normed,
                &mut residual,
                &hidden,
                &norm_weights[li].1,
                eps,
            );
            t_norm2 += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rai_infer::layers::swiglu_mlp(
                &mut mlp_out,
                &normed,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                &mut mlp_work,
                cfg.activation,
                &Default::default(),
            );
            t_mlp += t.elapsed().as_secs_f64();

            let t = Instant::now();
            rai_infer::layers::vec_add(&mut hidden, &residual, &mlp_out);
            t_res2 += t.elapsed().as_secs_f64();
        }

        let t = Instant::now();
        rai_infer::layers::rms_norm(&mut normed, &hidden, &final_norm_weights, eps);
        t_final_norm += t.elapsed().as_secs_f64();

        // Untied models project through their own 4-bit lm_head; tied models
        // re-read the 8-bit embedding table. Timing the wrong one here would
        // measure a kernel decode never runs for this model.
        if model.has_separate_lm_head {
            let lm = file.lm_head()?.context("lm_head section missing")?;
            let t = Instant::now();
            rai_infer::gemm::w4a8_matvec(
                &mut logits,
                lm.nibble_data,
                lm.group_params,
                &normed,
                lm.rows,
                lm.cols,
                lm.group_size,
            );
            t_lm_head += t.elapsed().as_secs_f64();
        } else {
            let embed = file.embedding()?;
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
            t_lm_head += t.elapsed().as_secs_f64();
        }
    }

    // Uninstrumented end-to-end control: if the sum of parts does not match
    // this, the per-op timings are distorted and must not be trusted.
    let mut control_scratch = Scratch::new();
    let t = Instant::now();
    for _ in 0..iters {
        model.embed_token(42, &mut hidden)?;
        model.forward_from_hidden(&mut hidden, pos, &mut kv_cache, true, &mut control_scratch)?;
        control_scratch.normed.resize(hs, 0.0);
        let mut normed_ctl = vec![0.0f32; hs];
        model.hidden_to_logits_into(&hidden, &mut normed_ctl, &mut logits)?;
    }
    let control_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let n = iters as f64;
    let us = |total: f64| total / n * 1e6;
    let rows: [(&str, f64); 12] = [
        ("RMSNorm input", us(t_norm1)),
        ("Fused QKV", us(t_qkv)),
        ("RoPE", us(t_rope)),
        ("KV store", us(t_kv_store)),
        ("Attention", us(t_attn)),
        ("O_proj", us(t_oproj)),
        ("Residual add1", us(t_res1)),
        ("RMSNorm post", us(t_norm2)),
        ("MLP (gate+up+silu+down)", us(t_mlp)),
        ("Residual add2", us(t_res2)),
        ("Final norm", us(t_final_norm)),
        ("LM head", us(t_lm_head)),
    ];
    let total: f64 = rows.iter().map(|r| r.1).sum();

    eprintln!("\n=== DECODE breakdown at pos={pos} (avg of {iters} iters) ===");
    for (name, value) in rows {
        eprintln!("{name:<26} {value:9.1} μs  {:5.1}%", value / total * 100.0);
    }
    eprintln!("{:-<49}", "");
    eprintln!(
        "Sum of parts               {total:9.1} μs  ({:.1} tok/s)",
        1e6 / total
    );
    eprintln!(
        "End-to-end control         {control_us:9.1} μs  ({:.1} tok/s)  [instrumentation overhead {:+.1}%]",
        1e6 / control_us,
        (total - control_us) / control_us * 100.0
    );

    let gemm = us(t_qkv) + us(t_oproj) + us(t_mlp) + us(t_lm_head);
    let attn_side = us(t_attn) + us(t_rope) + us(t_kv_store);
    eprintln!(
        "GEMM {:.1} μs ({:.1}%)   attention+rope+store {:.1} μs ({:.1}%)   elementwise {:.1} μs ({:.1}%)",
        gemm,
        gemm / total * 100.0,
        attn_side,
        attn_side / total * 100.0,
        total - gemm - attn_side,
        (total - gemm - attn_side) / total * 100.0
    );
    Ok(())
}

fn prefill_profile(model: &RaiModel, prompt_len: usize, max_ctx: usize, eps: f32) -> Result<()> {
    let cfg = &model.config;
    let hs = cfg.hidden_size as usize;
    let nh = cfg.num_heads as usize;
    let nkv = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let nl = cfg.num_layers as usize;
    let inter = cfg.intermediate_size as usize;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;

    // generate.rs prefills all but the final prompt token.
    let batch = (prompt_len - 1).min(max_ctx - 1);
    let file = model.file_ref();
    let rope =
        rai_infer::layers::RoPETable::with_scaling(hd, max_ctx, cfg.rope_theta, cfg.rope_scaling)
            .unwrap();

    let mut norm_weights = Vec::with_capacity(nl);
    for i in 0..nl {
        let l = file.layer(i)?;
        norm_weights.push((
            rai_infer::format::read_norm_weights(&l.input_layernorm),
            rai_infer::format::read_norm_weights(&l.post_attn_layernorm),
        ));
    }

    let positions: Vec<usize> = (0..batch).collect();
    let mut hiddens = vec![0.0f32; batch * hs];
    for i in 0..batch {
        model.embed_token(42, &mut hiddens[i * hs..(i + 1) * hs])?;
    }

    // Warmup so the weights are resident before timing.
    {
        let mut kv = model.create_kv_cache(max_ctx)?;
        let mut bs = BatchScratch::new();
        let mut warm = hiddens.clone();
        model.forward_batch(&mut warm, &positions, &mut kv, &mut bs)?;
    }

    let mut kv_cache = model.create_kv_cache(max_ctx)?;
    let mut normed = vec![0.0f32; batch * hs];
    let mut residual = vec![0.0f32; batch * hs];
    let mut q_batch = vec![0.0f32; batch * q_dim];
    let mut k_batch = vec![0.0f32; batch * kv_dim];
    let mut v_batch = vec![0.0f32; batch * kv_dim];
    let mut attn_out = vec![0.0f32; batch * q_dim];
    let mut o_out = vec![0.0f32; batch * hs];
    let mut gate_batch = vec![0.0f32; batch * inter];
    let mut up_batch = vec![0.0f32; batch * inter];
    let mut mlp_out = vec![0.0f32; batch * hs];
    let mut work = hiddens.clone();

    let mut t_norm = 0.0f64;
    let mut t_qkv = 0.0f64;
    let mut t_rope_store = 0.0f64;
    let mut t_attn = 0.0f64;
    let mut t_oproj = 0.0f64;
    let mut t_gate_up = 0.0f64;
    let mut t_silu = 0.0f64;
    let mut t_down = 0.0f64;
    let mut t_res = 0.0f64;

    let t_all = Instant::now();
    for li in 0..nl {
        let layer = file.layer(li)?;

        let t = Instant::now();
        for b in 0..batch {
            rai_infer::layers::rms_norm_with_residual(
                &mut normed[b * hs..(b + 1) * hs],
                &mut residual[b * hs..(b + 1) * hs],
                &work[b * hs..(b + 1) * hs],
                &norm_weights[li].0,
                eps,
            );
        }
        t_norm += t.elapsed().as_secs_f64();

        let t = Instant::now();
        rai_infer::gemm::w4a8_matmul(
            &mut q_batch,
            layer.q_proj.nibble_data,
            layer.q_proj.group_params,
            &normed,
            q_dim,
            hs,
            batch,
            layer.q_proj.group_size,
        );
        rai_infer::gemm::w4a8_matmul(
            &mut k_batch,
            layer.k_proj.nibble_data,
            layer.k_proj.group_params,
            &normed,
            kv_dim,
            hs,
            batch,
            layer.k_proj.group_size,
        );
        rai_infer::gemm::w4a8_matmul(
            &mut v_batch,
            layer.v_proj.nibble_data,
            layer.v_proj.group_params,
            &normed,
            kv_dim,
            hs,
            batch,
            layer.v_proj.group_size,
        );
        t_qkv += t.elapsed().as_secs_f64();

        // Mirror forward_batch: RoPE + store for every token, then attend.
        let t = Instant::now();
        for b in 0..batch {
            let pos = positions[b];
            rope.apply(&mut q_batch[b * q_dim..(b + 1) * q_dim], nh, pos);
            rope.apply(&mut k_batch[b * kv_dim..(b + 1) * kv_dim], nkv, pos);
            kv_cache.store(
                li,
                pos,
                &k_batch[b * kv_dim..(b + 1) * kv_dim],
                &v_batch[b * kv_dim..(b + 1) * kv_dim],
            );
        }
        t_rope_store += t.elapsed().as_secs_f64();

        let t = Instant::now();
        rai_infer::layers::attend_batch(
            &mut attn_out[..batch * q_dim],
            &q_batch[..batch * q_dim],
            &kv_cache,
            li,
            &positions,
            nh,
            nkv,
            hd,
            rai_infer::layers::ScoreShaping::plain(hd),
        );
        t_attn += t.elapsed().as_secs_f64();

        let t = Instant::now();
        rai_infer::gemm::w4a8_matmul(
            &mut o_out,
            layer.o_proj.nibble_data,
            layer.o_proj.group_params,
            &attn_out,
            hs,
            q_dim,
            batch,
            layer.o_proj.group_size,
        );
        t_oproj += t.elapsed().as_secs_f64();

        let t = Instant::now();
        for b in 0..batch {
            rai_infer::layers::vec_add(
                &mut work[b * hs..(b + 1) * hs],
                &residual[b * hs..(b + 1) * hs],
                &o_out[b * hs..(b + 1) * hs],
            );
        }
        for b in 0..batch {
            rai_infer::layers::rms_norm_with_residual(
                &mut normed[b * hs..(b + 1) * hs],
                &mut residual[b * hs..(b + 1) * hs],
                &work[b * hs..(b + 1) * hs],
                &norm_weights[li].1,
                eps,
            );
        }
        t_res += t.elapsed().as_secs_f64();

        let t = Instant::now();
        rai_infer::gemm::w4a8_matmul(
            &mut gate_batch,
            layer.gate_proj.nibble_data,
            layer.gate_proj.group_params,
            &normed,
            inter,
            hs,
            batch,
            layer.gate_proj.group_size,
        );
        rai_infer::gemm::w4a8_matmul(
            &mut up_batch,
            layer.up_proj.nibble_data,
            layer.up_proj.group_params,
            &normed,
            inter,
            hs,
            batch,
            layer.up_proj.group_size,
        );
        t_gate_up += t.elapsed().as_secs_f64();

        let t = Instant::now();
        for b in 0..batch {
            // The model's own activation, so a GeGLU model is not profiled
            // as if it ran SwiGLU.
            rai_infer::layers::glu_mul_inplace(
                cfg.activation,
                &mut gate_batch[b * inter..(b + 1) * inter],
                &up_batch[b * inter..(b + 1) * inter],
                inter,
            );
        }
        t_silu += t.elapsed().as_secs_f64();

        let t = Instant::now();
        rai_infer::gemm::w4a8_matmul(
            &mut mlp_out,
            layer.down_proj.nibble_data,
            layer.down_proj.group_params,
            &gate_batch,
            hs,
            inter,
            batch,
            layer.down_proj.group_size,
        );
        t_down += t.elapsed().as_secs_f64();

        let t = Instant::now();
        for b in 0..batch {
            rai_infer::layers::vec_add(
                &mut work[b * hs..(b + 1) * hs],
                &residual[b * hs..(b + 1) * hs],
                &mlp_out[b * hs..(b + 1) * hs],
            );
        }
        t_res += t.elapsed().as_secs_f64();
    }
    let instrumented_s = t_all.elapsed().as_secs_f64();

    // Uninstrumented control through the real forward_batch.
    let mut kv_ctl = model.create_kv_cache(max_ctx)?;
    let mut bs = BatchScratch::new();
    let mut hid_ctl = hiddens.clone();
    let t = Instant::now();
    model.forward_batch(&mut hid_ctl, &positions, &mut kv_ctl, &mut bs)?;
    let control_s = t.elapsed().as_secs_f64();

    let ms = |v: f64| v * 1e3;
    let rows: [(&str, f64); 9] = [
        ("RMSNorm (per token)", t_norm),
        ("QKV matmul (batched)", t_qkv),
        ("RoPE + KV store (serial)", t_rope_store),
        ("Attention (parallel over tokens)", t_attn),
        ("O_proj matmul (batched)", t_oproj),
        ("gate+up matmul (batched)", t_gate_up),
        ("SiLU*up (per token)", t_silu),
        ("down matmul (batched)", t_down),
        ("Residual adds (per token)", t_res),
    ];
    let total: f64 = rows.iter().map(|r| r.1).sum();

    eprintln!("\n=== PREFILL breakdown, {batch} tokens batched, {nl} layers ===");
    for (name, value) in rows {
        eprintln!(
            "{name:<32} {:9.1} ms  {:5.1}%",
            ms(value),
            value / total * 100.0
        );
    }
    eprintln!("{:-<57}", "");
    eprintln!(
        "Sum of parts                     {:9.1} ms  ({:.1} tok/s)",
        ms(total),
        batch as f64 / total
    );
    eprintln!(
        "Instrumented walk                {:9.1} ms",
        ms(instrumented_s)
    );
    eprintln!(
        "forward_batch control            {:9.1} ms  ({:.1} tok/s)  [instrumentation {:+.1}%]",
        ms(control_s),
        batch as f64 / control_s,
        (instrumented_s - control_s) / control_s * 100.0
    );
    Ok(())
}

fn ensure_profile_cpu_support() -> Result<()> {
    if !rai_infer::gemm::has_avx2() {
        anyhow::bail!("profile_fwd requires an x86_64 CPU with AVX2, FMA, and F16C support");
    }
    Ok(())
}
