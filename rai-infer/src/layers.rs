//! Transformer layer operations: RMSNorm, RoPE, SiLU, GQA attention, SwiGLU MLP.
//!
//! All operations are pure f32, hand-written with AVX2 SIMD acceleration.
//! No external linear algebra dependencies.

// Attention kernels intentionally mirror scalar/SIMD index arithmetic, and their explicit
// dimension arguments document and validate each buffer contract at the call boundary.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::format::QuantizedLinear;
use crate::gemm::{has_avx2, w4a32_fused_gate_up, w4a32_fused_qkv, w4a32_matvec};
use crate::kv_cache::KVCache;

const MAX_ROPE_TABLE_BYTES: usize = 512 * 1024 * 1024;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMSNorm: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
pub fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    let n = input.len();
    assert!(n > 0, "RMSNorm input must not be empty");
    assert_eq!(n, weight.len(), "RMSNorm weight length mismatch");
    assert_eq!(n, output.len(), "RMSNorm output length mismatch");
    assert!(
        eps.is_finite() && eps > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                rms_norm_avx2(output, input, weight, n, eps);
            }
            return;
        }
    }

    let mut sum_sq = 0.0f32;
    for &v in input.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// AVX2 RMSNorm: SIMD sum-of-squares + SIMD multiply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_avx2(output: &mut [f32], input: &[f32], weight: &[f32], n: usize, eps: f32) {
    let chunks8 = n / 8;
    let inp = input.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();

    // Phase 1: SIMD sum of squares
    let mut sq0 = _mm256_setzero_ps();
    let mut sq1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let chunks16 = chunks8 / 2;
    for _ in 0..chunks16 {
        let v0 = _mm256_loadu_ps(inp.add(i));
        let v1 = _mm256_loadu_ps(inp.add(i + 8));
        sq0 = _mm256_fmadd_ps(v0, v0, sq0);
        sq1 = _mm256_fmadd_ps(v1, v1, sq1);
        i += 16;
    }
    if i + 8 <= n {
        let v0 = _mm256_loadu_ps(inp.add(i));
        sq0 = _mm256_fmadd_ps(v0, v0, sq0);
        i += 8;
    }
    let sq_sum = _mm256_add_ps(sq0, sq1);
    let hi = _mm256_extractf128_ps(sq_sum, 1);
    let lo = _mm256_castps256_ps128(sq_sum);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s);
    let s2 = _mm_add_ps(s, shuf);
    let hi2 = _mm_movehl_ps(s2, s2);
    let mut sum_sq = _mm_cvtss_f32(_mm_add_ss(s2, hi2));
    while i < n {
        sum_sq += *inp.add(i) * *inp.add(i);
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let inv_v = _mm256_set1_ps(inv_rms);

    // Phase 2: SIMD multiply: out = input * inv_rms * weight
    i = 0;
    for _ in 0..chunks8 {
        let v = _mm256_loadu_ps(inp.add(i));
        let w = _mm256_loadu_ps(wt.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w));
        i += 8;
    }
    while i < n {
        *out.add(i) = *inp.add(i) * inv_rms * *wt.add(i);
        i += 1;
    }
}

/// Fused residual save + RMSNorm: reads `hidden` once, writes both `residual` (copy)
/// and `normed` (normalized). Saves one full read of hidden per call.
pub fn rms_norm_with_residual(
    normed: &mut [f32],
    residual: &mut [f32],
    hidden: &[f32],
    weight: &[f32],
    eps: f32,
) {
    let n = hidden.len();
    assert!(n > 0, "RMSNorm input must not be empty");
    assert_eq!(n, weight.len(), "RMSNorm weight length mismatch");
    assert_eq!(n, normed.len(), "RMSNorm output length mismatch");
    assert_eq!(n, residual.len(), "residual output length mismatch");
    assert!(
        eps.is_finite() && eps > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                rms_norm_with_residual_avx2(normed, residual, hidden, weight, n, eps);
            }
            return;
        }
    }

    // Scalar fallback
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let v = hidden[i];
        residual[i] = v;
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        normed[i] = hidden[i] * inv_rms * weight[i];
    }
}

/// AVX2 fused residual save + RMSNorm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_with_residual_avx2(
    normed: &mut [f32],
    residual: &mut [f32],
    hidden: &[f32],
    weight: &[f32],
    n: usize,
    eps: f32,
) {
    let chunks8 = n / 8;
    let inp = hidden.as_ptr();
    let wt = weight.as_ptr();
    let out = normed.as_mut_ptr();
    let res = residual.as_mut_ptr();

    // Phase 1: Copy to residual + compute sum of squares
    let mut sq0 = _mm256_setzero_ps();
    let mut sq1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let chunks16 = chunks8 / 2;
    for _ in 0..chunks16 {
        let v0 = _mm256_loadu_ps(inp.add(i));
        let v1 = _mm256_loadu_ps(inp.add(i + 8));
        _mm256_storeu_ps(res.add(i), v0);
        _mm256_storeu_ps(res.add(i + 8), v1);
        sq0 = _mm256_fmadd_ps(v0, v0, sq0);
        sq1 = _mm256_fmadd_ps(v1, v1, sq1);
        i += 16;
    }
    if i + 8 <= n {
        let v0 = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(res.add(i), v0);
        sq0 = _mm256_fmadd_ps(v0, v0, sq0);
        i += 8;
    }
    let sq_sum_v = _mm256_add_ps(sq0, sq1);
    let hi = _mm256_extractf128_ps(sq_sum_v, 1);
    let lo = _mm256_castps256_ps128(sq_sum_v);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s);
    let s2 = _mm_add_ps(s, shuf);
    let hi2 = _mm_movehl_ps(s2, s2);
    let mut sum_sq = _mm_cvtss_f32(_mm_add_ss(s2, hi2));
    while i < n {
        let v = *inp.add(i);
        *res.add(i) = v;
        sum_sq += v * v;
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let inv_v = _mm256_set1_ps(inv_rms);

    // Phase 2: Multiply input * inv_rms * weight → normed
    i = 0;
    for _ in 0..chunks8 {
        let v = _mm256_loadu_ps(inp.add(i));
        let w = _mm256_loadu_ps(wt.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w));
        i += 8;
    }
    while i < n {
        *out.add(i) = *inp.add(i) * inv_rms * *wt.add(i);
        i += 1;
    }
}

/// In-place RMSNorm.
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert!(n > 0, "RMSNorm input must not be empty");
    assert_eq!(n, weight.len(), "RMSNorm weight length mismatch");
    assert!(
        eps.is_finite() && eps > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// SIMD vector add: hidden[i] = residual[i] + addition[i]
pub fn vec_add(hidden: &mut [f32], residual: &[f32], addition: &[f32]) {
    let n = hidden.len();
    assert_eq!(n, residual.len(), "residual length mismatch");
    assert_eq!(n, addition.len(), "addition length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                let mut i = 0usize;
                let chunks8 = n / 8;
                for _ in 0..chunks8 {
                    let r = _mm256_loadu_ps(residual.as_ptr().add(i));
                    let a = _mm256_loadu_ps(addition.as_ptr().add(i));
                    _mm256_storeu_ps(hidden.as_mut_ptr().add(i), _mm256_add_ps(r, a));
                    i += 8;
                }
                while i < n {
                    *hidden.as_mut_ptr().add(i) =
                        *residual.as_ptr().add(i) + *addition.as_ptr().add(i);
                    i += 1;
                }
            }
            return;
        }
    }
    for i in 0..n {
        hidden[i] = residual[i] + addition[i];
    }
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings)
// ---------------------------------------------------------------------------

/// Pre-computed cos/sin tables for RoPE.
pub struct RoPETable {
    /// `cos_table[pos * head_dim/2 + i]` for pos in 0..max_ctx, i in 0..head_dim/2
    pub cos: Vec<f32>,
    /// `sin_table[pos * head_dim/2 + i]`
    pub sin: Vec<f32>,
    pub head_dim: usize,
    pub max_ctx: usize,
}

impl RoPETable {
    /// Build RoPE tables with the given theta and maximum context length.
    pub fn new(head_dim: usize, max_ctx: usize, theta: f32) -> Self {
        assert!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "RoPE head_dim must be positive and even"
        );
        assert!(max_ctx > 0, "RoPE context must be non-zero");
        assert!(
            theta.is_finite() && theta > 0.0,
            "RoPE theta must be finite and positive"
        );
        let half_dim = head_dim / 2;
        let elements = max_ctx
            .checked_mul(half_dim)
            .expect("RoPE table dimensions overflow");
        let bytes = elements
            .checked_mul(2)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .expect("RoPE table byte size overflows");
        assert!(
            bytes <= MAX_ROPE_TABLE_BYTES,
            "RoPE table exceeds the memory budget"
        );
        let mut cos = Vec::new();
        cos.try_reserve_exact(elements)
            .expect("unable to allocate RoPE cosine table");
        cos.resize(elements, 0.0);
        let mut sin = Vec::new();
        sin.try_reserve_exact(elements)
            .expect("unable to allocate RoPE sine table");
        sin.resize(elements, 0.0);

        for pos in 0..max_ctx {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            cos,
            sin,
            head_dim,
            max_ctx,
        }
    }

    /// Apply RoPE rotation to a set of heads at the given position.
    ///
    /// `heads` is `[num_heads * head_dim]`. Each head's pairs (x[2i], x[2i+1])
    /// are rotated by the position-dependent angle.
    pub fn apply(&self, heads: &mut [f32], num_heads: usize, pos: usize) {
        let hd = self.head_dim;
        let half = hd / 2;
        assert!(
            pos < self.max_ctx,
            "RoPE position is outside the context window"
        );
        let expected = num_heads
            .checked_mul(hd)
            .expect("RoPE head dimensions overflow");
        assert_eq!(heads.len(), expected, "RoPE head buffer length mismatch");

        let cos_row = &self.cos[pos * half..(pos + 1) * half];
        let sin_row = &self.sin[pos * half..(pos + 1) * half];

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() && self.head_dim.is_multiple_of(8) {
                unsafe {
                    self.apply_avx2(heads, num_heads, cos_row, sin_row);
                }
                return;
            }
        }

        for h in 0..num_heads {
            let base = h * hd;
            for i in 0..half {
                let x0 = heads[base + i];
                let x1 = heads[base + half + i];
                heads[base + i] = x0 * cos_row[i] - x1 * sin_row[i];
                heads[base + half + i] = x1 * cos_row[i] + x0 * sin_row[i];
            }
        }
    }

    /// AVX2 RoPE: processes 8 pairs per iteration using FMA.
    /// For head_dim=64: 4 iterations per head instead of 32 scalar ops.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn apply_avx2(
        &self,
        heads: &mut [f32],
        num_heads: usize,
        cos_row: &[f32],
        sin_row: &[f32],
    ) {
        let hd = self.head_dim;
        let half = hd / 2;
        let chunks8 = half / 8;
        let hp = heads.as_mut_ptr();
        let cp = cos_row.as_ptr();
        let sp = sin_row.as_ptr();

        for h in 0..num_heads {
            let base = h * hd;
            for c in 0..chunks8 {
                let i = c * 8;
                let cos_v = _mm256_loadu_ps(cp.add(i));
                let sin_v = _mm256_loadu_ps(sp.add(i));
                let x0 = _mm256_loadu_ps(hp.add(base + i));
                let x1 = _mm256_loadu_ps(hp.add(base + half + i));
                // x0*cos - x1*sin
                let r0 = _mm256_fmsub_ps(x0, cos_v, _mm256_mul_ps(x1, sin_v));
                // x1*cos + x0*sin
                let r1 = _mm256_fmadd_ps(x1, cos_v, _mm256_mul_ps(x0, sin_v));
                _mm256_storeu_ps(hp.add(base + i), r0);
                _mm256_storeu_ps(hp.add(base + half + i), r1);
            }
            // Scalar tail
            for i in (chunks8 * 8)..half {
                let x0 = *hp.add(base + i);
                let x1 = *hp.add(base + half + i);
                *hp.add(base + i) = x0 * *cp.add(i) - x1 * *sp.add(i);
                *hp.add(base + half + i) = x1 * *cp.add(i) + x0 * *sp.add(i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SiLU activation
// ---------------------------------------------------------------------------

/// SiLU activation: `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place SiLU on a vector.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// Fused SiLU(gate) * up, result written to gate. SIMD-accelerated.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32], n: usize) {
    assert!(n <= gate.len(), "SiLU gate buffer is too small");
    assert!(n <= up.len(), "SiLU up buffer is too small");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                silu_mul_avx2(gate, up, n);
            }
            return;
        }
    }
    for i in 0..n {
        gate[i] = silu(gate[i]) * up[i];
    }
}

/// AVX2 fast SiLU(x) * y using Schraudolph exp approximation.
/// SiLU(x) = x / (1 + exp(-x)). Approximation error < 2% for |x| < 10.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(gate: &mut [f32], up: &[f32], n: usize) {
    let chunks8 = n / 8;
    let neg_one = _mm256_set1_ps(-1.0);
    let one = _mm256_set1_ps(1.0);
    // Schraudolph constants for exp(): float_bits = a * x + b
    let exp_a = _mm256_set1_ps(12102203.0); // 2^23 / ln(2)
    let exp_b = _mm256_set1_ps(1065353216.0 - 486411.0); // 2^23 * 127 - correction
    let exp_lo = _mm256_set1_ps(0.0);
    let exp_hi = _mm256_set1_ps(2139095040.0);

    for i in 0..chunks8 {
        let off = i * 8;
        let x = _mm256_loadu_ps(gate.as_ptr().add(off));
        let u = _mm256_loadu_ps(up.as_ptr().add(off));

        // exp(-x) via Schraudolph trick
        let neg_x = _mm256_mul_ps(x, neg_one);
        let t = _mm256_add_ps(_mm256_mul_ps(exp_a, neg_x), exp_b);
        let t = _mm256_max_ps(_mm256_min_ps(t, exp_hi), exp_lo);
        let exp_neg_x = _mm256_castsi256_ps(_mm256_cvtps_epi32(t));

        // silu(x) = x / (1 + exp(-x))
        let denom = _mm256_add_ps(one, exp_neg_x);
        let silu_x = _mm256_div_ps(x, denom);

        // result = silu(x) * up
        _mm256_storeu_ps(gate.as_mut_ptr().add(off), _mm256_mul_ps(silu_x, u));
    }

    // Scalar tail
    for i in (chunks8 * 8)..n {
        gate[i] = silu(gate[i]) * up[i];
    }
}

// ---------------------------------------------------------------------------
// GQA Attention (single-token decode path)
// ---------------------------------------------------------------------------

/// Grouped-Query Attention for single-token decode.
///
/// 1. Project input → Q[num_heads*head_dim], K[num_kv_heads*head_dim], V[num_kv_heads*head_dim]
/// 2. Apply RoPE to Q and K
/// 3. Store K,V in cache at `pos`
/// 4. For each query head, attend over all cached K,V (with GQA head mapping)
/// 5. Project attention output through o_proj
///
/// `work` is a reusable workspace to avoid allocations.
pub fn gqa_attention_decode(
    output: &mut [f32], // [hidden_size]
    input: &[f32],      // [hidden_size]
    q_proj: &QuantizedLinear<'_>,
    k_proj: &QuantizedLinear<'_>,
    v_proj: &QuantizedLinear<'_>,
    o_proj: &QuantizedLinear<'_>,
    rope: &RoPETable,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    work: &mut AttentionWork,
    store_kv: bool, // false = probe pass, don't write K,V to cache
) {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && head_dim > 0,
        "attention dimensions must be non-zero"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "query heads must be divisible by KV heads"
    );
    let hidden = num_heads
        .checked_mul(head_dim)
        .expect("attention dimensions overflow");
    let kv_dim = num_kv_heads
        .checked_mul(head_dim)
        .expect("KV dimensions overflow");
    assert_eq!(input.len(), hidden, "attention input length mismatch");
    assert_eq!(output.len(), hidden, "attention output length mismatch");
    assert_eq!(
        rope.head_dim, head_dim,
        "RoPE and attention head_dim mismatch"
    );
    assert!(
        pos < rope.max_ctx,
        "attention position exceeds RoPE context"
    );
    assert!(
        kv_cache.supports_attention(layer_idx, num_kv_heads, pos, head_dim),
        "KV cache dimensions do not match attention"
    );
    // Attention reads positions 0..=pos; every one of them must hold real
    // data. A storing pass writes `pos` itself below, a probe pass (ponder)
    // relies on a previous pass having stored it.
    let filled = kv_cache.filled(layer_idx);
    assert!(
        if store_kv { pos <= filled } else { pos < filled },
        "attention at position {pos} would read unwritten KV entries (layer {layer_idx}, filled {filled})"
    );

    // 1. Projections: Q, K, V via W4A32 matvec
    work.q.resize(hidden, 0.0);
    work.k.resize(kv_dim, 0.0);
    work.v.resize(kv_dim, 0.0);

    // Seed prefetcher for O_proj data (accessed after attention)
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let ptr = o_proj.nibble_data.as_ptr();
        for offset in (0..o_proj.nibble_data.len()).step_by(64).take(4) {
            // `offset` is proven in-bounds above; prefetch does not dereference the pointer.
            _mm_prefetch(ptr.add(offset) as *const i8, _MM_HINT_T1);
        }
    }

    w4a32_fused_qkv(
        &mut work.q,
        &mut work.k,
        &mut work.v,
        q_proj,
        k_proj,
        v_proj,
        input,
    );

    // 2. Apply RoPE to Q (num_heads heads) and K (num_kv_heads heads)
    rope.apply(&mut work.q, num_heads, pos);
    rope.apply(&mut work.k, num_kv_heads, pos);

    // 3. Store K, V in cache (conditional — probe passes skip this)
    if store_kv {
        kv_cache.store(layer_idx, pos, &work.k, &work.v);
    }

    // 4. Scaled dot-product attention with GQA
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;
    work.attn_out.resize(hidden, 0.0);
    work.scores.resize(pos + 1, 0.0);

    for qh in 0..num_heads {
        let kvh = qh / heads_per_kv;
        let q_head = &work.q[qh * head_dim..(qh + 1) * head_dim];

        // Compute attention scores and weighted value sum (SIMD-accelerated)
        let out_head = &mut work.attn_out[qh * head_dim..(qh + 1) * head_dim];
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() && head_dim.is_multiple_of(8) {
                unsafe {
                    attention_head_avx2(
                        q_head,
                        out_head,
                        &mut work.scores,
                        kv_cache,
                        layer_idx,
                        kvh,
                        pos,
                        head_dim,
                        scale,
                    );
                }
                continue;
            }
        }

        // Scalar fallback
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..=pos {
            let k_cached = kv_cache.get_k(layer_idx, kvh, t, head_dim);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_cached[d];
            }
            work.scores[t] = dot * scale;
            if work.scores[t] > max_score {
                max_score = work.scores[t];
            }
        }
        let mut sum_exp = 0.0f32;
        for t in 0..=pos {
            work.scores[t] = (work.scores[t] - max_score).exp();
            sum_exp += work.scores[t];
        }
        let inv_sum = 1.0 / sum_exp;
        for t in 0..=pos {
            work.scores[t] *= inv_sum;
        }
        for d in 0..head_dim {
            out_head[d] = 0.0;
        }
        for t in 0..=pos {
            let v_cached = kv_cache.get_v(layer_idx, kvh, t, head_dim);
            let score = work.scores[t];
            for d in 0..head_dim {
                out_head[d] += score * v_cached[d];
            }
        }
    }

    // 5. Output projection
    w4a32_matvec(
        output,
        o_proj.nibble_data,
        o_proj.group_params,
        &work.attn_out,
        o_proj.rows,
        o_proj.cols,
        o_proj.group_size,
    );
}

/// AVX2 attention head: score computation + softmax + value accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attention_head_avx2(
    q_head: &[f32],
    out_head: &mut [f32],
    scores: &mut [f32],
    kv_cache: &KVCache,
    layer_idx: usize,
    kvh: usize,
    pos: usize,
    head_dim: usize,
    scale: f32,
) {
    let _scale_v = _mm256_set1_ps(scale);
    let chunks8 = head_dim / 8;

    // Compute dot(Q, K_cached) for all positions using AVX2
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
        // Horizontal sum
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

    // Softmax with Schraudolph fast-exp AVX2
    let max_v = _mm256_set1_ps(max_score);
    let exp_a = _mm256_set1_ps(12102203.0); // 2^23 / ln(2)
    let exp_b = _mm256_set1_ps(1065353216.0 - 486411.0); // 2^23 * 127 - correction
    let exp_lo = _mm256_set1_ps(0.0);
    let exp_hi = _mm256_set1_ps(2139095040.0);
    let mut sum_acc = _mm256_setzero_ps();
    let sp = scores.as_mut_ptr();
    let n_pos = pos + 1;
    let chunks8_s = n_pos / 8;
    for i in 0..chunks8_s {
        let off = i * 8;
        let sv = _mm256_loadu_ps(sp.add(off));
        let shifted = _mm256_sub_ps(sv, max_v);
        let t = _mm256_add_ps(_mm256_mul_ps(exp_a, shifted), exp_b);
        let t = _mm256_max_ps(_mm256_min_ps(t, exp_hi), exp_lo);
        let ev = _mm256_castsi256_ps(_mm256_cvtps_epi32(t));
        _mm256_storeu_ps(sp.add(off), ev);
        sum_acc = _mm256_add_ps(sum_acc, ev);
    }
    // Scalar tail for exp
    let mut sum_exp = {
        let hi = _mm256_extractf128_ps(sum_acc, 1);
        let lo = _mm256_castps256_ps128(sum_acc);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let s2 = _mm_add_ps(s, shuf);
        let hi2 = _mm_movehl_ps(s2, s2);
        _mm_cvtss_f32(_mm_add_ss(s2, hi2))
    };
    for t in (chunks8_s * 8)..n_pos {
        let v = (*sp.add(t) - max_score).exp();
        *sp.add(t) = v;
        sum_exp += v;
    }
    // Normalize
    let inv_sum = 1.0 / sum_exp;
    let inv_v = _mm256_set1_ps(inv_sum);
    for i in 0..chunks8_s {
        let off = i * 8;
        let sv = _mm256_loadu_ps(sp.add(off));
        _mm256_storeu_ps(sp.add(off), _mm256_mul_ps(sv, inv_v));
    }
    for t in (chunks8_s * 8)..n_pos {
        *sp.add(t) *= inv_sum;
    }

    // Weighted sum of V using AVX2
    // Zero the output
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

/// Standalone attention for batched forward: takes pre-computed Q, K, V,
/// applies RoPE, stores K/V in cache, computes attention, writes output.
/// Does NOT do QKV or O projections (those are batched separately).
pub fn compute_attention(
    attn_out: &mut [f32], // [num_heads * head_dim]
    q: &mut [f32],        // [num_heads * head_dim], RoPE applied in-place
    k: &mut [f32],        // [num_kv_heads * head_dim], RoPE applied in-place
    v: &[f32],            // [num_kv_heads * head_dim]
    rope: &RoPETable,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scores: &mut Vec<f32>,
) {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && head_dim > 0,
        "attention dimensions must be non-zero"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "query heads must be divisible by KV heads"
    );
    let hidden = num_heads
        .checked_mul(head_dim)
        .expect("attention dimensions overflow");
    let kv_dim = num_kv_heads
        .checked_mul(head_dim)
        .expect("KV dimensions overflow");
    assert_eq!(attn_out.len(), hidden, "attention output length mismatch");
    assert_eq!(q.len(), hidden, "query length mismatch");
    assert_eq!(k.len(), kv_dim, "key length mismatch");
    assert_eq!(v.len(), kv_dim, "value length mismatch");
    assert_eq!(
        rope.head_dim, head_dim,
        "RoPE and attention head_dim mismatch"
    );
    assert!(
        pos < rope.max_ctx,
        "attention position exceeds RoPE context"
    );
    assert!(
        kv_cache.supports_attention(layer_idx, num_kv_heads, pos, head_dim),
        "KV cache dimensions do not match attention"
    );
    // Apply RoPE
    rope.apply(q, num_heads, pos);
    rope.apply(k, num_kv_heads, pos);

    // Store K, V in cache
    kv_cache.store(layer_idx, pos, k, v);

    // Attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;
    scores.resize(pos + 1, 0.0);

    for qh in 0..num_heads {
        let kvh = qh / heads_per_kv;
        let q_head = &q[qh * head_dim..(qh + 1) * head_dim];
        let out_head = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() && head_dim.is_multiple_of(8) {
                unsafe {
                    attention_head_avx2(
                        q_head, out_head, scores, kv_cache, layer_idx, kvh, pos, head_dim, scale,
                    );
                }
                continue;
            }
        }

        // Scalar fallback
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..=pos {
            let k_cached = kv_cache.get_k(layer_idx, kvh, t, head_dim);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_cached[d];
            }
            scores[t] = dot * scale;
            if scores[t] > max_score {
                max_score = scores[t];
            }
        }
        let mut sum_exp = 0.0f32;
        for t in 0..=pos {
            scores[t] = (scores[t] - max_score).exp();
            sum_exp += scores[t];
        }
        let inv_sum = 1.0 / sum_exp;
        for t in 0..=pos {
            scores[t] *= inv_sum;
        }
        for d in 0..head_dim {
            out_head[d] = 0.0;
        }
        for t in 0..=pos {
            let v_cached = kv_cache.get_v(layer_idx, kvh, t, head_dim);
            let score = scores[t];
            for d in 0..head_dim {
                out_head[d] += score * v_cached[d];
            }
        }
    }
}

/// Reusable workspace for attention to avoid per-call allocations.
#[derive(Default)]
pub struct AttentionWork {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub scores: Vec<f32>,
}

impl AttentionWork {
    pub fn new() -> Self {
        Self {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            scores: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU MLP forward pass.
///
/// `hidden = silu(gate_proj(input)) * up_proj(input)`
/// `output = down_proj(hidden)`
pub fn swiglu_mlp(
    output: &mut [f32], // [hidden_size]
    input: &[f32],      // [hidden_size]
    gate_proj: &QuantizedLinear<'_>,
    up_proj: &QuantizedLinear<'_>,
    down_proj: &QuantizedLinear<'_>,
    work: &mut MlpWork,
) {
    let intermediate = gate_proj.rows;

    work.gate.resize(intermediate, 0.0);
    work.up.resize(intermediate, 0.0);

    // gate = gate_proj(input), up = up_proj(input) — fused and overlapped
    w4a32_fused_gate_up(&mut work.gate, &mut work.up, gate_proj, up_proj, input);

    // hidden = silu(gate) * up — SIMD-vectorized when available
    silu_mul_inplace(&mut work.gate, &work.up, intermediate);

    // output = down_proj(hidden)
    w4a32_matvec(
        output,
        down_proj.nibble_data,
        down_proj.group_params,
        &work.gate,
        down_proj.rows,
        down_proj.cols,
        down_proj.group_size,
    );
}

/// Reusable workspace for MLP.
#[derive(Default)]
pub struct MlpWork {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl MlpWork {
    pub fn new() -> Self {
        Self {
            gate: Vec::new(),
            up: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        rms_norm(&mut output, &input, &weight, 1e-5);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| x / rms).collect();
        for i in 0..4 {
            assert!(
                (output[i] - expected[i]).abs() < 1e-5,
                "idx {i}: got {} expected {}",
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.001);
        // silu(-1.0) = -1.0 / (1 + exp(1)) ≈ -0.2689
        assert!((silu(-1.0) - (-0.2689)).abs() < 0.001);
    }

    #[test]
    fn test_rope_basic() {
        let rope = RoPETable::new(4, 16, 10000.0);
        // At position 0, all angles are 0, so cos=1, sin=0 → no rotation
        let mut heads = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, head_dim=4
        let orig = heads.clone();
        rope.apply(&mut heads, 1, 0);
        for i in 0..4 {
            assert!(
                (heads[i] - orig[i]).abs() < 1e-5,
                "pos=0 should be identity"
            );
        }

        // At position > 0, values should change
        let mut heads2 = vec![1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut heads2, 1, 5);
        let changed = heads2
            .iter()
            .zip(orig.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(changed, "RoPE at pos=5 should rotate values");
    }

    #[test]
    #[should_panic(expected = "RMSNorm weight length mismatch")]
    fn rms_norm_rejects_mismatched_buffers_before_simd() {
        let mut output = [0.0; 8];
        rms_norm(&mut output, &[1.0; 8], &[1.0; 7], 1e-5);
    }

    #[test]
    #[should_panic(expected = "residual length mismatch")]
    fn vec_add_rejects_mismatched_buffers_before_simd() {
        let mut hidden = [0.0; 8];
        vec_add(&mut hidden, &[1.0; 7], &[1.0; 8]);
    }

    #[test]
    #[should_panic(expected = "SiLU up buffer is too small")]
    fn silu_mul_rejects_mismatched_buffers_before_simd() {
        silu_mul_inplace(&mut [1.0; 8], &[1.0; 7], 8);
    }

    #[test]
    #[should_panic(expected = "RoPE head buffer length mismatch")]
    fn rope_rejects_mismatched_head_buffer_before_simd() {
        let rope = RoPETable::new(8, 4, 10_000.0);
        rope.apply(&mut [0.0; 7], 1, 0);
    }

    #[test]
    #[should_panic(expected = "KV cache dimensions do not match attention")]
    fn attention_rejects_mismatched_cache_before_simd() {
        let rope = RoPETable::new(16, 4, 10_000.0);
        let mut cache = KVCache::new(1, 1, 4, 8).unwrap();
        let mut output = [0.0; 16];
        let mut q = [0.0; 16];
        let mut k = [0.0; 16];
        compute_attention(
            &mut output,
            &mut q,
            &mut k,
            &[0.0; 16],
            &rope,
            &mut cache,
            0,
            0,
            1,
            1,
            16,
            &mut Vec::new(),
        );
    }

    #[test]
    fn attention_uses_scalar_tail_for_non_multiple_of_eight_head_dim() {
        let rope = RoPETable::new(6, 2, 10_000.0);
        let mut cache = KVCache::new(1, 1, 2, 6).unwrap();
        let mut output = [99.0; 6];
        let mut q = [1.0; 6];
        let mut k = [1.0; 6];
        compute_attention(
            &mut output,
            &mut q,
            &mut k,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &rope,
            &mut cache,
            0,
            0,
            1,
            1,
            6,
            &mut Vec::new(),
        );
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
