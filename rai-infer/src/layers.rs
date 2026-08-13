//! Transformer layer operations: RMSNorm, RoPE, SiLU, GQA attention, SwiGLU MLP.
//!
//! All operations are pure f32, hand-written with AVX2 SIMD acceleration.
//! No external linear algebra dependencies.

// Attention kernels intentionally mirror scalar/SIMD index arithmetic, and their explicit
// dimension arguments document and validate each buffer contract at the call boundary.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use anyhow::{anyhow, ensure, Result};

use crate::format::QuantizedLinear;
use crate::gemm::{has_avx2, w4a8_fused_gate_up, w4a8_fused_qkv, w4a8_matvec};
use crate::kv_cache::KVCache;
use rayon::prelude::*;

/// Upper bound on the precomputed RoPE cos/sin table allocation.
///
/// `format.rs` imports this constant: `.raimodel` validation at load time is
/// the single gate that guarantees a loaded model can never ask
/// `RoPETable::new` for an over-budget table, so the limit is defined once
/// here at its point of enforcement.
pub(crate) const MAX_ROPE_TABLE_BYTES: usize = 512 * 1024 * 1024;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Capability selectors carried by the `.raimodel` v2 header
// ---------------------------------------------------------------------------

/// Which non-linearity the gated MLP uses.
///
/// Llama/Mistral/Qwen2 use SwiGLU (`silu(gate) * up`); Gemma uses GeGLU with
/// the tanh approximation of GeLU (`gelu_tanh(gate) * up`). The container
/// stores this as a `u8` so a v2 reader refuses an activation it does not
/// implement instead of quietly running the wrong one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// `silu(x) = x * sigmoid(x)` — SwiGLU.
    #[default]
    Silu,
    /// `gelu_tanh(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))` — GeGLU.
    GeluTanh,
}

impl Activation {
    /// Decode the header byte. Unknown values are rejected, never ignored.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Activation::Silu),
            1 => Some(Activation::GeluTanh),
            _ => None,
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Activation::Silu => 0,
            Activation::GeluTanh => 1,
        }
    }
}

/// How the RoPE inverse frequencies are transformed before the cos/sin tables
/// are built.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScaling {
    /// Plain RoPE straight from `rope_theta`.
    #[default]
    None,
    /// The Llama-3.1/3.2 wavelength-banded rescaling.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: u32,
    },
}

impl RopeScaling {
    pub fn code(self) -> u8 {
        match self {
            RopeScaling::None => 0,
            RopeScaling::Llama3 { .. } => 1,
        }
    }
}

/// One layer's bias vectors, in `format::PROJECTION_NAMES` order
/// (q, k, v, o, gate, up, down). `None` where a projection has no bias.
pub type ProjectionBiases<'a> = [Option<&'a [f32]>; 7];

/// Add a projection's bias to its output vector.
///
/// Deliberately a separate pass over `output` rather than a term inside the
/// W4A8 inner loop: the loop is `rows * cols` integer MACs against `rows`
/// float adds here, so folding the bias in would complicate the hottest kernel
/// in the model to save a fraction of a percent. It also keeps the quantized
/// path untouched, so a biased model and an unbiased one run the same GEMM.
///
/// # Panics
/// Panics if the bias length does not match the output length.
#[inline]
pub fn add_bias(output: &mut [f32], bias: Option<&[f32]>) {
    let Some(bias) = bias else {
        return;
    };
    assert_eq!(
        output.len(),
        bias.len(),
        "projection bias length does not match its output"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: `has_avx2()` gates AVX2, and the assert above proves both
            // slices hold `output.len()` values.
            unsafe {
                let n = output.len();
                let mut i = 0usize;
                for _ in 0..(n / 8) {
                    let o = _mm256_loadu_ps(output.as_ptr().add(i));
                    let b = _mm256_loadu_ps(bias.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(o, b));
                    i += 8;
                }
                while i < n {
                    *output.as_mut_ptr().add(i) += *bias.as_ptr().add(i);
                    i += 1;
                }
            }
            return;
        }
    }
    for (value, &b) in output.iter_mut().zip(bias) {
        *value += b;
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMSNorm: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
///
/// # Panics
/// Panics if `input` is empty, the buffer lengths differ, or `eps` is not
/// finite and positive.
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
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
/// - `output`, `input`, and `weight` must each hold at least `n` values;
///   `rms_norm` asserts exactly these lengths before dispatching here.
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
///
/// # Panics
/// Panics if `hidden` is empty, any buffer length differs from `hidden`'s, or
/// `eps` is not finite and positive.
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
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
/// - `normed`, `residual`, `hidden`, and `weight` must each hold at least `n`
///   values; `rms_norm_with_residual` asserts exactly these lengths before
///   dispatching here.
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

/// SIMD vector add: hidden[i] = residual[i] + addition[i]
///
/// # Panics
/// Panics if the three buffer lengths differ.
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
    ///
    /// Returns an error rather than aborting when the table does not fit: a
    /// `.raimodel` only a few kilobytes long can legally declare a very large
    /// `max_context`, and a model file must never be able to kill the process.
    pub fn new(head_dim: usize, max_ctx: usize, theta: f32) -> Result<Self> {
        Self::with_scaling(head_dim, max_ctx, theta, RopeScaling::None)
    }

    /// Build RoPE tables, optionally applying a frequency-rescaling scheme.
    ///
    /// Same error conditions as [`RoPETable::new`], plus non-finite or
    /// non-positive llama3 scaling parameters.
    pub fn with_scaling(
        head_dim: usize,
        max_ctx: usize,
        theta: f32,
        scaling: RopeScaling,
    ) -> Result<Self> {
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "RoPE head_dim must be positive and even, got {head_dim}"
        );
        ensure!(max_ctx > 0, "RoPE context must be non-zero");
        ensure!(
            theta.is_finite() && theta > 0.0,
            "RoPE theta must be finite and positive, got {theta}"
        );
        let half_dim = head_dim / 2;
        let elements = max_ctx
            .checked_mul(half_dim)
            .ok_or_else(|| anyhow!("RoPE table dimensions overflow"))?;
        let bytes = elements
            .checked_mul(2)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow!("RoPE table byte size overflows"))?;
        ensure!(
            bytes <= MAX_ROPE_TABLE_BYTES,
            "RoPE table needs {} MiB for max_context {max_ctx}, over the {} MiB budget",
            bytes >> 20,
            MAX_ROPE_TABLE_BYTES >> 20
        );
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        cos.try_reserve_exact(elements)
            .and_then(|()| sin.try_reserve_exact(elements))
            .map_err(|_| anyhow!("cannot allocate {} MiB for the RoPE table", bytes >> 20))?;
        cos.resize(elements, 0.0);
        sin.resize(elements, 0.0);

        // The inverse frequencies are position-independent, so they are
        // computed once and then rescaled once, rather than per position.
        let mut inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        if let RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } = scaling
        {
            apply_llama3_scaling(
                &mut inv_freq,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            );
        }

        for pos in 0..max_ctx {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Ok(Self {
            cos,
            sin,
            head_dim,
            max_ctx,
        })
    }

    /// Apply RoPE rotation to a set of heads at the given position.
    ///
    /// `heads` is `[num_heads * head_dim]`. Each head's pairs (x[2i], x[2i+1])
    /// are rotated by the position-dependent angle.
    ///
    /// # Panics
    /// Panics if `pos >= max_ctx`, the head-count product overflows, or
    /// `heads.len() != num_heads * head_dim`.
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
    ///
    /// # Safety
    /// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
    /// - `heads` must hold `num_heads * head_dim` values with `head_dim` a
    ///   multiple of 8, and `cos_row`/`sin_row` must hold `head_dim / 2`
    ///   values; `apply` asserts these bounds before dispatching here.
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

/// Rescale RoPE inverse frequencies with the Llama-3.1/3.2 scheme, in place.
///
/// Ported line for line from HuggingFace `transformers`
/// `src/transformers/modeling_rope_utils.py::_compute_llama3_parameters`
/// (verified against the copy in the clean-room venv):
///
/// ```text
/// low_freq_wavelen  = old_context_len / low_freq_factor
/// high_freq_wavelen = old_context_len / high_freq_factor
/// wavelen           = 2 * pi / inv_freq
/// inv_freq_llama    = where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
/// smooth_factor     = (old_context_len / wavelen - low_freq_factor)
///                     / (high_freq_factor - low_freq_factor)
/// smoothed          = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
/// is_medium_freq    = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
/// inv_freq_llama    = where(is_medium_freq, smoothed, inv_freq_llama)
/// ```
///
/// Two details of that reference are load-bearing and easy to get wrong:
///
/// * the smoothing blends `inv_freq_llama`, not `inv_freq` — but in the medium
///   band `wavelen <= low_freq_wavelen`, so the first `where` left it equal to
///   `inv_freq` and the two spellings coincide. The code below keeps the
///   reference's variable so the equivalence is visible rather than assumed;
/// * the band test is `~(w < high) & ~(w > low)`, i.e. the *closed* interval
///   `high <= w <= low`. The boundary values are smoothed, not passed through.
///
/// # Panics
/// Panics if any parameter is non-finite, `factor` is not positive,
/// `original_max_position` is zero, or `high_freq_factor == low_freq_factor`
/// (which would divide by zero in the smoothing term).
fn apply_llama3_scaling(
    inv_freq: &mut [f32],
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: u32,
) {
    assert!(
        factor.is_finite() && factor > 0.0,
        "llama3 RoPE factor must be finite and positive"
    );
    assert!(
        low_freq_factor.is_finite() && high_freq_factor.is_finite(),
        "llama3 RoPE frequency factors must be finite"
    );
    assert!(
        (high_freq_factor - low_freq_factor).abs() > 0.0,
        "llama3 RoPE high_freq_factor must differ from low_freq_factor"
    );
    assert!(
        original_max_position > 0,
        "llama3 RoPE original_max_position_embeddings must be non-zero"
    );

    let old_context_len = original_max_position as f32;
    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    for value in inv_freq.iter_mut() {
        let wavelen = 2.0 * std::f32::consts::PI / *value;
        let banded = if wavelen > low_freq_wavelen {
            *value / factor
        } else {
            *value
        };
        // The reference writes this as `~(w < high) & ~(w > low)`. Every
        // wavelength here is finite and positive (`theta` is validated finite
        // and positive, so no inverse frequency is zero or NaN), which makes
        // the negations equivalent to the closed band spelled directly.
        let is_medium = wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen;
        *value = if is_medium {
            let smooth = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            (1.0 - smooth) * banded / factor + smooth * banded
        } else {
            banded
        };
    }
}

// ---------------------------------------------------------------------------
// Gated-MLP activations
// ---------------------------------------------------------------------------

/// SiLU activation: `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fused SiLU(gate) * up, result written to gate. SIMD-accelerated.
///
/// # Panics
/// Panics if `gate` or `up` holds fewer than `n` values.
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
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
/// - `gate` and `up` must each hold at least `n` values; `silu_mul_inplace`
///   asserts these bounds before dispatching here.
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

/// GeLU, tanh approximation — the one Gemma trains against
/// (`gelu_pytorch_tanh`):
///
/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`
///
/// Uses [`tanh_poly`], not `f32::tanh`, so this is bit-identical to the scalar
/// tail the AVX2 kernel falls back on and to the no-AVX2 path.
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + tanh_poly(SQRT_2_OVER_PI * (x + COEFF * x * x * x)))
}

/// Fused `gelu_tanh(gate) * up`, result written to gate. SIMD-accelerated.
///
/// # Panics
/// Panics if `gate` or `up` holds fewer than `n` values.
pub fn gelu_tanh_mul_inplace(gate: &mut [f32], up: &[f32], n: usize) {
    assert!(n <= gate.len(), "GeLU gate buffer is too small");
    assert!(n <= up.len(), "GeLU up buffer is too small");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                gelu_tanh_mul_avx2(gate, up, n);
            }
            return;
        }
    }
    for i in 0..n {
        gate[i] = gelu_tanh(gate[i]) * up[i];
    }
}

/// AVX2 `gelu_tanh(x) * y`.
///
/// The SiLU path next door uses Schraudolph's bit-trick `exp`, whose ~2%
/// relative error is tolerable there because `silu` is a *bounded* rescaling of
/// its input. It is not tolerable here: `gelu_tanh` multiplies the saturating
/// factor by `x` itself, so a 2% error on the factor is a 2% error on an
/// unbounded output. This uses the accurate rational `tanh` in [`tanh_avx2`]
/// instead — same eight-lanes-per-iteration structure, no accuracy give-away
/// (measured max absolute error against `f64` `gelu_tanh`: 1.9e-6 over
/// [-30, 30], which is f32 rounding at that magnitude).
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
/// - `gate` and `up` must each hold at least `n` values;
///   `gelu_tanh_mul_inplace` asserts these bounds before dispatching here.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_tanh_mul_avx2(gate: &mut [f32], up: &[f32], n: usize) {
    let chunks8 = n / 8;
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let sqrt_2_over_pi = _mm256_set1_ps(0.797_884_6);
    let coeff = _mm256_set1_ps(0.044_715);

    for i in 0..chunks8 {
        let off = i * 8;
        let x = _mm256_loadu_ps(gate.as_ptr().add(off));
        let u = _mm256_loadu_ps(up.as_ptr().add(off));

        // t = sqrt(2/pi) * (x + 0.044715 x^3)
        let x2 = _mm256_mul_ps(x, x);
        let inner = _mm256_fmadd_ps(_mm256_mul_ps(coeff, x2), x, x);
        let t = _mm256_mul_ps(sqrt_2_over_pi, inner);

        let th = tanh_avx2(t);
        // 0.5 * x * (1 + tanh(t)) * up
        let g = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, th));
        _mm256_storeu_ps(gate.as_mut_ptr().add(off), _mm256_mul_ps(g, u));
    }

    // Scalar tail
    for i in (chunks8 * 8)..n {
        gate[i] = gelu_tanh(gate[i]) * up[i];
    }
}

/// `tanh` clamp point. `tanh(9) = 1 - 3.0e-8`, which already rounds to `1.0f32`,
/// so clamping the argument to ±9 is exact in f32 and keeps `t^12` inside
/// range.
const TANH_CLAMP: f32 = 9.0;

/// Numerator `P(u)` of the `tanh` approximation, `u = t^2`, ascending powers.
///
/// `tanh(t) ~= t * P(t^2) / Q(t^2)` for `|t| <= 9`, `P(0) = Q(0) = 1` so the
/// form is exactly odd and exactly `t` at the origin.
///
/// These are not borrowed constants: they were fitted here with an
/// iteratively-reweighted (Loeb) rational least-squares solve of `tanh` on
/// `[0, 9]`, run in the variable `u/81` for conditioning and converted back.
/// The fit's own error is 2.0e-8; evaluated in f32 with this Horner order the
/// end-to-end error against `f64::tanh` is 3.2e-7 over `[-9, 9]` — i.e. f32
/// rounding, not the approximation. `gelu_tanh_matches_high_precision_reference`
/// re-checks that bound at test time, so a bad edit cannot pass silently.
const TANH_P: [f32; 7] = [
    1.0,
    0.130_885_48,
    0.003_109_051_1,
    1.120_845_9e-5,
    -2.045_068_6e-8,
    5.389_845_5e-11,
    -8.784_743e-14,
];

/// Denominator `Q(u)` of the same approximation, ascending powers.
const TANH_Q: [f32; 4] = [1.0, 0.464_218_77, 0.024_515_394, 0.000_255_361_32];

/// Eight-lane `tanh` via the odd rational form `t * P(t^2) / Q(t^2)`, saturating
/// to exactly ±1 outside ±[`TANH_CLAMP`].
///
/// The saturation is a select, not just an argument clamp, and that distinction
/// is load-bearing: `gelu_tanh` evaluates `1 + tanh(t)`, so at `t <= -9` the
/// polynomial's last-ulp error (`-0.99999994` instead of `-1.0`) survives as
/// `6e-8`, which the following `0.5 * x` multiplies back up by `|x|`. Returning
/// exactly `-1.0` makes the cancellation exact instead. `tanh(9) = 1 - 3e-8`
/// already rounds to `1.0f32`, so nothing is lost.
///
/// # Safety
/// Must only be called from an AVX2+FMA context.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(t: __m256) -> __m256 {
    let limit = _mm256_set1_ps(TANH_CLAMP);
    let sign_mask = _mm256_set1_ps(-0.0);
    let abs_t = _mm256_andnot_ps(sign_mask, t);
    // Clamp before squaring so `t^12` cannot overflow on the discarded lanes.
    let tc = _mm256_min_ps(_mm256_max_ps(t, _mm256_set1_ps(-TANH_CLAMP)), limit);
    let u = _mm256_mul_ps(tc, tc);

    let mut p = _mm256_set1_ps(TANH_P[6]);
    for k in (0..6).rev() {
        p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(TANH_P[k]));
    }
    let mut q = _mm256_set1_ps(TANH_Q[3]);
    for k in (0..3).rev() {
        q = _mm256_fmadd_ps(q, u, _mm256_set1_ps(TANH_Q[k]));
    }
    let poly = _mm256_mul_ps(tc, _mm256_div_ps(p, q));

    let saturated = _mm256_or_ps(_mm256_and_ps(sign_mask, t), _mm256_set1_ps(1.0));
    let use_poly = _mm256_cmp_ps(abs_t, limit, _CMP_LT_OQ);
    _mm256_blendv_ps(saturated, poly, use_poly)
}

/// Scalar twin of [`tanh_avx2`], evaluated in the same order so the SIMD body
/// and the scalar tail of a buffer cannot disagree.
#[inline]
fn tanh_poly(t: f32) -> f32 {
    // NaN is routed to the saturating branch on purpose, matching the AVX2
    // path's `_CMP_LT_OQ` (which is false for NaN). A NaN activation means the
    // model is already broken; this at least does not turn it into a plausible
    // number by running it through the polynomial.
    if t.is_nan() || t.abs() >= TANH_CLAMP {
        return 1.0f32.copysign(t);
    }
    let u = t * t;
    let mut p = TANH_P[6];
    for k in (0..6).rev() {
        p = p * u + TANH_P[k];
    }
    let mut q = TANH_Q[3];
    for k in (0..3).rev() {
        q = q * u + TANH_Q[k];
    }
    t * (p / q)
}

/// Apply the model's gated-MLP activation: `act(gate) * up`, into `gate`.
///
/// # Panics
/// Panics if `gate` or `up` holds fewer than `n` values.
#[inline]
pub fn glu_mul_inplace(activation: Activation, gate: &mut [f32], up: &[f32], n: usize) {
    match activation {
        Activation::Silu => silu_mul_inplace(gate, up, n),
        Activation::GeluTanh => gelu_tanh_mul_inplace(gate, up, n),
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
///
/// # Panics
/// Panics if any dimension is zero, `num_heads` is not divisible by
/// `num_kv_heads`, a buffer length does not match the dimensions, `pos` is
/// outside the RoPE context, the KV cache does not match the attention
/// dimensions, any position `0..=pos` needed by attention is not yet filled
/// in the KV cache (the misuse-proofing watermark), or a projection fails the
/// GEMM buffer checks.
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
    biases: &ProjectionBiases<'_>,
) {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && head_dim > 0,
        "attention dimensions must be non-zero"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "query heads must be divisible by KV heads"
    );
    // The attention interior is `num_heads * head_dim` wide; the block's input
    // and output are `hidden_size` wide. Those are equal for Llama-shaped
    // models and not for Gemma-shaped ones, so each is taken from the
    // projection that actually defines it.
    let q_dim = num_heads
        .checked_mul(head_dim)
        .expect("attention dimensions overflow");
    let kv_dim = num_kv_heads
        .checked_mul(head_dim)
        .expect("KV dimensions overflow");
    assert_eq!(
        q_proj.rows, q_dim,
        "q_proj does not produce num_heads heads"
    );
    assert_eq!(
        k_proj.rows, kv_dim,
        "k_proj does not produce num_kv_heads heads"
    );
    assert_eq!(
        v_proj.rows, kv_dim,
        "v_proj does not produce num_kv_heads heads"
    );
    assert_eq!(
        o_proj.cols, q_dim,
        "o_proj does not consume the attention width"
    );
    assert_eq!(input.len(), q_proj.cols, "attention input length mismatch");
    assert_eq!(
        output.len(),
        o_proj.rows,
        "attention output length mismatch"
    );
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

    // 1. Projections: Q, K, V via W4A8 matvec
    work.q.resize(q_dim, 0.0);
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

    w4a8_fused_qkv(
        &mut work.q,
        &mut work.k,
        &mut work.v,
        q_proj,
        k_proj,
        v_proj,
        input,
    );

    // 1b. Projection biases (Qwen2/Qwen2.5 carry them on q/k/v). Added before
    //     RoPE, exactly where `q_proj(x) + b` sits in the reference model.
    add_bias(&mut work.q, biases[0]);
    add_bias(&mut work.k, biases[1]);
    add_bias(&mut work.v, biases[2]);

    // 2. Apply RoPE to Q (num_heads heads) and K (num_kv_heads heads)
    rope.apply(&mut work.q, num_heads, pos);
    rope.apply(&mut work.k, num_kv_heads, pos);

    // 3. Store K, V in cache (conditional — probe passes skip this)
    if store_kv {
        kv_cache.store(layer_idx, pos, &work.k, &work.v);
    }

    // 4. Scaled dot-product attention with GQA.
    //
    // Query heads are independent, and this is the one decode cost that grows
    // with context (measured: 0.6% of a decode step at pos=8, 7.7% at pos=512
    // on TinyLlama-1.1B), so it is spread across the pool rather than run on
    // one core. `scores` is one flat buffer sliced per head, which keeps the
    // parallel path allocation-free.
    work.attn_out.resize(q_dim, 0.0);
    work.scores.resize(num_heads * (pos + 1), 0.0);
    attention_all_heads(
        &mut work.attn_out,
        &mut work.scores,
        &work.q,
        kv_cache,
        layer_idx,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    // 5. Output projection
    w4a8_matvec(
        output,
        o_proj.nibble_data,
        o_proj.group_params,
        &work.attn_out,
        o_proj.rows,
        o_proj.cols,
        o_proj.group_size,
    );
    add_bias(output, biases[3]);
}

/// AVX2 attention head: score computation + softmax + value accumulation.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA).
/// - `q_head` and `out_head` must each hold at least `head_dim` values with
///   `head_dim` a multiple of 8.
/// - `scores.len() >= pos + 1` (the kernel writes every index `0..=pos`
///   unchecked).
/// - `layer_idx`/`kvh` must be in range for `kv_cache` and every position
///   `0..=pos` must already be stored (`supports_attention` plus the filled
///   watermark, asserted by the callers, establish this).
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

/// One query head of scaled dot-product attention over the cached K/V.
///
/// Both the decode path ([`gqa_attention_decode`]) and the batched path
/// ([`compute_attention`]) funnel through here, so the SIMD kernel and its
/// scalar fallback are defined once instead of once per caller.
///
/// `scores` is scratch for `pos + 1` values and is fully overwritten.
#[inline]
fn attention_head(
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
    debug_assert_eq!(q_head.len(), head_dim, "query head length mismatch");
    debug_assert_eq!(out_head.len(), head_dim, "attention head output mismatch");
    debug_assert!(scores.len() > pos, "score scratch is too small");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && head_dim.is_multiple_of(8) {
            // SAFETY: `has_avx2()` gates AVX2+FMA, `head_dim` is a multiple of
            // 8, the head buffers hold `head_dim` values, `scores` holds more
            // than `pos` values, and the callers assert the cache dimensions
            // and the filled watermark covering `0..=pos`.
            unsafe {
                attention_head_avx2(
                    q_head, out_head, scores, kv_cache, layer_idx, kvh, pos, head_dim, scale,
                );
            }
            return;
        }
    }

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

/// Context length at which spreading decode attention across the rayon pool
/// starts paying for the dispatch.
///
/// Query heads are independent, and attention is the one decode cost that
/// grows with context, so parallelizing it looks free — it is not. Measured
/// per-head decode attention on TinyLlama-1.1B (32 heads, 22 layers, isolated
/// A/B in one binary):
///
/// | pos | serial      | parallel   |
/// |-----|-------------|------------|
/// |   8 |  488–548 μs |  3372 μs   |  parallel ~6× *worse*
/// | 512 | 14.7–20.8 ms| 11.3 ms    |  parallel ~1.3–1.8× better
///
/// At 32 tiny work items the rayon split costs ~130 μs per layer, which swamps
/// the whole attention phase at short context. Break-even sits near pos≈130;
/// this threshold keeps a 2× margin above it so the common short-context case
/// never pays. The numbers came off a heavily contended machine — re-tune on a
/// quiet one before trusting the exact constant.
const PARALLEL_ATTENTION_MIN_POS: usize = 256;

/// All query heads of single-position decode attention.
///
/// Runs serially below [`PARALLEL_ATTENTION_MIN_POS`] and across the rayon
/// pool above it. `scores` is one flat `num_heads * (pos + 1)` buffer sliced
/// per head, which keeps the parallel path allocation-free. Heads never
/// interact, so both paths are bit-identical.
///
/// # Panics
/// Panics if the buffers are smaller than the head dimensions imply.
#[inline]
fn attention_all_heads(
    attn_out: &mut [f32],
    scores: &mut [f32],
    q: &[f32],
    kv_cache: &KVCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let row = pos + 1;
    let hidden = num_heads * head_dim;
    assert!(attn_out.len() >= hidden, "attention output is too small");
    assert!(q.len() >= hidden, "attention query buffer is too small");
    assert!(
        scores.len() >= num_heads * row,
        "attention score scratch is too small"
    );
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    if pos < PARALLEL_ATTENTION_MIN_POS || crate::gemm::serial_decode_attention_enabled() {
        for qh in 0..num_heads {
            attention_head(
                &q[qh * head_dim..(qh + 1) * head_dim],
                &mut attn_out[qh * head_dim..(qh + 1) * head_dim],
                &mut scores[qh * row..(qh + 1) * row],
                kv_cache,
                layer_idx,
                qh / heads_per_kv,
                pos,
                head_dim,
                scale,
            );
        }
        return;
    }

    attn_out[..hidden]
        .par_chunks_mut(head_dim)
        .zip(scores[..num_heads * row].par_chunks_mut(row))
        .enumerate()
        .for_each(|(qh, (out_head, head_scores))| {
            attention_head(
                &q[qh * head_dim..(qh + 1) * head_dim],
                out_head,
                head_scores,
                kv_cache,
                layer_idx,
                qh / heads_per_kv,
                pos,
                head_dim,
                scale,
            );
        });
}

/// Profiling hook: runs the whole per-position decode attention phase exactly
/// as `gqa_attention_decode` does, so `profile-fwd` times the shipping code
/// path instead of a copy that drifts from it.
///
/// # Panics
/// Panics if the buffers are smaller than the head dimensions imply.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn attention_all_heads_for_profiling(
    attn_out: &mut [f32],
    scores: &mut [f32],
    q: &[f32],
    kv_cache: &KVCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    attention_all_heads(
        attn_out,
        scores,
        q,
        kv_cache,
        layer_idx,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
    );
}

/// Standalone attention for batched forward: takes pre-computed Q, K, V,
/// applies RoPE, stores K/V in cache, computes attention, writes output.
/// Does NOT do QKV or O projections (those are batched separately).
///
/// # Panics
/// Panics if any dimension is zero, `num_heads` is not divisible by
/// `num_kv_heads`, a buffer length does not match the dimensions, `pos` is
/// outside the RoPE context, the KV cache does not match the attention
/// dimensions, or storing at `pos` would leave a gap below the KV cache's
/// filled watermark.
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
        attention_head(
            q_head, out_head, scores, kv_cache, layer_idx, kvh, pos, head_dim, scale,
        );
    }
}

/// Attention for a whole batch of query positions against an already-populated
/// KV cache, parallel over tokens.
///
/// The batched forward used to interleave "store this token's K/V" with
/// "attend for this token", which forced the entire attention phase to run
/// sequentially — for a 300-token prefill that is 300 serial attention calls
/// per layer on one core. Once every token's K/V is stored up front, token `b`
/// still reads only positions `0..=b`, so the causal structure is unchanged
/// and the reads become independent.
///
/// `q` must already have RoPE applied and the cache must already hold every
/// position in `positions`. Each token's arithmetic is identical to the
/// sequential path, so results are bit-identical.
///
/// # Panics
/// Panics if any dimension is zero, `num_heads` is not divisible by
/// `num_kv_heads`, or the buffers do not match `positions.len()` tokens.
pub fn attend_batch(
    attn_out: &mut [f32], // [batch * num_heads * head_dim]
    q: &[f32],            // [batch * num_heads * head_dim], RoPE applied
    kv_cache: &KVCache,
    layer_idx: usize,
    positions: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && head_dim > 0,
        "attention dimensions must be non-zero"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "query heads must be divisible by KV heads"
    );
    let q_dim = num_heads
        .checked_mul(head_dim)
        .expect("attention dimensions overflow");
    let batch = positions.len();
    assert_eq!(
        attn_out.len(),
        batch * q_dim,
        "batched attention output length mismatch"
    );
    assert_eq!(q.len(), batch * q_dim, "batched query length mismatch");
    assert!(
        kv_cache.supports_attention(layer_idx, num_kv_heads, 0, head_dim),
        "KV cache dimensions do not match attention"
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let attend_token = |out_tok: &mut [f32], q_tok: &[f32], pos: usize| {
        // Per-token scratch: the sequential path shared one buffer, which
        // is exactly what prevented this loop from being parallel.
        let mut scores = vec![0.0f32; pos + 1];
        for qh in 0..num_heads {
            let kvh = qh / heads_per_kv;
            attention_head(
                &q_tok[qh * head_dim..(qh + 1) * head_dim],
                &mut out_tok[qh * head_dim..(qh + 1) * head_dim],
                &mut scores,
                kv_cache,
                layer_idx,
                kvh,
                pos,
                head_dim,
                scale,
            );
        }
    };

    if crate::gemm::serial_batch_attention_enabled() {
        for b in 0..batch {
            let (out_tok, q_tok) = (
                &mut attn_out[b * q_dim..(b + 1) * q_dim],
                &q[b * q_dim..(b + 1) * q_dim],
            );
            attend_token(out_tok, q_tok, positions[b]);
        }
        return;
    }

    attn_out
        .par_chunks_mut(q_dim)
        .zip(q.par_chunks(q_dim))
        .zip(positions.par_iter())
        .for_each(|((out_tok, q_tok), &pos)| attend_token(out_tok, q_tok, pos));
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

/// Gated MLP forward pass.
///
/// `hidden = activation(gate_proj(input)) * up_proj(input)`
/// `output = down_proj(hidden)`
///
/// `activation` is [`Activation::Silu`] for SwiGLU (Llama, Mistral, Qwen2) and
/// [`Activation::GeluTanh`] for GeGLU (Gemma). The name is kept for continuity
/// with every caller and benchmark that already refers to it.
///
/// # Panics
/// Panics if the projections disagree on dimensions or any buffer fails the
/// GEMM checks (see [`crate::gemm::w4a8_fused_gate_up`] and
/// [`crate::gemm::w4a8_matvec`]).
pub fn swiglu_mlp(
    output: &mut [f32], // [hidden_size]
    input: &[f32],      // [hidden_size]
    gate_proj: &QuantizedLinear<'_>,
    up_proj: &QuantizedLinear<'_>,
    down_proj: &QuantizedLinear<'_>,
    work: &mut MlpWork,
    activation: Activation,
    biases: &ProjectionBiases<'_>,
) {
    let intermediate = gate_proj.rows;

    work.gate.resize(intermediate, 0.0);
    work.up.resize(intermediate, 0.0);

    // gate = gate_proj(input), up = up_proj(input) — fused and overlapped
    w4a8_fused_gate_up(&mut work.gate, &mut work.up, gate_proj, up_proj, input);
    add_bias(&mut work.gate, biases[4]);
    add_bias(&mut work.up, biases[5]);

    // hidden = activation(gate) * up — SIMD-vectorized when available
    glu_mul_inplace(activation, &mut work.gate, &work.up, intermediate);

    // output = down_proj(hidden)
    w4a8_matvec(
        output,
        down_proj.nibble_data,
        down_proj.group_params,
        &work.gate,
        down_proj.rows,
        down_proj.cols,
        down_proj.group_size,
    );
    add_bias(output, biases[6]);
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
        let rope = RoPETable::new(4, 16, 10000.0).unwrap();
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
    fn an_oversized_rope_request_is_an_error_not_an_abort() {
        // A .raimodel only a few kilobytes long can legally declare a very
        // large max_context. Building its table must fail with a message
        // rather than take the process down.
        let text = match RoPETable::new(128, 50_000_000, 10_000.0) {
            Ok(_) => panic!("a 50M-position table must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(text.contains("budget"), "unexpected message: {text}");
        assert!(
            text.contains("50000000"),
            "message should name the context: {text}"
        );

        // Degenerate arguments are errors too, not panics.
        assert!(RoPETable::new(0, 16, 10_000.0).is_err());
        assert!(RoPETable::new(7, 16, 10_000.0).is_err());
        assert!(RoPETable::new(8, 0, 10_000.0).is_err());
        assert!(RoPETable::new(8, 16, f32::NAN).is_err());
        assert!(RoPETable::new(8, 16, -1.0).is_err());
    }

    #[test]
    #[should_panic(expected = "RoPE head buffer length mismatch")]
    fn rope_rejects_mismatched_head_buffer_before_simd() {
        let rope = RoPETable::new(8, 4, 10_000.0).unwrap();
        rope.apply(&mut [0.0; 7], 1, 0);
    }

    #[test]
    #[should_panic(expected = "KV cache dimensions do not match attention")]
    fn attention_rejects_mismatched_cache_before_simd() {
        let rope = RoPETable::new(16, 4, 10_000.0).unwrap();
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
        let rope = RoPETable::new(6, 2, 10_000.0).unwrap();
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

    // -----------------------------------------------------------------------
    // GeGLU
    // -----------------------------------------------------------------------

    /// `gelu_tanh` in f64, straight from the definition — the reference every
    /// approximation below is measured against.
    fn gelu_tanh_reference(x: f64) -> f64 {
        0.5 * x
            * (1.0 + ((2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    /// A wide, deliberately awkward sweep: both signs, the saturation
    /// shoulders, and magnitudes far past anything an MLP produces.
    fn gelu_probe_points() -> Vec<f32> {
        let mut points: Vec<f32> = Vec::new();
        let mut x = -60.0f64;
        while x <= 60.0 {
            points.push(x as f32);
            x += 0.001;
        }
        points.extend([
            0.0, -0.0, 1e-8, -1e-8, 1e-4, -1e-4, 0.5, -0.5, 1.702, -1.702, 2.0, -2.0, 3.0, -3.0,
            5.0, -5.0, 8.0, -8.0, 10.0, -10.0, 1e3, -1e3, 1e6, -1e6,
        ]);
        points
    }

    #[test]
    fn gelu_tanh_matches_high_precision_reference() {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        for &x in &gelu_probe_points() {
            let reference = gelu_tanh_reference(x as f64);
            let error = (gelu_tanh(x) as f64 - reference).abs();
            if error > worst {
                worst = error;
                worst_at = x;
            }
        }
        // The bound is dominated by f32 rounding of the result itself: at
        // |x| = 60 one ulp is already 4e-6.
        assert!(
            worst < 1e-5,
            "scalar gelu_tanh is off by {worst} at x={worst_at}"
        );

        // Relative accuracy over the range where the result is not dominated
        // by the `1 + tanh` cancellation. Past |x| ~ 3 the true value is a
        // difference of two near-equal numbers and *any* f32 evaluation loses
        // relative precision there; the absolute bound above is what matters.
        // This is the check a Schraudolph-style fast exp would fail outright.
        for &x in &[-2.0f32, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0] {
            let reference = gelu_tanh_reference(x as f64);
            let relative = ((gelu_tanh(x) as f64 - reference) / reference).abs();
            assert!(relative < 1e-5, "gelu_tanh({x}) relative error {relative}");
        }

        // Saturation: gelu_tanh(x) -> 0 for x << 0 and -> x for x >> 0.
        assert!(gelu_tanh(-40.0).abs() < 1e-4, "{}", gelu_tanh(-40.0));
        assert!((gelu_tanh(40.0) - 40.0).abs() < 1e-4);
        assert_eq!(gelu_tanh(0.0), 0.0);
    }

    #[test]
    fn gelu_tanh_mul_agrees_with_the_scalar_definition() {
        // A length that is not a multiple of 8, so the SIMD body and the
        // scalar tail are both exercised in one call.
        let n = 8 * 5 + 3;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.37).collect();
        let up: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let mut got = gate.clone();
        gelu_tanh_mul_inplace(&mut got, &up, n);
        for i in 0..n {
            let reference = gelu_tanh_reference(gate[i] as f64) * up[i] as f64;
            assert!(
                (got[i] as f64 - reference).abs() < 1e-5,
                "index {i}: got {} expected {reference}",
                got[i]
            );
        }
    }

    #[test]
    fn glu_selects_the_activation() {
        let gate = [-2.0f32, -0.5, 0.5, 2.0, 3.0, -3.0, 0.0, 1.0];
        let up = [1.0f32; 8];

        let mut silu_out = gate;
        glu_mul_inplace(Activation::Silu, &mut silu_out, &up, 8);
        let mut gelu_out = gate;
        glu_mul_inplace(Activation::GeluTanh, &mut gelu_out, &up, 8);

        for i in 0..8 {
            assert!(
                (gelu_out[i] as f64 - gelu_tanh_reference(gate[i] as f64)).abs() < 1e-5,
                "GeGLU lane {i}"
            );
        }
        // The two must actually differ, or the selector is not selecting.
        assert!(
            (silu_out[0] - gelu_out[0]).abs() > 1e-3,
            "SiLU and GeLU produced the same value at x=-2"
        );
    }

    #[test]
    fn activation_codes_round_trip_and_reject_the_unknown() {
        assert_eq!(Activation::from_code(0), Some(Activation::Silu));
        assert_eq!(Activation::from_code(1), Some(Activation::GeluTanh));
        assert_eq!(Activation::from_code(2), None);
        assert_eq!(Activation::Silu.code(), 0);
        assert_eq!(Activation::GeluTanh.code(), 1);
    }

    // -----------------------------------------------------------------------
    // llama3 RoPE
    // -----------------------------------------------------------------------

    /// `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["llama3"]`
    /// evaluated on the Llama-3.2-1B-Instruct config (head_dim 64,
    /// rope_theta 500000, factor 32, low 1, high 4, original 8192) with
    /// transformers 5.15.0 / torch 2.13.0, printed at full f64 precision.
    const LLAMA3_1B_INV_FREQ: [f64; 32] = [
        1.0,
        0.663_601_279_258_728,
        0.440_366_625_785_827_64,
        0.292_227_834_463_119_5,
        0.193_922_758_102_417,
        0.128_687_381_744_384_77,
        0.085_397_101_938_724_52,
        0.056_669_618_934_392_93,
        0.037_606_030_702_590_94,
        0.024_955_408_647_656_44,
        0.016_560_440_883_040_428,
        0.010_989_529_080_688_953,
        0.007_292_665_075_510_74,
        0.004_839_421_249_926_09,
        0.003_211_446_106_433_868_4,
        0.001_290_548_010_729_253_3,
        0.000_429_556_705_057_621,
        9.708_286_233_944_818e-5,
        1.946_163_865_795_824_7e-5,
        1.291_476_746_700_937e-5,
        8.570_255_886_297_673e-6,
        5.687_232_260_243_036e-6,
        3.774_054_448_513_197_7e-6,
        2.504_467_147_446_121e-6,
        1.661_967_417_021_514_8e-6,
        1.102_883_629_755_524_5e-6,
        7.318_749_339_901_842e-7,
        4.856_731_266_045_244e-7,
        3.222_932_889_457_297_3e-7,
        2.138_742_303_259_277_8e-7,
        1.419_272_024_349_993_4e-7,
        9.418_306_490_260_875e-8,
    ];

    #[test]
    fn llama3_rope_matches_the_transformers_reference() {
        let scaling = RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8_192,
        };
        let table = RoPETable::with_scaling(64, 4, 500_000.0, scaling).unwrap();

        // At pos = 1 the angle *is* the inverse frequency, so the table's
        // cos/sin row is a direct read-out of the transform.
        for (i, &reference) in LLAMA3_1B_INV_FREQ.iter().enumerate() {
            let expected_cos = reference.cos() as f32;
            let expected_sin = reference.sin() as f32;
            assert!(
                (table.cos[32 + i] - expected_cos).abs() < 2e-7,
                "cos[{i}]: {} != {expected_cos}",
                table.cos[32 + i]
            );
            assert!(
                (table.sin[32 + i] - expected_sin).abs() < 2e-7,
                "sin[{i}]: {} != {expected_sin}",
                table.sin[32 + i]
            );
        }

        // All three bands must be represented, or the test would pass on an
        // implementation that only handles one of them.
        let unscaled = RoPETable::with_scaling(64, 4, 500_000.0, RopeScaling::None).unwrap();
        let old_context = 8_192.0f64;
        let (mut untouched, mut divided, mut smoothed) = (0, 0, 0);
        for (i, &reference) in LLAMA3_1B_INV_FREQ.iter().enumerate() {
            let plain = 1.0f64 / 500_000f64.powf(2.0 * i as f64 / 64.0);
            let wavelen = 2.0 * std::f64::consts::PI / plain;
            if wavelen < old_context / 4.0 {
                untouched += 1;
                assert!((reference / plain - 1.0).abs() < 1e-6, "band A at {i}");
            } else if wavelen > old_context / 1.0 {
                divided += 1;
                assert!(
                    (reference * 32.0 / plain - 1.0).abs() < 1e-6,
                    "band C at {i}"
                );
            } else {
                smoothed += 1;
                // Strictly between the two extremes.
                assert!(
                    reference < plain && reference > plain / 32.0,
                    "band B at {i}"
                );
            }
        }
        assert!(untouched > 0 && divided > 0 && smoothed > 0);
        // And the scaled table must actually differ from the plain one. `cos`
        // is 1.0 either way for the slowest frequencies, so this reads `sin`,
        // which is ~= the frequency itself at pos 1.
        let differing = (0..32)
            .filter(|&i| (table.sin[32 + i] - unscaled.sin[32 + i]).abs() > 1e-9)
            .count();
        assert!(
            differing >= 16,
            "llama3 scaling changed only {differing} of 32 frequencies"
        );
    }

    #[test]
    fn default_rope_scaling_leaves_the_table_untouched() {
        let plain = RoPETable::new(16, 8, 10_000.0).unwrap();
        let explicit = RoPETable::with_scaling(16, 8, 10_000.0, RopeScaling::None).unwrap();
        assert_eq!(plain.cos, explicit.cos);
        assert_eq!(plain.sin, explicit.sin);
    }

    // -----------------------------------------------------------------------
    // Bias
    // -----------------------------------------------------------------------

    #[test]
    fn add_bias_covers_simd_body_and_scalar_tail() {
        let n = 8 * 3 + 5;
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let mut output: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
        let before = output.clone();
        add_bias(&mut output, Some(&bias));
        for i in 0..n {
            assert_eq!(output[i], before[i] + bias[i], "index {i}");
        }

        // No bias is a no-op, not a zero-fill.
        let mut untouched = before.clone();
        add_bias(&mut untouched, None);
        assert_eq!(untouched, before);
    }

    #[test]
    #[should_panic(expected = "projection bias length")]
    fn add_bias_rejects_a_mismatched_vector() {
        let mut output = [0.0f32; 4];
        add_bias(&mut output, Some(&[1.0, 2.0]));
    }
}
