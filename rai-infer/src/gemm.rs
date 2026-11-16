//! High-performance W4A8 GEMM kernels for 4-bit quantized inference:
//! 4-bit weights × int8-quantized activations (f32 inputs are quantized
//! per group on entry, so callers still pass and receive f32).
//!
//! Optimizations:
//! 1. **Explicit AVX2+FMA+F16C SIMD** — 32 elements/iter in the inner loop.
//! 2. **Hardware F16C conversion** — `_mm_cvtph_ps` for scale/zero.
//! 3. **Fused projections** — QKV and gate+up quantize their shared input once
//!    and run as one parallel job list.
//! 4. **Factored dequant** — w*x = scale*dot(codes,x) + zero*sum(x_group), with
//!    both terms over the same (quantized, on AVX2) activations.
//! 5. **Rayon row chunks** for large matrices, single-thread for small, on the
//!    AVX2 and the scalar kernels alike.

// These kernels deliberately use numeric indexing and fixed, explicit argument lists so the
// scalar and SIMD implementations stay structurally comparable. Rewriting them as iterators or
// parameter objects would obscure bounds reasoning and can change hot-loop code generation.
#![allow(
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use half::f16;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::sync::OnceLock;

use crate::format::QuantizedLinear;

// ---------------------------------------------------------------------------
// Cached feature detection — eliminates per-call is_x86_feature_detected overhead
// ---------------------------------------------------------------------------

static HAS_AVX2_FMA_F16C: OnceLock<bool> = OnceLock::new();

/// Fast cached check: single pointer deref after first call.
#[inline(always)]
pub fn has_avx2() -> bool {
    *HAS_AVX2_FMA_F16C.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Send-safe pointer wrapper for parallel GEMM dispatch
// ---------------------------------------------------------------------------

/// The batched GEMM's output, shared across rayon workers.
///
/// Each worker writes the rows of its own chunk for every token. Those writes
/// are disjoint but strided by `rows`, so they cannot be handed out as
/// `&mut` slices the way the single-token paths hand out row chunks; this is
/// the one place a raw pointer crosses threads.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
struct SendPtr(*mut f32);
// SAFETY: every worker writes a disjoint set of elements (see above), and the
// buffer outlives the parallel loop that uses the pointer.
#[cfg(target_arch = "x86_64")]
unsafe impl Send for SendPtr {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for SendPtr {}
#[cfg(target_arch = "x86_64")]
impl SendPtr {
    #[inline(always)]
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn read_f16_le(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16::from_bits(bits).to_f32()
}

/// Per-group sums of an f32 activation vector, for the scalar kernels'
/// factored dequantization `w·x = scale·dot(codes, x) + zero·sum(x_group)`.
///
/// The AVX2 kernels never use these: they apply the zero point to the sum of
/// the *quantized* activations instead (see `quantize_input_split`).
fn group_sums_into(input: &[f32], sums: &mut [f32], cols: usize, group_size: usize) {
    for (g, sum) in sums.iter_mut().enumerate().take(cols.div_ceil(group_size)) {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(cols);
        *sum = input[start..end].iter().sum();
    }
}

/// Per-group int8 quantization with even/odd column split, for the AVX2
/// kernels. `input_even[k] = quantize(input[2k])`,
/// `input_odd[k] = quantize(input[2k+1])`.
///
/// Also writes, per group, the sum of the *quantized* activations in their
/// own units: `input_qsums[g] = scale_g * sum(q_i)`. The kernels apply each
/// group's zero point to that sum rather than to the exact f32 one. With the
/// exact sum, the zero term carried no quantization error while the code term
/// did, so the activation error landed weighted by the raw codes `c_i` (all
/// positive, mean ~7.5) instead of by the dequantized weights `s*c_i + z`
/// (centred near zero): about 2.7x larger than it has to be, in every
/// projection. Both terms from the same quantized activations compute exactly
/// `sum(w_i * xq_i)`, whose error is the weights' own.
///
/// Only the AVX2 kernels consume int8 activations; the scalar kernels work on
/// the f32 input directly.
#[cfg(target_arch = "x86_64")]
#[inline]
fn quantize_input_split(
    input: &[f32],
    input_even: &mut [i8],
    input_odd: &mut [i8],
    input_scales: &mut [f32],
    input_qsums: &mut [f32],
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    debug_assert!(
        has_avx2(),
        "int8 activations exist only for the AVX2 kernels"
    );
    // SAFETY: the callers only quantize after `has_avx2()` returned true, and
    // `validate_weight_buffers` established the buffer bounds.
    unsafe {
        quantize_input_split_avx2(
            input,
            input_even,
            input_odd,
            input_scales,
            input_qsums,
            cols,
            group_size,
            num_groups,
        );
    }
}

/// AVX2 fused quantize+split: absmax → scale → quantize → deinterleave even/odd.
/// Processes 8 floats (4 pairs) per iteration using 128-bit packs to avoid
/// AVX2 cross-lane ordering issues.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2 is required).
/// - `input` must hold at least `cols` values; `input_even` and `input_odd`
///   must each hold at least `cols / 2` values (`cols` even).
/// - `input_scales` and `input_qsums` must each hold `num_groups` values.
/// - `num_groups` must equal `cols.div_ceil(group_size)` and be at most
///   `MAX_GROUPS`, with `group_size` even. The public GEMM entry points
///   establish these bounds via `validate_weight_buffers`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_input_split_avx2(
    input: &[f32],
    input_even: &mut [i8],
    input_odd: &mut [i8],
    input_scales: &mut [f32],
    input_qsums: &mut [f32],
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
    // Shuffle masks to extract even/odd bytes from 8 packed i8 values
    let even_shuf = _mm_setr_epi8(0, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let odd_shuf = _mm_setr_epi8(1, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(cols);
        let len = end - start;
        let inp = input.as_ptr().add(start);

        // Phase 1: AVX2 absmax
        let chunks8 = len / 8;
        let mut vmax = _mm256_setzero_ps();
        for i in 0..chunks8 {
            let v = _mm256_loadu_ps(inp.add(i * 8));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
        }
        // Horizontal max
        let hi128 = _mm256_extractf128_ps(vmax, 1);
        let lo128 = _mm256_castps256_ps128(vmax);
        let m128 = _mm_max_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(m128);
        let m64 = _mm_max_ps(m128, shuf);
        let hi64 = _mm_movehl_ps(m64, m64);
        let mut absmax = _mm_cvtss_f32(_mm_max_ss(m64, hi64));
        for i in (chunks8 * 8)..len {
            let a = (*inp.add(i)).abs();
            if a > absmax {
                absmax = a;
            }
        }

        let scale = if absmax > 1e-10 { absmax / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        input_scales[g] = scale;

        // Phase 2: Quantize + deinterleave even/odd
        // Process 8 floats (4 pairs) per iteration using 128-bit packs
        let inv_v = _mm256_set1_ps(inv_scale);
        let min_v = _mm256_set1_ps(-128.0);
        let max_v = _mm256_set1_ps(127.0);
        let iters8 = len / 8;
        let even_base = start / 2;
        // Running sum of the quantized codes; |q| <= 128 and a group holds at
        // most `cols` values, so an i32 lane cannot overflow.
        let mut qsum_v = _mm256_setzero_si256();

        for i in 0..iters8 {
            let off = i * 8;
            let v = _mm256_loadu_ps(inp.add(off));
            let scaled = _mm256_mul_ps(v, inv_v);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_v), min_v);
            let i32s = _mm256_cvtps_epi32(clamped);
            qsum_v = _mm256_add_epi32(qsum_v, i32s);

            // Split into 128-bit halves and pack within lanes (no cross-lane issues)
            let lo = _mm256_castsi256_si128(i32s); // 4 i32 [0..3]
            let hi = _mm256_extracti128_si256(i32s, 1); // 4 i32 [4..7]
            let packed16 = _mm_packs_epi32(lo, hi); // 8 i16 in order
            let packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128()); // 8 i8 in lower bytes

            // Deinterleave: even=[0,2,4,6], odd=[1,3,5,7]
            let even_bytes = _mm_shuffle_epi8(packed8, even_shuf);
            let odd_bytes = _mm_shuffle_epi8(packed8, odd_shuf);

            // Store 4 bytes each (4 pairs)
            let dst_even = input_even.as_mut_ptr().add(even_base + i * 4) as *mut u32;
            let dst_odd = input_odd.as_mut_ptr().add(even_base + i * 4) as *mut u32;
            dst_even.write_unaligned(_mm_cvtsi128_si32(even_bytes) as u32);
            dst_odd.write_unaligned(_mm_cvtsi128_si32(odd_bytes) as u32);
        }
        let mut qsum = hsum_epi32(qsum_v);

        // Scalar tail for remaining pairs
        let mut col = start + iters8 * 8;
        while col + 1 < end {
            let k = col / 2;
            let even = quantize_i8(*input.as_ptr().add(col), inv_scale);
            let odd = quantize_i8(*input.as_ptr().add(col + 1), inv_scale);
            input_even[k] = even;
            input_odd[k] = odd;
            qsum += i32::from(even) + i32::from(odd);
            col += 2;
        }
        input_qsums[g] = scale * qsum as f32;
    }
}

/// One activation quantized to int8 exactly as the SIMD body does it:
/// round half to even, clamp to the i8 range.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn quantize_i8(value: f32, inv_scale: f32) -> i8 {
    (value * inv_scale).round_ties_even().clamp(-128.0, 127.0) as i8
}

/// Horizontal sum of eight i32 lanes.
///
/// # Safety
/// Must only be called from an AVX2 context.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b10_11_00_01));
    _mm_cvtsi128_si32(sum32)
}

// ---------------------------------------------------------------------------
// AVX2 W4A8 GEMM kernels
// ---------------------------------------------------------------------------

// NOTE: matvec_chunk_avx2 (f32 input path) removed — superseded by matvec_chunk_i8.
// NOTE: lm_head_chunk_avx2 (f32 hidden path) removed — superseded by lm_head_chunk_i8.

/// AVX2 W4A8 chunk processor: zero port 5 inner loop.
/// Uses even/odd input split to avoid all unpack/shuffle instructions.
/// lo_nibbles × input_even + hi_nibbles × input_odd via PMADDUBSW.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA+F16C).
/// - `output` must be valid for `chunk_len` writes.
/// - `nibble_data` must hold at least `(start_row + chunk_len) * cols / 2`
///   bytes and `group_params` at least
///   `(start_row + chunk_len) * num_groups * 4` bytes.
/// - `input_even`/`input_odd` must hold `cols / 2` int8 values produced by
///   `quantize_input_split` (values within ±127, so PMADDUBSW pair sums cannot
///   saturate against 4-bit codes), and `input_scales`/`input_qsums` its
///   `num_groups` per-group scales and quantized sums.
/// - `num_groups == cols.div_ceil(group_size) <= MAX_GROUPS`, `cols` and
///   `group_size` even. `validate_weight_buffers` at every public entry point
///   establishes exactly these preconditions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matvec_chunk_i8(
    output: *mut f32,
    nibble_data: *const u8,
    group_params: &[u8],
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: &[f32],
    input_qsums: &[f32],
    start_row: usize,
    chunk_len: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
    scratch: &mut [f32],
) {
    let row_bytes = cols / 2;
    let mask_0f = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);
    // Adaptive prefetch distance: 3 rows for small rows (~15ns compute each),
    // 1 row for large rows (7B MLP: 7168 bytes/row, plenty of compute to hide latency).
    let pf_dist: usize = if row_bytes > 2048 { 1 } else { 3 };
    // Caller-owned, reused across every row in the chunk. This used to be a
    // `[f32; MAX_GROUPS]` declared inside the row loop, which zeroed the whole
    // array once per row: invisible while the bound was 128, a per-decode-step
    // memset proportional to the bound once it is not.
    let w_scales = &mut scratch[..num_groups];

    for local in 0..chunk_len {
        let row = start_row + local;
        let nib_row = nibble_data.add(row * row_bytes);
        let row_param_base = row * num_groups * 4;

        // Prefetch ahead into L1 to hide DRAM latency (~50-70ns).
        if local + pf_dist < chunk_len {
            let pf_nib = nibble_data.add((row + pf_dist) * row_bytes);
            let mut pf = 0;
            while pf < row_bytes {
                _mm_prefetch(pf_nib.add(pf) as *const i8, _MM_HINT_T0);
                pf += 64;
            }
        }

        // Phase 1: Pre-compute zero correction and extract weight scales.
        // Use unchecked pointer reads to eliminate bounds-check overhead
        // (~275K checks per token in the original version).
        let mut zero_corr = 0.0f32;
        let gp_ptr = group_params.as_ptr();
        for g in 0..num_groups {
            let param_off = row_param_base + g * 4;
            let bits = (gp_ptr.add(param_off) as *const u32).read_unaligned();
            let v = _mm_cvtsi32_si128(bits as i32);
            let f = _mm_cvtph_ps(v);
            *w_scales.get_unchecked_mut(g) = _mm_cvtss_f32(f);
            let f1 = _mm_shuffle_ps(f, f, 1);
            zero_corr += _mm_cvtss_f32(f1) * *input_qsums.get_unchecked(g);
        }

        // Phase 2: Integer dot product per group using 256-bit PMADDUBSW.
        // Processes 64 weight elements per iteration (2× wider than 128-bit).
        let mut float_acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            let col_start = g * group_size;
            let actual_gs = ((g + 1) * group_size).min(cols) - col_start;
            let nib_off = col_start / 2;
            let even_off = col_start / 2;
            let chunks64 = actual_gs / 64;

            let mut iacc = _mm256_setzero_si256();

            for c in 0..chunks64 {
                // Load 32 weight bytes (64 nibbles) — 2× wider than 128-bit
                let raw = _mm256_loadu_si256(nib_row.add(nib_off + c * 32) as *const __m256i);
                let lo = _mm256_and_si256(raw, mask_0f);
                let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0f);

                // Load 32 pre-split int8 input values each (even/odd)
                let inp_e = _mm256_loadu_si256(input_even.add(even_off + c * 32) as *const __m256i);
                let inp_o = _mm256_loadu_si256(input_odd.add(even_off + c * 32) as *const __m256i);

                // 256-bit PMADDUBSW: 32 pairs each for even and odd
                let prod_e = _mm256_maddubs_epi16(lo, inp_e);
                let prod_o = _mm256_maddubs_epi16(hi, inp_o);
                let combined = _mm256_add_epi16(prod_e, prod_o);
                let sums = _mm256_madd_epi16(combined, ones_16);
                iacc = _mm256_add_epi32(iacc, sums);
            }

            // Finalize group: convert 256-bit int to float, scale, accumulate
            let combined_scale = *w_scales.get_unchecked(g) * *input_scales.get_unchecked(g);
            let dot_f = _mm256_mul_ps(_mm256_cvtepi32_ps(iacc), _mm256_set1_ps(combined_scale));
            float_acc = _mm256_add_ps(float_acc, dot_f);

            // Handle remaining 32-element block with 128-bit PMADDUBSW
            let simd_done = chunks64 * 64;
            if simd_done + 32 <= actual_gs {
                let mask128 = _mm_set1_epi8(0x0F);
                let ones128 = _mm_set1_epi16(1);
                let raw = _mm_loadu_si128(nib_row.add(nib_off + simd_done / 2) as *const __m128i);
                let lo = _mm_and_si128(raw, mask128);
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask128);
                let inp_e =
                    _mm_loadu_si128(input_even.add(even_off + simd_done / 2) as *const __m128i);
                let inp_o =
                    _mm_loadu_si128(input_odd.add(even_off + simd_done / 2) as *const __m128i);
                let prod_e = _mm_maddubs_epi16(lo, inp_e);
                let prod_o = _mm_maddubs_epi16(hi, inp_o);
                let combined = _mm_add_epi16(prod_e, prod_o);
                let sums = _mm_madd_epi16(combined, ones128);
                let tail_f = _mm_mul_ps(_mm_cvtepi32_ps(sums), _mm_set1_ps(combined_scale));
                let tail_256 = _mm256_insertf128_ps(_mm256_setzero_ps(), tail_f, 0);
                float_acc = _mm256_add_ps(float_acc, tail_256);
            }

            // Scalar tail for the remaining pairs, on the same int8 values
            // the SIMD body used: the zero term counts these quantized
            // activations too, so the code term must not see the exact ones.
            let simd_done2 = simd_done + if simd_done + 32 <= actual_gs { 32 } else { 0 };
            if simd_done2 < actual_gs {
                let tail = int8_pair_dot(
                    nib_row.add(nib_off + simd_done2 / 2),
                    input_even.add(even_off + simd_done2 / 2),
                    input_odd.add(even_off + simd_done2 / 2),
                    (actual_gs - simd_done2) / 2,
                );
                let tail_v = _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    tail as f32 * combined_scale,
                );
                float_acc = _mm256_add_ps(float_acc, tail_v);
            }
        }

        // Phase 3: Horizontal sum of 8-wide float accumulator
        let hi128 = _mm256_extractf128_ps(float_acc, 1);
        let lo128 = _mm256_castps256_ps128(float_acc);
        let s128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(s128);
        let s = _mm_add_ps(s128, shuf);
        let hi = _mm_movehl_ps(s, s);
        let dot = _mm_cvtss_f32(_mm_add_ss(s, hi));

        *output.add(local) = dot + zero_corr;
    }
}

/// `sum(lo(b_k) * even_k + hi(b_k) * odd_k)` over `pairs` packed bytes: the
/// scalar remainder of the int8 kernels, on exactly the values PMADDUBSW
/// multiplies in their SIMD bodies.
///
/// # Safety
/// `codes`, `even` and `odd` must each be readable for `pairs` values.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn int8_pair_dot(codes: *const u8, even: *const i8, odd: *const i8, pairs: usize) -> i32 {
    let mut sum = 0i32;
    for k in 0..pairs {
        let byte = *codes.add(k);
        sum += i32::from(byte & 0x0F) * i32::from(*even.add(k))
            + i32::from(byte >> 4) * i32::from(*odd.add(k));
    }
    sum
}

/// One weight row against `T` tokens, sharing a single unpack of the weights.
///
/// The 4-bit codes are unpacked once per 32-byte block and reused by all `T`
/// tokens, so the load/AND/shift work that dominated the single-token inner
/// loop is paid once per tile instead of once per token.
///
/// Each token accumulates over the same groups in the same order as
/// [`matvec_chunk_i8`], so the results are bit-identical to the single-token
/// kernel — this is a scheduling change, not a numerical one.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA+F16C).
/// - `nib_row` must point at `cols / 2` readable bytes for this row.
/// - `w_scales`/`w_zeros` must hold `num_groups` decoded values for this row.
/// - `input_even`/`input_odd` must hold `num_tokens * half_cols` int8 values,
///   token-major.
/// - `input_scales`/`input_qsums` must point at `num_tokens` arrays of
///   `num_groups` values.
/// - `t_base + T <= num_tokens`, and `output` must be valid for the
///   `t * rows + row` writes those tokens imply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matmul_row_tile<const T: usize>(
    output: *mut f32,
    nib_row: *const u8,
    w_scales: &[f32],
    w_zeros: &[f32],
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: *const f32,
    input_qsums: *const f32,
    t_base: usize,
    row: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    let half_cols = cols / 2;
    let mask_0f = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);
    let mut float_acc = [_mm256_setzero_ps(); T];

    for g in 0..num_groups {
        let col_start = g * group_size;
        let actual_gs = ((g + 1) * group_size).min(cols) - col_start;
        let nib_off = col_start / 2;
        let chunks64 = actual_gs / 64;

        let mut iacc = [_mm256_setzero_si256(); T];

        // Unpack each 32-byte weight block once, consume it T times.
        for c in 0..chunks64 {
            let raw = _mm256_loadu_si256(nib_row.add(nib_off + c * 32) as *const __m256i);
            let lo = _mm256_and_si256(raw, mask_0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0f);
            for j in 0..T {
                let base = (t_base + j) * half_cols + nib_off + c * 32;
                let inp_e = _mm256_loadu_si256(input_even.add(base) as *const __m256i);
                let inp_o = _mm256_loadu_si256(input_odd.add(base) as *const __m256i);
                let prod_e = _mm256_maddubs_epi16(lo, inp_e);
                let prod_o = _mm256_maddubs_epi16(hi, inp_o);
                let combined = _mm256_add_epi16(prod_e, prod_o);
                iacc[j] = _mm256_add_epi32(iacc[j], _mm256_madd_epi16(combined, ones_16));
            }
        }

        let simd_done = chunks64 * 64;
        let has_block32 = simd_done + 32 <= actual_gs;
        let simd_done2 = simd_done + if has_block32 { 32 } else { 0 };

        for j in 0..T {
            let t = t_base + j;
            let combined_scale = *w_scales.get_unchecked(g) * *input_scales.add(t * num_groups + g);
            float_acc[j] = _mm256_add_ps(
                float_acc[j],
                _mm256_mul_ps(_mm256_cvtepi32_ps(iacc[j]), _mm256_set1_ps(combined_scale)),
            );

            if has_block32 {
                let mask128 = _mm_set1_epi8(0x0F);
                let ones128 = _mm_set1_epi16(1);
                let raw = _mm_loadu_si128(nib_row.add(nib_off + simd_done / 2) as *const __m128i);
                let lo = _mm_and_si128(raw, mask128);
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask128);
                let base = t * half_cols + nib_off + simd_done / 2;
                let inp_e = _mm_loadu_si128(input_even.add(base) as *const __m128i);
                let inp_o = _mm_loadu_si128(input_odd.add(base) as *const __m128i);
                let prod_e = _mm_maddubs_epi16(lo, inp_e);
                let prod_o = _mm_maddubs_epi16(hi, inp_o);
                let combined = _mm_add_epi16(prod_e, prod_o);
                let sums = _mm_madd_epi16(combined, ones128);
                let tail_f = _mm_mul_ps(_mm_cvtepi32_ps(sums), _mm_set1_ps(combined_scale));
                float_acc[j] = _mm256_add_ps(
                    float_acc[j],
                    _mm256_insertf128_ps(_mm256_setzero_ps(), tail_f, 0),
                );
            }

            if simd_done2 < actual_gs {
                let base = t * half_cols + nib_off + simd_done2 / 2;
                let tail = int8_pair_dot(
                    nib_row.add(nib_off + simd_done2 / 2),
                    input_even.add(base),
                    input_odd.add(base),
                    (actual_gs - simd_done2) / 2,
                );
                let tail_v = _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    tail as f32 * combined_scale,
                );
                float_acc[j] = _mm256_add_ps(float_acc[j], tail_v);
            }
        }
    }

    for j in 0..T {
        let t = t_base + j;
        let hi128 = _mm256_extractf128_ps(float_acc[j], 1);
        let lo128 = _mm256_castps256_ps128(float_acc[j]);
        let s128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(s128);
        let s = _mm_add_ps(s128, shuf);
        let hi = _mm_movehl_ps(s, s);
        let dot = _mm_cvtss_f32(_mm_add_ss(s, hi));

        let sums = core::slice::from_raw_parts(input_qsums.add(t * num_groups), num_groups);
        let mut zero_corr = 0.0f32;
        for g in 0..num_groups {
            zero_corr += *w_zeros.get_unchecked(g) * *sums.get_unchecked(g);
        }
        *output.add(t * rows + row) = dot + zero_corr;
    }
}

/// AVX2 W4A8 batched chunk processor: weight-stationary across the whole batch.
///
/// Driving [`matvec_chunk_i8`] once per token made the batch pay three
/// token-independent costs `num_tokens` times over: the per-row f16 scale/zero
/// decode, the row prefetch, and the unpacking of every 4-bit weight block.
/// This hoists the first two out of the token loop entirely and amortizes the
/// third across a tile of four tokens.
///
/// # Safety
/// Same buffer contract as [`matmul_row_tile`], for every row in
/// `start_row..start_row + chunk_len`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matmul_chunk_i8(
    output: *mut f32,
    nibble_data: *const u8,
    group_params: &[u8],
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: *const f32,
    input_qsums: *const f32,
    start_row: usize,
    chunk_len: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
    num_tokens: usize,
    scratch: &mut [f32],
) {
    let row_bytes = cols / 2;
    let pf_dist: usize = if row_bytes > 2048 { 1 } else { 3 };
    let gp_ptr = group_params.as_ptr();

    // Caller-owned and reused across every row in the chunk. These were
    // `[f32; MAX_GROUPS]` locals: correct, but their memset scaled with the
    // bound rather than with the group size the model was quantized at.
    let (w_scales, w_zeros) = scratch.split_at_mut(num_groups);

    for local in 0..chunk_len {
        let row = start_row + local;
        let nib_row = nibble_data.add(row * row_bytes);

        // Prefetch once per row. The single-token kernel re-issued this for
        // every token, hammering L1 with lines it had already pulled in.
        if local + pf_dist < chunk_len {
            let pf_nib = nibble_data.add((row + pf_dist) * row_bytes);
            let mut pf = 0;
            while pf < row_bytes {
                _mm_prefetch(pf_nib.add(pf) as *const i8, _MM_HINT_T0);
                pf += 64;
            }
        }

        // Decode this row's f16 scale/zero pairs once for the whole batch.
        let row_param_base = row * num_groups * 4;
        for g in 0..num_groups {
            let bits = (gp_ptr.add(row_param_base + g * 4) as *const u32).read_unaligned();
            let v = _mm_cvtsi32_si128(bits as i32);
            let f = _mm_cvtph_ps(v);
            *w_scales.get_unchecked_mut(g) = _mm_cvtss_f32(f);
            *w_zeros.get_unchecked_mut(g) = _mm_cvtss_f32(_mm_shuffle_ps(f, f, 1));
        }

        let mut t = 0usize;
        while t + 4 <= num_tokens {
            matmul_row_tile::<4>(
                output,
                nib_row,
                &*w_scales,
                &*w_zeros,
                input_even,
                input_odd,
                input_scales,
                input_qsums,
                t,
                row,
                rows,
                cols,
                group_size,
                num_groups,
            );
            t += 4;
        }
        while t < num_tokens {
            matmul_row_tile::<1>(
                output,
                nib_row,
                &*w_scales,
                &*w_zeros,
                input_even,
                input_odd,
                input_scales,
                input_qsums,
                t,
                row,
                rows,
                cols,
                group_size,
                num_groups,
            );
            t += 1;
        }
    }
}

/// Hidden states quantized for the AVX2 LM head, token-major for
/// `num_tokens` tokens: int8 values in [-63, 63], per-group scales and
/// quantized sums.
#[cfg(target_arch = "x86_64")]
struct Int8Hidden {
    values: Vec<i8>,
    scales: Vec<f32>,
    qsums: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
impl Int8Hidden {
    /// Only called after `has_avx2()` returned true and the buffers were
    /// validated.
    fn quantize(hidden: &[f32], tokens: usize, hidden_size: usize, group_size: usize) -> Self {
        let num_groups = hidden_size.div_ceil(group_size);
        let mut out = Self {
            values: vec![0; tokens * hidden_size],
            scales: vec![0.0; tokens * num_groups],
            qsums: vec![0.0; tokens * num_groups],
        };
        for t in 0..tokens {
            // SAFETY: `has_avx2()` returned true (see above); each slice is
            // exactly one token's `hidden_size` values or `num_groups` groups.
            unsafe {
                quantize_hidden_i8_avx2(
                    &hidden[t * hidden_size..(t + 1) * hidden_size],
                    &mut out.values[t * hidden_size..(t + 1) * hidden_size],
                    &mut out.scales[t * num_groups..(t + 1) * num_groups],
                    &mut out.qsums[t * num_groups..(t + 1) * num_groups],
                    hidden_size,
                    group_size,
                    num_groups,
                );
            }
        }
        out
    }
}

/// AVX2 LM head using PMADDUBSW for 8-bit codes × int8 hidden.
///
/// The hidden state is quantized to [-63, 63] so PMADDUBSW's i16 pair sums
/// cannot saturate against 8-bit codes (255 · 63 · 2 = 32130 < 32767). A
/// 16-bit hidden through PMADDWD was measured as an alternative: about 1% less
/// divergence from the reference model, for 11% less decode speed on a
/// 152k-token vocabulary, where the head is the largest read of every token.
///
/// Each row's scales and zeros are decoded once and the row is then dotted
/// with every token's hidden state while its bytes are still in L1, so a batch
/// of tokens reads the table from memory once rather than once per token.
/// Logit `(t, v)` is written to `logits[t * vocab_stride + v]`.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA+F16C).
/// - `logits` must be valid for writes at `t * vocab_stride + v` for every
///   token `t < hidden.values.len() / hidden_size` and row
///   `v in start_row..start_row + chunk_len`, and no other thread may write
///   those elements.
/// - `embed_data` must hold at least `(start_row + chunk_len) * hidden_size`
///   bytes and `embed_params` at least
///   `(start_row + chunk_len) * num_groups * 4` bytes.
/// - `hidden` must come from [`Int8Hidden::quantize`] with this
///   `hidden_size` and `group_size`, and `scratch` must hold
///   `2 * num_groups` values.
/// - `num_groups == hidden_size.div_ceil(group_size) <= MAX_GROUPS`.
///   `tied_lm_head` asserts these bounds before dispatching here.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn lm_head_chunk_i8(
    logits: *mut f32,
    vocab_stride: usize,
    embed_data: *const u8,
    embed_params: &[u8],
    hidden: &Int8Hidden,
    start_row: usize,
    chunk_len: usize,
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
    scratch: &mut [f32],
) {
    let num_tokens = hidden.values.len() / hidden_size;
    // Reused across every row in the chunk; see `matvec_chunk_i8`. This kernel
    // walks the whole vocabulary, so a per-row memset here is the most
    // expensive place in the engine to leave one.
    let (w_scales, w_zeros) = scratch.split_at_mut(num_groups);

    for local in 0..chunk_len {
        let v = start_row + local;
        let row_data = embed_data.add(v * hidden_size);
        let row_param_base = v * num_groups * 4;

        // Prefetch 3 rows ahead to hide DRAM latency
        if local + 3 < chunk_len {
            let pf_data = embed_data.add((v + 3) * hidden_size);
            let mut pf = 0;
            while pf < hidden_size {
                _mm_prefetch(pf_data.add(pf) as *const i8, _MM_HINT_T0);
                pf += 64;
            }
        }

        let ep_ptr = embed_params.as_ptr();
        for g in 0..num_groups {
            let bits = (ep_ptr.add(row_param_base + g * 4) as *const u32).read_unaligned();
            let ff = _mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32));
            *w_scales.get_unchecked_mut(g) = _mm_cvtss_f32(ff);
            *w_zeros.get_unchecked_mut(g) = _mm_cvtss_f32(_mm_shuffle_ps(ff, ff, 1));
        }

        for t in 0..num_tokens {
            let dot = lm_head_row_i8(
                row_data,
                w_scales,
                w_zeros,
                hidden.values.as_ptr().add(t * hidden_size),
                &hidden.scales[t * num_groups..(t + 1) * num_groups],
                &hidden.qsums[t * num_groups..(t + 1) * num_groups],
                hidden_size,
                group_size,
                num_groups,
            );
            *logits.add(t * vocab_stride + v) = dot;
        }
    }
}

/// One vocabulary row against one token's int8 hidden state.
///
/// The zero term is applied to the quantized hidden's own sums, so code and
/// zero terms see the same activations (see `quantize_input_split`).
///
/// # Safety
/// Must only be called from an AVX2+FMA context, with `row_data` and
/// `hidden_i8` readable for `hidden_size` values, `hidden_i8` within
/// [-63, 63], and the four slices holding `num_groups` values each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn lm_head_row_i8(
    row_data: *const u8,
    w_scales: &[f32],
    w_zeros: &[f32],
    hidden_i8: *const i8,
    hidden_scales: &[f32],
    hidden_qsums: &[f32],
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) -> f32 {
    let ones_16 = _mm256_set1_epi16(1);
    let mut zero_corr = 0.0f32;
    let mut float_acc = _mm256_setzero_ps();
    for g in 0..num_groups {
        zero_corr += *w_zeros.get_unchecked(g) * *hidden_qsums.get_unchecked(g);

        let col_start = g * group_size;
        let actual_gs = ((g + 1) * group_size).min(hidden_size) - col_start;
        let codes = row_data.add(col_start);
        let hidden = hidden_i8.add(col_start);
        let chunks32 = actual_gs / 32;

        // PMADDUBSW: unsigned codes (arg 1) × signed hidden (arg 2) → i16
        // pair sums; PMADDWD widens them to i32. A lane adds at most
        // 2 · 32130 per 32 codes, so any group stays far below 2^31.
        let mut iacc = _mm256_setzero_si256();
        for c in 0..chunks32 {
            let off = c * 32;
            let code_v = _mm256_loadu_si256(codes.add(off) as *const __m256i);
            let hid_v = _mm256_loadu_si256(hidden.add(off) as *const __m256i);
            iacc = _mm256_add_epi32(
                iacc,
                _mm256_madd_epi16(_mm256_maddubs_epi16(code_v, hid_v), ones_16),
            );
        }
        // Embedding groups are almost always a multiple of 32 wide, so the
        // remainder is skipped rather than folded in as zeros.
        if chunks32 * 32 < actual_gs {
            let mut tail = 0i32;
            for i in (chunks32 * 32)..actual_gs {
                tail += i32::from(*codes.add(i)) * i32::from(*hidden.add(i));
            }
            iacc = _mm256_add_epi32(iacc, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, tail));
        }
        let combined = *w_scales.get_unchecked(g) * *hidden_scales.get_unchecked(g);
        float_acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc),
            _mm256_set1_ps(combined),
            float_acc,
        );
    }

    let hi128 = _mm256_extractf128_ps(float_acc, 1);
    let lo128 = _mm256_castps256_ps128(float_acc);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let s64 = _mm_add_ps(s128, shuf);
    let hi64 = _mm_movehl_ps(s64, s64);
    _mm_cvtss_f32(_mm_add_ss(s64, hi64)) + zero_corr
}

/// AVX2 hidden quantization for the LM head: per group, absmax → scale →
/// round half to even → clamp to [-63, 63] → i8 (no even/odd split needed).
///
/// Writes the per-group scales and, as `quantize_input_split` does, the
/// per-group sums of the quantized values in their own units, which the head
/// applies its zero points to. Only the AVX2 head quantizes; the scalar head
/// reads the f32 hidden directly.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2 is required).
/// - `hidden` and `output` must each hold at least `hidden_size` values;
///   `scales` and `qsums` at least `num_groups`.
/// - `num_groups` must equal `hidden_size.div_ceil(group_size)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_hidden_i8_avx2(
    hidden: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    qsums: &mut [f32],
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) {
    const LIMIT: f32 = 63.0;
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(hidden_size);
        let len = end - start;
        let inp = hidden.as_ptr().add(start);

        // AVX2 absmax
        let chunks8 = len / 8;
        let mut vmax = _mm256_setzero_ps();
        for i in 0..chunks8 {
            let v = _mm256_loadu_ps(inp.add(i * 8));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
        }
        let hi128 = _mm256_extractf128_ps(vmax, 1);
        let lo128 = _mm256_castps256_ps128(vmax);
        let m128 = _mm_max_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(m128);
        let m64 = _mm_max_ps(m128, shuf);
        let hi64 = _mm_movehl_ps(m64, m64);
        let mut absmax = _mm_cvtss_f32(_mm_max_ss(m64, hi64));
        for i in (chunks8 * 8)..len {
            absmax = absmax.max((*inp.add(i)).abs());
        }

        let scale = if absmax > 1e-10 { absmax / LIMIT } else { 1.0 };
        let inv_scale = 1.0 / scale;
        scales[g] = scale;

        let inv_v = _mm256_set1_ps(inv_scale);
        let min_v = _mm256_set1_ps(-LIMIT);
        let max_v = _mm256_set1_ps(LIMIT);
        let out_ptr = output.as_mut_ptr().add(start);
        let mut qsum_v = _mm256_setzero_si256();

        for i in 0..chunks8 {
            let off = i * 8;
            let v = _mm256_loadu_ps(inp.add(off));
            let rounded = _mm256_round_ps(
                _mm256_mul_ps(v, inv_v),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );
            let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_v), min_v);
            let i32s = _mm256_cvtps_epi32(clamped);
            qsum_v = _mm256_add_epi32(qsum_v, i32s);
            // Pack i32 → i16 → i8 within 128-bit halves (no cross-lane issue).
            let packed16 = _mm_packs_epi32(
                _mm256_castsi256_si128(i32s),
                _mm256_extracti128_si256(i32s, 1),
            );
            let packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
            (out_ptr.add(off) as *mut u64).write_unaligned(_mm_cvtsi128_si64(packed8) as u64);
        }
        let mut qsum = hsum_epi32(qsum_v);

        for i in (chunks8 * 8)..len {
            let q = (*inp.add(i) * inv_scale)
                .round_ties_even()
                .clamp(-LIMIT, LIMIT) as i8;
            *out_ptr.add(i) = q;
            qsum += i32::from(q);
        }
        qsums[g] = scale * qsum as f32;
    }
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

fn matvec_row_scalar(
    nibble_data: &[u8],
    group_params: &[u8],
    input: &[f32],
    input_sums: &[f32],
    row: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
) -> f32 {
    let row_bytes = cols / 2;
    let row_param_base = row * num_groups * 4;
    let mut acc = 0.0f32;
    let mut byte_off = row * row_bytes;
    let mut col = 0usize;

    for g in 0..num_groups {
        let param_off = row_param_base + g * 4;
        let scale = read_f16_le(group_params, param_off);
        let zero = read_f16_le(group_params, param_off + 2);
        let actual_gs = ((g + 1) * group_size).min(cols) - g * group_size;
        let half_gs = actual_gs / 2;

        let mut dot = 0.0f32;
        for b in 0..half_gs {
            let byte = nibble_data[byte_off + b];
            dot += (byte & 0x0F) as f32 * input[col + b * 2]
                + (byte >> 4) as f32 * input[col + b * 2 + 1];
        }
        byte_off += half_gs;
        col += actual_gs;

        acc += scale * dot + zero * input_sums[g];
    }
    acc
}

fn lm_head_row_scalar(
    embed_data: &[u8],
    embed_params: &[u8],
    hidden: &[f32],
    hidden_sums: &[f32],
    v: usize,
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) -> f32 {
    let row_param_base = v * num_groups * 4;
    let mut acc = 0.0f32;
    for g in 0..num_groups {
        let param_off = row_param_base + g * 4;
        let scale = read_f16_le(embed_params, param_off);
        let zero = read_f16_le(embed_params, param_off + 2);
        let col_start = g * group_size;
        let actual_gs = ((g + 1) * group_size).min(hidden_size) - col_start;

        let mut dot = 0.0f32;
        for i in 0..actual_gs {
            dot += embed_data[v * hidden_size + col_start + i] as f32 * hidden[col_start + i];
        }
        acc += scale * dot + zero * hidden_sums[g];
    }
    acc
}

// ---------------------------------------------------------------------------
// Dispatch policy
// ---------------------------------------------------------------------------

/// Vocabulary rows per parallel chunk of the LM head.
const LM_CHUNK: usize = 512;
/// Fewest rows a decode-time matvec needs before it is worth parallelizing.
const PAR_THRESHOLD: usize = 256;

/// Smallest weight matrix, in bytes of packed nibbles, worth handing to rayon
/// during single-token decode.
///
/// A row count alone is the wrong question: 576 rows of 576 columns is 165 KB,
/// which one thread streams in ~17 µs, while a rayon dispatch costs 20-27 µs
/// before any work happens. `bw-bench` measures both — steps 3 to 5 — and the
/// parallel read of that matrix comes back 10-20x slower than the serial one.
/// Below this size the fan-out is pure loss, so decode walks the rows itself.
///
/// Prefill is unaffected: it batches many tokens per weight chunk, so the work
/// per dispatch is large enough that the overhead disappears into it.
const PAR_MIN_BYTES: usize = 512 * 1024;

/// Whether a decode-time matvec of this shape is worth parallelizing.
fn matvec_should_parallelize(rows: usize, cols: usize) -> bool {
    rows >= PAR_THRESHOLD && rows * cols / 2 >= PAR_MIN_BYTES
}

/// Adaptive chunk sizing: target ~24KB per chunk to fit L1 cache (32KB).
/// Each row needs cols/2 bytes of nibble data.
/// SmolLM (cols=256): 24576/(128)=192→clamped to 64 rows.
/// Mistral hidden (cols=4096): 24576/(2048)=12 rows.
/// Mistral MLP (cols=14336): 24576/(7168)=3→clamped to 4 rows.
#[inline]
fn chunk_rows_for(cols: usize) -> usize {
    (24_576 / (cols / 2).max(1)).clamp(4, 64)
}

/// Maximum number of quantization groups per GEMM call.
///
/// This is a validation bound, not a working size: no buffer in the hot path is
/// dimensioned by it. Group data travels as slices of the model's real
/// `num_groups`, and per-row scratch is allocated once per worker thread, so a
/// model quantized at group size 128 pays for 128 whatever this constant says.
///
/// 1024 covers every architecture the converter accepts at the finest group
/// size it offers: the widest advertised MLP is Qwen3-32B's 25600, which needs
/// 800 groups at group size 32. It was 128, which was enough only for group
/// size 128 on a 7B and silently made the shipped default the sole option
/// there.
///
/// `format.rs` imports this constant: `.raimodel` validation at load time is
/// the single gate that keeps accepted model files within kernel capacity, so
/// the limit is defined once here at its point of enforcement.
pub const MAX_GROUPS: usize = 1024;

fn validate_weight_buffers(
    output_len: usize,
    nibble_len: usize,
    parameter_len: usize,
    input_len: usize,
    rows: usize,
    cols: usize,
    num_tokens: usize,
    group_size: usize,
) {
    assert!(rows > 0, "rows must be non-zero");
    assert!(
        cols > 0 && cols.is_multiple_of(2),
        "cols must be non-zero and even"
    );
    assert!(
        group_size > 0 && group_size.is_multiple_of(2),
        "group_size must be non-zero and even"
    );
    assert!(num_tokens > 0, "num_tokens must be non-zero");
    let num_groups = cols.div_ceil(group_size);
    assert!(
        num_groups <= MAX_GROUPS,
        "quantization group count exceeds kernel capacity"
    );
    let elements = rows.checked_mul(cols).expect("weight dimensions overflow");
    let required_nibbles = elements.div_ceil(2);
    let required_parameters = rows
        .checked_mul(num_groups)
        .and_then(|value| value.checked_mul(4))
        .expect("weight parameter dimensions overflow");
    let required_input = num_tokens
        .checked_mul(cols)
        .expect("input dimensions overflow");
    let required_output = num_tokens
        .checked_mul(rows)
        .expect("output dimensions overflow");
    assert!(nibble_len >= required_nibbles, "weight data is truncated");
    assert!(
        parameter_len >= required_parameters,
        "weight parameters are truncated"
    );
    assert!(input_len >= required_input, "input buffer is too small");
    assert!(output_len >= required_output, "output buffer is too small");
}

fn validate_projection(output: &[f32], projection: &QuantizedLinear<'_>, input: &[f32]) {
    validate_weight_buffers(
        output.len(),
        projection.nibble_data.len(),
        projection.group_params.len(),
        input.len(),
        projection.rows,
        projection.cols,
        1,
        projection.group_size,
    );
}

// ---------------------------------------------------------------------------
// Activations, prepared once per call
// ---------------------------------------------------------------------------

/// Tokens' activations quantized for the AVX2 int8 kernels, token-major: the
/// even/odd column split, and per group the scale and the quantized sum
/// `quantize_input_split` describes.
#[cfg(target_arch = "x86_64")]
struct Int8Activations {
    even: Vec<i8>,
    odd: Vec<i8>,
    scales: Vec<f32>,
    qsums: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
impl Int8Activations {
    /// Quantize `tokens` rows of `cols` values each. Only called after
    /// `has_avx2()` returned true and the buffers were validated.
    fn quantize(input: &[f32], tokens: usize, cols: usize, group_size: usize) -> Self {
        let num_groups = cols.div_ceil(group_size);
        let half = cols / 2;
        let mut act = Self {
            even: vec![0; tokens * half],
            odd: vec![0; tokens * half],
            scales: vec![0.0; tokens * num_groups],
            qsums: vec![0.0; tokens * num_groups],
        };
        for t in 0..tokens {
            quantize_input_split(
                &input[t * cols..(t + 1) * cols],
                &mut act.even[t * half..(t + 1) * half],
                &mut act.odd[t * half..(t + 1) * half],
                &mut act.scales[t * num_groups..(t + 1) * num_groups],
                &mut act.qsums[t * num_groups..(t + 1) * num_groups],
                cols,
                group_size,
                num_groups,
            );
        }
        act
    }
}

/// One token through projections that share its activations: an output slice
/// paired with the projection that fills it.
type MatvecJobs<'o, 'p, 'w> = Vec<(&'o mut [f32], &'p QuantizedLinear<'w>)>;

/// Run [`MatvecJobs`] on whichever kernel this CPU has.
///
/// Serial when the matrices together are under the decode dispatch floor
/// ([`PAR_MIN_BYTES`]), otherwise one rayon job list over row chunks of all of
/// them, so small K/V projections ride along with Q instead of each paying a
/// dispatch of their own.
fn matvec_dispatch(jobs: MatvecJobs<'_, '_, '_>, input: &[f32], cols: usize, group_size: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let act = Int8Activations::quantize(input, 1, cols, group_size);
            matvec_avx2(jobs, &act, cols, group_size);
            return;
        }
    }
    matvec_scalar(jobs, input, cols, group_size);
}

/// Split every job's output into row chunks, remembering each chunk's first
/// row, so one parallel loop can cover several projections.
fn row_chunks<'o, 'p, 'w>(
    jobs: MatvecJobs<'o, 'p, 'w>,
    rows_per_chunk: usize,
) -> Vec<(&'o mut [f32], &'p QuantizedLinear<'w>, usize)> {
    let mut chunks = Vec::new();
    for (output, projection) in jobs {
        for (index, chunk) in output[..projection.rows]
            .chunks_mut(rows_per_chunk)
            .enumerate()
        {
            chunks.push((chunk, projection, index * rows_per_chunk));
        }
    }
    chunks
}

#[cfg(target_arch = "x86_64")]
fn matvec_avx2(
    jobs: MatvecJobs<'_, '_, '_>,
    act: &Int8Activations,
    cols: usize,
    group_size: usize,
) {
    let num_groups = cols.div_ceil(group_size);
    let total_rows: usize = jobs.iter().map(|(_, projection)| projection.rows).sum();
    let run = |output: &mut [f32],
               projection: &QuantizedLinear<'_>,
               start: usize,
               scratch: &mut [f32]| {
        // SAFETY: `act` exists only after `has_avx2()` returned true;
        // `validate_projection` proved the projection holds `rows * cols / 2`
        // code bytes and `rows * num_groups * 4` parameter bytes, `output` is
        // its rows `start..start + output.len()`, `act` holds one token's
        // `cols / 2` split activations and `num_groups` scales and sums, and
        // `scratch` holds `num_groups` values.
        unsafe {
            matvec_chunk_i8(
                output.as_mut_ptr(),
                projection.nibble_data.as_ptr(),
                projection.group_params,
                act.even.as_ptr(),
                act.odd.as_ptr(),
                &act.scales,
                &act.qsums,
                start,
                output.len(),
                cols,
                group_size,
                num_groups,
                scratch,
            );
        }
    };
    if !matvec_should_parallelize(total_rows, cols) {
        let mut scratch = vec![0.0f32; num_groups];
        for (output, projection) in jobs {
            run(&mut output[..projection.rows], projection, 0, &mut scratch);
        }
        return;
    }
    row_chunks(jobs, chunk_rows_for(cols))
        .into_par_iter()
        .for_each_init(
            || vec![0.0f32; num_groups],
            |scratch, (output, projection, start)| run(output, projection, start, scratch),
        );
}

/// The scalar kernels read the f32 input directly, so nothing is quantized;
/// rows are spread across the pool exactly as the AVX2 path spreads them.
fn matvec_scalar(jobs: MatvecJobs<'_, '_, '_>, input: &[f32], cols: usize, group_size: usize) {
    let num_groups = cols.div_ceil(group_size);
    let mut sums = vec![0.0f32; num_groups];
    group_sums_into(input, &mut sums, cols, group_size);
    let total_rows: usize = jobs.iter().map(|(_, projection)| projection.rows).sum();
    let run = |output: &mut [f32], projection: &QuantizedLinear<'_>, start: usize| {
        for (offset, value) in output.iter_mut().enumerate() {
            *value = matvec_row_scalar(
                projection.nibble_data,
                projection.group_params,
                input,
                &sums,
                start + offset,
                cols,
                group_size,
                num_groups,
            );
        }
    };
    if !matvec_should_parallelize(total_rows, cols) {
        for (output, projection) in jobs {
            run(&mut output[..projection.rows], projection, 0);
        }
        return;
    }
    row_chunks(jobs, chunk_rows_for(cols))
        .into_par_iter()
        .for_each(|(output, projection, start)| run(output, projection, start));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// W4A8 matrix–vector product: `output[r] = dot(dequant(weights[r]), input)`.
///
/// On AVX2 hosts the f32 input is quantized to int8 per group internally; the
/// scalar kernels use it as f32. Weights are 4-bit codes with per-group f16
/// scale/zero parameters.
///
/// # Panics
/// Panics if `rows` is zero, `cols` or `group_size` is zero or odd, the group
/// count exceeds [`MAX_GROUPS`], any dimension product overflows, or any
/// buffer is smaller than the dimensions imply.
pub fn w4a8_matvec(
    output: &mut [f32],
    nibble_data: &[u8],
    group_params: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) {
    validate_weight_buffers(
        output.len(),
        nibble_data.len(),
        group_params.len(),
        input.len(),
        rows,
        cols,
        1,
        group_size,
    );
    let projection = QuantizedLinear {
        rows,
        cols,
        group_params,
        nibble_data,
        group_size,
    };
    matvec_dispatch(vec![(output, &projection)], input, cols, group_size);
}

/// Fused Q/K/V projections: the input is quantized once and all three
/// matrices share one parallel dispatch, so the small K/V projections ride
/// along with Q rather than each paying for a dispatch of their own.
///
/// The decode dispatch floor applies to the combined matrix: when Q+K+V packed
/// together stay under `PAR_MIN_BYTES`, one thread walks all three projections
/// itself — the same policy `w4a8_matvec` applies, for the same measured
/// reason (`bw-bench` steps 3-5).
///
/// # Panics
/// Panics if any projection fails the [`w4a8_matvec`] buffer/dimension
/// checks, or if the three projections disagree on `cols` or `group_size`.
pub fn w4a8_fused_qkv(
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    q_proj: &QuantizedLinear<'_>,
    k_proj: &QuantizedLinear<'_>,
    v_proj: &QuantizedLinear<'_>,
    input: &[f32],
) {
    validate_projection(q_out, q_proj, input);
    validate_projection(k_out, k_proj, input);
    validate_projection(v_out, v_proj, input);
    assert_eq!(q_proj.cols, k_proj.cols, "Q/K column mismatch");
    assert_eq!(q_proj.cols, v_proj.cols, "Q/V column mismatch");
    assert_eq!(q_proj.group_size, k_proj.group_size, "Q/K group mismatch");
    assert_eq!(q_proj.group_size, v_proj.group_size, "Q/V group mismatch");
    matvec_dispatch(
        vec![(q_out, q_proj), (k_out, k_proj), (v_out, v_proj)],
        input,
        q_proj.cols,
        q_proj.group_size,
    );
}

/// Fused gate+up projections: one quantization of the input and one parallel
/// dispatch over both matrices. The combined gate+up matrix is held to the
/// same `PAR_MIN_BYTES` decode floor as [`w4a8_matvec`]; under it, one thread
/// walks both projections.
///
/// # Panics
/// Panics if either projection fails the [`w4a8_matvec`] buffer/dimension
/// checks, or if the two projections disagree on `cols` or `group_size`.
pub fn w4a8_fused_gate_up(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    gate_proj: &QuantizedLinear<'_>,
    up_proj: &QuantizedLinear<'_>,
    input: &[f32],
) {
    validate_projection(gate_out, gate_proj, input);
    validate_projection(up_out, up_proj, input);
    assert_eq!(gate_proj.cols, up_proj.cols, "gate/up column mismatch");
    assert_eq!(
        gate_proj.group_size, up_proj.group_size,
        "gate/up group mismatch"
    );
    matvec_dispatch(
        vec![(gate_out, gate_proj), (up_out, up_proj)],
        input,
        gate_proj.cols,
        gate_proj.group_size,
    );
}

/// Benchmark-only A/B switch for the batched GEMM restructure.
///
/// Set `RAI_BENCH_LEGACY_MATMUL=1` to drive a batch through the single-token
/// kernel once per token, the way [`w4a8_matmul`] worked before the
/// weight-stationary rewrite. Both paths compute each token over the same
/// groups in the same order, so this changes speed and nothing else — it
/// exists so before/after can be measured in one binary on one machine
/// instead of across two builds under different background load.
///
/// x86-64 only: the switch selects between two AVX2 batch strategies, and no
/// other target has a second strategy to select. On those targets the batched
/// path is the scalar row loop either way, so the variable is inert and the
/// function is not compiled rather than silently ignored.
#[cfg(target_arch = "x86_64")]
fn legacy_matmul_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RAI_BENCH_LEGACY_MATMUL").is_ok_and(|v| v == "1"))
}

/// Benchmark-only A/B switch for the parallel prefill attention phase.
///
/// Set `RAI_BENCH_SERIAL_ATTN=1` to attend one token at a time, as the batched
/// forward did before the store-then-attend split. Numerically identical.
pub fn serial_batch_attention_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RAI_BENCH_SERIAL_ATTN").is_ok_and(|v| v == "1"))
}

/// Benchmark-only A/B switch for parallel decode attention.
///
/// Set `RAI_BENCH_SERIAL_DECODE_ATTN=1` to run the per-head decode attention
/// loop on one core, as it did before. Numerically identical: query heads do
/// not interact.
pub fn serial_decode_attention_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RAI_BENCH_SERIAL_DECODE_ATTN").is_ok_and(|v| v == "1"))
}

/// Configure the rayon thread pool for optimal inference performance.
/// Call once at program start. Respects the `RAYON_NUM_THREADS` environment
/// variable; defaults to the machine's performance cores plus one (see
/// `topology`), capped at the CPUs this process may use.
pub fn configure_thread_pool() {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(crate::topology::default_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

/// Batched W4A8 matrix multiply: weight-stationary with L2 cache reuse.
///
/// Processes all tokens per weight chunk so weight data (read from DRAM for token 0)
/// stays in L1/L2 cache for tokens 1..B-1. This gives near-1-token bandwidth cost
/// for B tokens of output.
///
/// Layout: input[t * cols + c], output[t * rows + r] (row-major per token).
///
/// # Panics
/// Panics if `num_tokens` is zero or any [`w4a8_matvec`] buffer/dimension
/// check fails for the batched buffer sizes.
pub fn w4a8_matmul(
    output: &mut [f32],
    nibble_data: &[u8],
    group_params: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
    num_tokens: usize,
    group_size: usize,
) {
    validate_weight_buffers(
        output.len(),
        nibble_data.len(),
        group_params.len(),
        input.len(),
        rows,
        cols,
        num_tokens,
        group_size,
    );
    if num_tokens == 1 {
        return w4a8_matvec(
            output,
            nibble_data,
            group_params,
            input,
            rows,
            cols,
            group_size,
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // All tokens are quantized up front into flat token-major buffers
            // (four allocations, not four per token). The group buffers are
            // strided by the model's real `num_groups`, so one cache line
            // usually covers the same group across several tokens — exactly
            // how the weight-stationary kernel reads them.
            let act = Int8Activations::quantize(input, num_tokens, cols, group_size);
            return matmul_avx2(
                output,
                nibble_data,
                group_params,
                &act,
                rows,
                cols,
                num_tokens,
                group_size,
            );
        }
    }

    // Scalar: each token's outputs are one contiguous row of `output`, filled
    // by the same row-parallel matvec a single token uses.
    let projection = QuantizedLinear {
        rows,
        cols,
        group_params,
        nibble_data,
        group_size,
    };
    for (t, token_output) in output[..num_tokens * rows].chunks_mut(rows).enumerate() {
        matvec_scalar(
            vec![(token_output, &projection)],
            &input[t * cols..(t + 1) * cols],
            cols,
            group_size,
        );
    }
}

/// Several projections of one batch of activations, which are quantized once
/// for all of them: the batched forward's Q/K/V and its gate/up. Each output
/// is laid out as [`w4a8_matmul`]'s, and computed exactly as it would be.
///
/// # Panics
/// Panics if any projection fails the [`w4a8_matmul`] checks for
/// `num_tokens` tokens, or the projections disagree on `cols` or
/// `group_size`.
pub fn w4a8_matmul_shared(
    jobs: Vec<(&mut [f32], &QuantizedLinear<'_>)>,
    input: &[f32],
    num_tokens: usize,
) {
    let Some((_, first)) = jobs.first() else {
        return;
    };
    let (cols, group_size) = (first.cols, first.group_size);
    for (output, projection) in &jobs {
        assert_eq!(projection.cols, cols, "projections disagree on cols");
        assert_eq!(
            projection.group_size, group_size,
            "projections disagree on group size"
        );
        validate_weight_buffers(
            output.len(),
            projection.nibble_data.len(),
            projection.group_params.len(),
            input.len(),
            projection.rows,
            cols,
            num_tokens,
            group_size,
        );
    }
    if num_tokens == 1 {
        return matvec_dispatch(jobs, input, cols, group_size);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let act = Int8Activations::quantize(input, num_tokens, cols, group_size);
            for (output, projection) in jobs {
                matmul_avx2(
                    output,
                    projection.nibble_data,
                    projection.group_params,
                    &act,
                    projection.rows,
                    cols,
                    num_tokens,
                    group_size,
                );
            }
            return;
        }
    }
    for (output, projection) in jobs {
        w4a8_matmul(
            output,
            projection.nibble_data,
            projection.group_params,
            input,
            projection.rows,
            cols,
            num_tokens,
            group_size,
        );
    }
}

/// The AVX2 half of [`w4a8_matmul`], on already-quantized activations.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn matmul_avx2(
    output: &mut [f32],
    nibble_data: &[u8],
    group_params: &[u8],
    act: &Int8Activations,
    rows: usize,
    cols: usize,
    num_tokens: usize,
    group_size: usize,
) {
    let num_groups = cols.div_ceil(group_size);
    let half_cols = cols / 2;
    let cr = chunk_rows_for(cols);
    let num_chunks = rows.div_ceil(cr);
    // Every chunk writes its own rows for every token: disjoint, but strided
    // by `rows`, so the output travels as a raw pointer rather than as slices.
    let out_ptr = SendPtr(output.as_mut_ptr());

    if legacy_matmul_enabled() {
        // Benchmark-only: drive the batch through the single-token kernel once
        // per token, as this function did before the weight-stationary rewrite.
        (0..num_chunks).into_par_iter().for_each_init(
            || vec![0.0f32; num_groups],
            |scratch, ci| {
                let start = ci * cr;
                let len = cr.min(rows - start);
                for t in 0..num_tokens {
                    // SAFETY: as in `matvec_avx2`, for token `t`'s slice of
                    // the activations and output row `t`.
                    unsafe {
                        matvec_chunk_i8(
                            out_ptr.ptr().add(t * rows + start),
                            nibble_data.as_ptr(),
                            group_params,
                            act.even.as_ptr().add(t * half_cols),
                            act.odd.as_ptr().add(t * half_cols),
                            &act.scales[t * num_groups..(t + 1) * num_groups],
                            &act.qsums[t * num_groups..(t + 1) * num_groups],
                            start,
                            len,
                            cols,
                            group_size,
                            num_groups,
                            scratch,
                        );
                    }
                }
            },
        );
        return;
    }

    // Weight-stationary parallel dispatch: each thread takes a chunk of rows
    // and drives the whole batch through it, so the weight data it pulls into
    // L1 is unpacked once and reused by every token.
    (0..num_chunks).into_par_iter().for_each_init(
        || vec![0.0f32; 2 * num_groups],
        |scratch, ci| {
            let start = ci * cr;
            let len = cr.min(rows - start);
            // SAFETY: `validate_weight_buffers` proved the weights, the
            // `num_tokens * rows` output and the `num_tokens * cols` input
            // (quantized into `act`) all fit; chunks cover disjoint rows.
            unsafe {
                matmul_chunk_i8(
                    out_ptr.ptr(),
                    nibble_data.as_ptr(),
                    group_params,
                    act.even.as_ptr(),
                    act.odd.as_ptr(),
                    act.scales.as_ptr(),
                    act.qsums.as_ptr(),
                    start,
                    len,
                    rows,
                    cols,
                    group_size,
                    num_groups,
                    num_tokens,
                    scratch,
                );
            }
        },
    );
}

/// Dequantize one embedding row (8-bit codes, per-group f16 scale/zero) into `output`.
///
/// # Panics
/// Panics if `group_size` is zero, `token_id >= vocab_size`, any dimension
/// product overflows, or any buffer is smaller than the dimensions imply.
pub fn embed_lookup(
    output: &mut [f32],
    token_id: usize,
    embed_data: &[u8],
    embed_params: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    group_size: usize,
) {
    assert!(group_size > 0, "group_size must be non-zero");
    assert!(token_id < vocab_size, "token id is out of range");
    assert!(output.len() >= hidden_size, "embedding output is too small");
    let num_groups = hidden_size.div_ceil(group_size);
    assert!(
        embed_data.len()
            >= vocab_size
                .checked_mul(hidden_size)
                .expect("embedding dimensions overflow"),
        "embedding data is truncated"
    );
    assert!(
        embed_params.len()
            >= vocab_size
                .checked_mul(num_groups)
                .and_then(|value| value.checked_mul(4))
                .expect("embedding parameter dimensions overflow"),
        "embedding parameters are truncated"
    );
    let row_data_start = token_id * hidden_size;
    let row_param_start = token_id * num_groups * 4;

    for g in 0..num_groups {
        let param_off = row_param_start + g * 4;
        let scale = read_f16_le(embed_params, param_off);
        let zero = read_f16_le(embed_params, param_off + 2);

        let col_start = g * group_size;
        let col_end = ((g + 1) * group_size).min(hidden_size);

        for c in col_start..col_end {
            let code = embed_data[row_data_start + c] as f32;
            output[c] = code * scale + zero;
        }
    }
}

/// Project a hidden state through the (tied) 8-bit embedding matrix to logits.
///
/// On AVX2 hosts the hidden state is quantized to int8 (range ±63) per group
/// for the integer PMADDUBSW path; the scalar fallback uses the f32 hidden
/// directly. Both spread the vocabulary across the pool in [`LM_CHUNK`]-row
/// chunks.
///
/// # Panics
/// Panics if `group_size` is zero, `hidden_size` is zero or odd, the group
/// count exceeds [`MAX_GROUPS`], any dimension product overflows, or any
/// buffer is smaller than the dimensions imply.
pub fn tied_lm_head(
    logits: &mut [f32],
    hidden: &[f32],
    embed_data: &[u8],
    embed_params: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    group_size: usize,
) {
    lm_head_dispatch(
        logits,
        hidden,
        embed_data,
        embed_params,
        vocab_size,
        hidden_size,
        group_size,
        1,
    );
}

/// [`tied_lm_head`] for `num_tokens` hidden states at once, token-major:
/// `hidden[t * hidden_size..]` in, `logits[t * vocab_size..]` out.
///
/// The table is the largest single read in a small model (a 151,936-token
/// vocabulary at hidden size 896 is 136 MB), and verifying several drafted
/// tokens, or scoring a chunk of text, used to stream it once per token. Here
/// each row is read once per batch and dotted with every token while it is in
/// L1. Every logit is computed exactly as [`tied_lm_head`] computes it.
///
/// # Panics
/// The [`tied_lm_head`] conditions, for `num_tokens` tokens, or if
/// `num_tokens` is zero.
pub fn tied_lm_head_batch(
    logits: &mut [f32],
    hidden: &[f32],
    embed_data: &[u8],
    embed_params: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    group_size: usize,
    num_tokens: usize,
) {
    lm_head_dispatch(
        logits,
        hidden,
        embed_data,
        embed_params,
        vocab_size,
        hidden_size,
        group_size,
        num_tokens,
    );
}

#[allow(clippy::too_many_arguments)]
fn lm_head_dispatch(
    logits: &mut [f32],
    hidden: &[f32],
    embed_data: &[u8],
    embed_params: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    group_size: usize,
    num_tokens: usize,
) {
    assert!(group_size > 0, "group_size must be non-zero");
    assert!(num_tokens > 0, "num_tokens must be non-zero");
    assert!(
        hidden_size > 0 && hidden_size.is_multiple_of(2),
        "hidden_size must be non-zero and even"
    );
    let hidden_len = num_tokens
        .checked_mul(hidden_size)
        .expect("hidden dimensions overflow");
    let logits_len = num_tokens
        .checked_mul(vocab_size)
        .expect("logit dimensions overflow");
    assert!(hidden.len() >= hidden_len, "hidden buffer is too small");
    assert!(logits.len() >= logits_len, "logit buffer is too small");
    let num_groups = hidden_size.div_ceil(group_size);
    assert!(
        num_groups <= MAX_GROUPS,
        "quantization groups exceed the kernel maximum"
    );
    assert!(
        embed_data.len()
            >= vocab_size
                .checked_mul(hidden_size)
                .expect("embedding dimensions overflow"),
        "embedding data is truncated"
    );
    assert!(
        embed_params.len()
            >= vocab_size
                .checked_mul(num_groups)
                .and_then(|value| value.checked_mul(4))
                .expect("embedding parameter dimensions overflow"),
        "embedding parameters are truncated"
    );
    let logits = &mut logits[..logits_len];
    let hidden = &hidden[..hidden_len];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let quantized = Int8Hidden::quantize(hidden, num_tokens, hidden_size, group_size);
            // Each chunk writes its own vocabulary rows for every token:
            // disjoint, but strided by `vocab_size` across tokens.
            let out = SendPtr(logits.as_mut_ptr());
            (0..vocab_size.div_ceil(LM_CHUNK))
                .into_par_iter()
                .for_each_init(
                    || vec![0.0f32; 2 * num_groups],
                    |scratch, ci| {
                        let start = ci * LM_CHUNK;
                        // SAFETY: `has_avx2()` returned true; the asserts above
                        // proved the table holds `vocab_size` rows of codes and
                        // parameters and the logits `num_tokens * vocab_size`
                        // values; chunks cover disjoint vocabulary rows.
                        unsafe {
                            lm_head_chunk_i8(
                                out.ptr(),
                                vocab_size,
                                embed_data.as_ptr(),
                                embed_params,
                                &quantized,
                                start,
                                LM_CHUNK.min(vocab_size - start),
                                hidden_size,
                                group_size,
                                num_groups,
                                scratch,
                            );
                        }
                    },
                );
            return;
        }
    }

    for (token_logits, token_hidden) in logits
        .chunks_exact_mut(vocab_size)
        .zip(hidden.chunks_exact(hidden_size))
    {
        let mut sums = vec![0.0f32; num_groups];
        group_sums_into(token_hidden, &mut sums, hidden_size, group_size);
        token_logits
            .par_chunks_mut(LM_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                for (offset, logit) in chunk.iter_mut().enumerate() {
                    *logit = lm_head_row_scalar(
                        embed_data,
                        embed_params,
                        token_hidden,
                        &sums,
                        ci * LM_CHUNK + offset,
                        hidden_size,
                        group_size,
                        num_groups,
                    );
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_w4a8_matvec_identity_like() {
        let rows = 2;
        let cols = 4;
        let group_size = 4;
        let num_groups = 1;

        let scale_f16 = f16::from_f32(1.0);
        let zero_f16 = f16::from_f32(0.0);
        let mut group_params = vec![0u8; rows * num_groups * 4];
        for r in 0..rows {
            let off = r * 4;
            group_params[off..off + 2].copy_from_slice(&scale_f16.to_le_bytes());
            group_params[off + 2..off + 4].copy_from_slice(&zero_f16.to_le_bytes());
        }

        let nibble_data = vec![0x11u8; rows * cols / 2];
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];

        w4a8_matvec(
            &mut output,
            &nibble_data,
            &group_params,
            &input,
            rows,
            cols,
            group_size,
        );

        assert!((output[0] - 4.0).abs() < 0.05, "got {}", output[0]);
        assert!((output[1] - 4.0).abs() < 0.05, "got {}", output[1]);
    }

    #[test]
    fn test_w4a8_matvec_with_scale_zero() {
        let rows = 1;
        let cols = 2;
        let group_size = 2;

        let scale_f16 = f16::from_f32(0.5);
        let zero_f16 = f16::from_f32(-1.0);
        let mut group_params = vec![0u8; 4];
        group_params[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        group_params[2..4].copy_from_slice(&zero_f16.to_le_bytes());

        let nibble_data = vec![0x55u8];
        let input = vec![2.0f32, 3.0];
        let mut output = vec![0.0f32; 1];

        w4a8_matvec(
            &mut output,
            &nibble_data,
            &group_params,
            &input,
            rows,
            cols,
            group_size,
        );

        assert!((output[0] - 7.5).abs() < 0.05, "got {}", output[0]);
    }

    #[test]
    fn test_w4a8_matmul_matches_matvec() {
        let rows = 4;
        let cols = 8;
        let num_tokens = 3;
        let group_size = 4;
        let num_groups = 2;

        let scale_vals = [0.1f32, 0.2, 0.15, 0.25, 0.12, 0.18, 0.22, 0.14];
        let zero_vals = [-0.5f32, -0.3, -0.4, -0.6, -0.35, -0.45, -0.55, -0.25];
        let mut group_params = vec![0u8; rows * num_groups * 4];
        for r in 0..rows {
            for g in 0..num_groups {
                let idx = r * num_groups + g;
                let off = idx * 4;
                let s = f16::from_f32(scale_vals[idx % scale_vals.len()]);
                let z = f16::from_f32(zero_vals[idx % zero_vals.len()]);
                group_params[off..off + 2].copy_from_slice(&s.to_le_bytes());
                group_params[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
            }
        }

        let nibble_data: Vec<u8> = (0..rows * cols / 2)
            .map(|i| ((i as u8 * 7 + 3) % 16) | (((i as u8 * 11 + 5) % 16) << 4))
            .collect();
        let input: Vec<f32> = (0..num_tokens * cols)
            .map(|i| i as f32 * 0.1 - 1.0)
            .collect();

        let mut output_mm = vec![0.0f32; num_tokens * rows];
        w4a8_matmul(
            &mut output_mm,
            &nibble_data,
            &group_params,
            &input,
            rows,
            cols,
            num_tokens,
            group_size,
        );

        for t in 0..num_tokens {
            let mut output_mv = vec![0.0f32; rows];
            w4a8_matvec(
                &mut output_mv,
                &nibble_data,
                &group_params,
                &input[t * cols..(t + 1) * cols],
                rows,
                cols,
                group_size,
            );
            for r in 0..rows {
                let diff = (output_mm[t * rows + r] - output_mv[r]).abs();
                assert!(
                    diff < 1e-3,
                    "token {t} row {r}: matmul={} vs matvec={}, diff={diff}",
                    output_mm[t * rows + r],
                    output_mv[r]
                );
            }
        }
    }

    /// Build a deterministic 4-bit matrix and its group parameters.
    ///
    /// Values are spread across the whole nibble range and the scales differ
    /// per row and per group, so a kernel that silently used group 0's scale
    /// for every group — the exact failure a scratch-buffer mistake produces —
    /// cannot pass by accident.
    fn synthetic_weights(rows: usize, cols: usize, group_size: usize) -> (Vec<u8>, Vec<u8>) {
        let num_groups = cols.div_ceil(group_size);
        let mut group_params = vec![0u8; rows * num_groups * 4];
        for r in 0..rows {
            for g in 0..num_groups {
                let off = (r * num_groups + g) * 4;
                let scale = f16::from_f32(0.05 + 0.01 * ((r + g) % 7) as f32);
                let zero = f16::from_f32(-0.4 + 0.05 * ((r * 3 + g) % 9) as f32);
                group_params[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                group_params[off + 2..off + 4].copy_from_slice(&zero.to_le_bytes());
            }
        }

        let mut nibble_data = vec![0u8; rows * cols / 2];
        for (i, byte) in nibble_data.iter_mut().enumerate() {
            let lo = (i * 7 + 3) % 16;
            let hi = (i * 11 + 5) % 16;
            *byte = (lo | (hi << 4)) as u8;
        }
        (nibble_data, group_params)
    }

    fn synthetic_input(num_tokens: usize, cols: usize) -> Vec<f32> {
        (0..num_tokens * cols)
            .map(|i| {
                let x = (i % 97) as f32 / 97.0 - 0.5;
                x * (1.0 + (i % 13) as f32 / 13.0)
            })
            .collect()
    }

    /// The batch kernel and the single-token kernel must agree, at every group
    /// size a model can be quantized at.
    ///
    /// `w4a8_matmul` and `w4a8_matvec` share the input quantization but not the
    /// inner loop, so this compares two independent implementations rather than
    /// a function against itself. The tolerance is for f32 summation order,
    /// which legitimately differs between the two: the batch kernel accumulates
    /// per group across tokens, the single-token kernel accumulates per row.
    #[test]
    fn batched_and_single_token_gemm_agree_at_every_group_size() {
        let rows = 96;
        let cols = 512;
        let num_tokens = 5;
        let input = synthetic_input(num_tokens, cols);

        // 512 divides evenly by 32/64/128/256; 96 and 160 leave a short tail
        // group, which the kernels handle on a separate path.
        for &group_size in &[32usize, 64, 96, 128, 160, 256] {
            let (nibble_data, group_params) = synthetic_weights(rows, cols, group_size);

            let mut batched = vec![0.0f32; num_tokens * rows];
            w4a8_matmul(
                &mut batched,
                &nibble_data,
                &group_params,
                &input,
                rows,
                cols,
                num_tokens,
                group_size,
            );

            for t in 0..num_tokens {
                let mut single = vec![0.0f32; rows];
                w4a8_matvec(
                    &mut single,
                    &nibble_data,
                    &group_params,
                    &input[t * cols..(t + 1) * cols],
                    rows,
                    cols,
                    group_size,
                );
                for row in 0..rows {
                    let a = batched[t * rows + row];
                    let b = single[row];
                    assert!(
                        (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                        "group_size {group_size}, token {t}, row {row}: \
                         batched {a} vs single-token {b}"
                    );
                }
            }
        }
    }

    /// The fused projections must match running each projection on its own.
    ///
    /// The fused paths have their own chunk scheduler and their own scratch,
    /// and they are what decode actually calls; a group-size bug that only
    /// appeared there would be invisible to the test above.
    #[test]
    fn fused_projections_match_separate_ones_at_every_group_size() {
        let cols = 256;
        let q_rows = 64;
        let k_rows = 16;
        let v_rows = 16;
        let input = synthetic_input(1, cols);

        for &group_size in &[32usize, 64, 128, 192] {
            let (q_nib, q_par) = synthetic_weights(q_rows, cols, group_size);
            let (k_nib, k_par) = synthetic_weights(k_rows, cols, group_size);
            let (v_nib, v_par) = synthetic_weights(v_rows, cols, group_size);

            let mut q_fused = vec![0.0f32; q_rows];
            let mut k_fused = vec![0.0f32; k_rows];
            let mut v_fused = vec![0.0f32; v_rows];
            let q = crate::format::QuantizedLinear {
                rows: q_rows,
                cols,
                group_params: &q_par,
                nibble_data: &q_nib,
                group_size,
            };
            let k = crate::format::QuantizedLinear {
                rows: k_rows,
                cols,
                group_params: &k_par,
                nibble_data: &k_nib,
                group_size,
            };
            let v = crate::format::QuantizedLinear {
                rows: v_rows,
                cols,
                group_params: &v_par,
                nibble_data: &v_nib,
                group_size,
            };
            w4a8_fused_qkv(&mut q_fused, &mut k_fused, &mut v_fused, &q, &k, &v, &input);

            for (fused, nib, par, rows, name) in [
                (&q_fused, &q_nib, &q_par, q_rows, "q"),
                (&k_fused, &k_nib, &k_par, k_rows, "k"),
                (&v_fused, &v_nib, &v_par, v_rows, "v"),
            ] {
                let mut separate = vec![0.0f32; rows];
                w4a8_matvec(&mut separate, nib, par, &input, rows, cols, group_size);
                for row in 0..rows {
                    let a = fused[row];
                    let b = separate[row];
                    assert!(
                        (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                        "group_size {group_size}, {name}[{row}]: fused {a} vs separate {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_embed_lookup() {
        let vocab = 3;
        let hidden = 4;
        let gs = 2;
        let num_groups = 2;

        let mut embed_params = vec![0u8; vocab * num_groups * 4];
        let off = num_groups * 4;
        embed_params[off..off + 2].copy_from_slice(&f16::from_f32(0.1).to_le_bytes());
        embed_params[off + 2..off + 4].copy_from_slice(&f16::from_f32(-0.5).to_le_bytes());
        let off2 = off + 4;
        embed_params[off2..off2 + 2].copy_from_slice(&f16::from_f32(0.2).to_le_bytes());
        embed_params[off2 + 2..off2 + 4].copy_from_slice(&f16::from_f32(-1.0).to_le_bytes());

        let mut embed_data = vec![0u8; vocab * hidden];
        embed_data[hidden] = 10;
        embed_data[hidden + 1] = 20;
        embed_data[hidden + 2] = 30;
        embed_data[hidden + 3] = 40;

        let mut output = vec![0.0f32; hidden];
        embed_lookup(
            &mut output,
            1,
            &embed_data,
            &embed_params,
            vocab,
            hidden,
            gs,
        );

        assert!((output[0] - 0.5).abs() < 0.05, "got {}", output[0]);
        assert!((output[1] - 1.5).abs() < 0.05, "got {}", output[1]);
        assert!((output[2] - 5.0).abs() < 0.05, "got {}", output[2]);
        assert!((output[3] - 7.0).abs() < 0.05, "got {}", output[3]);
    }

    #[test]
    fn test_w4a8_matvec_large_factored() {
        let rows = 16;
        let cols = 128;
        let group_size = 128;

        let scale = 0.1f32;
        let zero = -0.8f32;
        let scale_f16 = f16::from_f32(scale);
        let zero_f16 = f16::from_f32(zero);

        let mut group_params = vec![0u8; rows * 4];
        for r in 0..rows {
            let off = r * 4;
            group_params[off..off + 2].copy_from_slice(&scale_f16.to_le_bytes());
            group_params[off + 2..off + 4].copy_from_slice(&zero_f16.to_le_bytes());
        }

        let nibble_data: Vec<u8> = (0..rows * cols / 2)
            .map(|i| {
                let lo = ((i * 7 + 3) % 16) as u8;
                let hi = ((i * 11 + 5) % 16) as u8;
                lo | (hi << 4)
            })
            .collect();

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let mut output = vec![0.0f32; rows];

        w4a8_matvec(
            &mut output,
            &nibble_data,
            &group_params,
            &input,
            rows,
            cols,
            group_size,
        );

        let scale_rt = scale_f16.to_f32();
        let zero_rt = zero_f16.to_f32();
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                let byte_idx = r * (cols / 2) + c / 2;
                let nibble = if c % 2 == 0 {
                    nibble_data[byte_idx] & 0x0F
                } else {
                    nibble_data[byte_idx] >> 4
                };
                let w = nibble as f32 * scale_rt + zero_rt;
                expected += w * input[c];
            }
            let diff = (output[r] - expected).abs();
            assert!(
                diff < 0.01,
                "row {r}: got {} expected {}, diff={diff}",
                output[r],
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "weight data is truncated")]
    fn truncated_weights_are_rejected_before_kernel_dispatch() {
        let mut output = vec![0.0; 4];
        let input = vec![1.0; 8];
        let params = vec![0_u8; 16];
        w4a8_matvec(&mut output, &[], &params, &input, 4, 8, 8);
    }

    #[test]
    #[should_panic(expected = "cols must be non-zero and even")]
    fn odd_dimensions_are_rejected_before_kernel_dispatch() {
        let mut output = vec![0.0; 1];
        let input = vec![1.0; 3];
        let params = vec![0_u8; 4];
        let weights = vec![0_u8; 2];
        w4a8_matvec(&mut output, &weights, &params, &input, 1, 3, 3);
    }

    // -----------------------------------------------------------------------
    // Parallel-path reference tests. The shapes here clear both dispatch
    // gates -- row count and packed-weight size -- so the rayon branch is the
    // one under test on AVX2 hosts; on non-AVX2 hosts the same assertions
    // cover the scalar fallback.
    // -----------------------------------------------------------------------

    /// Uniform per-row/per-group f16 (scale, zero) parameters.
    fn pack_group_params(rows: usize, num_groups: usize, scale: f32, zero: f32) -> Vec<u8> {
        let s = f16::from_f32(scale);
        let z = f16::from_f32(zero);
        let mut params = vec![0u8; rows * num_groups * 4];
        for idx in 0..rows * num_groups {
            let off = idx * 4;
            params[off..off + 2].copy_from_slice(&s.to_le_bytes());
            params[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
        }
        params
    }

    /// Deterministic packed-nibble pattern (same style as the tests above);
    /// `seed` decorrelates multiple weight matrices.
    fn pack_nibbles(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        (0..rows * cols / 2)
            .map(|i| {
                let lo = ((i * 7 + 3 + seed) % 16) as u8;
                let hi = ((i * 11 + 5 + seed) % 16) as u8;
                lo | (hi << 4)
            })
            .collect()
    }

    /// Input whose per-group int8 quantization is exact: every value is
    /// `t / limit` for integer `t` with `|t| <= limit`, and each group
    /// attains absmax 1.0, so the quantization scale is exactly `1/limit`
    /// and `round(v / scale)` recovers `t`. This keeps the integer kernel
    /// path bit-comparable with an f64 dequantized reference.
    fn exact_quant_input(cols: usize, group_size: usize, limit: i32) -> Vec<f32> {
        (0..cols)
            .map(|i| {
                let j = i % group_size;
                let t: i32 = if j == 0 {
                    limit
                } else {
                    ((j * 37) % (2 * limit as usize + 1)) as i32 - limit
                };
                t as f32 / limit as f32
            })
            .collect()
    }

    /// f64 reference: `out[r] = sum_c (code * scale + zero) * input[c]` with
    /// f16-roundtripped scale/zero, matching the kernels' factored formula.
    fn reference_matvec(
        nibbles: &[u8],
        scale: f32,
        zero: f32,
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let s = f64::from(f16::from_f32(scale).to_f32());
        let z = f64::from(f16::from_f32(zero).to_f32());
        (0..rows)
            .map(|r| {
                let mut acc = 0.0f64;
                for c in 0..cols {
                    let byte = nibbles[r * cols / 2 + c / 2];
                    let code = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    acc += (f64::from(code) * s + z) * f64::from(input[c]);
                }
                acc as f32
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= allowed,
                "{what}[{i}]: got {a}, expected {e} (|diff| {} > {allowed})",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn matvec_parallel_path_matches_dequantized_reference() {
        let rows = 4288; // parallel branch on AVX2 hosts
        let cols = 256;
        let group_size = 64;
        assert!(matvec_should_parallelize(rows, cols));

        let nibbles = pack_nibbles(rows, cols, 0);
        let params = pack_group_params(rows, cols / group_size, 0.5, -4.0);
        let input = exact_quant_input(cols, group_size, 127);
        let mut output = vec![0.0f32; rows];

        w4a8_matvec(
            &mut output,
            &nibbles,
            &params,
            &input,
            rows,
            cols,
            group_size,
        );

        let expected = reference_matvec(&nibbles, 0.5, -4.0, &input, rows, cols);
        assert_close(&output, &expected, 5e-3, "parallel matvec");
    }

    #[test]
    fn fused_qkv_parallel_path_matches_per_matvec() {
        let q_rows = 4288;
        let kv_rows = 64; // GQA-style narrow K/V
        let cols = 256;
        let group_size = 64;
        assert!(matvec_should_parallelize(q_rows, cols));

        let q_nib = pack_nibbles(q_rows, cols, 0);
        let k_nib = pack_nibbles(kv_rows, cols, 5);
        let v_nib = pack_nibbles(kv_rows, cols, 9);
        let q_par = pack_group_params(q_rows, cols / group_size, 0.25, -2.0);
        let k_par = pack_group_params(kv_rows, cols / group_size, 0.5, -4.0);
        let v_par = pack_group_params(kv_rows, cols / group_size, 0.125, -1.0);
        let input = exact_quant_input(cols, group_size, 127);

        let q_proj = QuantizedLinear {
            rows: q_rows,
            cols,
            group_params: &q_par,
            nibble_data: &q_nib,
            group_size,
        };
        let k_proj = QuantizedLinear {
            rows: kv_rows,
            cols,
            group_params: &k_par,
            nibble_data: &k_nib,
            group_size,
        };
        let v_proj = QuantizedLinear {
            rows: kv_rows,
            cols,
            group_params: &v_par,
            nibble_data: &v_nib,
            group_size,
        };

        let mut q_out = vec![0.0f32; q_rows];
        let mut k_out = vec![0.0f32; kv_rows];
        let mut v_out = vec![0.0f32; kv_rows];
        w4a8_fused_qkv(
            &mut q_out, &mut k_out, &mut v_out, &q_proj, &k_proj, &v_proj, &input,
        );

        // Per-projection reference through the single-matrix entry point.
        let mut q_ref = vec![0.0f32; q_rows];
        let mut k_ref = vec![0.0f32; kv_rows];
        let mut v_ref = vec![0.0f32; kv_rows];
        w4a8_matvec(&mut q_ref, &q_nib, &q_par, &input, q_rows, cols, group_size);
        w4a8_matvec(
            &mut k_ref, &k_nib, &k_par, &input, kv_rows, cols, group_size,
        );
        w4a8_matvec(
            &mut v_ref, &v_nib, &v_par, &input, kv_rows, cols, group_size,
        );
        assert_close(&q_out, &q_ref, 1e-4, "fused Q vs matvec");
        assert_close(&k_out, &k_ref, 1e-4, "fused K vs matvec");
        assert_close(&v_out, &v_ref, 1e-4, "fused V vs matvec");

        // And against the dequantized f64 reference.
        let q_expected = reference_matvec(&q_nib, 0.25, -2.0, &input, q_rows, cols);
        assert_close(&q_out, &q_expected, 5e-3, "fused Q vs reference");
    }

    #[test]
    fn fused_qkv_decode_size_stays_serial_and_matches() {
        // SmolLM-class shape: Q+K+V packed together is 276 KB, under the
        // dispatch floor, so the fused path must walk the rows on one thread.
        let q_rows = 576;
        let kv_rows = 192;
        let cols = 576;
        let group_size = 64;
        assert!(!matvec_should_parallelize(q_rows + 2 * kv_rows, cols));

        let q_nib = pack_nibbles(q_rows, cols, 3);
        let k_nib = pack_nibbles(kv_rows, cols, 7);
        let v_nib = pack_nibbles(kv_rows, cols, 11);
        let q_par = pack_group_params(q_rows, cols / group_size, 0.25, -2.0);
        let k_par = pack_group_params(kv_rows, cols / group_size, 0.5, -4.0);
        let v_par = pack_group_params(kv_rows, cols / group_size, 0.125, -1.0);
        let input = exact_quant_input(cols, group_size, 127);

        let q_proj = QuantizedLinear {
            rows: q_rows,
            cols,
            group_params: &q_par,
            nibble_data: &q_nib,
            group_size,
        };
        let k_proj = QuantizedLinear {
            rows: kv_rows,
            cols,
            group_params: &k_par,
            nibble_data: &k_nib,
            group_size,
        };
        let v_proj = QuantizedLinear {
            rows: kv_rows,
            cols,
            group_params: &v_par,
            nibble_data: &v_nib,
            group_size,
        };

        let mut q_out = vec![0.0f32; q_rows];
        let mut k_out = vec![0.0f32; kv_rows];
        let mut v_out = vec![0.0f32; kv_rows];
        w4a8_fused_qkv(
            &mut q_out, &mut k_out, &mut v_out, &q_proj, &k_proj, &v_proj, &input,
        );

        let mut q_ref = vec![0.0f32; q_rows];
        let mut k_ref = vec![0.0f32; kv_rows];
        let mut v_ref = vec![0.0f32; kv_rows];
        w4a8_matvec(&mut q_ref, &q_nib, &q_par, &input, q_rows, cols, group_size);
        w4a8_matvec(
            &mut k_ref, &k_nib, &k_par, &input, kv_rows, cols, group_size,
        );
        w4a8_matvec(
            &mut v_ref, &v_nib, &v_par, &input, kv_rows, cols, group_size,
        );
        assert_close(&q_out, &q_ref, 1e-4, "serial fused Q vs matvec");
        assert_close(&k_out, &k_ref, 1e-4, "serial fused K vs matvec");
        assert_close(&v_out, &v_ref, 1e-4, "serial fused V vs matvec");

        let q_expected = reference_matvec(&q_nib, 0.25, -2.0, &input, q_rows, cols);
        assert_close(&q_out, &q_expected, 5e-3, "serial fused Q vs reference");
    }

    #[test]
    fn fused_gate_up_parallel_path_matches_per_matvec() {
        let rows = 4288;
        let cols = 256;
        let group_size = 64;
        assert!(matvec_should_parallelize(rows, cols));

        let gate_nib = pack_nibbles(rows, cols, 1);
        let up_nib = pack_nibbles(rows, cols, 7);
        let gate_par = pack_group_params(rows, cols / group_size, 0.25, -2.0);
        let up_par = pack_group_params(rows, cols / group_size, 0.5, -3.0);
        let input = exact_quant_input(cols, group_size, 127);

        let gate_proj = QuantizedLinear {
            rows,
            cols,
            group_params: &gate_par,
            nibble_data: &gate_nib,
            group_size,
        };
        let up_proj = QuantizedLinear {
            rows,
            cols,
            group_params: &up_par,
            nibble_data: &up_nib,
            group_size,
        };

        let mut gate_out = vec![0.0f32; rows];
        let mut up_out = vec![0.0f32; rows];
        w4a8_fused_gate_up(&mut gate_out, &mut up_out, &gate_proj, &up_proj, &input);

        let mut gate_ref = vec![0.0f32; rows];
        let mut up_ref = vec![0.0f32; rows];
        w4a8_matvec(
            &mut gate_ref,
            &gate_nib,
            &gate_par,
            &input,
            rows,
            cols,
            group_size,
        );
        w4a8_matvec(
            &mut up_ref,
            &up_nib,
            &up_par,
            &input,
            rows,
            cols,
            group_size,
        );
        assert_close(&gate_out, &gate_ref, 1e-4, "fused gate vs matvec");
        assert_close(&up_out, &up_ref, 1e-4, "fused up vs matvec");

        let gate_expected = reference_matvec(&gate_nib, 0.25, -2.0, &input, rows, cols);
        assert_close(&gate_out, &gate_expected, 5e-3, "fused gate vs reference");
    }

    #[test]
    fn fused_gate_up_decode_size_stays_serial_and_matches() {
        // 331 KB combined: under the dispatch floor, serial branch.
        let rows = 576;
        let cols = 576;
        let group_size = 64;
        assert!(!matvec_should_parallelize(2 * rows, cols));

        let gate_nib = pack_nibbles(rows, cols, 2);
        let up_nib = pack_nibbles(rows, cols, 13);
        let gate_par = pack_group_params(rows, cols / group_size, 0.25, -2.0);
        let up_par = pack_group_params(rows, cols / group_size, 0.5, -3.0);
        let input = exact_quant_input(cols, group_size, 127);

        let gate_proj = QuantizedLinear {
            rows,
            cols,
            group_params: &gate_par,
            nibble_data: &gate_nib,
            group_size,
        };
        let up_proj = QuantizedLinear {
            rows,
            cols,
            group_params: &up_par,
            nibble_data: &up_nib,
            group_size,
        };

        let mut gate_out = vec![0.0f32; rows];
        let mut up_out = vec![0.0f32; rows];
        w4a8_fused_gate_up(&mut gate_out, &mut up_out, &gate_proj, &up_proj, &input);

        let mut gate_ref = vec![0.0f32; rows];
        let mut up_ref = vec![0.0f32; rows];
        w4a8_matvec(
            &mut gate_ref,
            &gate_nib,
            &gate_par,
            &input,
            rows,
            cols,
            group_size,
        );
        w4a8_matvec(
            &mut up_ref,
            &up_nib,
            &up_par,
            &input,
            rows,
            cols,
            group_size,
        );
        assert_close(&gate_out, &gate_ref, 1e-4, "serial fused gate vs matvec");
        assert_close(&up_out, &up_ref, 1e-4, "serial fused up vs matvec");

        let gate_expected = reference_matvec(&gate_nib, 0.25, -2.0, &input, rows, cols);
        assert_close(
            &gate_out,
            &gate_expected,
            5e-3,
            "serial fused gate vs reference",
        );
    }

    #[test]
    fn tied_lm_head_parallel_path_matches_scalar_reference() {
        // vocab > 2 * LM_CHUNK → at least three parallel chunks, one partial.
        let vocab = 1200;
        let hidden_size = 256;
        let group_size = 64;
        assert!(vocab > 2 * LM_CHUNK);

        let embed_data: Vec<u8> = (0..vocab * hidden_size)
            .map(|i| ((i * 31 + 7) % 256) as u8)
            .collect();
        let embed_params = pack_group_params(vocab, hidden_size / group_size, 0.03125, -4.0);
        // limit 63 matches the LM head's i8 quantization range, so the
        // AVX2 integer path is exact for this hidden vector.
        let hidden = exact_quant_input(hidden_size, group_size, 63);

        let mut logits = vec![0.0f32; vocab];
        tied_lm_head(
            &mut logits,
            &hidden,
            &embed_data,
            &embed_params,
            vocab,
            hidden_size,
            group_size,
        );

        let s = f64::from(f16::from_f32(0.03125).to_f32());
        let z = f64::from(f16::from_f32(-4.0).to_f32());
        let expected: Vec<f32> = (0..vocab)
            .map(|v| {
                let mut acc = 0.0f64;
                for c in 0..hidden_size {
                    let code = embed_data[v * hidden_size + c];
                    acc += (f64::from(code) * s + z) * f64::from(hidden[c]);
                }
                acc as f32
            })
            .collect();
        assert_close(&logits, &expected, 5e-3, "tied_lm_head vs reference");
    }

    // -----------------------------------------------------------------------
    // Activation quantization must not inflate the error
    // -----------------------------------------------------------------------

    /// A deterministic, non-repeating sequence in [-1, 1): activations that do
    /// NOT quantize exactly, unlike `exact_quant_input`, so the quantization
    /// error is real and its treatment is what the tests below measure.
    fn messy_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    /// Weights quantized like the converter does it: per-group min/max codes,
    /// so each group's zero point is its (negative) minimum — the layout under
    /// which an inconsistent zero term costs the most.
    fn min_max_weights(
        rows: usize,
        cols: usize,
        group_size: usize,
        seed: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let num_groups = cols.div_ceil(group_size);
        let mut nibbles = vec![0u8; rows * cols / 2];
        let mut params = vec![0u8; rows * num_groups * 4];
        let codes = messy_values(rows * cols, seed);
        for r in 0..rows {
            for g in 0..num_groups {
                let scale = f16::from_f32(0.02 + 0.001 * ((r + g) % 5) as f32);
                let zero = f16::from_f32(-0.15 - 0.01 * ((r * 3 + g) % 4) as f32);
                let off = (r * num_groups + g) * 4;
                params[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                params[off + 2..off + 4].copy_from_slice(&zero.to_le_bytes());
            }
            for c in 0..cols {
                let code = ((codes[r * cols + c] + 1.0) * 7.999) as u8 & 0x0F;
                let byte = &mut nibbles[(r * cols + c) / 2];
                if c % 2 == 0 {
                    *byte |= code;
                } else {
                    *byte |= code << 4;
                }
            }
        }
        (nibbles, params)
    }

    /// `dequant(weights) · activations` in f64, with the activations first put
    /// through exactly the int8 rounding the AVX2 kernels apply.
    fn reference_over_quantized_input(
        nibbles: &[u8],
        params: &[u8],
        input: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let num_groups = cols.div_ceil(group_size);
        let mut quantized = vec![0.0f64; cols];
        for g in 0..num_groups {
            let (start, end) = (g * group_size, ((g + 1) * group_size).min(cols));
            let absmax = input[start..end].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if absmax > 1e-10 { absmax / 127.0 } else { 1.0 };
            let inv = 1.0 / scale;
            for i in start..end {
                let q = (input[i] * inv).round_ties_even().clamp(-128.0, 127.0);
                quantized[i] = q as f64 * scale as f64;
            }
        }
        (0..rows)
            .map(|r| {
                let mut acc = 0.0f64;
                for c in 0..cols {
                    let off = (r * num_groups + c / group_size) * 4;
                    let scale = read_f16_le(params, off) as f64;
                    let zero = read_f16_le(params, off + 2) as f64;
                    let byte = nibbles[(r * cols + c) / 2];
                    let code = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 } as f64;
                    acc += (scale * code + zero) * quantized[c];
                }
                acc as f32
            })
            .collect()
    }

    /// The AVX2 kernels compute exactly `Σ w·x̂`: the zero term must see the
    /// same quantized activations as the code term. With the exact f32 sum in
    /// the zero term instead, the result is off by `zero · Σ(x − x̂)` per
    /// group, two orders of magnitude outside this tolerance.
    #[test]
    fn int8_kernels_compute_the_product_of_the_quantized_activations() {
        if !has_avx2() {
            return;
        }
        let (rows, cols, group_size, tokens) = (96, 384, 64, 5);
        let (nibbles, params) = min_max_weights(rows, cols, group_size, 7);
        let input = messy_values(tokens * cols, 11);

        let mut single = vec![0.0f32; rows];
        w4a8_matvec(
            &mut single,
            &nibbles,
            &params,
            &input[..cols],
            rows,
            cols,
            group_size,
        );
        let expected = reference_over_quantized_input(
            &nibbles,
            &params,
            &input[..cols],
            rows,
            cols,
            group_size,
        );
        assert_close(&single, &expected, 2e-4, "matvec over quantized input");

        let mut batched = vec![0.0f32; tokens * rows];
        w4a8_matmul(
            &mut batched,
            &nibbles,
            &params,
            &input,
            rows,
            cols,
            tokens,
            group_size,
        );
        for t in 0..tokens {
            let expected = reference_over_quantized_input(
                &nibbles,
                &params,
                &input[t * cols..(t + 1) * cols],
                rows,
                cols,
                group_size,
            );
            assert_close(
                &batched[t * rows..(t + 1) * rows],
                &expected,
                2e-4,
                "matmul over quantized input",
            );
        }
    }

    /// The AVX2 head computes exactly `Σ w·ĥ` over its ±63-quantized hidden
    /// state: the zero term sees the quantized hidden too. With the exact f32
    /// sum in the zero term, the logits are off by `zero · Σ(h − ĥ)` per
    /// group, far outside this tolerance.
    #[test]
    fn tied_lm_head_computes_the_product_of_the_quantized_hidden() {
        if !has_avx2() {
            return;
        }
        let (vocab, hidden_size, group_size) = (700, 256, 64);
        let num_groups = hidden_size / group_size;
        let codes: Vec<u8> = messy_values(vocab * hidden_size, 5)
            .iter()
            .map(|v| ((v + 1.0) * 127.99) as u8)
            .collect();
        let mut params = vec![0u8; vocab * num_groups * 4];
        for (i, chunk) in params.chunks_exact_mut(4).enumerate() {
            chunk[..2]
                .copy_from_slice(&f16::from_f32(0.004 + 0.0001 * (i % 7) as f32).to_le_bytes());
            chunk[2..].copy_from_slice(&f16::from_f32(-0.5 - 0.01 * (i % 3) as f32).to_le_bytes());
        }
        let hidden = messy_values(hidden_size, 9);
        let mut logits = vec![0.0f32; vocab];
        tied_lm_head(
            &mut logits,
            &hidden,
            &codes,
            &params,
            vocab,
            hidden_size,
            group_size,
        );
        // The hidden state as the head sees it: per-group absmax/63, rounded
        // half to even.
        let mut quantized = vec![0.0f64; hidden_size];
        for g in 0..num_groups {
            let group = &hidden[g * group_size..(g + 1) * group_size];
            let absmax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if absmax > 1e-10 { absmax / 63.0 } else { 1.0 };
            let inv = 1.0 / scale;
            for (i, value) in group.iter().enumerate() {
                let q = (value * inv).round_ties_even().clamp(-63.0, 63.0);
                quantized[g * group_size + i] = q as f64 * scale as f64;
            }
        }
        let expected: Vec<f32> = (0..vocab)
            .map(|v| {
                let mut acc = 0.0f64;
                for c in 0..hidden_size {
                    let off = (v * num_groups + c / group_size) * 4;
                    let w = read_f16_le(&params, off) as f64 * codes[v * hidden_size + c] as f64
                        + read_f16_le(&params, off + 2) as f64;
                    acc += w * quantized[c];
                }
                acc as f32
            })
            .collect();
        assert_close(&logits, &expected, 2e-4, "tied head over quantized hidden");
    }

    /// The scalar kernels (aarch64, x86 without AVX2) spread rows across the
    /// pool like the AVX2 ones, and stay exact against the f32 input.
    #[test]
    fn scalar_matvec_is_parallel_and_exact() {
        let (rows, cols, group_size) = (2048, 1024, 128);
        assert!(matvec_should_parallelize(rows, cols));
        let (nibbles, params) = min_max_weights(rows, cols, group_size, 3);
        let input = messy_values(cols, 4);
        let projection = QuantizedLinear {
            rows,
            cols,
            group_params: &params,
            nibble_data: &nibbles,
            group_size,
        };
        let mut output = vec![0.0f32; rows];
        matvec_scalar(vec![(&mut output, &projection)], &input, cols, group_size);
        let num_groups = cols / group_size;
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                let mut acc = 0.0f64;
                for c in 0..cols {
                    let off = (r * num_groups + c / group_size) * 4;
                    let byte = nibbles[(r * cols + c) / 2];
                    let code = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 } as f64;
                    let w = read_f16_le(&params, off) as f64 * code
                        + read_f16_le(&params, off + 2) as f64;
                    acc += w * input[c] as f64;
                }
                acc as f32
            })
            .collect();
        assert_close(&output, &expected, 1e-4, "parallel scalar matvec");
    }

    /// The batched head is a scheduling change only: every logit equals the
    /// single-token head's, bit for bit, across more than one vocabulary chunk.
    #[test]
    fn batched_tied_head_matches_the_single_token_head_exactly() {
        let (vocab, hidden_size, group_size, tokens) = (2 * LM_CHUNK + 77, 192, 64, 5);
        let num_groups = hidden_size / group_size;
        let codes: Vec<u8> = messy_values(vocab * hidden_size, 21)
            .iter()
            .map(|v| ((v + 1.0) * 127.99) as u8)
            .collect();
        let mut params = vec![0u8; vocab * num_groups * 4];
        for (i, chunk) in params.chunks_exact_mut(4).enumerate() {
            chunk[..2]
                .copy_from_slice(&f16::from_f32(0.003 + 0.0002 * (i % 5) as f32).to_le_bytes());
            chunk[2..].copy_from_slice(&f16::from_f32(-0.4 - 0.02 * (i % 3) as f32).to_le_bytes());
        }
        let hidden = messy_values(tokens * hidden_size, 22);
        let mut batched = vec![0.0f32; tokens * vocab];
        tied_lm_head_batch(
            &mut batched,
            &hidden,
            &codes,
            &params,
            vocab,
            hidden_size,
            group_size,
            tokens,
        );
        for t in 0..tokens {
            let mut single = vec![0.0f32; vocab];
            tied_lm_head(
                &mut single,
                &hidden[t * hidden_size..(t + 1) * hidden_size],
                &codes,
                &params,
                vocab,
                hidden_size,
                group_size,
            );
            assert_eq!(
                &batched[t * vocab..(t + 1) * vocab],
                &single[..],
                "token {t}"
            );
        }
    }

    /// Quantizing a shared input once for several projections changes nothing
    /// about any of them.
    #[test]
    fn shared_projections_match_separate_matmuls_exactly() {
        let (cols, group_size, tokens) = (256, 64, 6);
        let input = messy_values(tokens * cols, 31);
        let shapes = [(96usize, 1u64), (32, 2), (32, 3)];
        let weights: Vec<(Vec<u8>, Vec<u8>)> = shapes
            .iter()
            .map(|&(rows, seed)| min_max_weights(rows, cols, group_size, seed))
            .collect();
        let projections: Vec<QuantizedLinear<'_>> = shapes
            .iter()
            .zip(&weights)
            .map(|(&(rows, _), (nibbles, params))| QuantizedLinear {
                rows,
                cols,
                group_params: params,
                nibble_data: nibbles,
                group_size,
            })
            .collect();
        let mut shared: Vec<Vec<f32>> = shapes
            .iter()
            .map(|&(rows, _)| vec![0.0f32; rows * tokens])
            .collect();
        w4a8_matmul_shared(
            shared
                .iter_mut()
                .zip(&projections)
                .map(|(output, projection)| (&mut output[..], projection))
                .collect(),
            &input,
            tokens,
        );
        for (output, projection) in shared.iter().zip(&projections) {
            let mut separate = vec![0.0f32; projection.rows * tokens];
            w4a8_matmul(
                &mut separate,
                projection.nibble_data,
                projection.group_params,
                &input,
                projection.rows,
                cols,
                tokens,
                group_size,
            );
            assert_eq!(output, &separate);
        }
    }
}
