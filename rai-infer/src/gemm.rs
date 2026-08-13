//! High-performance W4A8 GEMM kernels for 4-bit quantized inference:
//! 4-bit weights × int8-quantized activations (f32 inputs are quantized
//! per group on entry, so callers still pass and receive f32).
//!
//! Optimizations:
//! 1. **Explicit AVX2+FMA+F16C SIMD** — 32 elements/iter in the inner loop.
//! 2. **Hardware F16C conversion** — `_mm_cvtph_ps` for scale/zero.
//! 3. **Fused projections** — QKV and gate+up share input_sums, overlap via rayon::join.
//! 4. **Factored dequant** — w*x = scale*dot(codes,x) + zero*sum(x_group).
//! 5. **Rayon par_chunks_mut** for large matrices, single-thread for small.

// These kernels deliberately use numeric indexing and fixed, explicit argument lists so the
// scalar and SIMD implementations stay structurally comparable. Rewriting them as iterators or
// parameter objects would obscure bounds reasoning and can change hot-loop code generation.
#![allow(
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use half::f16;

// Every rayon dispatch in this file is inside the AVX2 path, so on a target
// without those kernels the import itself is unused. `-D warnings` is a gate
// contributors run on their own machine, and an Apple Silicon or ARM-server
// contributor must be able to pass it.
#[cfg(target_arch = "x86_64")]
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

/// Wrappers to allow sending raw pointers across rayon threads.
/// Safety: caller must ensure no data races (non-overlapping regions per thread).
///
/// These exist only to feed the parallel AVX2 chunk kernels. The scalar
/// fallback walks rows on one thread and never forms a raw pointer, so the
/// whole group is compiled out on non-x86-64 targets rather than sitting there
/// as dead code that trips `-D warnings`.
#[cfg(target_arch = "x86_64")]
mod parallel_ptr {
    use super::MAX_GROUPS;

    #[derive(Copy, Clone)]
    pub(super) struct SendPtr(pub(super) *mut f32);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    impl SendPtr {
        #[inline(always)]
        pub(super) fn ptr(self) -> *mut f32 {
            self.0
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct SyncU8Ptr(pub(super) *const u8);
    unsafe impl Send for SyncU8Ptr {}
    unsafe impl Sync for SyncU8Ptr {}
    impl SyncU8Ptr {
        #[inline(always)]
        pub(super) fn ptr(self) -> *const u8 {
            self.0
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct SyncF32Ptr(pub(super) *const f32);
    unsafe impl Send for SyncF32Ptr {}
    unsafe impl Sync for SyncF32Ptr {}
    impl SyncF32Ptr {
        #[inline(always)]
        pub(super) fn ptr(self) -> *const f32 {
            self.0
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct SyncI8Ptr(pub(super) *const i8);
    unsafe impl Send for SyncI8Ptr {}
    unsafe impl Sync for SyncI8Ptr {}
    impl SyncI8Ptr {
        #[inline(always)]
        pub(super) fn ptr(self) -> *const i8 {
            self.0
        }
    }

    /// Per-token quantization parameter arrays shared read-only by every worker.
    #[derive(Copy, Clone)]
    pub(super) struct SyncGroupPtr(pub(super) *const [f32; MAX_GROUPS]);
    unsafe impl Send for SyncGroupPtr {}
    unsafe impl Sync for SyncGroupPtr {}
    impl SyncGroupPtr {
        #[inline(always)]
        pub(super) fn ptr(self) -> *const [f32; MAX_GROUPS] {
            self.0
        }
    }
}

#[cfg(target_arch = "x86_64")]
use parallel_ptr::{SendPtr, SyncF32Ptr, SyncGroupPtr, SyncI8Ptr, SyncU8Ptr};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn read_f16_le(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16::from_bits(bits).to_f32()
}

/// Compute per-group input sums for the factored dequant formula.
#[inline]
fn compute_input_sums(input: &[f32], cols: usize, group_size: usize) -> [f32; MAX_GROUPS] {
    let num_groups = cols.div_ceil(group_size);
    let mut sums = [0.0f32; MAX_GROUPS];

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                compute_input_sums_avx2(input, &mut sums, cols, group_size, num_groups);
            }
            return sums;
        }
    }

    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(cols);
        let mut s = 0.0f32;
        for &v in &input[start..end] {
            s += v;
        }
        sums[g] = s;
    }
    sums
}

/// AVX2 per-group input sums.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2 is required).
/// - `input` must hold at least `cols` values.
/// - `num_groups` must equal `cols.div_ceil(group_size)` and be at most
///   `MAX_GROUPS` (the `sums` array bound).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_input_sums_avx2(
    input: &[f32],
    sums: &mut [f32; MAX_GROUPS],
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    let inp = input.as_ptr();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(cols);
        let len = end - start;
        let chunks8 = len / 8;
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks8 {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(inp.add(start + i * 8)));
        }
        // Horizontal sum
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s128);
        let s2 = _mm_add_ps(s128, shuf);
        let hi2 = _mm_movehl_ps(s2, s2);
        let mut total = _mm_cvtss_f32(_mm_add_ss(s2, hi2));
        for i in (chunks8 * 8)..len {
            total += *inp.add(start + i);
        }
        sums[g] = total;
    }
}

/// Per-group int8 quantization with even/odd column split.
/// Separates input into even-indexed and odd-indexed columns for zero-port5 GEMM.
/// input_even[k] = quantize(input[2k]), input_odd[k] = quantize(input[2k+1]).
#[inline]
fn quantize_input_split(
    input: &[f32],
    input_even: &mut [i8],
    input_odd: &mut [i8],
    input_scales: &mut [f32; MAX_GROUPS],
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                quantize_input_split_avx2(
                    input,
                    input_even,
                    input_odd,
                    input_scales,
                    cols,
                    group_size,
                    num_groups,
                );
            }
            return;
        }
    }
    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(cols);
        let mut absmax = 0.0f32;
        for i in start..end {
            let a = input[i].abs();
            if a > absmax {
                absmax = a;
            }
        }
        let scale = if absmax > 1e-10 { absmax / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        input_scales[g] = scale;
        let mut col = start;
        while col + 1 < end {
            let k = col / 2;
            input_even[k] = (input[col] * inv_scale).round().max(-128.0).min(127.0) as i8;
            input_odd[k] = (input[col + 1] * inv_scale).round().max(-128.0).min(127.0) as i8;
            col += 2;
        }
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
/// - `num_groups` must equal `cols.div_ceil(group_size)` and be at most
///   `MAX_GROUPS`, with `group_size` even. The public GEMM entry points
///   establish these bounds via `validate_weight_buffers`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_input_split_avx2(
    input: &[f32],
    input_even: &mut [i8],
    input_odd: &mut [i8],
    input_scales: &mut [f32; MAX_GROUPS],
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

        for i in 0..iters8 {
            let off = i * 8;
            let v = _mm256_loadu_ps(inp.add(off));
            let scaled = _mm256_mul_ps(v, inv_v);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_v), min_v);
            let i32s = _mm256_cvtps_epi32(clamped);

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

        // Scalar tail for remaining pairs
        let mut col = start + iters8 * 8;
        while col + 1 < end {
            let k = col / 2;
            input_even[k] = (*input.as_ptr().add(col) * inv_scale)
                .round()
                .max(-128.0)
                .min(127.0) as i8;
            input_odd[k] = (*input.as_ptr().add(col + 1) * inv_scale)
                .round()
                .max(-128.0)
                .min(127.0) as i8;
            col += 2;
        }
    }
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
/// - `input_f32` must hold `cols` values; `input_even`/`input_odd` must hold
///   `cols / 2` int8 values produced by `quantize_input_split` (values within
///   ±127, so PMADDUBSW pair sums cannot saturate against 4-bit codes).
/// - `num_groups == cols.div_ceil(group_size) <= MAX_GROUPS`, `cols` and
///   `group_size` even. `validate_weight_buffers` at every public entry point
///   establishes exactly these preconditions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matvec_chunk_i8(
    output: *mut f32,
    nibble_data: *const u8,
    group_params: &[u8],
    input_f32: *const f32,
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: &[f32; MAX_GROUPS],
    input_sums: &[f32; MAX_GROUPS],
    start_row: usize,
    chunk_len: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    let row_bytes = cols / 2;
    let mask_0f = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);
    // Adaptive prefetch distance: 3 rows for small rows (~15ns compute each),
    // 1 row for large rows (7B MLP: 7168 bytes/row, plenty of compute to hide latency).
    let pf_dist: usize = if row_bytes > 2048 { 1 } else { 3 };

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
        let mut w_scales = [0.0f32; MAX_GROUPS];
        let gp_ptr = group_params.as_ptr();
        for g in 0..num_groups {
            let param_off = row_param_base + g * 4;
            let bits = (gp_ptr.add(param_off) as *const u32).read_unaligned();
            let v = _mm_cvtsi32_si128(bits as i32);
            let f = _mm_cvtph_ps(v);
            w_scales[g] = _mm_cvtss_f32(f);
            let f1 = _mm_shuffle_ps(f, f, 1);
            zero_corr += _mm_cvtss_f32(f1) * input_sums[g];
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
            let combined_scale = w_scales[g] * input_scales[g];
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

            // Scalar tail for remaining elements
            let simd_done2 = simd_done + if simd_done + 32 <= actual_gs { 32 } else { 0 };
            if simd_done2 < actual_gs {
                let mut tail_acc = 0.0f32;
                let mut tc = simd_done2;
                while tc + 1 < actual_gs {
                    let b = *nib_row.add(nib_off + tc / 2);
                    tail_acc += (b & 0x0F) as f32 * *input_f32.add(col_start + tc)
                        + (b >> 4) as f32 * *input_f32.add(col_start + tc + 1);
                    tc += 2;
                }
                let tail_v =
                    _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, w_scales[g] * tail_acc);
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
/// - `input_even`/`input_odd` must hold `num_tokens * half_cols` int8 values
///   and `input_f32` `num_tokens * cols` floats, token-major.
/// - `input_scales`/`input_sums` must point at `num_tokens` arrays.
/// - `t_base + T <= num_tokens`, and `output` must be valid for the
///   `t * rows + row` writes those tokens imply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matmul_row_tile<const T: usize>(
    output: *mut f32,
    nib_row: *const u8,
    w_scales: &[f32; MAX_GROUPS],
    w_zeros: &[f32; MAX_GROUPS],
    input_f32: *const f32,
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: *const [f32; MAX_GROUPS],
    input_sums: *const [f32; MAX_GROUPS],
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
            let combined_scale = w_scales[g] * (*input_scales.add(t)).get_unchecked(g);
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
                let tok_f32 = input_f32.add(t * cols);
                let mut tail_acc = 0.0f32;
                let mut tc = simd_done2;
                while tc + 1 < actual_gs {
                    let b = *nib_row.add(nib_off + tc / 2);
                    tail_acc += (b & 0x0F) as f32 * *tok_f32.add(col_start + tc)
                        + (b >> 4) as f32 * *tok_f32.add(col_start + tc + 1);
                    tc += 2;
                }
                let tail_v =
                    _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, w_scales[g] * tail_acc);
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

        let sums = &*input_sums.add(t);
        let mut zero_corr = 0.0f32;
        for g in 0..num_groups {
            zero_corr += w_zeros[g] * sums.get_unchecked(g);
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
    input_f32: *const f32,
    input_even: *const i8,
    input_odd: *const i8,
    input_scales: *const [f32; MAX_GROUPS],
    input_sums: *const [f32; MAX_GROUPS],
    start_row: usize,
    chunk_len: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
    num_tokens: usize,
) {
    let row_bytes = cols / 2;
    let pf_dist: usize = if row_bytes > 2048 { 1 } else { 3 };
    let gp_ptr = group_params.as_ptr();

    // Hoisted out of the row loop: these are 512 bytes each, and re-declaring
    // them per row costs a 1 KB stack memset for every row in the chunk. Only
    // the first `num_groups` entries are ever read, and every row overwrites
    // exactly those before use.
    let mut w_scales = [0.0f32; MAX_GROUPS];
    let mut w_zeros = [0.0f32; MAX_GROUPS];

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
            w_scales[g] = _mm_cvtss_f32(f);
            w_zeros[g] = _mm_cvtss_f32(_mm_shuffle_ps(f, f, 1));
        }

        let mut t = 0usize;
        while t + 4 <= num_tokens {
            matmul_row_tile::<4>(
                output,
                nib_row,
                &w_scales,
                &w_zeros,
                input_f32,
                input_even,
                input_odd,
                input_scales,
                input_sums,
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
                &w_scales,
                &w_zeros,
                input_f32,
                input_even,
                input_odd,
                input_scales,
                input_sums,
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

/// AVX2 LM head using PMADDUBSW for 8-bit codes × i8 hidden.
/// Hidden is quantized to [-63, 63] to prevent i16 saturation in PMADDUBSW
/// (max product: 255*63 = 16065, max pair sum: 32130 < 32767).
/// Processes 32 elements per iteration vs 8 for the FMA path.
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2+FMA+F16C).
/// - `logits` must be valid for `chunk_len` writes.
/// - `embed_data` must hold at least `(start_row + chunk_len) * hidden_size`
///   bytes and `embed_params` at least
///   `(start_row + chunk_len) * num_groups * 4` bytes.
/// - `hidden_i8` must hold `hidden_size` values produced by
///   `quantize_hidden_i8` (range [-63, 63]; anything wider can saturate the
///   i16 PMADDUBSW pair sums against 8-bit codes).
/// - `num_groups == hidden_size.div_ceil(group_size) <= MAX_GROUPS`.
///   `tied_lm_head` asserts these bounds before dispatching here.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn lm_head_chunk_i8(
    logits: *mut f32,
    embed_data: *const u8,
    embed_params: &[u8],
    hidden_i8: *const u8, // quantized hidden, treated as unsigned for PMADDUBSW arg2
    hidden_sums: &[f32; MAX_GROUPS],
    hidden_scales: &[f32; MAX_GROUPS], // per-group quantization scales
    start_row: usize,
    chunk_len: usize,
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) {
    let ones_16 = _mm256_set1_epi16(1);

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

        // Phase 1: Extract scales and zero corrections (unchecked reads)
        let mut zero_corr = 0.0f32;
        let mut w_scales = [0.0f32; MAX_GROUPS];
        let ep_ptr = embed_params.as_ptr();
        for g in 0..num_groups {
            let param_off = row_param_base + g * 4;
            let bits = (ep_ptr.add(param_off) as *const u32).read_unaligned();
            let fv = _mm_cvtsi32_si128(bits as i32);
            let ff = _mm_cvtph_ps(fv);
            w_scales[g] = _mm_cvtss_f32(ff);
            let ff1 = _mm_shuffle_ps(ff, ff, 1);
            zero_corr += _mm_cvtss_f32(ff1) * hidden_sums[g];
        }

        // Phase 2: Integer dot products using PMADDUBSW (256-bit AVX2)
        // codes (u8) × hidden_i8 (i8 stored as u8 with bias 128)
        // PMADDUBSW treats arg1 as unsigned, arg2 as signed.
        // We use codes as arg1 (unsigned 0-255) and hidden_i8 as arg2 (signed -63..63).
        let mut float_acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            let col_start = g * group_size;
            let actual_gs = ((g + 1) * group_size).min(hidden_size) - col_start;
            let codes = row_data.add(col_start);
            let hidden = hidden_i8.add(col_start);
            let chunks32 = actual_gs / 32;

            let mut iacc = _mm256_setzero_si256();

            for c in 0..chunks32 {
                let off = c * 32;
                // Load 32 code bytes and 32 hidden i8 bytes
                let code_v = _mm256_loadu_si256(codes.add(off) as *const __m256i);
                let hid_v = _mm256_loadu_si256(hidden.add(off) as *const __m256i);

                // PMADDUBSW: unsigned codes × signed hidden → i16 pair sums
                let products = _mm256_maddubs_epi16(code_v, hid_v);
                // PMADDWD: sum adjacent i16 pairs → i32
                let sums = _mm256_madd_epi16(products, ones_16);
                iacc = _mm256_add_epi32(iacc, sums);
            }

            // Convert to float, multiply by combined scale, accumulate
            let combined_scale = w_scales[g] * hidden_scales[g];
            let dot_f = _mm256_mul_ps(_mm256_cvtepi32_ps(iacc), _mm256_set1_ps(combined_scale));
            float_acc = _mm256_add_ps(float_acc, dot_f);

            // Scalar tail for remaining elements
            let simd_done = chunks32 * 32;
            if simd_done < actual_gs {
                let mut tail = 0.0f32;
                for i in simd_done..actual_gs {
                    tail += *codes.add(i) as f32 * *(hidden as *const i8).add(i) as f32;
                }
                float_acc = _mm256_add_ps(
                    float_acc,
                    _mm256_set_ps(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        w_scales[g] * hidden_scales[g] * tail,
                    ),
                );
            }
        }

        // Phase 3: Horizontal sum
        let hi128 = _mm256_extractf128_ps(float_acc, 1);
        let lo128 = _mm256_castps256_ps128(float_acc);
        let s128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(s128);
        let s64 = _mm_add_ps(s128, shuf);
        let hi64 = _mm_movehl_ps(s64, s64);
        let dot = _mm_cvtss_f32(_mm_add_ss(s64, hi64));

        *logits.add(local) = dot + zero_corr;
    }
}

/// Quantize f32 hidden state to i8 for use with PMADDUBSW in LM head.
/// Uses [-63, 63] range to prevent saturation with 8-bit codes (max 255).
/// Returns per-group scales.
///
/// Only the AVX2 `tied_lm_head` path needs an i8 hidden state; the scalar
/// fallback reads f32 directly, so this is not compiled on other targets.
#[cfg(target_arch = "x86_64")]
fn quantize_hidden_i8(
    hidden: &[f32],
    output: &mut [i8],
    scales: &mut [f32; MAX_GROUPS],
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                quantize_hidden_i8_avx2(
                    hidden,
                    output,
                    scales,
                    hidden_size,
                    group_size,
                    num_groups,
                );
            }
            return;
        }
    }
    for g in 0..num_groups {
        let start = g * group_size;
        let end = ((g + 1) * group_size).min(hidden_size);
        let mut absmax = 0.0f32;
        for i in start..end {
            let a = hidden[i].abs();
            if a > absmax {
                absmax = a;
            }
        }
        let scale = if absmax > 1e-10 { absmax / 63.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        scales[g] = scale;
        for i in start..end {
            output[i] = (hidden[i] * inv_scale).round().max(-63.0).min(63.0) as i8;
        }
    }
}

/// AVX2 hidden quantization: absmax + scale + quantize to i8 (no even/odd split needed).
///
/// # Safety
/// - Must only be called after `has_avx2()` returned true (AVX2 is required).
/// - `hidden` and `output` must each hold at least `hidden_size` values.
/// - `num_groups` must equal `hidden_size.div_ceil(group_size)` and be at
///   most `MAX_GROUPS` (the `scales` array bound).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_hidden_i8_avx2(
    hidden: &[f32],
    output: &mut [i8],
    scales: &mut [f32; MAX_GROUPS],
    hidden_size: usize,
    group_size: usize,
    num_groups: usize,
) {
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
            let a = (*inp.add(i)).abs();
            if a > absmax {
                absmax = a;
            }
        }

        let scale = if absmax > 1e-10 { absmax / 63.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        scales[g] = scale;

        // AVX2 quantize: float → round → clamp → i32 → pack to i8
        let inv_v = _mm256_set1_ps(inv_scale);
        let min_v = _mm256_set1_ps(-63.0);
        let max_v = _mm256_set1_ps(63.0);
        let out_ptr = output.as_mut_ptr().add(start);

        for i in 0..chunks8 {
            let off = i * 8;
            let v = _mm256_loadu_ps(inp.add(off));
            let scaled = _mm256_mul_ps(v, inv_v);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_v), min_v);
            let i32s = _mm256_cvtps_epi32(clamped);

            // Pack i32 → i16 → i8 (128-bit to avoid cross-lane issues)
            let lo = _mm256_castsi256_si128(i32s);
            let hi = _mm256_extracti128_si256(i32s, 1);
            let packed16 = _mm_packs_epi32(lo, hi); // 8 i16 in order
            let packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128()); // 8 i8 in lower bytes

            // Store 8 bytes
            let dst = out_ptr.add(off) as *mut u64;
            dst.write_unaligned(_mm_cvtsi128_si64(packed8) as u64);
        }

        // Scalar tail
        for i in (chunks8 * 8)..len {
            *(out_ptr.add(i)) = (*inp.add(i) * inv_scale).round().max(-63.0).min(63.0) as i8;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

fn matvec_row_scalar(
    nibble_data: &[u8],
    group_params: &[u8],
    input: &[f32],
    input_sums: &[f32; MAX_GROUPS],
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
    hidden_sums: &[f32; MAX_GROUPS],
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
// Internal dispatch (shared input_sums, no recomputation)
// ---------------------------------------------------------------------------

// These describe the parallel AVX2 dispatch policy, which only exists on
// x86-64; the scalar fallback walks every row on one thread. They stay visible
// to `cfg(test)` because the reference tests size their inputs relative to
// `PAR_THRESHOLD` and `LM_CHUNK` to force the parallel branch on AVX2 hosts.
#[cfg(any(target_arch = "x86_64", test))]
const LM_CHUNK: usize = 512;
#[cfg(any(target_arch = "x86_64", test))]
const PAR_THRESHOLD: usize = 256;

/// Adaptive chunk sizing: target ~24KB per chunk to fit L1 cache (32KB).
/// Each row needs cols/2 bytes of nibble data.
/// SmolLM (cols=256): 24576/(128)=192→clamped to 64 rows.
/// Mistral hidden (cols=4096): 24576/(2048)=12 rows.
/// Mistral MLP (cols=14336): 24576/(7168)=3→clamped to 4 rows.
#[cfg(target_arch = "x86_64")]
#[inline]
fn chunk_rows_for(cols: usize) -> usize {
    (24_576 / (cols / 2).max(1)).clamp(4, 64)
}
/// Maximum number of quantization groups per GEMM call (a 7B MLP at group
/// size 128 needs 14336/128 = 112 groups, so 128 leaves headroom).
///
/// `format.rs` imports this constant: `.raimodel` validation at load time is
/// the single gate that keeps accepted model files within kernel capacity, so
/// the limit is defined once here at its point of enforcement.
pub const MAX_GROUPS: usize = 128;

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

/// Inner dispatch with pre-computed input data. Uses W4A8 integer GEMM when available.
/// Uses dynamic chunk sizing: exactly num_threads chunks for optimal load balance
/// and minimal atomic overhead.
fn w4a8_matvec_inner(
    output: &mut [f32],
    nibble_data: &[u8],
    group_params: &[u8],
    input: &[f32],
    input_even: &[i8],
    input_odd: &[i8],
    input_scales: &[f32; MAX_GROUPS],
    input_sums: &[f32; MAX_GROUPS],
    rows: usize,
    cols: usize,
    group_size: usize,
    num_groups: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            if rows >= PAR_THRESHOLD {
                // Adaptive chunk sizing: keeps per-chunk weight data in L1 (32KB).
                let cr = chunk_rows_for(cols);
                let num_chunks = rows.div_ceil(cr);
                let out_ptr = SendPtr(output.as_mut_ptr());
                let nib_ptr = SyncU8Ptr(nibble_data.as_ptr());
                let inp_f32_ptr = SyncF32Ptr(input.as_ptr());
                let inp_even_ptr = SyncI8Ptr(input_even.as_ptr());
                let inp_odd_ptr = SyncI8Ptr(input_odd.as_ptr());
                let scales = *input_scales;
                let sums = *input_sums;
                (0..num_chunks).into_par_iter().for_each(|ci| {
                    let start = ci * cr;
                    let len = cr.min(rows - start);
                    unsafe {
                        matvec_chunk_i8(
                            out_ptr.ptr().add(start),
                            nib_ptr.ptr(),
                            group_params,
                            inp_f32_ptr.ptr(),
                            inp_even_ptr.ptr(),
                            inp_odd_ptr.ptr(),
                            &scales,
                            &sums,
                            start,
                            len,
                            cols,
                            group_size,
                            num_groups,
                        );
                    }
                });
            } else {
                unsafe {
                    matvec_chunk_i8(
                        output.as_mut_ptr(),
                        nibble_data.as_ptr(),
                        group_params,
                        input.as_ptr(),
                        input_even.as_ptr(),
                        input_odd.as_ptr(),
                        input_scales,
                        input_sums,
                        0,
                        rows,
                        cols,
                        group_size,
                        num_groups,
                    );
                }
            }
            return;
        }
    }

    // The scalar row kernel dequantizes against the f32 input and the per-group
    // sums, so the int8 split of the input is only ever consumed by the AVX2
    // kernels above.
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (input_even, input_odd, input_scales);

    for row in 0..rows {
        output[row] = matvec_row_scalar(
            nibble_data,
            group_params,
            input,
            input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// W4A8 matrix–vector product: `output[r] = dot(dequant(weights[r]), input)`.
///
/// The f32 input is quantized to int8 per group internally; weights are 4-bit
/// codes with per-group f16 scale/zero parameters.
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
    let num_groups = cols.div_ceil(group_size);
    let input_sums = compute_input_sums(input, cols, group_size);
    let half_cols = cols / 2;
    let mut input_even = vec![0i8; half_cols];
    let mut input_odd = vec![0i8; half_cols];
    let mut input_scales = [0.0f32; MAX_GROUPS];
    quantize_input_split(
        input,
        &mut input_even,
        &mut input_odd,
        &mut input_scales,
        cols,
        group_size,
        num_groups,
    );
    w4a8_matvec_inner(
        output,
        nibble_data,
        group_params,
        input,
        &input_even,
        &input_odd,
        &input_scales,
        &input_sums,
        rows,
        cols,
        group_size,
        num_groups,
    );
}

/// Fused Q/K/V projections: single parallel dispatch over all Q+K+V rows.
/// Shared input_sums computed once. K/V rows (below PAR_THRESHOLD individually)
/// become parallel alongside Q rows via work-stealing.
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
    let cols = q_proj.cols;
    let group_size = q_proj.group_size;
    let num_groups = cols.div_ceil(group_size);
    let input_sums = compute_input_sums(input, cols, group_size);
    let half_cols = cols / 2;
    let mut input_even = vec![0i8; half_cols];
    let mut input_odd = vec![0i8; half_cols];
    let mut input_scales = [0.0f32; MAX_GROUPS];
    quantize_input_split(
        input,
        &mut input_even,
        &mut input_odd,
        &mut input_scales,
        cols,
        group_size,
        num_groups,
    );

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let cr = chunk_rows_for(cols);
            let q_chunks = q_proj.rows.div_ceil(cr);
            let k_chunks = k_proj.rows.div_ceil(cr);
            let v_chunks = v_proj.rows.div_ceil(cr);
            let total_chunks = q_chunks + k_chunks + v_chunks;

            let q_ptr = SendPtr(q_out.as_mut_ptr());
            let k_ptr = SendPtr(k_out.as_mut_ptr());
            let v_ptr = SendPtr(v_out.as_mut_ptr());
            let q_nib = SyncU8Ptr(q_proj.nibble_data.as_ptr());
            let k_nib = SyncU8Ptr(k_proj.nibble_data.as_ptr());
            let v_nib = SyncU8Ptr(v_proj.nibble_data.as_ptr());
            let q_par = q_proj.group_params;
            let k_par = k_proj.group_params;
            let v_par = v_proj.group_params;
            let q_rows = q_proj.rows;
            let k_rows = k_proj.rows;
            let v_rows = v_proj.rows;
            let inp_f32_ptr = SyncF32Ptr(input.as_ptr());
            let inp_even_ptr = SyncI8Ptr(input_even.as_ptr());
            let inp_odd_ptr = SyncI8Ptr(input_odd.as_ptr());
            let scales = input_scales;
            let sums = input_sums;

            (0..total_chunks).into_par_iter().for_each(|chunk_idx| {
                let (out_ptr, nibble_data, group_params, rows, start_row) = if chunk_idx < q_chunks
                {
                    let start = chunk_idx * cr;
                    (q_ptr, q_nib, q_par, q_rows, start)
                } else if chunk_idx < q_chunks + k_chunks {
                    let ki = chunk_idx - q_chunks;
                    let start = ki * cr;
                    (k_ptr, k_nib, k_par, k_rows, start)
                } else {
                    let vi = chunk_idx - q_chunks - k_chunks;
                    let start = vi * cr;
                    (v_ptr, v_nib, v_par, v_rows, start)
                };

                let chunk_len = cr.min(rows - start_row);
                unsafe {
                    matvec_chunk_i8(
                        out_ptr.ptr().add(start_row),
                        nibble_data.ptr(),
                        group_params,
                        inp_f32_ptr.ptr(),
                        inp_even_ptr.ptr(),
                        inp_odd_ptr.ptr(),
                        &scales,
                        &sums,
                        start_row,
                        chunk_len,
                        cols,
                        group_size,
                        num_groups,
                    );
                }
            });
            return;
        }
    }

    // Scalar fallback
    for row in 0..q_proj.rows {
        q_out[row] = matvec_row_scalar(
            q_proj.nibble_data,
            q_proj.group_params,
            input,
            &input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
    for row in 0..k_proj.rows {
        k_out[row] = matvec_row_scalar(
            k_proj.nibble_data,
            k_proj.group_params,
            input,
            &input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
    for row in 0..v_proj.rows {
        v_out[row] = matvec_row_scalar(
            v_proj.nibble_data,
            v_proj.group_params,
            input,
            &input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
}

/// Fused gate+up projections: single parallel dispatch over all gate+up rows.
/// Shared input_sums computed once.
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
    let cols = gate_proj.cols;
    let group_size = gate_proj.group_size;
    let num_groups = cols.div_ceil(group_size);
    let input_sums = compute_input_sums(input, cols, group_size);
    let half_cols = cols / 2;
    let mut input_even = vec![0i8; half_cols];
    let mut input_odd = vec![0i8; half_cols];
    let mut input_scales = [0.0f32; MAX_GROUPS];
    quantize_input_split(
        input,
        &mut input_even,
        &mut input_odd,
        &mut input_scales,
        cols,
        group_size,
        num_groups,
    );

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let cr = chunk_rows_for(cols);
            let g_chunks = gate_proj.rows.div_ceil(cr);
            let u_chunks = up_proj.rows.div_ceil(cr);
            let total_chunks = g_chunks + u_chunks;

            let g_ptr = SendPtr(gate_out.as_mut_ptr());
            let u_ptr = SendPtr(up_out.as_mut_ptr());
            let g_nib = SyncU8Ptr(gate_proj.nibble_data.as_ptr());
            let u_nib = SyncU8Ptr(up_proj.nibble_data.as_ptr());
            let g_par = gate_proj.group_params;
            let u_par = up_proj.group_params;
            let g_rows = gate_proj.rows;
            let u_rows = up_proj.rows;
            let inp_f32_ptr = SyncF32Ptr(input.as_ptr());
            let inp_even_ptr = SyncI8Ptr(input_even.as_ptr());
            let inp_odd_ptr = SyncI8Ptr(input_odd.as_ptr());
            let scales = input_scales;
            let sums = input_sums;

            (0..total_chunks).into_par_iter().for_each(|chunk_idx| {
                let (out_ptr, nibble_data, group_params, rows, start_row) = if chunk_idx < g_chunks
                {
                    let start = chunk_idx * cr;
                    (g_ptr, g_nib, g_par, g_rows, start)
                } else {
                    let ui = chunk_idx - g_chunks;
                    let start = ui * cr;
                    (u_ptr, u_nib, u_par, u_rows, start)
                };

                let chunk_len = cr.min(rows - start_row);
                unsafe {
                    matvec_chunk_i8(
                        out_ptr.ptr().add(start_row),
                        nibble_data.ptr(),
                        group_params,
                        inp_f32_ptr.ptr(),
                        inp_even_ptr.ptr(),
                        inp_odd_ptr.ptr(),
                        &scales,
                        &sums,
                        start_row,
                        chunk_len,
                        cols,
                        group_size,
                        num_groups,
                    );
                }
            });
            return;
        }
    }

    // Scalar fallback
    for row in 0..gate_proj.rows {
        gate_out[row] = matvec_row_scalar(
            gate_proj.nibble_data,
            gate_proj.group_params,
            input,
            &input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
    for row in 0..up_proj.rows {
        up_out[row] = matvec_row_scalar(
            up_proj.nibble_data,
            up_proj.group_params,
            input,
            &input_sums,
            row,
            cols,
            group_size,
            num_groups,
        );
    }
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
/// Call once at program start. Respects RAYON_NUM_THREADS env var for tuning;
/// defaults to physical core count + 1.
pub fn configure_thread_pool() {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            // Physical cores + 1: Best balance for mixed GEMM+attention workloads.
            // More threads help bandwidth-bound GEMMs but hurt sequential attention.
            (cpus / 2 + 1).max(2)
        });
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

    let num_groups = cols.div_ceil(group_size);
    let half_cols = cols / 2;

    // Pre-quantize all token inputs into flat contiguous buffers (2 allocs, not 2*num_tokens)
    let mut all_sums: Vec<[f32; MAX_GROUPS]> = Vec::with_capacity(num_tokens);
    let mut all_even_flat: Vec<i8> = vec![0i8; num_tokens * half_cols];
    let mut all_odd_flat: Vec<i8> = vec![0i8; num_tokens * half_cols];
    let mut all_scales: Vec<[f32; MAX_GROUPS]> = Vec::with_capacity(num_tokens);

    for t in 0..num_tokens {
        let inp = &input[t * cols..(t + 1) * cols];
        all_sums.push(compute_input_sums(inp, cols, group_size));
        let even = &mut all_even_flat[t * half_cols..(t + 1) * half_cols];
        let odd = &mut all_odd_flat[t * half_cols..(t + 1) * half_cols];
        let mut scales = [0.0f32; MAX_GROUPS];
        quantize_input_split(inp, even, odd, &mut scales, cols, group_size, num_groups);
        all_scales.push(scales);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let cr = chunk_rows_for(cols);
            let num_chunks = rows.div_ceil(cr);

            let out_ptr = SendPtr(output.as_mut_ptr());
            let nib_ptr = SyncU8Ptr(nibble_data.as_ptr());
            let inp_f32_ptr = SyncF32Ptr(input.as_ptr());
            let inp_even_ptr = SyncI8Ptr(all_even_flat.as_ptr());
            let inp_odd_ptr = SyncI8Ptr(all_odd_flat.as_ptr());
            let scales_ptr = SyncGroupPtr(all_scales.as_ptr());
            let sums_ptr = SyncGroupPtr(all_sums.as_ptr());

            if legacy_matmul_enabled() {
                // Benchmark-only: drive the batch through the single-token
                // kernel once per token, as this function did before the
                // weight-stationary rewrite.
                (0..num_chunks).into_par_iter().for_each(|ci| {
                    let start = ci * cr;
                    let len = cr.min(rows - start);
                    for t in 0..num_tokens {
                        unsafe {
                            matvec_chunk_i8(
                                out_ptr.ptr().add(t * rows + start),
                                nib_ptr.ptr(),
                                group_params,
                                inp_f32_ptr.ptr().add(t * cols),
                                inp_even_ptr.ptr().add(t * half_cols),
                                inp_odd_ptr.ptr().add(t * half_cols),
                                &*scales_ptr.ptr().add(t),
                                &*sums_ptr.ptr().add(t),
                                start,
                                len,
                                cols,
                                group_size,
                                num_groups,
                            );
                        }
                    }
                });
                return;
            }

            // Weight-stationary parallel dispatch: each thread takes a chunk of
            // rows and drives the whole batch through it, so the weight data it
            // pulls into L1 is unpacked once and reused by every token.
            (0..num_chunks).into_par_iter().for_each(|ci| {
                let start = ci * cr;
                let len = cr.min(rows - start);
                unsafe {
                    matmul_chunk_i8(
                        out_ptr.ptr(),
                        nib_ptr.ptr(),
                        group_params,
                        inp_f32_ptr.ptr(),
                        inp_even_ptr.ptr(),
                        inp_odd_ptr.ptr(),
                        scales_ptr.ptr(),
                        sums_ptr.ptr(),
                        start,
                        len,
                        rows,
                        cols,
                        group_size,
                        num_groups,
                        num_tokens,
                    );
                }
            });
            return;
        }
    }

    // Scalar fallback
    for v in output[..num_tokens * rows].iter_mut() {
        *v = 0.0;
    }
    for row in 0..rows {
        for t in 0..num_tokens {
            let inp = &input[t * cols..(t + 1) * cols];
            let sums = &all_sums[t];
            output[t * rows + row] = matvec_row_scalar(
                nibble_data,
                group_params,
                inp,
                sums,
                row,
                cols,
                group_size,
                num_groups,
            );
        }
    }
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
/// On AVX2 hosts the hidden state is quantized to int8 (range ±63) for the
/// integer PMADDUBSW path; the scalar fallback uses the f32 hidden directly.
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
    assert!(group_size > 0, "group_size must be non-zero");
    assert!(
        hidden_size > 0 && hidden_size.is_multiple_of(2),
        "hidden_size must be non-zero and even"
    );
    assert!(hidden.len() >= hidden_size, "hidden buffer is too small");
    assert!(logits.len() >= vocab_size, "logit buffer is too small");
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

    let hidden_sums = compute_input_sums(hidden, hidden_size, group_size);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // Quantize hidden state to i8 for integer PMADDUBSW path
            let mut hidden_i8 = vec![0i8; hidden_size];
            let mut hidden_scales = [0.0f32; MAX_GROUPS];
            quantize_hidden_i8(
                hidden,
                &mut hidden_i8,
                &mut hidden_scales,
                hidden_size,
                group_size,
                num_groups,
            );

            let num_chunks = vocab_size.div_ceil(LM_CHUNK);
            let out_ptr = SendPtr(logits.as_mut_ptr());
            let emb_ptr = SyncU8Ptr(embed_data.as_ptr());
            let hid_i8_ptr = SyncU8Ptr(hidden_i8.as_ptr() as *const u8);
            let sums = hidden_sums;
            let hscales = hidden_scales;
            (0..num_chunks).into_par_iter().for_each(|ci| {
                let start = ci * LM_CHUNK;
                let len = LM_CHUNK.min(vocab_size - start);
                unsafe {
                    lm_head_chunk_i8(
                        out_ptr.ptr().add(start),
                        emb_ptr.ptr(),
                        embed_params,
                        hid_i8_ptr.ptr(),
                        &sums,
                        &hscales,
                        start,
                        len,
                        hidden_size,
                        group_size,
                        num_groups,
                    );
                }
            });
            return;
        }
    }

    for v in 0..vocab_size {
        logits[v] = lm_head_row_scalar(
            embed_data,
            embed_params,
            hidden,
            &hidden_sums,
            v,
            hidden_size,
            group_size,
            num_groups,
        );
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
    // Parallel-path reference tests (rows > PAR_THRESHOLD so the rayon
    // dispatch runs on AVX2 hosts; on non-AVX2 hosts the same assertions
    // cover the scalar fallback).
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
        let rows = 320; // > PAR_THRESHOLD → parallel branch on AVX2 hosts
        let cols = 256;
        let group_size = 64;
        assert!(rows > PAR_THRESHOLD);

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
        let q_rows = 320; // > PAR_THRESHOLD
        let kv_rows = 64; // GQA-style narrow K/V
        let cols = 256;
        let group_size = 64;
        assert!(q_rows > PAR_THRESHOLD);

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
    fn fused_gate_up_parallel_path_matches_per_matvec() {
        let rows = 320; // > PAR_THRESHOLD
        let cols = 256;
        let group_size = 64;
        assert!(rows > PAR_THRESHOLD);

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
}
