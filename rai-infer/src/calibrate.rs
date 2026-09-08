//! Calibration for GPTQ quantization, inside the converter.
//!
//! # What calibration is for
//!
//! Round-to-nearest picks each weight's 4-bit code by looking at the weight.
//! GPTQ picks it by looking at what the weight *does*: it minimizes the error
//! in the layer's output under the distribution of inputs the layer actually
//! sees, which needs a sample of those inputs. That sample is the Hessian
//! `H = XᵀX`, where `X` is the stack of activations entering a projection over
//! some calibration text.
//!
//! Measured on SmolLM2-1.7B, that difference is worth 8.80 perplexity against
//! 10.30 at an identical file size — most of the gap between this engine and
//! llama.cpp.
//!
//! # Why this lives in Rust and not in the Python exporter
//!
//! The Python exporter has done this since the beginning, and it is the reason
//! the good numbers were unreachable: it writes container v1, so it refuses
//! every checkpoint with projection biases, per-head QK norms, RoPE scaling or
//! MoE routing — Qwen2/2.5, Qwen3, Gemma, Llama-3.1. The engine's own converter
//! writes v2 and accepts all of them, and it needs no Python at all. Moving
//! calibration here is what makes the better quantizer the default rather than
//! a footnote most users cannot reach.
//!
//! # Sequential, not parallel
//!
//! `scripts/export_raimodel.py` registers forward hooks on the unquantized
//! model and collects every layer's Hessian in one pass. That is the simplest
//! thing to write and it has two costs. It holds all of them at once — for a
//! 7B that is about 32 GB, which is why it needs a large machine — and it
//! calibrates each layer against inputs produced by *unquantized* earlier
//! layers, which is not what the finished model will feed it.
//!
//! This calibrates one layer at a time and then propagates through the layer it
//! just quantized. One layer's Hessians are alive at once (about 1 GB for a
//! 7B), and every layer after the first is calibrated against the error the
//! earlier ones really introduce.
//!
//! # The cost that cannot be argued away
//!
//! `H` is `cols × cols`. For a 7B MLP that is 14336², or 1.6 GB in f64, and
//! accumulating it over `n` tokens costs `n · cols²` multiply-adds. Calibrated
//! conversion is therefore not the streaming, 85 MB operation that
//! round-to-nearest is; it is minutes to hours and gigabytes. That is the
//! price of the quality, it is paid once at conversion, and the run-time model
//! is byte-for-byte the same shape either way.

use anyhow::{ensure, Result};
use rayon::prelude::*;

/// Accumulate `Hᵀ += XᵀX` for one block of activation rows.
///
/// `x` is `rows × cols`, row-major; `h` is `cols × cols`, row-major, and only
/// the **upper triangle** is written — `H` is symmetric by construction and
/// computing both halves doubles the most expensive operation in calibration
/// for nothing. [`symmetrize`] mirrors it once at the end.
///
/// Accumulation is in f64 while the inputs stay f32. The sum runs over tens of
/// thousands of tokens, and an f32 accumulator loses the small contributions
/// once the running total is large — which is exactly the regime a Hessian
/// diagonal lives in.
pub fn accumulate_gram(h: &mut [f64], x: &[f32], rows: usize, cols: usize) -> Result<()> {
    ensure!(
        x.len() == rows.saturating_mul(cols),
        "activation block is {} floats, expected {rows}x{cols}",
        x.len()
    );
    ensure!(
        h.len() == cols.saturating_mul(cols),
        "Hessian buffer is {} floats, expected {cols}x{cols}",
        h.len()
    );
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    // Rows of `H` are independent, so the matrix splits into horizontal bands
    // that threads can own outright. The band is sized to give every worker
    // several pieces — an even split would leave the thread holding the first
    // band, whose rows are the longest, running alone at the end, because the
    // upper triangle makes row `i` cost `cols - i`.
    let threads = rayon::current_num_threads().max(1);
    let band = cols.div_ceil(threads * 8).max(1);

    h.par_chunks_mut(band * cols)
        .enumerate()
        .for_each(|(index, block)| {
            let first = index * band;
            let last = (first + block.len() / cols).min(cols);
            if first >= last {
                return;
            }

            #[cfg(target_arch = "x86_64")]
            {
                if crate::gemm::has_avx2() {
                    // SAFETY: AVX2 was detected, and `block` is exactly the
                    // rows `first..last` of an initialized `cols x cols`
                    // buffer whose lengths were checked above.
                    unsafe { accumulate_band_avx2(block, x, rows, cols, first, last) };
                    return;
                }
            }

            accumulate_band_scalar(block, x, rows, cols, first, last);
        });

    Ok(())
}

/// Portable band accumulation, used on non-AVX2 targets and by the tests.
///
/// `block` holds rows `first..last` of `H`; `first` is needed because the upper
/// triangle starts at the global column index, not at the band's own zero.
fn accumulate_band_scalar(
    block: &mut [f64],
    x: &[f32],
    rows: usize,
    cols: usize,
    first: usize,
    last: usize,
) {
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for i in first..last {
            let xi = f64::from(row[i]);
            if xi == 0.0 {
                continue;
            }
            let out = &mut block[(i - first) * cols..(i - first + 1) * cols];
            for j in i..cols {
                out[j] += xi * f64::from(row[j]);
            }
        }
    }
}

/// Blocked AVX2 accumulation.
///
/// The loop order is (row, i, j) rather than (i, j, row) so that each `x[r][i]`
/// is loaded once and broadcast across a whole row of `H`, and the `H` row is
/// walked contiguously. That turns the inner loop into a stream of FMAs over
/// memory that is already in cache from the previous `i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_band_avx2(
    block: &mut [f64],
    x: &[f32],
    rows: usize,
    cols: usize,
    first: usize,
    last: usize,
) {
    use std::arch::x86_64::*;

    let h_ptr = block.as_mut_ptr();
    let x_ptr = x.as_ptr();

    for r in 0..rows {
        let row = x_ptr.add(r * cols);
        for i in first..last {
            let xi = f64::from(*row.add(i));
            if xi == 0.0 {
                continue;
            }
            let broadcast = _mm256_set1_pd(xi);
            let out = h_ptr.add((i - first) * cols);

            // The upper triangle starts at j == i.
            let mut j = i;
            while j + 4 <= cols {
                // f32 activations widened to f64 four at a time.
                let xj = _mm256_cvtps_pd(_mm_loadu_ps(row.add(j)));
                let acc = _mm256_loadu_pd(out.add(j));
                _mm256_storeu_pd(out.add(j), _mm256_fmadd_pd(broadcast, xj, acc));
                j += 4;
            }
            while j < cols {
                *out.add(j) += xi * f64::from(*row.add(j));
                j += 1;
            }
        }
    }
}

/// Mirror the upper triangle into the lower one.
///
/// The solver checks symmetry before factorizing, so this is not cosmetic: a
/// Hessian left half-filled is rejected, and rightly.
pub fn symmetrize(h: &mut [f64], cols: usize) {
    for i in 0..cols {
        for j in (i + 1)..cols {
            h[j * cols + i] = h[i * cols + j];
        }
    }
}

/// Divide an accumulated gram matrix by the token count.
///
/// GPTQ's damping term is a fraction of the mean diagonal, so the Hessian has
/// to be a mean rather than a sum or the damping means something different for
/// every calibration size.
pub fn scale_by_tokens(h: &mut [f64], tokens: usize) -> Result<()> {
    ensure!(tokens > 0, "cannot average a Hessian over zero tokens");
    let inv = 1.0 / tokens as f64;
    for value in h.iter_mut() {
        *value *= inv;
    }
    Ok(())
}

/// Per-input-channel activation energy: the diagonal of `XᵀX`.
///
/// This is the whole statistic the default calibration path needs. The full
/// Hessian answers "how do these two channels interact", which is what buys
/// GPTQ its last fraction of a percent and costs a `cols × cols` factorization;
/// the diagonal answers "how much does this channel matter", which is what buys
/// most of the gain and costs one pass over the activations.
pub fn accumulate_channel_energy(energy: &mut [f64], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(energy.len(), cols);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for (slot, &value) in energy.iter_mut().zip(row.iter()) {
            let v = f64::from(value);
            *slot += v * v;
        }
    }
}

/// Quantize one group of a row to `bits`, returning the squared error weighted
/// by each column's activation energy.
///
/// The weighting is the point. An unweighted fit spends the group's whole
/// 4-bit range covering whichever weight happens to be largest, even when the
/// channel it multiplies is one the layer barely uses.
fn weighted_group_error(
    weights: &[f64],
    energy: &[f64],
    scale: f64,
    zero: f64,
    levels: f64,
) -> f64 {
    let mut total = 0.0;
    for (&w, &e) in weights.iter().zip(energy.iter()) {
        let code = ((w - zero) / scale).round().clamp(0.0, levels - 1.0);
        let delta = w - (code * scale + zero);
        total += e * delta * delta;
    }
    total
}

/// Choose a group's scale and zero point by minimising activation-weighted
/// error rather than by spanning the range.
///
/// Round-to-nearest takes `scale = (max - min) / (levels - 1)`, which is
/// optimal only if every weight in the group matters equally and the
/// distribution is uniform. Neither holds. Shrinking the range clips the
/// extremes and buys resolution everywhere else, and whether that trade is
/// worth taking depends on the activations — which is exactly what `energy`
/// carries.
///
/// The search is a grid over shrink factors. It is deliberately not a golden
/// section or a gradient step: the objective has small flat regions and local
/// minima where a code crosses a rounding boundary, and a coarse grid that is
/// evaluated exactly beats a clever method that converges to the wrong basin.
pub fn group_params_weighted(
    weights: &[f64],
    energy: &[f64],
    bits: u8,
    shrink_steps: usize,
) -> (f64, f64) {
    debug_assert_eq!(weights.len(), energy.len());
    let levels = f64::from(1u32 << bits);

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &w in weights {
        min = min.min(w);
        max = max.max(w);
    }
    let span = max - min;
    if !span.is_finite() || span < 1e-15 {
        // A constant group: any scale reproduces it, and 1.0 keeps the stored
        // f16 well away from the subnormal range.
        return (1.0, min);
    }

    let mut best = (span / (levels - 1.0), min);
    let mut best_error = weighted_group_error(weights, energy, best.0, best.1, levels);

    // Shrink toward the centre of mass rather than the midpoint: a group whose
    // weights cluster near one end loses less by clipping the far end.
    let weight_sum: f64 = energy.iter().sum();
    let centre = if weight_sum > 0.0 {
        weights
            .iter()
            .zip(energy.iter())
            .map(|(w, e)| w * e)
            .sum::<f64>()
            / weight_sum
    } else {
        0.5 * (min + max)
    };

    for step in 1..=shrink_steps {
        // 1.0 down to 0.5 of the original span.
        let factor = 1.0 - 0.5 * (step as f64 / shrink_steps as f64);
        let half = 0.5 * span * factor;
        let low = (centre - half).max(min);
        let high = (centre + half).min(max);
        let shrunk = high - low;
        if shrunk < 1e-15 {
            continue;
        }
        let scale = shrunk / (levels - 1.0);
        let error = weighted_group_error(weights, energy, scale, low, levels);
        if error < best_error {
            best_error = error;
            best = (scale, low);
        }
    }
    best
}

/// Search a per-input-channel scale that makes 4-bit rounding hurt less.
///
/// Scaling weight column `j` by `s[j]` and the matching input channel by
/// `1/s[j]` leaves the layer's output algebraically unchanged: the projections
/// fed by an RMSNorm can absorb `1/s` into that norm's weight vector, which the
/// container already stores per channel. What changes is the quantization: a
/// column scaled up occupies more of its group's range and survives rounding
/// better, at the cost of the columns sharing that group.
///
/// `alpha` is searched over a grid because the right exponent depends on how
/// heavy the activation distribution's tail is, and that differs per layer.
/// The objective is the diagonal-weighted output error, which is the same
/// quantity [`group_params_weighted`] minimises — the two are searched together
/// rather than one after the other, because a scale that helps only under the
/// old clipping is not a scale worth folding into the model.
pub fn search_channel_scales(
    weights: &[f64],
    energy: &[f64],
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
    alphas: &[f64],
) -> Vec<f64> {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(energy.len(), cols);

    let mut best_scales = vec![1.0f64; cols];
    let mut best_error = f64::INFINITY;
    let mut scaled = vec![0.0f64; rows * cols];
    let mut scratch_w = vec![0.0f64; group_size];
    let mut scratch_e = vec![0.0f64; group_size];

    for &alpha in alphas {
        // s[j] = mean(x_j^2)^(alpha/2), normalized to a geometric mean of 1 so
        // the folded norm weights stay near their original magnitude and the
        // f16 scales that end up in the file keep their precision.
        let mut scales: Vec<f64> = energy
            .iter()
            .map(|&e| (e.max(1e-12)).powf(0.5 * alpha))
            .collect();
        let log_mean = scales.iter().map(|s| s.ln()).sum::<f64>() / cols as f64;
        let norm = log_mean.exp();
        if !norm.is_finite() || norm <= 0.0 {
            continue;
        }
        for s in scales.iter_mut() {
            *s /= norm;
        }

        for r in 0..rows {
            for c in 0..cols {
                scaled[r * cols + c] = weights[r * cols + c] * scales[c];
            }
        }

        let mut error = 0.0;
        for r in 0..rows {
            let mut start = 0;
            while start < cols {
                let end = (start + group_size).min(cols);
                let len = end - start;
                scratch_w[..len].copy_from_slice(&scaled[r * cols + start..r * cols + end]);
                // The activation seen by the scaled column is x/s, so its
                // energy is divided by s^2.
                for (slot, c) in scratch_e[..len].iter_mut().zip(start..end) {
                    *slot = energy[c] / (scales[c] * scales[c]);
                }
                let (scale, zero) =
                    group_params_weighted(&scratch_w[..len], &scratch_e[..len], bits, 8);
                error += weighted_group_error(
                    &scratch_w[..len],
                    &scratch_e[..len],
                    scale,
                    zero,
                    f64::from(1u32 << bits),
                );
                start = end;
            }
        }

        if error < best_error {
            best_error = error;
            best_scales = scales;
        }
    }
    best_scales
}

/// `out[n][r] = sum_c input[n][c] * weights[r][c]`, with an optional bias.
///
/// The weight layout is row-major `[rows, cols]` — the same orientation
/// safetensors stores a projection in, so no transpose is needed between
/// reading a tensor and using it.
///
/// Parallel over output rows: each thread owns a disjoint set of `r`, reads the
/// whole input, and writes nothing another thread touches.
pub fn matmul_t(
    out: &mut [f32],
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    tokens: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    ensure!(input.len() == tokens * cols, "input is not {tokens}x{cols}");
    ensure!(
        weights.len() == rows * cols,
        "weights are not {rows}x{cols}"
    );
    ensure!(out.len() == tokens * rows, "output is not {tokens}x{rows}");
    if let Some(bias) = bias {
        ensure!(bias.len() == rows, "bias is not {rows} long");
    }

    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(token, row_out)| {
            let x = &input[token * cols..(token + 1) * cols];
            for (r, slot) in row_out.iter_mut().enumerate() {
                let w = &weights[r * cols..(r + 1) * cols];
                // f32 accumulation matches what the engine does at run time;
                // an f64 accumulator here would make calibration see a slightly
                // different model than inference does.
                let mut acc = 0.0f32;
                for (a, b) in x.iter().zip(w.iter()) {
                    acc += a * b;
                }
                *slot = acc + bias.map_or(0.0, |b| b[r]);
            }
        });
    Ok(())
}

/// Causal multi-head attention over one sequence, in f32.
///
/// `q` is `[seq, num_heads * head_dim]`, `k` and `v` are
/// `[seq, num_kv_heads * head_dim]`, and grouped-query attention maps query
/// head `h` to key/value head `h / (num_heads / num_kv_heads)`.
///
/// Softmax is computed with the row maximum subtracted. A calibration sequence
/// is hundreds of tokens of real text, and the logits it produces are large
/// enough that the naive form overflows f32 — which would fill the energy
/// statistics with NaN and quantize the model against nothing.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    ensure!(
        num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads),
        "{num_heads} query heads do not divide into {num_kv_heads} key/value heads"
    );
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    ensure!(q.len() == seq * q_dim, "queries are not {seq}x{q_dim}");
    ensure!(k.len() == seq * kv_dim, "keys are not {seq}x{kv_dim}");
    ensure!(v.len() == seq * kv_dim, "values are not {seq}x{kv_dim}");
    ensure!(out.len() == seq * q_dim, "output is not {seq}x{q_dim}");

    let group = num_heads / num_kv_heads;

    // Parallel over (token, head): every pair writes one disjoint span.
    out.par_chunks_mut(q_dim)
        .enumerate()
        .for_each(|(t, row_out)| {
            let mut scores = vec![0.0f32; t + 1];
            for h in 0..num_heads {
                let kv_head = h / group;
                let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];

                let mut max = f32::NEG_INFINITY;
                for (u, score) in scores.iter_mut().enumerate() {
                    let kh =
                        &k[u * kv_dim + kv_head * head_dim..u * kv_dim + (kv_head + 1) * head_dim];
                    let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                    *score = dot * scale;
                    max = max.max(*score);
                }

                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                let dst = &mut row_out[h * head_dim..(h + 1) * head_dim];
                dst.fill(0.0);
                for (u, &weight) in scores.iter().enumerate() {
                    let w = weight * inv;
                    let vh =
                        &v[u * kv_dim + kv_head * head_dim..u * kv_dim + (kv_head + 1) * head_dim];
                    for (slot, &value) in dst.iter_mut().zip(vh.iter()) {
                        *slot += w * value;
                    }
                }
            }
        });
    Ok(())
}

/// Apply rotary position embedding in place to one sequence.
///
/// `x` is `[seq, heads * head_dim]`. The rotation pairs element `i` with
/// `i + head_dim/2`, which is the layout the engine's own RoPE tables and
/// kernels use — a calibration pass that paired them the other way would
/// measure a model that does not exist.
pub fn apply_rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let width = heads * head_dim;
    ensure!(x.len() == seq * width, "tensor is not {seq}x{width}");
    ensure!(
        head_dim.is_multiple_of(2),
        "head_dim {head_dim} must be even"
    );
    let half = head_dim / 2;
    ensure!(
        cos.len() >= seq * half && sin.len() >= seq * half,
        "RoPE tables are shorter than {seq} positions"
    );

    for t in 0..seq {
        for h in 0..heads {
            let base = t * width + h * head_dim;
            for i in 0..half {
                let c = cos[t * half + i];
                let s = sin[t * half + i];
                let a = x[base + i];
                let b = x[base + i + half];
                x[base + i] = a * c - b * s;
                x[base + i + half] = a * s + b * c;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_gram(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut h = vec![0.0f64; cols * cols];
        for r in 0..rows {
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += f64::from(x[r * cols + i]) * f64::from(x[r * cols + j]);
                }
            }
        }
        h
    }

    fn synthetic(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let v = ((i * 37 % 211) as f32 / 211.0) - 0.5;
                // A third of the entries are exactly zero, which exercises the
                // skip in the accumulator.
                if i % 3 == 0 {
                    0.0
                } else {
                    v * (1.0 + (i % 7) as f32 / 7.0)
                }
            })
            .collect()
    }

    /// The fast path must agree with the obvious one.
    ///
    /// Column counts are chosen to leave a tail the 4-wide loop cannot cover
    /// (13, 17) as well as exact multiples (16), because the tail is where an
    /// accumulator like this goes wrong.
    #[test]
    fn accumulated_gram_matches_the_reference_at_every_width() {
        for &(rows, cols) in &[(1usize, 13usize), (5, 16), (9, 17), (33, 64), (7, 1)] {
            let x = synthetic(rows, cols);
            let mut got = vec![0.0f64; cols * cols];
            accumulate_gram(&mut got, &x, rows, cols).unwrap();
            symmetrize(&mut got, cols);
            let want = reference_gram(&x, rows, cols);
            for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "{rows}x{cols} entry {index}: got {a}, want {b}"
                );
            }
        }
    }

    /// Accumulating two blocks must equal accumulating their concatenation.
    ///
    /// Calibration feeds the corpus in chunks, so this is the property that
    /// lets it: without it the Hessian would depend on the batch size.
    #[test]
    fn accumulation_is_independent_of_how_the_corpus_is_split() {
        let (cols, first, second) = (24usize, 11usize, 7usize);
        let x = synthetic(first + second, cols);

        let mut split = vec![0.0f64; cols * cols];
        accumulate_gram(&mut split, &x[..first * cols], first, cols).unwrap();
        accumulate_gram(&mut split, &x[first * cols..], second, cols).unwrap();

        let mut whole = vec![0.0f64; cols * cols];
        accumulate_gram(&mut whole, &x, first + second, cols).unwrap();

        for (a, b) in split.iter().zip(whole.iter()) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn a_symmetrized_gram_matrix_is_symmetric() {
        let cols = 19;
        let x = synthetic(6, cols);
        let mut h = vec![0.0f64; cols * cols];
        accumulate_gram(&mut h, &x, 6, cols).unwrap();
        symmetrize(&mut h, cols);
        for i in 0..cols {
            for j in 0..cols {
                assert_eq!(h[i * cols + j], h[j * cols + i], "entry ({i},{j})");
            }
        }
    }

    #[test]
    fn mismatched_buffer_sizes_are_refused() {
        let mut h = vec![0.0f64; 9];
        assert!(accumulate_gram(&mut h, &[1.0, 2.0], 1, 3).is_err());
        let mut h = vec![0.0f64; 4];
        assert!(accumulate_gram(&mut h, &[1.0, 2.0, 3.0], 1, 3).is_err());
    }

    /// Weights drawn from a bell curve, activations with a heavy tail.
    ///
    /// This is the shape that makes activation-aware quantization worth doing:
    /// a handful of input channels carry most of the energy, and a quantizer
    /// that treats every channel alike spends its range on the ones that do
    /// not matter. A uniform-energy fixture would show no difference and prove
    /// nothing.
    fn heavy_tailed_fixture(rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
        let mut weights = vec![0.0f64; rows * cols];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for value in weights.iter_mut() {
            // Sum of three uniforms: cheap, deterministic, roughly normal.
            *value = (next() + next() + next() - 1.5) * 0.1;
        }
        // One channel in sixteen carries two orders of magnitude more energy.
        let energy: Vec<f64> = (0..cols)
            .map(|c| if c % 16 == 0 { 100.0 } else { 1.0 })
            .collect();
        (weights, energy)
    }

    /// Error of plain round-to-nearest, measured with the same weighting the
    /// calibrated path optimizes, so the two are comparable.
    fn rtn_weighted_error(
        weights: &[f64],
        energy: &[f64],
        rows: usize,
        cols: usize,
        bits: u8,
        group_size: usize,
    ) -> f64 {
        let levels = f64::from(1u32 << bits);
        let mut total = 0.0;
        for r in 0..rows {
            let mut start = 0;
            while start < cols {
                let end = (start + group_size).min(cols);
                let group = &weights[r * cols + start..r * cols + end];
                let min = group.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = group.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let span = max - min;
                let scale = if span < 1e-15 {
                    1.0
                } else {
                    span / (levels - 1.0)
                };
                for (offset, &w) in group.iter().enumerate() {
                    let code = ((w - min) / scale).round().clamp(0.0, levels - 1.0);
                    let delta = w - (code * scale + min);
                    total += energy[start + offset] * delta * delta;
                }
                start = end;
            }
        }
        total
    }

    /// Weighted clipping must beat round-to-nearest on weighted error.
    ///
    /// This is the claim the default path rests on. If it fails, the extra
    /// machinery is not worth its cost and should be deleted rather than
    /// shipped.
    #[test]
    fn weighted_clipping_beats_round_to_nearest() {
        let (rows, cols, group_size, bits) = (32usize, 128usize, 32usize, 4u8);
        let (weights, energy) = heavy_tailed_fixture(rows, cols);

        let baseline = rtn_weighted_error(&weights, &energy, rows, cols, bits, group_size);

        let mut calibrated = 0.0;
        for r in 0..rows {
            let mut start = 0;
            while start < cols {
                let end = (start + group_size).min(cols);
                let group = &weights[r * cols + start..r * cols + end];
                let slice = &energy[start..end];
                let (scale, zero) = group_params_weighted(group, slice, bits, 8);
                calibrated +=
                    super::weighted_group_error(group, slice, scale, zero, f64::from(1u32 << bits));
                start = end;
            }
        }

        println!(
            "weighted error: round-to-nearest {baseline:.6}, calibrated {calibrated:.6} ({:.1}% lower)",
            (baseline - calibrated) / baseline * 100.0
        );
        assert!(
            calibrated < baseline,
            "weighted clipping did not improve on round-to-nearest: \
             {calibrated} vs {baseline}"
        );
    }

    /// Folding a channel scale into the preceding norm is exact.
    ///
    /// The whole activation-aware path depends on this identity: dividing the
    /// RMSNorm weight by `s` and multiplying the weight column by `s` must
    /// leave the layer's output unchanged, or the model has been altered
    /// rather than requantized. Checked in f64 so any real discrepancy shows
    /// rather than hiding under float noise.
    #[test]
    fn folding_a_channel_scale_into_the_norm_weight_is_exact() {
        let (rows, cols) = (5usize, 12usize);
        let (weights, energy) = heavy_tailed_fixture(rows, cols);
        let norm: Vec<f64> = (0..cols).map(|c| 0.5 + 0.1 * (c % 5) as f64).collect();
        let hidden: Vec<f64> = (0..cols).map(|c| 0.3 - 0.05 * (c % 7) as f64).collect();

        let scales = search_channel_scales(
            &weights,
            &energy,
            rows,
            cols,
            4,
            4,
            &[0.0, 0.25, 0.5, 0.75, 1.0],
        );
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));

        for r in 0..rows {
            let original: f64 = (0..cols)
                .map(|c| weights[r * cols + c] * norm[c] * hidden[c])
                .sum();
            let folded: f64 = (0..cols)
                .map(|c| (weights[r * cols + c] * scales[c]) * (norm[c] / scales[c]) * hidden[c])
                .sum();
            assert!(
                (original - folded).abs() <= 1e-12 * original.abs().max(1.0),
                "row {r}: folding changed the output, {original} vs {folded}"
            );
        }
    }

    /// A constant group must not produce a degenerate scale.
    ///
    /// Real checkpoints contain them — a pruned head, a zeroed expert — and a
    /// scale of zero would divide by zero on the next line and an f16
    /// subnormal would quantize everything to one code.
    #[test]
    fn a_constant_group_gets_a_usable_scale() {
        let weights = vec![0.25f64; 16];
        let energy = vec![1.0f64; 16];
        let (scale, zero) = group_params_weighted(&weights, &energy, 4, 8);
        assert!(scale > 0.0 && scale.is_finite(), "scale {scale}");
        assert!((zero - 0.25).abs() < 1e-12, "zero {zero}");
    }

    /// The calibration RoPE must be the engine's RoPE, bit for bit.
    ///
    /// The engine splits each head in half and rotates element `i` against
    /// `i + head_dim/2`. An implementation that instead pairs `(x[2i], x[2i+1])`
    /// — which the engine's own comment used to describe — produces a
    /// different model from the same weights, and calibration built on it would
    /// quantize against activations the engine never sees.
    #[test]
    fn calibration_rope_matches_the_engine() {
        let (heads, head_dim, seq) = (3usize, 8usize, 5usize);
        let table = crate::layers::RoPETable::new(head_dim, seq, 10000.0).expect("RoPE table");

        let width = heads * head_dim;
        let original: Vec<f32> = (0..seq * width)
            .map(|i| ((i % 23) as f32 / 23.0) - 0.5)
            .collect();

        // The engine rotates one position at a time, in place.
        let mut engine = original.clone();
        for t in 0..seq {
            table.apply(&mut engine[t * width..(t + 1) * width], heads, t);
        }

        // Calibration rotates the whole sequence at once.
        let mut ours = original.clone();
        let half = head_dim / 2;
        apply_rope(
            &mut ours,
            &table.cos[..seq * half],
            &table.sin[..seq * half],
            seq,
            heads,
            head_dim,
        )
        .unwrap();

        for (index, (a, b)) in ours.iter().zip(engine.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "element {index}: calibration {a}, engine {b}"
            );
        }
    }

    /// Attention must reduce to the obvious thing on a one-token sequence.
    ///
    /// With a single position the causal mask leaves one score, the softmax of
    /// one value is 1, and the output is exactly the value vector. Anything
    /// else means the mask, the softmax or the head indexing is wrong.
    #[test]
    fn single_token_attention_returns_the_value_vector() {
        let (heads, head_dim) = (2usize, 4usize);
        let q = vec![0.3f32; heads * head_dim];
        let k = vec![0.7f32; heads * head_dim];
        let v: Vec<f32> = (0..heads * head_dim).map(|i| i as f32 * 0.25).collect();
        let mut out = vec![0.0f32; heads * head_dim];

        causal_attention(&mut out, &q, &k, &v, 1, heads, heads, head_dim, 0.5).unwrap();
        for (index, (got, want)) in out.iter().zip(v.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "element {index}: {got} vs {want}"
            );
        }
    }

    /// A later token must never attend to an earlier one's future.
    ///
    /// Changing the last token's value vector must leave every earlier
    /// output untouched. A mask that leaks the future would raise perplexity
    /// only slightly while making the calibration statistics wrong in a way no
    /// aggregate number reveals.
    #[test]
    fn attention_is_causal() {
        let (seq, heads, head_dim) = (4usize, 2usize, 4usize);
        let width = heads * head_dim;
        let q: Vec<f32> = (0..seq * width).map(|i| (i % 7) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..seq * width).map(|i| (i % 5) as f32 * 0.2).collect();
        let mut v: Vec<f32> = (0..seq * width).map(|i| (i % 11) as f32 * 0.3).collect();

        let mut before = vec![0.0f32; seq * width];
        causal_attention(&mut before, &q, &k, &v, seq, heads, heads, head_dim, 0.25).unwrap();

        for slot in v[(seq - 1) * width..].iter_mut() {
            *slot += 100.0;
        }
        let mut after = vec![0.0f32; seq * width];
        causal_attention(&mut after, &q, &k, &v, seq, heads, heads, head_dim, 0.25).unwrap();

        for t in 0..seq - 1 {
            for i in 0..width {
                let (a, b) = (before[t * width + i], after[t * width + i]);
                assert!(
                    (a - b).abs() < 1e-6,
                    "token {t} saw the future: {a} became {b}"
                );
            }
        }
    }

    /// Grouped-query attention must map query heads onto the right KV head.
    ///
    /// With two query heads per KV head, heads 0 and 1 share KV head 0. Giving
    /// the two KV heads clearly different values makes a mis-mapping obvious
    /// instead of merely slightly wrong.
    #[test]
    fn grouped_query_heads_read_their_own_key_value_head() {
        let (heads, kv_heads, head_dim) = (4usize, 2usize, 4usize);
        let q = vec![1.0f32; heads * head_dim];
        let k = vec![1.0f32; kv_heads * head_dim];
        let mut v = vec![0.0f32; kv_heads * head_dim];
        v[..head_dim].fill(2.0);
        v[head_dim..].fill(9.0);

        let mut out = vec![0.0f32; heads * head_dim];
        causal_attention(&mut out, &q, &k, &v, 1, heads, kv_heads, head_dim, 1.0).unwrap();

        for h in 0..heads {
            let want = if h < 2 { 2.0 } else { 9.0 };
            for i in 0..head_dim {
                let got = out[h * head_dim + i];
                assert!(
                    (got - want).abs() < 1e-6,
                    "head {h} element {i}: {got} vs {want}"
                );
            }
        }
    }

    /// The f32 matmul must agree with a plain triple loop, bias included.
    #[test]
    fn matmul_matches_the_obvious_loop() {
        let (tokens, rows, cols) = (5usize, 7usize, 9usize);
        let input: Vec<f32> = (0..tokens * cols)
            .map(|i| (i % 13) as f32 * 0.1 - 0.5)
            .collect();
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| (i % 17) as f32 * 0.05 - 0.3)
            .collect();
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();

        let mut got = vec![0.0f32; tokens * rows];
        matmul_t(&mut got, &input, &weights, Some(&bias), tokens, rows, cols).unwrap();

        for t in 0..tokens {
            for r in 0..rows {
                let mut want = bias[r];
                for c in 0..cols {
                    want += input[t * cols + c] * weights[r * cols + c];
                }
                let g = got[t * rows + r];
                assert!((g - want).abs() <= 1e-5, "({t},{r}): {g} vs {want}");
            }
        }
    }

    #[test]
    fn averaging_over_zero_tokens_is_refused() {
        let mut h = vec![1.0f64; 4];
        assert!(scale_by_tokens(&mut h, 0).is_err());
        assert!(scale_by_tokens(&mut h, 2).is_ok());
        assert_eq!(h[0], 0.5);
    }
}
