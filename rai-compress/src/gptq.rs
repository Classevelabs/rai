//! GPTQ: Calibration-based weight quantization guided by the Hessian.
//!
//! GPTQ quantizes weight matrices column-by-column, using the inverse Hessian
//! (H = X^T @ X from calibration data) to propagate each column's quantization
//! error to remaining columns. This ensures that errors in "important" weight
//! directions (those multiplied by large activations) are minimized.
//!
//! Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
//! Generative Pre-trained Transformers" (2022).

use nalgebra::DMatrix;

/// Parameters for a quantization group (per-row, per group of columns).
#[derive(Debug, Clone)]
pub struct GroupParams {
    pub scale: f64,
    pub zero_point: f64,
}

/// Statistics from GPTQ quantization.
#[derive(Debug, Clone)]
pub struct GptqStats {
    pub mse: f64,
    pub max_error: f64,
    pub bits_per_weight: f64,
    pub compressed_bytes: usize,
}

/// Result of GPTQ quantization.
#[derive(Debug, Clone)]
pub struct GptqResult {
    /// Quantized codes, stored as [row * cols + col] (row-major).
    pub quantized_codes: Vec<u32>,
    /// Per-row-per-group quantization parameters, indexed as [gid * rows + row].
    pub group_params: Vec<GroupParams>,
    /// Bits per weight.
    pub bits: u8,
    /// Matrix dimensions.
    pub rows: usize,
    pub cols: usize,
    /// Group size used.
    pub group_size: usize,
    /// Compression statistics.
    pub stats: GptqStats,
}

impl GptqResult {
    /// Compute compressed size in bytes.
    ///
    /// Data: rows * cols * bits / 8
    /// Metadata: rows * num_groups * 4 bytes (FP16 scale + FP16 zero per row per group)
    /// At 4-bit, group_size=128: metadata overhead ~ 0.25 bpw
    pub fn compressed_bytes(&self) -> usize {
        let data_bits = self.rows * self.cols * self.bits as usize;
        let data_bytes = data_bits.div_ceil(8);
        let num_groups = self.cols.div_ceil(self.group_size);
        let meta_bytes = self.rows * num_groups * 4;
        data_bytes + meta_bytes
    }
}

/// GPTQ quantization of a weight matrix guided by the Hessian.
///
/// # Arguments
/// * `weights` - Weight matrix \[out_features, in_features\]
/// * `hessian` - Hessian matrix \[in_features, in_features\] (H = X^T @ X / n_samples)
/// * `bits` - Number of bits per weight (2-8)
/// * `block_size` - Column block size for blocked error propagation (typically 128)
/// * `group_size` - Number of columns sharing quantization parameters per row (typically 128)
pub fn gptq_quantize(
    weights: &DMatrix<f64>,
    hessian: &DMatrix<f64>,
    bits: u8,
    block_size: usize,
    group_size: usize,
) -> GptqResult {
    let rows = weights.nrows();
    let cols = weights.ncols();
    assert_eq!(hessian.nrows(), cols, "Hessian rows must match weight cols");
    assert_eq!(hessian.ncols(), cols, "Hessian must be square");
    assert!((2..=8).contains(&bits), "bits must be 2-8");

    let levels = (1u32 << bits) as f64;

    // Working copy of weights (modified during error propagation)
    let mut w = weights.clone();

    let num_groups = cols.div_ceil(group_size);

    // Group params will be computed lazily from current (error-compensated) weights
    // at the start of each group. Indexed as group_params[gid * rows + r].
    let mut group_params = vec![
        GroupParams {
            scale: 1.0,
            zero_point: 0.0
        };
        num_groups * rows
    ];

    // Damp the Hessian: H += 0.01 * mean(diag(H)) * I
    let mut h = hessian.clone();
    let diag_mean = h.diagonal().iter().sum::<f64>() / cols as f64;
    let damp = 0.01 * diag_mean.max(1e-6);
    for i in 0..cols {
        h[(i, i)] += damp;
    }

    // Cholesky decomposition and inverse
    let chol = h
        .cholesky()
        .expect("Damped Hessian must be positive definite");
    let h_inv = chol.inverse();

    // Output quantized codes
    let mut codes = vec![0u32; rows * cols];

    // Process columns in blocks
    let mut block_start = 0;
    while block_start < cols {
        let block_end = (block_start + block_size).min(cols);
        let bsize = block_end - block_start;

        // Error accumulator for inter-block update: [rows, bsize]
        let mut err_block = DMatrix::zeros(rows, bsize);

        for j in 0..bsize {
            let col = block_start + j;
            let gid = col / group_size;
            let d = h_inv[(col, col)];

            // At the start of each group, compute scale/zero from CURRENT
            // (error-compensated) weight values. This is critical: after error
            // propagation, values may drift from original range. Using current
            // values avoids clipping waste and matches AutoGPTQ behavior.
            if col % group_size == 0 {
                let g_end = (col + group_size).min(cols);
                for r in 0..rows {
                    let mut min_val = f64::INFINITY;
                    let mut max_val = f64::NEG_INFINITY;
                    for c in col..g_end {
                        let v = w[(r, c)];
                        min_val = min_val.min(v);
                        max_val = max_val.max(v);
                    }
                    let range = max_val - min_val;
                    let scale = if range < 1e-15 {
                        1.0
                    } else {
                        range / (levels - 1.0)
                    };
                    group_params[gid * rows + r] = GroupParams {
                        scale,
                        zero_point: min_val,
                    };
                }
            }

            for r in 0..rows {
                let gp = &group_params[gid * rows + r];
                let val = w[(r, col)];

                // Quantize
                let q = ((val - gp.zero_point) / gp.scale)
                    .round()
                    .max(0.0)
                    .min(levels - 1.0) as u32;
                codes[r * cols + col] = q;

                // Dequantize
                let w_hat = q as f64 * gp.scale + gp.zero_point;

                // Error divided by diagonal Hessian inverse element
                let err = (val - w_hat) / d;
                err_block[(r, j)] = err;

                // Propagate error to remaining columns within this block
                for k in (j + 1)..bsize {
                    w[(r, block_start + k)] -= err * h_inv[(col, block_start + k)];
                }
            }
        }

        // Inter-block error propagation:
        // W[:, block_end:] -= err_block @ H_inv[block_start:block_end, block_end:]
        if block_end < cols {
            let remaining = cols - block_end;
            let h_inv_cross = h_inv.view((block_start, block_end), (bsize, remaining));
            let update = &err_block * h_inv_cross;
            for r in 0..rows {
                for c in 0..remaining {
                    w[(r, block_end + c)] -= update[(r, c)];
                }
            }
        }

        block_start = block_end;
    }

    // Compute MSE vs original weights
    let mut total_sq_err = 0.0f64;
    let mut max_err = 0.0f64;
    for c in 0..cols {
        let gid = c / group_size;
        for r in 0..rows {
            let gp = &group_params[gid * rows + r];
            let orig = weights[(r, c)];
            let q = codes[r * cols + c];
            let recon = q as f64 * gp.scale + gp.zero_point;
            let e = (orig - recon).abs();
            total_sq_err += e * e;
            max_err = max_err.max(e);
        }
    }
    let n = (rows * cols) as f64;
    let mse = total_sq_err / n;

    let compressed_bytes = {
        let data_bits = rows * cols * bits as usize;
        let data_bytes = data_bits.div_ceil(8);
        let meta_bytes = rows * num_groups * 4;
        data_bytes + meta_bytes
    };
    let bits_per_weight = (compressed_bytes * 8) as f64 / n;

    GptqResult {
        quantized_codes: codes,
        group_params,
        bits,
        rows,
        cols,
        group_size,
        stats: GptqStats {
            mse,
            max_error: max_err,
            bits_per_weight,
            compressed_bytes,
        },
    }
}

/// Compute Hessian-weighted MSE: trace((W - Q)^T @ (W - Q) @ H) / (rows * cols).
///
/// This is the actual metric GPTQ optimizes — it measures output error, not raw
/// weight error. Columns multiplied by large activations (high H diagonal) contribute
/// more to this metric.
pub fn hessian_weighted_mse(
    original: &DMatrix<f64>,
    quantized: &DMatrix<f64>,
    hessian: &DMatrix<f64>,
) -> f64 {
    let rows = original.nrows();
    let cols = original.ncols();
    let mut total = 0.0f64;
    for r in 0..rows {
        // Error vector for this row
        let err: Vec<f64> = (0..cols)
            .map(|c| original[(r, c)] - quantized[(r, c)])
            .collect();
        // Quadratic form: e^T @ H @ e
        for i in 0..cols {
            for j in 0..cols {
                total += err[i] * hessian[(i, j)] * err[j];
            }
        }
    }
    total / (rows * cols) as f64
}

/// Decompress a GPTQ result back to a dense matrix.
pub fn gptq_decompress(result: &GptqResult) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(result.rows, result.cols);
    for c in 0..result.cols {
        let gid = c / result.group_size;
        for r in 0..result.rows {
            let gp = &result.group_params[gid * result.rows + r];
            let q = result.quantized_codes[r * result.cols + c];
            out[(r, c)] = q as f64 * gp.scale + gp.zero_point;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::compress_uniform_4bit;
    use rand::Rng;

    /// With identity Hessian, GPTQ should degrade gracefully.
    /// H_inv = I means no cross-column error propagation, so GPTQ ~ uniform.
    #[test]
    fn identity_hessian_degrades_gracefully() {
        let mut rng = rand::thread_rng();
        let rows = 64;
        let cols = 128;
        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 4, 128, 128);

        assert!(
            result.stats.mse < 0.1,
            "MSE too high with identity H: {}",
            result.stats.mse
        );
        assert_eq!(result.rows, rows);
        assert_eq!(result.cols, cols);

        // With identity H, GPTQ should be roughly similar to uniform
        let uniform = compress_uniform_4bit(&weights, 64);
        let ratio = result.stats.mse / uniform.mse;
        println!(
            "Identity H: GPTQ MSE={:.6e}, Uniform MSE={:.6e}, ratio={:.2}",
            result.stats.mse, uniform.mse, ratio
        );
        assert!(
            ratio < 2.0,
            "GPTQ shouldn't be much worse than uniform with identity H (ratio={ratio:.2})"
        );
    }

    /// With a correlated Hessian, GPTQ should beat uniform on Hessian-weighted MSE.
    /// This is the key validation: calibration-guided quantization > naive quantization.
    ///
    /// GPTQ optimizes output error (Hessian-weighted MSE), not raw weight MSE.
    /// We verify GPTQ achieves lower Hessian-weighted error by redistributing
    /// quantization error from important to unimportant weight directions.
    #[test]
    fn correlated_hessian_beats_uniform() {
        let rows = 256;
        let cols = 256;

        // Simulate activations where some features are 10x more active.
        let x = DMatrix::from_fn(2048, cols, |i, j| {
            let base = ((i as f64 * 0.01 + j as f64 * 0.03).sin()) * 0.5;
            if j < 64 {
                base * 10.0
            } else {
                base
            }
        });
        let hessian = x.transpose() * &x / 2048.0;

        // Weights with realistic structure
        let weights = DMatrix::from_fn(rows, cols, |i, j| {
            ((i as f64 * 0.07 + j as f64 * 0.03).sin()) * 0.5
                + ((i as f64 * 0.11 + j as f64 * 0.17).cos()) * 0.3
                + ((i * 7 + j * 13) as f64 * 0.001).sin() * 0.05
        });

        let gptq_result = gptq_quantize(&weights, &hessian, 4, 128, 128);
        let gptq_decompressed = gptq_decompress(&gptq_result);

        // Reconstruct uniform quantized matrix for weighted comparison
        let flat: Vec<f64> = weights.iter().cloned().collect();
        let mut uniform_recon = vec![0.0f64; rows * cols];
        for (i, chunk) in flat.chunks(128).enumerate() {
            let block = crate::quantize::quantize_uniform(chunk, 4);
            let recovered = crate::quantize::dequantize(&block);
            let start = i * 128;
            for (j, &v) in recovered.iter().enumerate() {
                if start + j < uniform_recon.len() {
                    uniform_recon[start + j] = v;
                }
            }
        }
        let uniform_mat = DMatrix::from_iterator(rows, cols, uniform_recon.iter().cloned());

        let gptq_wmse = hessian_weighted_mse(&weights, &gptq_decompressed, &hessian);
        let uniform_wmse = hessian_weighted_mse(&weights, &uniform_mat, &hessian);
        let improvement = uniform_wmse / gptq_wmse;

        println!(
            "Hessian-weighted MSE: GPTQ={:.6e}, Uniform={:.6e}, GPTQ is {:.2}x better",
            gptq_wmse, uniform_wmse, improvement
        );
        println!(
            "Raw MSE: GPTQ={:.6e}, Uniform={:.6e}",
            gptq_result.stats.mse,
            weights
                .iter()
                .zip(uniform_mat.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / (rows * cols) as f64
        );

        assert!(
            gptq_wmse < uniform_wmse,
            "GPTQ weighted MSE ({:.2e}) should beat uniform ({:.2e})",
            gptq_wmse,
            uniform_wmse
        );
    }

    /// 3-bit quantization should work without panics.
    #[test]
    fn three_bit_works() {
        let mut rng = rand::thread_rng();
        let rows = 64;
        let cols = 128;
        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 3, 128, 128);

        assert_eq!(result.bits, 3);
        assert!(
            result.stats.mse < 1.0,
            "3-bit MSE too high: {}",
            result.stats.mse
        );
        // All codes should be in range [0, 7]
        assert!(result.quantized_codes.iter().all(|&c| c < 8));
    }

    /// Decompress roundtrip should reproduce quantized values exactly.
    #[test]
    fn decompress_roundtrip() {
        let mut rng = rand::thread_rng();
        let rows = 32;
        let cols = 64;

        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-2.0..2.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 4, 64, 64);
        let decompressed = gptq_decompress(&result);

        // Decompressed should exactly match quantized/dequantized values
        let mut max_diff = 0.0f64;
        for c in 0..cols {
            let gid = c / result.group_size;
            for r in 0..rows {
                let gp = &result.group_params[gid * rows + r];
                let q = result.quantized_codes[r * cols + c];
                let expected = q as f64 * gp.scale + gp.zero_point;
                let got = decompressed[(r, c)];
                max_diff = max_diff.max((expected - got).abs());
            }
        }

        assert!(max_diff < 1e-12, "Decompress roundtrip error: {max_diff}");
    }
}
