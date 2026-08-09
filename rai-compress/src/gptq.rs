//! GPTQ: Calibration-based weight quantization guided by the Hessian.
//!
//! GPTQ quantizes weight matrices column-by-column, propagating each column's
//! quantization error to the remaining columns through the upper-triangular
//! Cholesky factor U of the inverse Hessian (H = X^T @ X from calibration
//! data, H^-1 = U^T @ U). This ensures that errors in "important" weight
//! directions (those multiplied by large activations) are minimized.
//!
//! Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
//! Generative Pre-trained Transformers" (2022).

use nalgebra::DMatrix;

/// Errors returned when caller input or a serialized GPTQ representation is invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GptqError {
    #[error("invalid GPTQ input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid GPTQ representation: {0}")]
    InvalidRepresentation(&'static str),
    #[error("GPTQ dimensions overflow")]
    SizeOverflow,
    #[error("unable to allocate GPTQ output")]
    AllocationFailed,
    #[error("damped Hessian is not positive definite")]
    HessianNotPositiveDefinite,
    #[error("GPTQ numerical failure: {0}")]
    NumericalFailure(&'static str),
}

fn checked_num_groups(cols: usize, group_size: usize) -> Result<usize, GptqError> {
    if group_size == 0 {
        return Err(GptqError::InvalidInput("group_size must be non-zero"));
    }
    Ok(cols / group_size + usize::from(!cols.is_multiple_of(group_size)))
}

fn modeled_compressed_bytes(
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
) -> Result<usize, GptqError> {
    if rows == 0 || cols == 0 {
        return Err(GptqError::InvalidRepresentation(
            "matrix shape must be non-zero",
        ));
    }
    if !(2..=8).contains(&bits) {
        return Err(GptqError::InvalidRepresentation("bits must be 2-8"));
    }
    let data_bits = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(bits as usize))
        .ok_or(GptqError::SizeOverflow)?;
    let data_bytes = data_bits.checked_add(7).ok_or(GptqError::SizeOverflow)? / 8;
    let num_groups = checked_num_groups(cols, group_size)?;
    let meta_bytes = rows
        .checked_mul(num_groups)
        .and_then(|value| value.checked_mul(4))
        .ok_or(GptqError::SizeOverflow)?;
    data_bytes
        .checked_add(meta_bytes)
        .ok_or(GptqError::SizeOverflow)
}

fn validate_result(result: &GptqResult) -> Result<(), GptqError> {
    let elements = result
        .rows
        .checked_mul(result.cols)
        .ok_or(GptqError::SizeOverflow)?;
    if result.rows == 0 || result.cols == 0 {
        return Err(GptqError::InvalidRepresentation(
            "matrix shape must be non-zero",
        ));
    }
    if !(2..=8).contains(&result.bits) {
        return Err(GptqError::InvalidRepresentation("bits must be 2-8"));
    }
    let groups = checked_num_groups(result.cols, result.group_size)
        .map_err(|_| GptqError::InvalidRepresentation("group_size must be non-zero"))?;
    if result.quantized_codes.len() != elements {
        return Err(GptqError::InvalidRepresentation("code length mismatch"));
    }
    let parameter_count = result
        .rows
        .checked_mul(groups)
        .ok_or(GptqError::SizeOverflow)?;
    if result.group_params.len() != parameter_count {
        return Err(GptqError::InvalidRepresentation(
            "parameter length mismatch",
        ));
    }
    let max_code = (1u32 << result.bits) - 1;
    if result.quantized_codes.iter().any(|&code| code > max_code) {
        return Err(GptqError::InvalidRepresentation(
            "quantized code exceeds configured bit width",
        ));
    }
    if result.group_params.iter().any(|params| {
        !params.scale.is_finite() || params.scale <= 0.0 || !params.zero_point.is_finite()
    }) {
        return Err(GptqError::InvalidRepresentation(
            "parameters must be finite with positive scales",
        ));
    }
    Ok(())
}

/// Parameters for a quantization group (per-row, per group of columns).
#[derive(Debug, Clone)]
pub struct GroupParams {
    pub scale: f64,
    pub zero_point: f64,
}

/// Modeled GPTQ statistics. Byte counts assume future bit-packing and FP16 metadata;
/// `GptqResult` currently stores unpacked u32 codes and f64 parameters.
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
    pub fn compressed_bytes(&self) -> Result<usize, GptqError> {
        validate_result(self)?;
        modeled_compressed_bytes(self.rows, self.cols, self.bits, self.group_size)
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
) -> Result<GptqResult, GptqError> {
    let rows = weights.nrows();
    let cols = weights.ncols();
    if rows == 0 || cols == 0 {
        return Err(GptqError::InvalidInput("weights must not be empty"));
    }
    if block_size == 0 {
        return Err(GptqError::InvalidInput("block_size must be non-zero"));
    }
    if group_size == 0 {
        return Err(GptqError::InvalidInput("group_size must be non-zero"));
    }
    if hessian.shape() != (cols, cols) {
        return Err(GptqError::InvalidInput(
            "Hessian shape must match the weight columns",
        ));
    }
    if !(2..=8).contains(&bits) {
        return Err(GptqError::InvalidInput("bits must be 2-8"));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::InvalidInput("weights must be finite"));
    }
    if hessian.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::InvalidInput("Hessian must be finite"));
    }
    // Symmetry is checked with a RELATIVE tolerance: H is accumulated as
    // X^T @ X, so its float asymmetry scales with the magnitude of its
    // entries and an absolute threshold rejects legitimately symmetric
    // Hessians with large entries. The small absolute floor keeps an
    // all-zero Hessian symmetric (damping decides its fate later).
    let max_abs_entry = hessian
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    let symmetry_tolerance = (1e-8 * max_abs_entry).max(1e-12);
    if !(0..cols).all(|row| {
        (0..cols).all(|col| (hessian[(row, col)] - hessian[(col, row)]).abs() <= symmetry_tolerance)
    }) {
        return Err(GptqError::InvalidInput("Hessian must be symmetric"));
    }

    let levels = (1u32 << bits) as f64;

    // Working copy of weights (modified during error propagation)
    let mut w = weights.clone();

    let num_groups = checked_num_groups(cols, group_size)?;
    let group_param_count = num_groups
        .checked_mul(rows)
        .ok_or(GptqError::SizeOverflow)?;

    // Group params will be computed lazily from current (error-compensated) weights
    // at the start of each group. Indexed as group_params[gid * rows + r].
    let mut group_params = Vec::new();
    group_params
        .try_reserve_exact(group_param_count)
        .map_err(|_| GptqError::AllocationFailed)?;
    group_params.resize(
        group_param_count,
        GroupParams {
            scale: 1.0,
            zero_point: 0.0,
        },
    );

    // Damp the Hessian: H += 0.01 * mean(diag(H)) * I
    let mut h = hessian.clone();
    let diag_mean = h.diagonal().iter().sum::<f64>() / cols as f64;
    let damp = 0.01 * diag_mean.max(1e-6);
    for i in 0..cols {
        h[(i, i)] += damp;
    }

    // GPTQ (Frantar et al., 2022, Algorithm 1) operates on the
    // UPPER-TRIANGULAR Cholesky factor U of the inverse Hessian
    // (H^-1 = U^T @ U), not on H^-1 itself: the per-column denominator is
    // U[j, j] and the error-propagation row is U[j, j+1..]. Compute H^-1
    // through the Cholesky solve, then factor it a second time.
    let chol = h.cholesky().ok_or(GptqError::HessianNotPositiveDefinite)?;
    let mut h_inv = chol.inverse();
    if h_inv.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::NumericalFailure(
            "inverse Hessian contains non-finite values",
        ));
    }
    // Symmetrize before factorizing: the triangular solves behind `inverse`
    // can leave tiny asymmetries that break the second Cholesky.
    for r in 0..cols {
        for c in (r + 1)..cols {
            let mean = 0.5 * (h_inv[(r, c)] + h_inv[(c, r)]);
            h_inv[(r, c)] = mean;
            h_inv[(c, r)] = mean;
        }
    }
    let u = h_inv
        .cholesky()
        .ok_or(GptqError::NumericalFailure(
            "Cholesky factorization of the inverse Hessian failed",
        ))?
        .l()
        .transpose();

    // Output quantized codes
    let code_count = rows.checked_mul(cols).ok_or(GptqError::SizeOverflow)?;
    let mut codes = Vec::new();
    codes
        .try_reserve_exact(code_count)
        .map_err(|_| GptqError::AllocationFailed)?;
    codes.resize(code_count, 0u32);

    // Process columns in blocks
    let mut block_start = 0;
    while block_start < cols {
        let block_end = block_start.saturating_add(block_size).min(cols);
        let bsize = block_end - block_start;

        // Error accumulator for inter-block update: [rows, bsize]
        let mut err_block = DMatrix::zeros(rows, bsize);

        for j in 0..bsize {
            let col = block_start + j;
            let gid = col / group_size;
            let d = u[(col, col)];
            if !d.is_finite() || d <= 0.0 {
                return Err(GptqError::NumericalFailure(
                    "Cholesky factor diagonal is not positive and finite",
                ));
            }

            // At the start of each group, compute scale/zero from CURRENT
            // (error-compensated) weight values. This is critical: after error
            // propagation, values may drift from original range. Using current
            // values avoids clipping waste and matches AutoGPTQ behavior.
            if col % group_size == 0 {
                let g_end = col.saturating_add(group_size).min(cols);
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

                // Error divided by the Cholesky factor's diagonal element
                let err = (val - w_hat) / d;
                if !err.is_finite() {
                    return Err(GptqError::NumericalFailure(
                        "error propagation produced a non-finite value",
                    ));
                }
                err_block[(r, j)] = err;

                // Propagate error to remaining columns within this block
                for k in (j + 1)..bsize {
                    let next = w[(r, block_start + k)] - err * u[(col, block_start + k)];
                    if !next.is_finite() {
                        return Err(GptqError::NumericalFailure(
                            "error propagation produced a non-finite weight",
                        ));
                    }
                    w[(r, block_start + k)] = next;
                }
            }
        }

        // Inter-block error propagation:
        // W[:, block_end:] -= err_block @ U[block_start:block_end, block_end:]
        if block_end < cols {
            let remaining = cols - block_end;
            let u_cross = u.view((block_start, block_end), (bsize, remaining));
            let update = &err_block * u_cross;
            for r in 0..rows {
                for c in 0..remaining {
                    let next = w[(r, block_end + c)] - update[(r, c)];
                    if !next.is_finite() {
                        return Err(GptqError::NumericalFailure(
                            "block update produced a non-finite weight",
                        ));
                    }
                    w[(r, block_end + c)] = next;
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
    if !total_sq_err.is_finite() || !max_err.is_finite() {
        return Err(GptqError::NumericalFailure(
            "quantization error statistics are non-finite",
        ));
    }
    let n = code_count as f64;
    let mse = total_sq_err / n;

    let compressed_bytes = modeled_compressed_bytes(rows, cols, bits, group_size)?;
    let bits_per_weight = compressed_bytes as f64 * 8.0 / n;

    Ok(GptqResult {
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
    })
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
) -> Result<f64, GptqError> {
    let rows = original.nrows();
    let cols = original.ncols();
    if rows == 0 || cols == 0 {
        return Err(GptqError::InvalidInput("matrices must not be empty"));
    }
    if quantized.shape() != original.shape() {
        return Err(GptqError::InvalidInput("quantized shape mismatch"));
    }
    if hessian.shape() != (cols, cols) {
        return Err(GptqError::InvalidInput("Hessian shape mismatch"));
    }
    if original.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::InvalidInput("original values must be finite"));
    }
    if quantized.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::InvalidInput("quantized values must be finite"));
    }
    if hessian.iter().any(|value| !value.is_finite()) {
        return Err(GptqError::InvalidInput("Hessian values must be finite"));
    }
    // trace(E^T @ E @ H) = sum over rows r of e_r^T @ H @ e_r, computed as
    // sum((E @ H) .* E) so the O(rows * cols^2) work goes through nalgebra's
    // matrix multiply instead of a scalar triple loop.
    let err = original - quantized;
    let total = (&err * hessian).component_mul(&err).sum();
    let elements = rows.checked_mul(cols).ok_or(GptqError::SizeOverflow)?;
    let value = total / elements as f64;
    if !value.is_finite() {
        return Err(GptqError::NumericalFailure("weighted error is non-finite"));
    }
    Ok(value)
}

/// Decompress a GPTQ result back to a dense matrix.
pub fn gptq_decompress(result: &GptqResult) -> Result<DMatrix<f64>, GptqError> {
    validate_result(result)?;
    let mut out = DMatrix::zeros(result.rows, result.cols);
    for c in 0..result.cols {
        let gid = c / result.group_size;
        for r in 0..result.rows {
            let gp = &result.group_params[gid * result.rows + r];
            let q = result.quantized_codes[r * result.cols + c];
            let value = q as f64 * gp.scale + gp.zero_point;
            if !value.is_finite() {
                return Err(GptqError::NumericalFailure(
                    "decompression produced a non-finite value",
                ));
            }
            out[(r, c)] = value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::compress_uniform_4bit;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// With identity Hessian, GPTQ should degrade gracefully.
    /// The Cholesky factor of H^-1 is diagonal, so there is no cross-column
    /// error propagation and GPTQ ~ uniform.
    #[test]
    fn identity_hessian_degrades_gracefully() {
        let mut rng = StdRng::seed_from_u64(0x5EED_1001);
        let rows = 32;
        let cols = 64;
        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 4, 32, 32).unwrap();

        assert!(
            result.stats.mse < 0.1,
            "MSE too high with identity H: {}",
            result.stats.mse
        );
        assert_eq!(result.rows, rows);
        assert_eq!(result.cols, cols);

        // With identity H, GPTQ should be roughly similar to uniform
        let uniform = compress_uniform_4bit(&weights, 32).unwrap();
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
        let rows = 64;
        let cols = 64;

        // Simulate activations where some features are 10x more active.
        let x = DMatrix::from_fn(512, cols, |i, j| {
            let base = ((i as f64 * 0.01 + j as f64 * 0.03).sin()) * 0.5;
            if j < 16 {
                base * 10.0
            } else {
                base
            }
        });
        let hessian = x.transpose() * &x / 512.0;

        // Weights with realistic structure
        let weights = DMatrix::from_fn(rows, cols, |i, j| {
            ((i as f64 * 0.07 + j as f64 * 0.03).sin()) * 0.5
                + ((i as f64 * 0.11 + j as f64 * 0.17).cos()) * 0.3
                + ((i * 7 + j * 13) as f64 * 0.001).sin() * 0.05
        });

        let gptq_result = gptq_quantize(&weights, &hessian, 4, 32, 32).unwrap();
        let gptq_decompressed = gptq_decompress(&gptq_result).unwrap();

        // Reconstruct uniform quantized matrix for weighted comparison
        let flat: Vec<f64> = weights.iter().cloned().collect();
        let mut uniform_recon = vec![0.0f64; rows * cols];
        for (i, chunk) in flat.chunks(32).enumerate() {
            let block = crate::quantize::quantize_uniform(chunk, 4).unwrap();
            let recovered = crate::quantize::dequantize(&block).unwrap();
            let start = i * 32;
            for (j, &v) in recovered.iter().enumerate() {
                if start + j < uniform_recon.len() {
                    uniform_recon[start + j] = v;
                }
            }
        }
        let uniform_mat = DMatrix::from_iterator(rows, cols, uniform_recon.iter().cloned());

        let gptq_wmse = hessian_weighted_mse(&weights, &gptq_decompressed, &hessian).unwrap();
        let uniform_wmse = hessian_weighted_mse(&weights, &uniform_mat, &hessian).unwrap();
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

    /// The matrix-product form of `hessian_weighted_mse`
    /// (sum((E @ H) .* E) / n) must agree with the definitional scalar form
    /// (sum over rows of e_r^T @ H @ e_r, divided by n) to float precision.
    #[test]
    fn hessian_weighted_mse_matches_scalar_reference() {
        let mut rng = StdRng::seed_from_u64(0x5EED_1004);
        let rows = 12;
        let cols = 17;
        let original = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let quantized = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        // Symmetric PSD-ish Hessian from random activations.
        let x = DMatrix::from_fn(40, cols, |_, _| rng.gen_range(-1.0..1.0));
        let hessian = x.transpose() * &x / 40.0;

        let fast = hessian_weighted_mse(&original, &quantized, &hessian).unwrap();

        let mut scalar_total = 0.0f64;
        for r in 0..rows {
            let err: Vec<f64> = (0..cols)
                .map(|c| original[(r, c)] - quantized[(r, c)])
                .collect();
            for i in 0..cols {
                for j in 0..cols {
                    scalar_total += err[i] * hessian[(i, j)] * err[j];
                }
            }
        }
        let scalar = scalar_total / (rows * cols) as f64;

        let denom = scalar.abs().max(1e-30);
        assert!(
            ((fast - scalar) / denom).abs() < 1e-12,
            "matrix form ({fast:.17e}) must match scalar form ({scalar:.17e})"
        );
    }

    /// With an isotropic Hessian H = c * I, the Cholesky factor of H^-1 is
    /// diagonal, so cross-column error propagation is exactly zero and GPTQ
    /// must bit-match plain round-to-nearest quantization with the same
    /// per-row-per-group parameters. (The Python exporter test suite asserts
    /// the same equivalence.)
    #[test]
    fn isotropic_hessian_matches_round_to_nearest() {
        let rows = 16;
        let cols = 100; // ragged final group: 100 = 3 * 32 + 4
        let group_size = 32;
        let bits = 4u8;
        let levels = (1u32 << bits) as f64;
        let weights = DMatrix::from_fn(rows, cols, |i, j| {
            ((i as f64 * 0.37 + j as f64 * 0.19).sin()) * 1.3
                + ((i * 3 + j * 7) as f64 * 0.01).cos() * 0.4
        });
        let hessian = DMatrix::from_diagonal_element(cols, cols, 3.7);

        let result = gptq_quantize(&weights, &hessian, bits, 64, group_size).unwrap();

        let num_groups = checked_num_groups(cols, group_size).unwrap();
        for gid in 0..num_groups {
            let g_start = gid * group_size;
            let g_end = (g_start + group_size).min(cols);
            for r in 0..rows {
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for c in g_start..g_end {
                    min_val = min_val.min(weights[(r, c)]);
                    max_val = max_val.max(weights[(r, c)]);
                }
                let range = max_val - min_val;
                let scale = if range < 1e-15 {
                    1.0
                } else {
                    range / (levels - 1.0)
                };
                let gp = &result.group_params[gid * rows + r];
                assert_eq!(
                    gp.scale, scale,
                    "scale mismatch for row {r} group {gid}: isotropic-Hessian GPTQ must use RTN group params"
                );
                assert_eq!(
                    gp.zero_point, min_val,
                    "zero-point mismatch for row {r} group {gid}"
                );
                for c in g_start..g_end {
                    let expected = ((weights[(r, c)] - min_val) / scale)
                        .round()
                        .max(0.0)
                        .min(levels - 1.0) as u32;
                    assert_eq!(
                        result.quantized_codes[r * cols + c],
                        expected,
                        "code mismatch at ({r}, {c}): isotropic-Hessian GPTQ must equal round-to-nearest"
                    );
                }
            }
        }
    }

    /// Symmetry must be judged with a RELATIVE tolerance: an H = X^T @ X with
    /// large-magnitude entries carries float accumulation asymmetry far above
    /// any absolute epsilon while remaining symmetric for every practical
    /// purpose, while a genuinely asymmetric Hessian (1% relative skew) must
    /// still be rejected.
    #[test]
    fn symmetry_tolerance_is_relative() {
        let rows = 8;
        let cols = 16;
        let weights = DMatrix::from_fn(rows, cols, |i, j| ((i * 5 + j * 3) as f64 * 0.1).sin());

        // H = X^T @ X with entries around 1e6.
        let x = DMatrix::from_fn(64, cols, |i, j| {
            ((i as f64 * 0.13 + j as f64 * 0.71).sin() + 1.5) * 100.0
        });
        let mut hessian = x.transpose() * &x;
        // Make the accumulation asymmetry explicit and deterministic: exactly
        // symmetrize, then reintroduce a float-scale skew. 1e-4 absolute is
        // ~1e-10 RELATIVE to these entries — far above the old 1e-10 absolute
        // threshold, far below the 1e-8 relative one.
        for r in 0..cols {
            for c in (r + 1)..cols {
                let mean = 0.5 * (hessian[(r, c)] + hessian[(c, r)]);
                hessian[(r, c)] = mean + 1e-4;
                hessian[(c, r)] = mean - 1e-4;
            }
        }
        let max_abs = hessian.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_abs > 1e6,
            "test wants large-magnitude entries, got {max_abs:.3e}"
        );
        assert!(
            (hessian[(0, 1)] - hessian[(1, 0)]).abs() > 1e-10,
            "test wants asymmetry above the old absolute threshold"
        );
        gptq_quantize(&weights, &hessian, 4, 16, 16)
            .expect("symmetric Hessian with float-scale asymmetry must be accepted");

        // A 1% relative skew on one entry is a genuinely asymmetric matrix.
        let mut asymmetric = hessian;
        asymmetric[(2, 5)] += 0.01 * asymmetric[(2, 5)].abs();
        assert_eq!(
            gptq_quantize(&weights, &asymmetric, 4, 16, 16).unwrap_err(),
            GptqError::InvalidInput("Hessian must be symmetric")
        );
    }

    /// 3-bit quantization should work without panics.
    #[test]
    fn three_bit_works() {
        let mut rng = StdRng::seed_from_u64(0x5EED_1002);
        let rows = 32;
        let cols = 64;
        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 3, 64, 64).unwrap();

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
        let mut rng = StdRng::seed_from_u64(0x5EED_1003);
        let rows = 16;
        let cols = 32;

        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-2.0..2.0));
        let hessian = DMatrix::identity(cols, cols);

        let result = gptq_quantize(&weights, &hessian, 4, 32, 32).unwrap();
        let decompressed = gptq_decompress(&result).unwrap();

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

    #[test]
    fn rejects_non_positive_definite_hessian() {
        let weights = DMatrix::from_element(1, 2, 0.5);
        let hessian = DMatrix::from_row_slice(2, 2, &[-10.0, 0.0, 0.0, -10.0]);
        assert_eq!(
            gptq_quantize(&weights, &hessian, 4, 2, 2).unwrap_err(),
            GptqError::HessianNotPositiveDefinite
        );
    }

    #[test]
    fn malformed_representation_is_rejected_without_panicking() {
        let malformed = GptqResult {
            quantized_codes: vec![16],
            group_params: vec![GroupParams {
                scale: 1.0,
                zero_point: 0.0,
            }],
            bits: 4,
            rows: 1,
            cols: 1,
            group_size: 1,
            stats: GptqStats {
                mse: 0.0,
                max_error: 0.0,
                bits_per_weight: 0.0,
                compressed_bytes: 0,
            },
        };

        assert!(matches!(
            gptq_decompress(&malformed),
            Err(GptqError::InvalidRepresentation(_))
        ));
        assert!(matches!(
            malformed.compressed_bytes(),
            Err(GptqError::InvalidRepresentation(_))
        ));
    }
}
