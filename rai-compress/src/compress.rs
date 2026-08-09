use crate::bitpack::BitPackError;
use crate::channel::{ChannelError, ChannelNorm};
use crate::gptq::GptqError;
use crate::prior::{PriorError, WeightPrior};
use crate::quantize::{choose_bits, dequantize, quantize_uniform, QuantizeError, QuantizedBlock};
use crate::sparse::SparseError;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Top-level error for the compression pipelines. Wraps every stage-specific
/// error in the crate ([`BitPackError`], [`QuantizeError`], [`ChannelError`],
/// [`SparseError`], [`PriorError`], [`GptqError`]) with the cause preserved
/// through [`std::error::Error::source`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompressionError {
    #[error("invalid compression input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid compressed representation: {0}")]
    InvalidRepresentation(&'static str),
    #[error("compressed dimensions overflow")]
    SizeOverflow,
    #[error("unable to allocate decompressed matrix")]
    AllocationFailed,
    #[error(transparent)]
    Prior(#[from] PriorError),
    #[error(transparent)]
    QuantizedBlock(#[from] QuantizeError),
    #[error(transparent)]
    Channel(#[from] ChannelError),
    #[error(transparent)]
    Sparse(#[from] SparseError),
    #[error(transparent)]
    BitPack(#[from] BitPackError),
    #[error(transparent)]
    Gptq(#[from] GptqError),
}

/// Compressed weight matrix using Resonance Compression.
///
/// Layout:
/// ┌──────────────────────────────┐
/// │ Prior (low-rank: U, S, V)    │  ← captures structure (FP16-equivalent)
/// ├──────────────────────────────┤
/// │ Residual blocks              │  ← only the surprise (2-3 bits avg)
/// │  [block_0] [block_1] ...     │
/// │  each block: scale + bits[]  │
/// └──────────────────────────────┘
///
/// Decompression: W ≈ Prior.predict() + dequant(residual_blocks)
/// Prior can be cached → hot path is just residual dequant.
#[derive(Debug, Clone)]
pub struct CompressedMatrix {
    /// Low-rank prior capturing weight structure.
    pub prior: WeightPrior,
    /// Quantized residual blocks.
    pub blocks: Vec<QuantizedBlock>,
    /// Block size used.
    pub block_size: usize,
    /// Original matrix shape.
    pub rows: usize,
    pub cols: usize,
    /// Compression statistics.
    pub stats: CompressionStats,
}

/// Modeled compression statistics. Byte counts assume future FP16 metadata/prior storage;
/// the current in-memory f64 representation is larger and is not a serialized roundtrip.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes (FP64).
    pub original_bytes: usize,
    /// Compressed size in bytes.
    pub compressed_bytes: usize,
    /// Compression ratio (original / compressed).
    pub ratio: f64,
    /// Effective bits per weight.
    pub bits_per_weight: f64,
    /// Fraction of variance captured by prior.
    pub variance_explained: f64,
    /// Average bits used for residual (per weight).
    pub avg_residual_bits: f64,
    /// Prior storage overhead (bytes).
    pub prior_bytes: usize,
    /// Max quantization error.
    pub max_error: f64,
    /// Mean squared error.
    pub mse: f64,
}

/// Configuration for Resonance Compression.
#[derive(Debug, Clone)]
pub struct RCConfig {
    /// Block size for residual quantization.
    pub block_size: usize,
    /// Prior rank (0 = auto-select).
    pub rank: usize,
    /// Absolute error tolerance per element for adaptive bit allocation.
    /// Smaller = more bits = higher accuracy. Typical: 0.001 - 0.01.
    pub abs_tol: f64,
}

impl Default for RCConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            rank: 0, // auto
            abs_tol: 0.005,
        }
    }
}

/// Compress a weight matrix using Resonance Compression.
///
/// # Errors
///
/// Returns [`CompressionError::InvalidInput`] when the matrix is empty or
/// non-finite, `block_size` is zero, `abs_tol` is not finite and positive, or
/// an explicit `rank` exceeds a matrix dimension; [`CompressionError::SizeOverflow`]
/// when dimensions overflow; and forwards any prior/quantization stage error.
pub fn compress(
    weights: &DMatrix<f64>,
    config: &RCConfig,
) -> Result<CompressedMatrix, CompressionError> {
    let rows = weights.nrows();
    let cols = weights.ncols();
    if rows == 0 || cols == 0 {
        return Err(CompressionError::InvalidInput(
            "weight matrix must not be empty",
        ));
    }
    if config.block_size == 0 {
        return Err(CompressionError::InvalidInput(
            "block_size must be non-zero",
        ));
    }
    if !config.abs_tol.is_finite() || config.abs_tol <= 0.0 {
        return Err(CompressionError::InvalidInput(
            "abs_tol must be finite and positive",
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(CompressionError::InvalidInput("weights must be finite"));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or(CompressionError::SizeOverflow)?;

    // Step 1: Learn prior (low-rank approximation)
    let rank = if config.rank == 0 {
        WeightPrior::optimal_rank(weights, config.block_size, 4.0)?
    } else {
        if config.rank > rows.min(cols) {
            return Err(CompressionError::InvalidInput(
                "rank must not exceed a matrix dimension",
            ));
        }
        config.rank
    };
    let prior = WeightPrior::from_weights(weights, rank)?;
    let variance_explained = prior.variance_explained(weights)?;

    // Step 2: Compute residual
    let residual = prior.residual(weights)?;
    let residual_flat: Vec<f64> = residual.iter().cloned().collect();

    // Step 3: Adaptive block-wise quantization
    let mut blocks = Vec::new();
    let mut total_residual_bits = 0usize;

    for chunk in residual_flat.chunks(config.block_size) {
        let bits = choose_bits(chunk, config.abs_tol)?;

        let block = quantize_uniform(chunk, bits)?;

        total_residual_bits += chunk.len() * bits as usize;
        blocks.push(block);
    }

    // Step 4: Compute statistics
    let prior_bytes = prior.prior_size_bytes()?;
    let residual_bytes = blocks.iter().map(|b| b.packed.size_bytes()).sum::<usize>();
    let block_metadata_bytes = blocks.len() * 5; // scale(2, FP16) + zero(2, FP16) + bits(1)
    let compressed_bytes = prior_bytes + residual_bytes + block_metadata_bytes;
    let original_bytes = total * 8; // FP64

    // Compute reconstruction error
    let reconstructed = decompress_matrix(&CompressedMatrix {
        prior: prior.clone(),
        blocks: blocks.clone(),
        block_size: config.block_size,
        rows,
        cols,
        stats: CompressionStats {
            original_bytes: 0,
            compressed_bytes: 0,
            ratio: 0.0,
            bits_per_weight: 0.0,
            variance_explained: 0.0,
            avg_residual_bits: 0.0,
            prior_bytes: 0,
            max_error: 0.0,
            mse: 0.0,
        },
    })?;

    let errors: Vec<f64> = weights
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
    let mse: f64 = errors.iter().map(|e| e * e).sum::<f64>() / total as f64;

    let stats = CompressionStats {
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f64 / compressed_bytes as f64,
        bits_per_weight: (compressed_bytes * 8) as f64 / total as f64,
        variance_explained,
        avg_residual_bits: total_residual_bits as f64 / total as f64,
        prior_bytes,
        max_error,
        mse,
    };

    Ok(CompressedMatrix {
        prior,
        blocks,
        block_size: config.block_size,
        rows,
        cols,
        stats,
    })
}

/// Decompress a matrix back to full precision.
pub fn decompress_matrix(compressed: &CompressedMatrix) -> Result<DMatrix<f64>, CompressionError> {
    if compressed.rows == 0 || compressed.cols == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "matrix shape must be non-zero",
        ));
    }
    if compressed.block_size == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "block_size must be non-zero",
        ));
    }
    let total = compressed
        .rows
        .checked_mul(compressed.cols)
        .ok_or(CompressionError::SizeOverflow)?;
    // Reconstruct prior
    let mut result = compressed.prior.predict()?;
    if result.shape() != (compressed.rows, compressed.cols) {
        return Err(CompressionError::InvalidRepresentation(
            "matrix shape does not match the prior",
        ));
    }

    // Add dequantized residuals
    let mut flat_residual = Vec::new();
    flat_residual
        .try_reserve_exact(total)
        .map_err(|_| CompressionError::AllocationFailed)?;
    for block in &compressed.blocks {
        let decoded = dequantize(block)?;
        let remaining = total.saturating_sub(flat_residual.len());
        let expected_len = compressed.block_size.min(remaining);
        if decoded.len() != expected_len {
            return Err(CompressionError::InvalidRepresentation(
                "residual block length does not match block_size or matrix shape",
            ));
        }
        flat_residual.extend(decoded);
    }
    if flat_residual.len() != total {
        return Err(CompressionError::InvalidRepresentation(
            "residual length does not match matrix shape",
        ));
    }

    for (i, val) in flat_residual.iter().enumerate() {
        let row = i % compressed.rows;
        let col = i / compressed.rows;
        let value = result[(row, col)] + val;
        if !value.is_finite() {
            return Err(CompressionError::InvalidRepresentation(
                "reconstruction contains non-finite values",
            ));
        }
        result[(row, col)] = value;
    }

    Ok(result)
}

/// Shared HRC/SAC decode path: validate the shape and channel metadata,
/// dequantize the dense blocks, apply the sparse outliers, and return the
/// matrix still in the normalized (channel) domain. The flat residual is
/// interpreted column-major, matching `DMatrix::iter` order on the encode
/// side.
pub(crate) fn decode_normalized_matrix(
    channel_norm: &ChannelNorm,
    dense_blocks: &[QuantizedBlock],
    sparse_indices: &[u32],
    sparse_values: &[f64],
    block_size: usize,
    rows: usize,
    cols: usize,
) -> Result<DMatrix<f64>, CompressionError> {
    if rows == 0 || cols == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "matrix shape must be non-zero",
        ));
    }
    if block_size == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "block_size must be non-zero",
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or(CompressionError::SizeOverflow)?;
    if channel_norm.rows != rows || channel_norm.cols != cols {
        return Err(CompressionError::InvalidRepresentation(
            "channel normalization dimensions do not match the matrix",
        ));
    }
    channel_norm.validate()?;

    // Dequantize dense blocks
    let mut dense_flat = Vec::new();
    dense_flat
        .try_reserve_exact(total)
        .map_err(|_| CompressionError::AllocationFailed)?;
    for block in dense_blocks {
        let decoded = dequantize(block)?;
        let remaining = total.saturating_sub(dense_flat.len());
        let expected_len = block_size.min(remaining);
        if decoded.len() != expected_len {
            return Err(CompressionError::InvalidRepresentation(
                "residual block length does not match block_size or matrix shape",
            ));
        }
        dense_flat.extend(decoded);
    }
    if dense_flat.len() != total {
        return Err(CompressionError::InvalidRepresentation(
            "dense residual length does not match the matrix shape",
        ));
    }

    // Apply sparse outliers
    if sparse_indices.len() != sparse_values.len() {
        return Err(CompressionError::InvalidRepresentation(
            "sparse index and value lengths differ",
        ));
    }
    let mut seen_indices = HashSet::new();
    seen_indices
        .try_reserve(sparse_indices.len())
        .map_err(|_| CompressionError::AllocationFailed)?;
    for (&idx, &val) in sparse_indices.iter().zip(sparse_values.iter()) {
        let index = idx as usize;
        if index >= total || !val.is_finite() || !seen_indices.insert(idx) {
            return Err(CompressionError::InvalidRepresentation(
                "sparse entries must be unique, in range, and finite",
            ));
        }
        let value = dense_flat[index] + val;
        if !value.is_finite() {
            return Err(CompressionError::InvalidRepresentation(
                "sparse reconstruction produced a non-finite value",
            ));
        }
        dense_flat[index] = value;
    }

    Ok(DMatrix::from_iterator(rows, cols, dense_flat))
}

/// Standard 4-bit uniform quantization (baseline for comparison).
///
/// # Errors
///
/// Returns [`CompressionError::InvalidInput`] when the matrix is empty or
/// non-finite or `block_size` is zero, and [`CompressionError::SizeOverflow`]
/// when dimensions overflow.
pub fn compress_uniform_4bit(
    weights: &DMatrix<f64>,
    block_size: usize,
) -> Result<CompressionStats, CompressionError> {
    let rows = weights.nrows();
    let cols = weights.ncols();
    if rows == 0 || cols == 0 {
        return Err(CompressionError::InvalidInput(
            "weight matrix must not be empty",
        ));
    }
    if block_size == 0 {
        return Err(CompressionError::InvalidInput(
            "block_size must be non-zero",
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(CompressionError::InvalidInput("weights must be finite"));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or(CompressionError::SizeOverflow)?;
    let flat: Vec<f64> = weights.iter().cloned().collect();

    let mut total_error = 0.0f64;
    let mut max_error = 0.0f64;
    let mut compressed_bytes = 0usize;

    for chunk in flat.chunks(block_size) {
        let block = quantize_uniform(chunk, 4)?;
        let recovered = dequantize(&block)?;
        compressed_bytes += block.packed.size_bytes() + 5; // data + metadata (scale+zero FP16 + bits)

        for (a, b) in chunk.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            total_error += err * err;
            max_error = max_error.max(err);
        }
    }

    Ok(CompressionStats {
        original_bytes: total * 8,
        compressed_bytes,
        ratio: (total * 8) as f64 / compressed_bytes as f64,
        bits_per_weight: (compressed_bytes * 8) as f64 / total as f64,
        variance_explained: 0.0,
        avg_residual_bits: 4.0,
        prior_bytes: 0,
        max_error,
        mse: total_error / total as f64,
    })
}

/// Compare RC vs standard 4-bit on a weight matrix.
///
/// # Errors
///
/// Forwards any error from [`compress`] or [`compress_uniform_4bit`].
pub fn compare(
    weights: &DMatrix<f64>,
    config: &RCConfig,
) -> Result<ComparisonReport, CompressionError> {
    let rc = compress(weights, config)?;
    let baseline = compress_uniform_4bit(weights, config.block_size)?;

    Ok(ComparisonReport {
        rc_stats: rc.stats,
        baseline_stats: baseline,
    })
}

/// Comparison between RC and baseline.
#[derive(Debug)]
pub struct ComparisonReport {
    pub rc_stats: CompressionStats,
    pub baseline_stats: CompressionStats,
}

impl std::fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔═══════════════════════════════════════════════════╗")?;
        writeln!(f, "║     Resonance Compression vs 4-bit Uniform       ║")?;
        writeln!(f, "╠═══════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Metric              │ 4-bit Uniform │ RC (ours)    ║")?;
        writeln!(f, "╟──────────────────────┼───────────────┼──────────────╢")?;
        writeln!(
            f,
            "║ Bits/weight          │ {:>13.2} │ {:>12.2} ║",
            self.baseline_stats.bits_per_weight, self.rc_stats.bits_per_weight
        )?;
        writeln!(
            f,
            "║ Compression ratio    │ {:>13.2}x│ {:>11.2}x ║",
            self.baseline_stats.ratio, self.rc_stats.ratio
        )?;
        writeln!(
            f,
            "║ MSE                  │ {:>13.2e} │ {:>12.2e} ║",
            self.baseline_stats.mse, self.rc_stats.mse
        )?;
        writeln!(
            f,
            "║ Max error            │ {:>13.4} │ {:>12.4} ║",
            self.baseline_stats.max_error, self.rc_stats.max_error
        )?;
        writeln!(
            f,
            "║ Size (bytes)         │ {:>13} │ {:>12} ║",
            self.baseline_stats.compressed_bytes, self.rc_stats.compressed_bytes
        )?;
        writeln!(
            f,
            "║ Prior variance       │           n/a │ {:>11.1}% ║",
            self.rc_stats.variance_explained * 100.0
        )?;
        writeln!(
            f,
            "║ Avg residual bits    │           4.0 │ {:>12.2} ║",
            self.rc_stats.avg_residual_bits
        )?;
        writeln!(f, "╚═══════════════════════════════════════════════════╝")?;

        let size_win = (1.0
            - self.rc_stats.compressed_bytes as f64 / self.baseline_stats.compressed_bytes as f64)
            * 100.0;
        let accuracy_win = if self.baseline_stats.mse > 1e-15 {
            (1.0 - self.rc_stats.mse / self.baseline_stats.mse) * 100.0
        } else {
            0.0
        };

        writeln!(f)?;
        if size_win > 0.0 {
            writeln!(f, "  RC is {size_win:.1}% SMALLER than 4-bit")?;
        } else {
            writeln!(f, "  RC is {:.1}% larger than 4-bit", -size_win)?;
        }
        if accuracy_win > 0.0 {
            writeln!(f, "  RC is {accuracy_win:.1}% MORE ACCURATE than 4-bit")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// Test: RC beats 4-bit on low-rank matrices (which LLM attention matrices are).
    #[test]
    fn rc_beats_4bit_on_structured_weights() {
        let mut rng = StdRng::seed_from_u64(0x5EED_2001);

        // Simulate an attention weight matrix: low-rank + small noise
        // Real attention matrices have effective rank << min(rows, cols).
        // Per-column frequencies keep the factor columns decorrelated at this
        // matrix size, so the true rank really is 3.
        let rows = 96;
        let cols = 96;
        let true_rank = 3;

        let u = DMatrix::from_fn(rows, true_rank, |i, j| {
            (i as f64 * 0.05 * (j as f64 + 1.0)).sin()
        });
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(true_rank, |i, _| {
            10.0 / (i as f64 + 1.0) // Decaying singular values
        }));
        let v = DMatrix::from_fn(cols, true_rank, |i, j| {
            (i as f64 * 0.06 * (j as f64 + 1.0) + 0.3 * j as f64).cos()
        });

        let clean = &u * s * v.transpose();
        let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.005..0.005));
        let weights = clean + noise;

        let config = RCConfig::default();
        let report = compare(&weights, &config).unwrap();

        println!("{report}");

        // RC should have LOWER MSE
        assert!(
            report.rc_stats.mse < report.baseline_stats.mse,
            "RC MSE ({:.2e}) should be less than 4-bit MSE ({:.2e})",
            report.rc_stats.mse,
            report.baseline_stats.mse,
        );

        // RC should use FEWER bits per weight
        assert!(
            report.rc_stats.bits_per_weight < report.baseline_stats.bits_per_weight,
            "RC bpw ({:.2}) should be less than 4-bit bpw ({:.2})",
            report.rc_stats.bits_per_weight,
            report.baseline_stats.bits_per_weight,
        );
    }

    /// Test: RC on truly random matrices (worst case — no structure to exploit).
    #[test]
    fn rc_on_random_weights() {
        let mut rng = StdRng::seed_from_u64(0x5EED_2002);
        let rows = 64;
        let cols = 64;

        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));

        let config = RCConfig::default();
        let report = compare(&weights, &config).unwrap();

        println!("{report}");

        // Even on random data, RC should at least not be catastrophically worse
        // (the prior still captures some variance via SVD)
        assert!(
            report.rc_stats.mse < report.baseline_stats.mse * 5.0,
            "RC shouldn't be much worse even on random data"
        );
    }

    #[test]
    fn malformed_compressed_shape_is_rejected() {
        let weights = DMatrix::from_element(2, 2, 1.0);
        let mut compressed = compress(&weights, &RCConfig::default()).unwrap();
        compressed.rows = 3;
        assert!(matches!(
            decompress_matrix(&compressed),
            Err(CompressionError::InvalidRepresentation(_)) | Err(CompressionError::Prior(_))
        ));
    }

    #[test]
    fn invalid_compress_inputs_are_rejected() {
        let empty = DMatrix::zeros(0, 0);
        assert!(matches!(
            compress(&empty, &RCConfig::default()),
            Err(CompressionError::InvalidInput(_))
        ));
        assert!(matches!(
            compress_uniform_4bit(&empty, 64),
            Err(CompressionError::InvalidInput(_))
        ));

        let weights = DMatrix::from_element(2, 2, 1.0);
        assert!(matches!(
            compress(
                &weights,
                &RCConfig {
                    block_size: 0,
                    ..RCConfig::default()
                }
            ),
            Err(CompressionError::InvalidInput(_))
        ));
        assert!(matches!(
            compress(
                &weights,
                &RCConfig {
                    abs_tol: f64::NAN,
                    ..RCConfig::default()
                }
            ),
            Err(CompressionError::InvalidInput(_))
        ));
        assert!(matches!(
            compress(
                &weights,
                &RCConfig {
                    rank: 3,
                    ..RCConfig::default()
                }
            ),
            Err(CompressionError::InvalidInput(_))
        ));
        assert!(matches!(
            compress(&DMatrix::from_element(2, 2, f64::NAN), &RCConfig::default()),
            Err(CompressionError::InvalidInput(_))
        ));
        assert!(matches!(
            compress_uniform_4bit(&weights, 0),
            Err(CompressionError::InvalidInput(_))
        ));
    }

    /// Test: simulate realistic LLM weight distribution.
    #[test]
    fn rc_on_mlp_weights() {
        let mut rng = StdRng::seed_from_u64(0x5EED_2003);
        let rows = 160;
        let cols = 64;

        // MLP weights: moderate rank structure + noise. Typical MLP has
        // effective rank well below min(rows, cols).
        let effective_rank = 8;
        let u = DMatrix::from_fn(rows, effective_rank, |i, j| {
            (i as f64 * 0.005 + j as f64 * 0.3).sin() * 0.5
        });
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(effective_rank, |i, _| {
            5.0 / (i as f64 + 1.0).sqrt()
        }));
        let v = DMatrix::from_fn(cols, effective_rank, |i, j| {
            (i as f64 * 0.008 + j as f64 * 0.15).cos() * 0.5
        });

        let structured = &u * s * v.transpose();
        let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.05..0.05));
        let weights = structured + noise;

        let config = RCConfig::default();
        let report = compare(&weights, &config).unwrap();

        println!("{report}");

        assert!(
            report.rc_stats.mse < report.baseline_stats.mse,
            "RC should beat 4-bit on structured MLP weights"
        );
    }
}
