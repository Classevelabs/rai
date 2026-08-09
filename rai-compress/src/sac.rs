/// Sparse-Adaptive Compression (SAC) — lean pipeline for real LLM weights.
///
/// Key insight from real-weight testing: SVD prior captures only 1-5% variance
/// on typical small LLM weights but costs 2-4 bpw in overhead. For real weights,
/// a leaner pipeline without SVD actually wins at iso-bitrate.
///
/// Pipeline:
/// ```text
/// W (original)
///   ↓
/// [Stage 1: Channel Normalization]  — per-row mean/scale (4 bytes/row)
///   ↓ W_norm (unit variance per row)
/// [Stage 2: Sparse Outlier Extract] — top k% at FP16 (6 bytes each)
///   ↓ dense_residual (much smaller dynamic range)
/// [Stage 3: Adaptive Block Quant]   — 1-8 bits per block of 64
///   ↓
/// Compressed
/// ```
///
/// Why this beats SVD-based approaches on real weights:
/// - Zero overhead from SVD storage (which doesn't pay for itself at low rank)
/// - Outlier extraction reduces dynamic range 65-85% (the single biggest win)
/// - Smaller blocks (64 vs 128) for more fine-grained bit allocation
/// - Per-row normalization handles the channel magnitude variation that SVD tries to capture
use crate::channel::ChannelNorm;
use crate::compress::CompressionError;
use crate::quantize::{choose_bits, dequantize, quantize_uniform, QuantizedBlock};
use crate::sparse::SparseDenseDecomp;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// SAC compressed representation.
#[derive(Debug, Clone)]
pub struct SACCompressed {
    /// Stage 1: per-row normalization.
    pub channel_norm: ChannelNorm,
    /// Stage 2: sparse outlier indices and values.
    pub sparse_indices: Vec<u32>,
    pub sparse_values: Vec<f64>,
    /// Stage 3: quantized dense blocks.
    pub dense_blocks: Vec<QuantizedBlock>,
    /// Block size.
    pub block_size: usize,
    /// Matrix shape.
    pub rows: usize,
    pub cols: usize,
    /// Stats.
    pub stats: SACStats,
}

#[derive(Debug, Clone)]
pub struct SACStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub ratio: f64,
    pub bits_per_weight: f64,
    pub mse: f64,
    pub max_error: f64,
    pub psnr_db: f64,
    pub channel_bytes: usize,
    pub sparse_bytes: usize,
    pub dense_bytes: usize,
    pub outlier_fraction: f64,
    pub avg_dense_bits: f64,
    pub range_reduction: f64,
    pub num_outliers: usize,
}

/// Configuration for SAC.
#[derive(Debug, Clone)]
pub struct SACConfig {
    /// Block size for quantization (smaller = finer-grained bit allocation).
    pub block_size: usize,
    /// Fraction of values to extract as sparse outliers.
    pub outlier_fraction: f64,
    /// Absolute error tolerance for adaptive bit allocation.
    pub abs_tol: f64,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            block_size: 64,          // Smaller blocks for finer adaptation
            outlier_fraction: 0.005, // 0.5% outliers (tuned for real weights)
            abs_tol: 0.01,
        }
    }
}

/// Compress using SAC pipeline.
pub fn sac_compress(weights: &DMatrix<f64>, config: &SACConfig) -> SACCompressed {
    let rows = weights.nrows();
    let cols = weights.ncols();
    assert!(rows > 0 && cols > 0, "weight matrix must not be empty");
    assert!(config.block_size > 0, "block_size must be non-zero");
    assert!(
        config.outlier_fraction.is_finite() && (0.0..=1.0).contains(&config.outlier_fraction),
        "outlier_fraction must be finite and between zero and one"
    );
    assert!(
        config.abs_tol.is_finite() && config.abs_tol > 0.0,
        "abs_tol must be finite and positive"
    );
    assert!(
        weights.iter().all(|value| value.is_finite()),
        "weights must be finite"
    );
    let total = rows.checked_mul(cols).expect("weight dimensions overflow");

    // === Stage 1: Channel normalization ===
    let (channel_norm, w_norm) = ChannelNorm::normalize(weights)
        .expect("validated weights must support channel normalization");
    let w_flat: Vec<f64> = w_norm.iter().cloned().collect();

    // === Stage 2: Sparse outlier extraction ===
    let decomp = SparseDenseDecomp::from_residual(&w_flat, config.outlier_fraction)
        .expect("validated residual must support sparse decomposition");
    let range_reduction = decomp
        .range_reduction(&w_flat)
        .expect("fresh sparse decomposition must support its source residual");

    // === Stage 3: Adaptive block quantization ===
    let mut dense_blocks = Vec::new();
    let mut total_dense_bits = 0usize;

    for chunk in decomp.dense.chunks(config.block_size) {
        let bits = choose_bits(chunk, config.abs_tol);
        let block = quantize_uniform(chunk, bits);
        total_dense_bits += chunk.len() * bits as usize;
        dense_blocks.push(block);
    }

    // === Compute sizes ===
    let channel_bytes = channel_norm
        .size_bytes()
        .expect("validated channel metadata size must fit in memory");
    let sparse_bytes = decomp
        .sparse_bytes()
        .expect("validated sparse metadata size must fit in memory");
    let dense_data_bytes: usize = dense_blocks.iter().map(|b| b.packed.size_bytes()).sum();
    let block_meta_bytes = dense_blocks.len() * 5; // scale(2) + zero(2) + bits(1)
    let dense_bytes = dense_data_bytes + block_meta_bytes;
    let compressed_bytes = channel_bytes + sparse_bytes + dense_bytes;

    // Build result for decompression + error measurement
    let result = SACCompressed {
        channel_norm,
        sparse_indices: decomp.sparse_indices,
        sparse_values: decomp.sparse_values,
        dense_blocks,
        block_size: config.block_size,
        rows,
        cols,
        stats: SACStats {
            original_bytes: 0,
            compressed_bytes: 0,
            ratio: 0.0,
            bits_per_weight: 0.0,
            mse: 0.0,
            max_error: 0.0,
            psnr_db: 0.0,
            channel_bytes: 0,
            sparse_bytes: 0,
            dense_bytes: 0,
            outlier_fraction: 0.0,
            avg_dense_bits: 0.0,
            range_reduction: 0.0,
            num_outliers: 0,
        },
    };

    let reconstructed = sac_decompress(&result)
        .expect("freshly compressed SAC matrix must be internally consistent");

    let mut max_error = 0.0f64;
    let mut total_error = 0.0f64;
    let mut signal_power = 0.0f64;
    for (a, b) in weights.iter().zip(reconstructed.iter()) {
        let err = (a - b).abs();
        max_error = max_error.max(err);
        total_error += err * err;
        signal_power += a * a;
    }
    let mse = total_error / total as f64;
    let psnr = if mse > 0.0 {
        10.0 * (signal_power / (total as f64 * mse)).log10()
    } else {
        f64::INFINITY
    };

    SACCompressed {
        stats: SACStats {
            original_bytes: total * 2, // vs FP16
            compressed_bytes,
            ratio: (total * 2) as f64 / compressed_bytes as f64,
            bits_per_weight: (compressed_bytes * 8) as f64 / total as f64,
            mse,
            max_error,
            psnr_db: psnr,
            channel_bytes,
            sparse_bytes,
            dense_bytes,
            outlier_fraction: config.outlier_fraction,
            avg_dense_bits: total_dense_bits as f64 / total as f64,
            range_reduction,
            num_outliers: result.sparse_indices.len(),
        },
        ..result
    }
}

/// Decompress a SAC-compressed matrix.
pub fn sac_decompress(compressed: &SACCompressed) -> Result<DMatrix<f64>, CompressionError> {
    let rows = compressed.rows;
    let cols = compressed.cols;
    if rows == 0 || cols == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "SAC matrix shape must be non-zero",
        ));
    }
    if compressed.block_size == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "SAC block_size must be non-zero",
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or(CompressionError::SizeOverflow)?;
    if compressed.channel_norm.rows != rows
        || compressed.channel_norm.cols != cols
        || compressed.channel_norm.means.len() != rows
        || compressed.channel_norm.scales.len() != rows
    {
        return Err(CompressionError::InvalidRepresentation(
            "channel normalization dimensions do not match the matrix",
        ));
    }
    if compressed
        .channel_norm
        .means
        .iter()
        .any(|value| !value.is_finite())
        || compressed
            .channel_norm
            .scales
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(CompressionError::InvalidRepresentation(
            "channel normalization parameters must be finite with positive scales",
        ));
    }

    // Stage 3: Dequantize dense blocks
    let mut dense_flat = Vec::new();
    dense_flat
        .try_reserve_exact(total)
        .map_err(|_| CompressionError::AllocationFailed)?;
    for block in &compressed.dense_blocks {
        let decoded = dequantize(block)?;
        let remaining = total.saturating_sub(dense_flat.len());
        let expected_len = compressed.block_size.min(remaining);
        if decoded.len() != expected_len {
            return Err(CompressionError::InvalidRepresentation(
                "SAC residual block length does not match block_size or matrix shape",
            ));
        }
        dense_flat.extend(decoded);
    }
    if dense_flat.len() != total {
        return Err(CompressionError::InvalidRepresentation(
            "SAC dense residual length does not match the matrix shape",
        ));
    }

    // Stage 2: Add sparse outliers
    if compressed.sparse_indices.len() != compressed.sparse_values.len() {
        return Err(CompressionError::InvalidRepresentation(
            "SAC sparse index and value lengths differ",
        ));
    }
    let mut seen_indices = HashSet::new();
    seen_indices
        .try_reserve(compressed.sparse_indices.len())
        .map_err(|_| CompressionError::AllocationFailed)?;
    for (&idx, &val) in compressed
        .sparse_indices
        .iter()
        .zip(compressed.sparse_values.iter())
    {
        let index = idx as usize;
        if index >= total || !val.is_finite() || !seen_indices.insert(idx) {
            return Err(CompressionError::InvalidRepresentation(
                "SAC sparse entries must be unique, in range, and finite",
            ));
        }
        let value = dense_flat[index] + val;
        if !value.is_finite() {
            return Err(CompressionError::InvalidRepresentation(
                "SAC sparse reconstruction produced a non-finite value",
            ));
        }
        dense_flat[index] = value;
    }

    // Stage 1: Denormalize
    // Convert flat back to matrix (column-major, matching nalgebra's iter() order)
    let w_norm = DMatrix::from_iterator(rows, cols, dense_flat);
    let output = compressed.channel_norm.denormalize(&w_norm)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(CompressionError::InvalidRepresentation(
            "SAC denormalization produced a non-finite value",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn sac_roundtrip() {
        let mut rng = rand::thread_rng();
        let w = DMatrix::from_fn(64, 64, |_, _| rng.gen_range(-1.0..1.0));
        let config = SACConfig::default();
        let compressed = sac_compress(&w, &config);
        let recovered = sac_decompress(&compressed).unwrap();

        let mse: f64 = w
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (64 * 64) as f64;

        assert!(
            mse < 0.1,
            "SAC roundtrip should be reasonably accurate, got MSE={mse}"
        );
        println!(
            "SAC: {:.2} bpw, MSE={:.2e}",
            compressed.stats.bits_per_weight, compressed.stats.mse
        );
    }

    #[test]
    fn sac_on_structured() {
        let w = DMatrix::from_fn(128, 128, |i, j| {
            (i as f64 * 0.01).sin() * (j as f64 * 0.02).cos() * 2.0
                + if (i * 128 + j) % 200 == 0 { 5.0 } else { 0.0 } // outliers
        });
        let config = SACConfig {
            abs_tol: 0.01,
            ..Default::default()
        };
        let compressed = sac_compress(&w, &config);
        println!(
            "SAC structured: {:.2} bpw, MSE={:.2e}, range reduction={:.1}%",
            compressed.stats.bits_per_weight,
            compressed.stats.mse,
            compressed.stats.range_reduction * 100.0
        );
    }

    #[test]
    fn sac_rejects_mismatched_sparse_entries() {
        let weights = DMatrix::from_element(4, 4, 1.0);
        let mut compressed = sac_compress(&weights, &SACConfig::default());
        compressed.sparse_indices.push(0);
        assert!(matches!(
            sac_decompress(&compressed),
            Err(CompressionError::InvalidRepresentation(_))
        ));
    }
}
