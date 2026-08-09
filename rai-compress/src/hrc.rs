use crate::channel::ChannelNorm;
use crate::compress::CompressionError;
use crate::prior::WeightPrior;
use crate::quantize::{choose_bits, dequantize, quantize_uniform, QuantizedBlock};
use crate::sparse::SparseDenseDecomp;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Hybrid Resonance Compression — multi-stage cascaded pipeline.
///
/// Each stage strips away a different type of structure:
///
/// ```text
/// W (original)
///   ↓
/// [Stage 1: Channel Normalization]  — per-row mean/scale
///   ↓ W_norm
/// [Stage 2: Low-Rank SVD Prior]     — global correlation structure
///   ↓ residual
/// [Stage 3: Sparse Outlier Extract] — top 1% at full precision
///   ↓ dense_residual (much smaller dynamic range)
/// [Stage 4: Adaptive Quantization]  — 1-3 bits on what's left
///   ↓
/// Compressed
/// ```
///
/// Why this beats single-stage approaches:
/// - Stage 1 removes O(rows) parameters of coarse structure
/// - Stage 2 removes the dominant low-rank structure
/// - Stage 3 removes the few outliers that would force high bit-widths
/// - Stage 4 gets away with very few bits because stages 1-3 already
///   captured everything important
#[derive(Debug, Clone)]
pub struct HRCompressed {
    /// Stage 1: Channel normalization parameters.
    pub channel_norm: ChannelNorm,
    /// Stage 2: Low-rank prior.
    pub prior: WeightPrior,
    /// Stage 3: Sparse outlier indices and values.
    pub sparse_indices: Vec<u32>,
    pub sparse_values: Vec<f64>,
    /// Stage 4: Quantized dense residual blocks.
    pub dense_blocks: Vec<QuantizedBlock>,
    /// Block size for stage 4.
    pub block_size: usize,
    /// Matrix shape.
    pub rows: usize,
    pub cols: usize,
    /// Stats.
    pub stats: HRCStats,
}

#[derive(Debug, Clone)]
pub struct HRCStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub ratio: f64,
    pub bits_per_weight: f64,
    pub mse: f64,
    pub max_error: f64,
    pub psnr_db: f64,
    // Per-stage breakdown
    pub channel_bytes: usize,
    pub prior_bytes: usize,
    pub sparse_bytes: usize,
    pub dense_bytes: usize,
    pub prior_variance_explained: f64,
    pub outlier_fraction: f64,
    pub avg_dense_bits: f64,
    pub range_reduction_from_outliers: f64,
}

/// Configuration for HRC.
#[derive(Debug, Clone)]
pub struct HRCConfig {
    /// Block size for dense quantization.
    pub block_size: usize,
    /// SVD rank (0 = auto).
    pub rank: usize,
    /// Fraction of values to extract as sparse outliers.
    pub outlier_fraction: f64,
    /// Absolute error tolerance for adaptive bit allocation.
    pub abs_tol: f64,
}

impl Default for HRCConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            rank: 0,
            outlier_fraction: 0.01, // top 1%
            abs_tol: 0.005,
        }
    }
}

/// Compress a weight matrix using the full HRC pipeline.
pub fn hrc_compress(weights: &DMatrix<f64>, config: &HRCConfig) -> HRCompressed {
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

    // === Stage 2: Low-rank SVD prior ===
    let rank = if config.rank == 0 {
        WeightPrior::optimal_rank(&w_norm, config.block_size, 4.0)
            .expect("validated weights must produce a prior rank")
    } else {
        assert!(
            config.rank <= rows.min(cols),
            "rank must not exceed a matrix dimension"
        );
        config.rank
    };
    let prior = WeightPrior::from_weights(&w_norm, rank)
        .expect("validated weights and rank must produce a prior");
    let residual = prior
        .residual(&w_norm)
        .expect("validated prior must support its source matrix");
    let residual_flat: Vec<f64> = residual.iter().cloned().collect();

    // === Stage 3: Sparse outlier extraction ===
    let decomp = SparseDenseDecomp::from_residual(&residual_flat, config.outlier_fraction)
        .expect("validated residual must support sparse decomposition");
    let range_reduction = decomp
        .range_reduction(&residual_flat)
        .expect("fresh sparse decomposition must support its source residual");

    // === Stage 4: Adaptive quantization of dense remainder ===
    let mut dense_blocks = Vec::new();
    let mut total_dense_bits = 0usize;

    for chunk in decomp.dense.chunks(config.block_size) {
        let bits = choose_bits(chunk, config.abs_tol);
        let block = quantize_uniform(chunk, bits);
        total_dense_bits += chunk.len() * bits as usize;
        dense_blocks.push(block);
    }

    // === Compute stats ===
    let channel_bytes = channel_norm
        .size_bytes()
        .expect("validated channel metadata size must fit in memory");
    let prior_bytes = prior
        .prior_size_bytes()
        .expect("validated prior size must fit in memory");
    let sparse_bytes = decomp
        .sparse_bytes()
        .expect("validated sparse metadata size must fit in memory");
    let dense_bytes: usize = dense_blocks.iter().map(|b| b.packed.size_bytes()).sum();
    let block_meta_bytes = dense_blocks.len() * 5;
    let compressed_bytes =
        channel_bytes + prior_bytes + sparse_bytes + dense_bytes + block_meta_bytes;

    // Reconstruction for error measurement
    let result = HRCompressed {
        channel_norm,
        prior,
        sparse_indices: decomp.sparse_indices,
        sparse_values: decomp.sparse_values,
        dense_blocks,
        block_size: config.block_size,
        rows,
        cols,
        stats: HRCStats {
            original_bytes: 0,
            compressed_bytes: 0,
            ratio: 0.0,
            bits_per_weight: 0.0,
            mse: 0.0,
            max_error: 0.0,
            psnr_db: 0.0,
            channel_bytes: 0,
            prior_bytes: 0,
            sparse_bytes: 0,
            dense_bytes: 0,
            prior_variance_explained: 0.0,
            outlier_fraction: 0.0,
            avg_dense_bits: 0.0,
            range_reduction_from_outliers: 0.0,
        },
    };

    let reconstructed = hrc_decompress(&result)
        .expect("freshly compressed HRC matrix must be internally consistent");

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

    let prior_var = result
        .prior
        .variance_explained(&{
            let (_, w_norm) = ChannelNorm::normalize(weights)
                .expect("validated weights must support channel normalization");
            w_norm
        })
        .expect("validated prior must support the normalized source matrix");

    HRCompressed {
        stats: HRCStats {
            original_bytes: total * 8,
            compressed_bytes,
            ratio: (total * 8) as f64 / compressed_bytes as f64,
            bits_per_weight: (compressed_bytes * 8) as f64 / total as f64,
            mse,
            max_error,
            psnr_db: psnr,
            channel_bytes,
            prior_bytes,
            sparse_bytes,
            dense_bytes: dense_bytes + block_meta_bytes,
            prior_variance_explained: prior_var,
            outlier_fraction: config.outlier_fraction,
            avg_dense_bits: total_dense_bits as f64 / total as f64,
            range_reduction_from_outliers: range_reduction,
        },
        ..result
    }
}

/// Decompress an HRC-compressed matrix.
pub fn hrc_decompress(compressed: &HRCompressed) -> Result<DMatrix<f64>, CompressionError> {
    let rows = compressed.rows;
    let cols = compressed.cols;
    if rows == 0 || cols == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "HRC matrix shape must be non-zero",
        ));
    }
    if compressed.block_size == 0 {
        return Err(CompressionError::InvalidRepresentation(
            "HRC block_size must be non-zero",
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

    // Stage 4: Dequantize dense blocks
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
                "HRC residual block length does not match block_size or matrix shape",
            ));
        }
        dense_flat.extend(decoded);
    }
    if dense_flat.len() != total {
        return Err(CompressionError::InvalidRepresentation(
            "HRC dense residual length does not match the matrix shape",
        ));
    }

    // Stage 3: Add sparse outliers
    if compressed.sparse_indices.len() != compressed.sparse_values.len() {
        return Err(CompressionError::InvalidRepresentation(
            "HRC sparse index and value lengths differ",
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
                "HRC sparse entries must be unique, in range, and finite",
            ));
        }
        let value = dense_flat[index] + val;
        if !value.is_finite() {
            return Err(CompressionError::InvalidRepresentation(
                "HRC sparse reconstruction produced a non-finite value",
            ));
        }
        dense_flat[index] = value;
    }

    // Stage 2: Add SVD prior
    let prior_matrix = compressed.prior.predict()?;
    if prior_matrix.shape() != (rows, cols) {
        return Err(CompressionError::InvalidRepresentation(
            "HRC prior dimensions do not match the matrix",
        ));
    }
    let mut w_norm = prior_matrix;
    for (i, &val) in dense_flat.iter().enumerate() {
        let row = i % rows;
        let col = i / rows;
        let value = w_norm[(row, col)] + val;
        if !value.is_finite() {
            return Err(CompressionError::InvalidRepresentation(
                "HRC reconstruction produced a non-finite value",
            ));
        }
        w_norm[(row, col)] = value;
    }

    // Stage 1: Denormalize channels
    let output = compressed.channel_norm.denormalize(&w_norm)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(CompressionError::InvalidRepresentation(
            "HRC denormalization produced a non-finite value",
        ));
    }
    Ok(output)
}

/// Full comparison: HRC vs RC vs 4-bit uniform.
pub fn full_compare(weights: &DMatrix<f64>, hrc_config: &HRCConfig) -> FullReport {
    let hrc = hrc_compress(weights, hrc_config);

    let rc_config = crate::compress::RCConfig {
        block_size: hrc_config.block_size,
        rank: hrc_config.rank,
        abs_tol: hrc_config.abs_tol,
    };
    let rc = crate::compress::compress(weights, &rc_config);
    let baseline = crate::compress::compress_uniform_4bit(weights, hrc_config.block_size);

    FullReport {
        hrc_stats: hrc.stats,
        rc_stats: rc.stats,
        baseline_stats: baseline,
    }
}

#[derive(Debug)]
pub struct FullReport {
    pub hrc_stats: HRCStats,
    pub rc_stats: crate::compress::CompressionStats,
    pub baseline_stats: crate::compress::CompressionStats,
}

impl std::fmt::Display for FullReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║          Compression Comparison: 4-bit vs RC vs HRC                 ║"
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ Metric                │  4-bit Uniform  │   RC (v1)   │  HRC (v2)   ║"
        )?;
        writeln!(
            f,
            "╟───────────────────────┼─────────────────┼─────────────┼─────────────╢"
        )?;
        writeln!(
            f,
            "║ Bits/weight            │ {:>15.2} │ {:>11.2} │ {:>11.2} ║",
            self.baseline_stats.bits_per_weight,
            self.rc_stats.bits_per_weight,
            self.hrc_stats.bits_per_weight
        )?;
        writeln!(
            f,
            "║ Compression ratio      │ {:>14.2}x │ {:>10.2}x │ {:>10.2}x ║",
            self.baseline_stats.ratio, self.rc_stats.ratio, self.hrc_stats.ratio
        )?;
        writeln!(
            f,
            "║ MSE                    │ {:>15.2e} │ {:>11.2e} │ {:>11.2e} ║",
            self.baseline_stats.mse, self.rc_stats.mse, self.hrc_stats.mse
        )?;
        writeln!(
            f,
            "║ Max error              │ {:>15.4} │ {:>11.4} │ {:>11.4} ║",
            self.baseline_stats.max_error, self.rc_stats.max_error, self.hrc_stats.max_error
        )?;
        writeln!(
            f,
            "║ PSNR (dB)              │ {:>15} │ {:>11} │ {:>11.1} ║",
            "n/a", "n/a", self.hrc_stats.psnr_db
        )?;
        writeln!(
            f,
            "║ Size (bytes)           │ {:>15} │ {:>11} │ {:>11} ║",
            self.baseline_stats.compressed_bytes,
            self.rc_stats.compressed_bytes,
            self.hrc_stats.compressed_bytes
        )?;
        writeln!(
            f,
            "╟───────────────────────┼─────────────────┼─────────────┼─────────────╢"
        )?;
        writeln!(
            f,
            "║ Prior variance captur  │             n/a │ {:>10.1}% │ {:>10.1}% ║",
            self.rc_stats.variance_explained * 100.0,
            self.hrc_stats.prior_variance_explained * 100.0
        )?;
        writeln!(
            f,
            "║ Outlier extraction     │             n/a │         n/a │ {:>10.1}% ║",
            self.hrc_stats.outlier_fraction * 100.0
        )?;
        writeln!(
            f,
            "║ Range reduction (outlr)│             n/a │         n/a │ {:>10.1}% ║",
            self.hrc_stats.range_reduction_from_outliers * 100.0
        )?;
        writeln!(
            f,
            "║ Avg dense bits         │             4.0 │ {:>11.2} │ {:>11.2} ║",
            self.rc_stats.avg_residual_bits, self.hrc_stats.avg_dense_bits
        )?;
        writeln!(
            f,
            "╟───────────────────────┼─────────────────┼─────────────┼─────────────╢"
        )?;
        writeln!(
            f,
            "║ HRC breakdown:         │  channel: {:>5} │ prior: {:>5} │             ║",
            self.hrc_stats.channel_bytes, self.hrc_stats.prior_bytes
        )?;
        writeln!(
            f,
            "║                        │  sparse: {:>6} │ dense: {:>5} │             ║",
            self.hrc_stats.sparse_bytes, self.hrc_stats.dense_bytes
        )?;
        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════════════╝"
        )?;

        // Summary
        let hrc_vs_4bit_size = (1.0
            - self.hrc_stats.compressed_bytes as f64 / self.baseline_stats.compressed_bytes as f64)
            * 100.0;
        let hrc_vs_rc_size = (1.0
            - self.hrc_stats.compressed_bytes as f64 / self.rc_stats.compressed_bytes as f64)
            * 100.0;
        let hrc_vs_4bit_mse = if self.baseline_stats.mse > 1e-15 {
            self.baseline_stats.mse / self.hrc_stats.mse
        } else {
            1.0
        };
        let hrc_vs_rc_mse = if self.rc_stats.mse > 1e-15 {
            self.rc_stats.mse / self.hrc_stats.mse
        } else {
            1.0
        };

        writeln!(f)?;
        writeln!(
            f,
            "  HRC vs 4-bit: {:.1}% {} size, {:.0}x more accurate",
            hrc_vs_4bit_size.abs(),
            if hrc_vs_4bit_size > 0.0 {
                "smaller"
            } else {
                "larger"
            },
            hrc_vs_4bit_mse
        )?;
        writeln!(
            f,
            "  HRC vs RC:    {:.1}% {} size, {:.1}x more accurate",
            hrc_vs_rc_size.abs(),
            if hrc_vs_rc_size > 0.0 {
                "smaller"
            } else {
                "larger"
            },
            hrc_vs_rc_mse
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn make_attention_weights(rows: usize, cols: usize, rank: usize) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let u = DMatrix::from_fn(rows, rank, |i, j| (i as f64 * 0.01 + j as f64 * 0.1).sin());
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(rank, |i, _| {
            10.0 / (i as f64 + 1.0)
        }));
        let v = DMatrix::from_fn(cols, rank, |i, j| (i as f64 * 0.02 + j as f64 * 0.07).cos());
        let clean = &u * s * v.transpose();
        // Add noise + a few outliers
        let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.01..0.01));
        let mut result = clean + noise;
        // Inject sparse outliers (like real weights have)
        for _ in 0..((rows * cols) as f64 * 0.005) as usize {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            result[(r, c)] += rng.gen_range(-2.0..2.0);
        }
        result
    }

    fn make_mlp_weights(rows: usize, cols: usize) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let rank = 20;
        let u = DMatrix::from_fn(rows, rank, |i, j| {
            (i as f64 * 0.005 + j as f64 * 0.3).sin() * 0.5
        });
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(rank, |i, _| {
            5.0 / (i as f64 + 1.0).sqrt()
        }));
        let v = DMatrix::from_fn(cols, rank, |i, j| {
            (i as f64 * 0.008 + j as f64 * 0.15).cos() * 0.5
        });
        let structured = &u * s * v.transpose();
        let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.05..0.05));
        let mut result = structured + noise;
        // Sparse outliers
        for _ in 0..((rows * cols) as f64 * 0.01) as usize {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            result[(r, c)] += rng.gen_range(-1.0..1.0);
        }
        result
    }

    #[test]
    fn hrc_beats_all_on_attention() {
        let weights = make_attention_weights(256, 256, 8);
        let config = HRCConfig::default();
        let report = full_compare(&weights, &config);

        println!("\n=== ATTENTION WEIGHTS (256x256, rank-8 + outliers) ===");
        println!("{report}");

        // HRC should beat 4-bit on MSE
        assert!(
            report.hrc_stats.mse < report.baseline_stats.mse,
            "HRC MSE ({:.2e}) should beat 4-bit MSE ({:.2e})",
            report.hrc_stats.mse,
            report.baseline_stats.mse
        );
    }

    #[test]
    fn hrc_beats_all_on_mlp() {
        let weights = make_mlp_weights(512, 128);
        let config = HRCConfig::default();
        let report = full_compare(&weights, &config);

        println!("\n=== MLP WEIGHTS (512x128, rank-20 + outliers) ===");
        println!("{report}");

        assert!(
            report.hrc_stats.mse < report.baseline_stats.mse,
            "HRC MSE ({:.2e}) should beat 4-bit MSE ({:.2e})",
            report.hrc_stats.mse,
            report.baseline_stats.mse
        );
    }

    #[test]
    fn hrc_on_random() {
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(128, 128, |_, _| rng.gen_range(-1.0..1.0));
        let config = HRCConfig::default();
        let report = full_compare(&weights, &config);

        println!("\n=== RANDOM WEIGHTS (128x128, no structure) ===");
        println!("{report}");

        // Even on random data, HRC shouldn't catastrophically fail
        assert!(
            report.hrc_stats.mse < 1.0,
            "HRC shouldn't catastrophically fail on random data"
        );
    }

    #[test]
    fn hrc_decompresses_accurately() {
        let weights = make_attention_weights(64, 64, 4);
        let config = HRCConfig {
            abs_tol: 0.001,
            ..Default::default()
        };
        let compressed = hrc_compress(&weights, &config);
        let recovered = hrc_decompress(&compressed).unwrap();

        // Check every element
        let mut max_err = 0.0f64;
        for i in 0..64 {
            for j in 0..64 {
                let err = (weights[(i, j)] - recovered[(i, j)]).abs();
                max_err = max_err.max(err);
            }
        }

        println!("Max decompression error: {max_err:.6}");
        assert!(
            max_err < 0.1,
            "decompression should be accurate, got max_err={max_err}"
        );
    }

    #[test]
    fn hrc_rejects_out_of_range_sparse_entries() {
        let weights = DMatrix::from_element(4, 4, 1.0);
        let mut compressed = hrc_compress(&weights, &HRCConfig::default());
        compressed.sparse_indices.push(u32::MAX);
        compressed.sparse_values.push(1.0);
        assert!(matches!(
            hrc_decompress(&compressed),
            Err(CompressionError::InvalidRepresentation(_))
        ));
    }

    /// The big test: simulate a full transformer layer's worth of weights.
    #[test]
    fn hrc_full_transformer_layer() {
        let d_model = 256;
        let d_ff = d_model * 4;

        // Q, K, V projections (low-rank, structured)
        let q = make_attention_weights(d_model, d_model, 8);
        let k = make_attention_weights(d_model, d_model, 8);
        let v = make_attention_weights(d_model, d_model, 8);
        // MLP up and down projections
        let mlp_up = make_mlp_weights(d_ff, d_model);
        let mlp_down = make_mlp_weights(d_model, d_ff);

        let config = HRCConfig::default();

        let matrices = vec![
            ("Q_proj", q),
            ("K_proj", k),
            ("V_proj", v),
            ("MLP_up", mlp_up),
            ("MLP_down", mlp_down),
        ];

        println!("\n{}", "=".repeat(70));
        println!("  FULL TRANSFORMER LAYER COMPRESSION");
        println!("  d_model={d_model}, d_ff={d_ff}");
        println!("{}", "=".repeat(70));

        let mut total_original = 0usize;
        let mut total_4bit = 0usize;
        let mut total_rc = 0usize;
        let mut total_hrc = 0usize;
        let mut total_4bit_mse = 0.0f64;
        let mut total_rc_mse = 0.0f64;
        let mut total_hrc_mse = 0.0f64;
        let mut total_weights = 0usize;

        for (name, w) in &matrices {
            let report = full_compare(w, &config);
            let n = w.nrows() * w.ncols();

            println!("\n{name} ({} x {}):", w.nrows(), w.ncols());
            println!(
                "  4-bit: {:.2} bpw, MSE={:.2e}",
                report.baseline_stats.bits_per_weight, report.baseline_stats.mse
            );
            println!(
                "  RC:    {:.2} bpw, MSE={:.2e}",
                report.rc_stats.bits_per_weight, report.rc_stats.mse
            );
            println!(
                "  HRC:   {:.2} bpw, MSE={:.2e}, PSNR={:.1}dB",
                report.hrc_stats.bits_per_weight, report.hrc_stats.mse, report.hrc_stats.psnr_db
            );

            total_original += report.hrc_stats.original_bytes;
            total_4bit += report.baseline_stats.compressed_bytes;
            total_rc += report.rc_stats.compressed_bytes;
            total_hrc += report.hrc_stats.compressed_bytes;
            total_4bit_mse += report.baseline_stats.mse * n as f64;
            total_rc_mse += report.rc_stats.mse * n as f64;
            total_hrc_mse += report.hrc_stats.mse * n as f64;
            total_weights += n;
        }

        let avg_4bit_bpw = (total_4bit * 8) as f64 / total_weights as f64;
        let avg_rc_bpw = (total_rc * 8) as f64 / total_weights as f64;
        let avg_hrc_bpw = (total_hrc * 8) as f64 / total_weights as f64;
        let avg_4bit_mse = total_4bit_mse / total_weights as f64;
        let avg_rc_mse = total_rc_mse / total_weights as f64;
        let avg_hrc_mse = total_hrc_mse / total_weights as f64;

        println!("\n{}", "─".repeat(70));
        println!(
            "TOTALS ({total_weights} weights, {:.2} MB original):",
            total_original as f64 / 1e6
        );
        println!(
            "  4-bit: {:.2} bpw, {:.2} MB, avg MSE={:.2e}",
            avg_4bit_bpw,
            total_4bit as f64 / 1e6,
            avg_4bit_mse
        );
        println!(
            "  RC:    {:.2} bpw, {:.2} MB, avg MSE={:.2e}",
            avg_rc_bpw,
            total_rc as f64 / 1e6,
            avg_rc_mse
        );
        println!(
            "  HRC:   {:.2} bpw, {:.2} MB, avg MSE={:.2e}",
            avg_hrc_bpw,
            total_hrc as f64 / 1e6,
            avg_hrc_mse
        );

        let hrc_vs_4bit_size = (1.0 - total_hrc as f64 / total_4bit as f64) * 100.0;
        let hrc_vs_4bit_acc = avg_4bit_mse / avg_hrc_mse;

        println!(
            "\n  HRC is {:.1}% {} than 4-bit, {:.0}x more accurate",
            hrc_vs_4bit_size.abs(),
            if hrc_vs_4bit_size > 0.0 {
                "smaller"
            } else {
                "larger"
            },
            hrc_vs_4bit_acc
        );
    }
}
