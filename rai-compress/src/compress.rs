use crate::prior::WeightPrior;
use crate::quantize::{choose_bits, dequantize, quantize_kmeans, quantize_uniform, QuantizedBlock};
use nalgebra::DMatrix;

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

/// Statistics about the compression.
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
    /// Use k-means quantization instead of uniform.
    pub use_kmeans: bool,
    /// K-means iterations (if enabled).
    pub kmeans_iters: usize,
}

impl Default for RCConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            rank: 0, // auto
            abs_tol: 0.005,
            use_kmeans: false,
            kmeans_iters: 10,
        }
    }
}

/// Compress a weight matrix using Resonance Compression.
pub fn compress(weights: &DMatrix<f64>, config: &RCConfig) -> CompressedMatrix {
    let rows = weights.nrows();
    let cols = weights.ncols();
    let total = rows * cols;

    // Step 1: Learn prior (low-rank approximation)
    let rank = if config.rank == 0 {
        WeightPrior::optimal_rank(weights, config.block_size, 4.0)
    } else {
        config.rank
    };
    let prior = WeightPrior::from_weights(weights, rank);
    let variance_explained = prior.variance_explained(weights);

    // Step 2: Compute residual
    let residual = prior.residual(weights);
    let residual_flat: Vec<f64> = residual.iter().cloned().collect();

    // Step 3: Adaptive block-wise quantization
    let mut blocks = Vec::new();
    let mut total_residual_bits = 0usize;

    for chunk in residual_flat.chunks(config.block_size) {
        let bits = choose_bits(chunk, config.abs_tol);

        let block = if config.use_kmeans {
            quantize_kmeans(chunk, bits, config.kmeans_iters)
        } else {
            quantize_uniform(chunk, bits)
        };

        total_residual_bits += chunk.len() * bits as usize;
        blocks.push(block);
    }

    // Step 4: Compute statistics
    let prior_bytes = prior.prior_size_bytes();
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
    });

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

    CompressedMatrix {
        prior,
        blocks,
        block_size: config.block_size,
        rows,
        cols,
        stats,
    }
}

/// Decompress a matrix back to full precision.
pub fn decompress_matrix(compressed: &CompressedMatrix) -> DMatrix<f64> {
    // Reconstruct prior
    let mut result = compressed.prior.predict();

    // Add dequantized residuals
    let mut flat_residual = Vec::with_capacity(compressed.rows * compressed.cols);
    for block in &compressed.blocks {
        flat_residual.extend(dequantize(block));
    }
    flat_residual.truncate(compressed.rows * compressed.cols);

    for (i, val) in flat_residual.iter().enumerate() {
        let row = i % compressed.rows;
        let col = i / compressed.rows;
        if row < result.nrows() && col < result.ncols() {
            result[(row, col)] += val;
        }
    }

    result
}

/// Standard 4-bit uniform quantization (baseline for comparison).
pub fn compress_uniform_4bit(weights: &DMatrix<f64>, block_size: usize) -> CompressionStats {
    let rows = weights.nrows();
    let cols = weights.ncols();
    let total = rows * cols;
    let flat: Vec<f64> = weights.iter().cloned().collect();

    let mut total_error = 0.0f64;
    let mut max_error = 0.0f64;
    let mut compressed_bytes = 0usize;

    for chunk in flat.chunks(block_size) {
        let block = quantize_uniform(chunk, 4);
        let recovered = dequantize(&block);
        compressed_bytes += block.packed.size_bytes() + 5; // data + metadata (scale+zero FP16 + bits)

        for (a, b) in chunk.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            total_error += err * err;
            max_error = max_error.max(err);
        }
    }

    CompressionStats {
        original_bytes: total * 8,
        compressed_bytes,
        ratio: (total * 8) as f64 / compressed_bytes as f64,
        bits_per_weight: (compressed_bytes * 8) as f64 / total as f64,
        variance_explained: 0.0,
        avg_residual_bits: 4.0,
        prior_bytes: 0,
        max_error,
        mse: total_error / total as f64,
    }
}

/// Compare RC vs standard 4-bit on a weight matrix.
pub fn compare(weights: &DMatrix<f64>, config: &RCConfig) -> ComparisonReport {
    let rc = compress(weights, config);
    let baseline = compress_uniform_4bit(weights, config.block_size);

    ComparisonReport {
        rc_stats: rc.stats,
        baseline_stats: baseline,
    }
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
    use rand::Rng;

    /// Test: RC beats 4-bit on low-rank matrices (which LLM attention matrices are).
    #[test]
    fn rc_beats_4bit_on_structured_weights() {
        let mut rng = rand::thread_rng();

        // Simulate an attention weight matrix: low-rank + small noise
        // Real attention matrices have effective rank << min(rows, cols)
        let rows = 256;
        let cols = 256;
        let true_rank = 8;

        let u = DMatrix::from_fn(rows, true_rank, |i, j| {
            (i as f64 * 0.01 + j as f64 * 0.1).sin()
        });
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(true_rank, |i, _| {
            10.0 / (i as f64 + 1.0) // Decaying singular values
        }));
        let v = DMatrix::from_fn(cols, true_rank, |i, j| {
            (i as f64 * 0.02 + j as f64 * 0.07).cos()
        });

        let clean = &u * s * v.transpose();
        let noise = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-0.01..0.01));
        let weights = clean + noise;

        let config = RCConfig::default();
        let report = compare(&weights, &config);

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
        let mut rng = rand::thread_rng();
        let rows = 128;
        let cols = 128;

        let weights = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));

        let config = RCConfig::default();
        let report = compare(&weights, &config);

        println!("{report}");

        // Even on random data, RC should at least not be catastrophically worse
        // (the prior still captures some variance via SVD)
        assert!(
            report.rc_stats.mse < report.baseline_stats.mse * 5.0,
            "RC shouldn't be much worse even on random data"
        );
    }

    /// Test: simulate realistic LLM weight distribution.
    #[test]
    fn rc_on_mlp_weights() {
        let mut rng = rand::thread_rng();
        let rows = 512;
        let cols = 128;

        // MLP weights: moderate rank structure + Gaussian noise
        // Typical MLP has effective rank around 30-60% of min(rows,cols)
        let effective_rank = 20;
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
        let report = compare(&weights, &config);

        println!("{report}");

        assert!(
            report.rc_stats.mse < report.baseline_stats.mse,
            "RC should beat 4-bit on structured MLP weights"
        );
    }
}
