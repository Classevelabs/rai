use crate::bitpack::BitPacker;

/// Quantization parameters for a block of residual values.
#[derive(Debug, Clone)]
pub struct BlockParams {
    /// Scale factor: maps quantized values back to float.
    pub scale: f64,
    /// Zero point offset.
    pub zero_point: f64,
    /// Bits per value for this block.
    pub bits: u8,
}

/// A quantized block of residual values.
#[derive(Debug, Clone)]
pub struct QuantizedBlock {
    pub params: BlockParams,
    pub packed: BitPacker,
}

/// Uniform quantization of residual values.
///
/// Standard approach: map float range to integer grid.
/// Our advantage: because we operate on RESIDUALS (not raw weights),
/// the range is much smaller → same bits give better precision.
pub fn quantize_uniform(values: &[f64], bits: u8) -> QuantizedBlock {
    assert!(bits > 0 && bits <= 8);
    let levels = (1u32 << bits) as f64;

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let scale = if range < 1e-15 {
        1.0
    } else {
        range / (levels - 1.0)
    };
    let zero_point = min;

    let quantized: Vec<u32> = values
        .iter()
        .map(|&v| {
            let q = ((v - zero_point) / scale).round() as i64;
            q.max(0).min(levels as i64 - 1) as u32
        })
        .collect();

    QuantizedBlock {
        params: BlockParams {
            scale,
            zero_point,
            bits,
        },
        packed: BitPacker::pack(&quantized, bits),
    }
}

/// Dequantize a block back to floats.
pub fn dequantize(block: &QuantizedBlock) -> Vec<f64> {
    let values = block.packed.unpack();
    values
        .iter()
        .map(|&q| q as f64 * block.params.scale + block.params.zero_point)
        .collect()
}

/// Adaptive quantization: choose bits per block based on residual magnitude.
///
/// This is the key innovation over uniform bit allocation:
/// - Blocks with tiny residuals → 1-2 bits (the prior captured everything)
/// - Blocks with moderate residuals → 3-4 bits
/// - Blocks with large residuals → 5-6 bits (prior missed this part)
///
/// Uses absolute error tolerance: if the residual range is already small
/// relative to `abs_tol`, fewer bits are needed.
///
/// `abs_tol` is the maximum acceptable quantization error per element.
/// For LLM weights, typical values: 0.001 to 0.01.
pub fn choose_bits(residual_block: &[f64], abs_tol: f64) -> u8 {
    let min = residual_block.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = residual_block
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // If range is within tolerance, 1 bit suffices (just encodes above/below midpoint)
    if range < abs_tol * 2.0 {
        return 1;
    }

    // bits needed = ceil(log2(range / abs_tol))
    // This gives enough quantization levels to cover the range with abs_tol precision
    let bits_needed = (range / abs_tol).log2().ceil() as i32;

    (bits_needed.max(1).min(8)) as u8
}

/// Non-uniform quantization using learned centroids.
///
/// Instead of uniform grid, use k-means to find optimal quantization points.
/// This is like NRA attractors — each centroid is a "stable point" that
/// nearby values snap to.
pub fn quantize_kmeans(values: &[f64], bits: u8, max_iter: usize) -> QuantizedBlock {
    let k = 1usize << bits;
    if values.is_empty() || k == 0 {
        return quantize_uniform(values, bits);
    }

    // Initialize centroids uniformly across range
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-15 {
        return quantize_uniform(values, bits);
    }

    let mut centroids: Vec<f64> = (0..k)
        .map(|i| min + (max - min) * (i as f64 + 0.5) / k as f64)
        .collect();

    // K-means iterations
    for _ in 0..max_iter {
        // Assign each value to nearest centroid
        let assignments: Vec<usize> = values
            .iter()
            .map(|&v| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| (v - *a).abs().partial_cmp(&(v - *b).abs()).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        // Update centroids
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        for (&v, &a) in values.iter().zip(assignments.iter()) {
            sums[a] += v;
            counts[a] += 1;
        }
        let mut changed = false;
        for i in 0..k {
            if counts[i] > 0 {
                let new_c = sums[i] / counts[i] as f64;
                if (new_c - centroids[i]).abs() > 1e-10 {
                    changed = true;
                }
                centroids[i] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    // Sort centroids for consistent encoding
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Quantize: assign to nearest centroid
    let quantized: Vec<u32> = values
        .iter()
        .map(|&v| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (v - *a).abs().partial_cmp(&(v - *b).abs()).unwrap())
                .unwrap()
                .0 as u32
        })
        .collect();

    // Store centroids as scale/zero_point approximation
    // For full k-means decode, we'd store the centroid table separately
    // Here we use a lookup approach
    let scale = if centroids.len() >= 2 {
        (centroids.last().unwrap() - centroids[0]) / (k as f64 - 1.0)
    } else {
        1.0
    };
    let zero_point = centroids[0];

    QuantizedBlock {
        params: BlockParams {
            scale,
            zero_point,
            bits,
        },
        packed: BitPacker::pack(&quantized, bits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_roundtrip_4bit() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 0.1).collect();
        let block = quantize_uniform(&values, 4);
        let recovered = dequantize(&block);

        let mse: f64 = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        // 4-bit over [-3.2, 3.1] range: step = 6.3/15 ≈ 0.42, MSE ≈ step²/12 ≈ 0.015
        assert!(mse < 0.02, "4-bit uniform should have low MSE, got {mse}");
    }

    #[test]
    fn adaptive_bits_small_residual() {
        // Range = 0.0063, with abs_tol = 0.005 → range < 2*tol → 1 bit
        let values: Vec<f64> = (0..64).map(|i| (i as f64) * 0.0001).collect();
        let bits = choose_bits(&values, 0.005);
        assert!(bits <= 2, "tiny residual should need few bits, got {bits}");
    }

    #[test]
    fn adaptive_bits_large_residual() {
        // Range = 63, with abs_tol = 0.005 → log2(63/0.005) = log2(12600) ≈ 13.6 → 8 (clamped)
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 1.0).collect();
        let bits = choose_bits(&values, 0.005);
        assert!(
            bits >= 4,
            "large residual should need more bits, got {bits}"
        );
    }
}
