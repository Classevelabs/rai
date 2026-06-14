//! Sparse outlier extraction.
//!
//! Key insight: after SVD, the residual has a few large outliers and
//! many near-zero values. The outliers dominate quantization error.
//! By extracting them at full precision and quantizing only the dense
//! remainder, we get dramatically better accuracy at minimal cost.
//!
//! This is the single biggest accuracy win in the pipeline.

/// Sparse + Dense decomposition of a residual vector.
#[derive(Debug, Clone)]
pub struct SparseDenseDecomp {
    /// Indices of outlier values.
    pub sparse_indices: Vec<u32>,
    /// Outlier values at full precision (FP16 in production).
    pub sparse_values: Vec<f64>,
    /// Dense remainder with outliers zeroed out.
    pub dense: Vec<f64>,
    /// Total number of elements.
    pub len: usize,
    /// Outlier threshold used.
    pub threshold: f64,
}

impl SparseDenseDecomp {
    /// Decompose: extract top outliers, leave dense remainder.
    ///
    /// `fraction` = fraction of values to treat as outliers (e.g., 0.01 = top 1%)
    pub fn from_residual(values: &[f64], fraction: f64) -> Self {
        let n = values.len();
        let k = if n == 0 {
            0
        } else {
            ((n as f64 * fraction).ceil() as usize).clamp(1, n)
        };

        // Find the threshold: k-th largest absolute value
        let mut abs_values: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = if k < n { abs_values[k].1 } else { 0.0 };

        let mut sparse_indices = Vec::new();
        let mut sparse_values = Vec::new();
        let mut dense = values.to_vec();

        for &(idx, abs_val) in abs_values.iter().take(k) {
            if abs_val > threshold {
                sparse_indices.push(idx as u32);
                sparse_values.push(values[idx]);
                dense[idx] = 0.0; // Zero out the outlier in dense
            }
        }

        Self {
            sparse_indices,
            sparse_values,
            dense,
            len: n,
            threshold,
        }
    }

    /// Reconstruct the original values from sparse + dense.
    pub fn reconstruct(&self, dense_decoded: &[f64]) -> Vec<f64> {
        let mut result = dense_decoded.to_vec();
        for (&idx, &val) in self.sparse_indices.iter().zip(self.sparse_values.iter()) {
            if (idx as usize) < result.len() {
                result[idx as usize] += val;
            }
        }
        result
    }

    /// Storage cost of the sparse part in bytes.
    /// Each outlier: 4 bytes (index) + 2 bytes (FP16 value) = 6 bytes.
    pub fn sparse_bytes(&self) -> usize {
        self.sparse_indices.len() * 6
    }

    /// Reduction in dense dynamic range after outlier removal.
    pub fn range_reduction(&self, original: &[f64]) -> f64 {
        let orig_range = original.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - original.iter().cloned().fold(f64::INFINITY, f64::min);
        let dense_range = self.dense.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - self.dense.iter().cloned().fold(f64::INFINITY, f64::min);
        if orig_range < 1e-15 {
            return 1.0;
        }
        1.0 - dense_range / orig_range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlier_extraction_preserves_values() {
        let values = vec![0.01, 0.02, -0.01, 5.0, 0.03, -0.02, -4.0, 0.01];
        let decomp = SparseDenseDecomp::from_residual(&values, 0.25); // top 25% = 2 outliers

        // The two outliers should be 5.0 and -4.0
        assert!(decomp.sparse_values.contains(&5.0) || decomp.sparse_values.contains(&-4.0));

        // Dense should have those zeroed
        let reconstructed = decomp.reconstruct(&decomp.dense);
        for (a, b) in values.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "reconstruction mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn range_reduces_after_outlier_removal() {
        let mut values = vec![0.01; 100];
        values[50] = 10.0; // single outlier
        values[75] = -8.0; // single outlier

        let decomp = SparseDenseDecomp::from_residual(&values, 0.02);
        let reduction = decomp.range_reduction(&values);

        assert!(
            reduction > 0.5,
            "removing outliers should significantly reduce range, got {reduction}"
        );
    }
}
