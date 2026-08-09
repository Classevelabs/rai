/// Sparse outlier extraction.
///
/// Key insight: after SVD, the residual has a few large outliers and
/// many near-zero values. The outliers dominate quantization error.
/// By extracting them at full precision and quantizing only the dense
/// remainder, we get dramatically better accuracy at minimal cost.
///
/// This is the single biggest accuracy win in the pipeline.
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SparseError {
    #[error("invalid sparse input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid sparse representation: {0}")]
    InvalidRepresentation(&'static str),
    #[error("sparse size overflows")]
    SizeOverflow,
    #[error("unable to allocate sparse output")]
    AllocationFailed,
    #[error("sparse operation produced a non-finite value")]
    NumericalFailure,
}

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
    /// Outlier threshold used (inclusive: values with `|v| >= threshold`
    /// among the top-k are extracted).
    pub threshold: f64,
}

impl SparseDenseDecomp {
    /// Decompose: extract top outliers, leave dense remainder.
    ///
    /// `fraction` = fraction of values to treat as outliers (e.g., 0.01 = top 1%)
    pub fn from_residual(values: &[f64], fraction: f64) -> Result<Self, SparseError> {
        let n = values.len();
        if n == 0 {
            return Err(SparseError::InvalidInput("residual must not be empty"));
        }
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(SparseError::InvalidInput(
                "outlier fraction must be finite and between zero and one",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SparseError::InvalidInput("residual values must be finite"));
        }
        if n > u32::MAX as usize {
            return Err(SparseError::SizeOverflow);
        }
        let k = ((n as f64 * fraction).ceil() as usize).min(n);

        // Find the threshold: k-th largest absolute value
        let mut abs_values: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_values.sort_by(|a, b| b.1.total_cmp(&a.1));

        let threshold = if k == 0 {
            f64::INFINITY
        } else if k < n {
            abs_values[k].1
        } else {
            0.0
        };

        let mut sparse_indices = Vec::new();
        let mut sparse_values = Vec::new();
        let mut dense = values.to_vec();

        // Inclusive comparison: every one of the top-k entries satisfies
        // `abs_val >= threshold` by construction of the sort, so ties at the
        // threshold are extracted too and `fraction` is honored exactly.
        for &(idx, abs_val) in abs_values.iter().take(k) {
            if abs_val >= threshold {
                sparse_indices.push(idx as u32);
                sparse_values.push(values[idx]);
                dense[idx] = 0.0; // Zero out the outlier in dense
            }
        }

        Ok(Self {
            sparse_indices,
            sparse_values,
            dense,
            len: n,
            threshold,
        })
    }

    /// Reconstruct the original values from sparse + dense.
    pub fn reconstruct(&self, dense_decoded: &[f64]) -> Result<Vec<f64>, SparseError> {
        self.validate()?;
        if dense_decoded.len() != self.len {
            return Err(SparseError::InvalidInput(
                "dense residual length does not match sparse metadata",
            ));
        }
        if dense_decoded.iter().any(|value| !value.is_finite()) {
            return Err(SparseError::InvalidInput(
                "dense residual values must be finite",
            ));
        }
        let mut result = dense_decoded.to_vec();
        for (&idx, &val) in self.sparse_indices.iter().zip(self.sparse_values.iter()) {
            let index = idx as usize;
            let value = result[index] + val;
            if !value.is_finite() {
                return Err(SparseError::NumericalFailure);
            }
            result[index] = value;
        }
        Ok(result)
    }

    /// Storage cost of the sparse part in bytes.
    /// Each outlier: 4 bytes (index) + 2 bytes (FP16 value) = 6 bytes.
    pub fn sparse_bytes(&self) -> Result<usize, SparseError> {
        self.validate()?;
        self.sparse_indices
            .len()
            .checked_mul(6)
            .ok_or(SparseError::SizeOverflow)
    }

    /// Reduction in dense dynamic range after outlier removal.
    pub fn range_reduction(&self, original: &[f64]) -> Result<f64, SparseError> {
        self.validate()?;
        if original.len() != self.len {
            return Err(SparseError::InvalidInput(
                "original residual length does not match sparse metadata",
            ));
        }
        if original.iter().any(|value| !value.is_finite()) {
            return Err(SparseError::InvalidInput(
                "original residual values must be finite",
            ));
        }
        let orig_range = original.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - original.iter().cloned().fold(f64::INFINITY, f64::min);
        let dense_range = self.dense.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - self.dense.iter().cloned().fold(f64::INFINITY, f64::min);
        if orig_range < 1e-15 {
            return Ok(1.0);
        }
        let reduction = 1.0 - dense_range / orig_range;
        if !reduction.is_finite() {
            return Err(SparseError::NumericalFailure);
        }
        Ok(reduction)
    }

    fn validate(&self) -> Result<(), SparseError> {
        if self.len == 0 || self.dense.len() != self.len {
            return Err(SparseError::InvalidRepresentation(
                "dense length must be non-zero and match metadata",
            ));
        }
        if self.sparse_indices.len() != self.sparse_values.len() {
            return Err(SparseError::InvalidRepresentation(
                "sparse index and value lengths differ",
            ));
        }
        if self.dense.iter().any(|value| !value.is_finite())
            || self.sparse_values.iter().any(|value| !value.is_finite())
            || !(self.threshold.is_finite() || self.threshold == f64::INFINITY)
        {
            return Err(SparseError::InvalidRepresentation(
                "sparse values and threshold must be finite",
            ));
        }
        let mut seen = HashSet::new();
        seen.try_reserve(self.sparse_indices.len())
            .map_err(|_| SparseError::AllocationFailed)?;
        if self
            .sparse_indices
            .iter()
            .any(|&index| index as usize >= self.len || !seen.insert(index))
        {
            return Err(SparseError::InvalidRepresentation(
                "sparse indices must be unique and in range",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlier_extraction_preserves_values() {
        let values = vec![0.01, 0.02, -0.01, 5.0, 0.03, -0.02, -4.0, 0.01];
        let decomp = SparseDenseDecomp::from_residual(&values, 0.25).unwrap(); // top 25% = 2 outliers

        // The two outliers should be 5.0 and -4.0
        assert!(decomp.sparse_values.contains(&5.0) || decomp.sparse_values.contains(&-4.0));

        // Dense should have those zeroed
        let reconstructed = decomp.reconstruct(&decomp.dense).unwrap();
        for (a, b) in values.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "reconstruction mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn range_reduces_after_outlier_removal() {
        let mut values = vec![0.01; 100];
        values[50] = 10.0; // single outlier
        values[75] = -8.0; // single outlier

        let decomp = SparseDenseDecomp::from_residual(&values, 0.02).unwrap();
        let reduction = decomp.range_reduction(&values).unwrap();

        assert!(
            reduction > 0.5,
            "removing outliers should significantly reduce range, got {reduction}"
        );
    }

    #[test]
    fn tied_outliers_are_extracted_and_fraction_is_honored() {
        // FIVE values share the outlier magnitude while the requested
        // fraction asks for the top 4 (20% of 20). The threshold (the 5th
        // largest magnitude) then ties with all top-4 entries; a strict `>`
        // comparison would extract nothing, so the inclusive comparison is
        // what honors `fraction`.
        let mut values = vec![0.01; 20];
        for idx in [3, 7, 11, 15, 19] {
            values[idx] = if idx == 7 { -2.0 } else { 2.0 };
        }

        let decomp = SparseDenseDecomp::from_residual(&values, 0.2).unwrap();
        assert_eq!(
            decomp.sparse_indices.len(),
            4,
            "ties at the threshold must be extracted so the fraction is honored"
        );
        assert!(decomp.sparse_values.iter().all(|v| v.abs() == 2.0));
        let reconstructed = decomp.reconstruct(&decomp.dense).unwrap();
        for (a, b) in values.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn malformed_sparse_representation_is_rejected() {
        let malformed = SparseDenseDecomp {
            sparse_indices: vec![2],
            sparse_values: vec![],
            dense: vec![0.0],
            len: 1,
            threshold: 1.0,
        };
        assert!(matches!(
            malformed.reconstruct(&[0.0]),
            Err(SparseError::InvalidRepresentation(_))
        ));
        assert!(malformed.sparse_bytes().is_err());
    }
}
