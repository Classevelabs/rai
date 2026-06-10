use nalgebra::DMatrix;

/// Per-channel (per-row) normalization.
///
/// Stage 0 of the pipeline: remove per-row mean and scale.
/// This captures the coarsest structure — some rows have larger
/// magnitudes than others, and removing this before SVD lets the
/// SVD focus on the actual correlation structure.
///
/// Cost: 2 values per row (mean + scale) = 16 bits/row = negligible.
#[derive(Debug, Clone)]
pub struct ChannelNorm {
    /// Per-row means.
    pub means: Vec<f64>,
    /// Per-row scales (std dev).
    pub scales: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of cols.
    pub cols: usize,
}

impl ChannelNorm {
    /// Compute and apply channel normalization.
    pub fn normalize(weights: &DMatrix<f64>) -> (Self, DMatrix<f64>) {
        let rows = weights.nrows();
        let cols = weights.ncols();
        let mut means = Vec::with_capacity(rows);
        let mut scales = Vec::with_capacity(rows);
        let mut normalized = weights.clone();

        for i in 0..rows {
            let row = weights.row(i);
            let mean: f64 = row.iter().sum::<f64>() / cols as f64;
            let var: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / cols as f64;
            let scale = var.sqrt().max(1e-10);

            means.push(mean);
            scales.push(scale);

            for j in 0..cols {
                normalized[(i, j)] = (weights[(i, j)] - mean) / scale;
            }
        }

        (
            Self {
                means,
                scales,
                rows,
                cols,
            },
            normalized,
        )
    }

    /// Denormalize: restore original scale.
    pub fn denormalize(&self, normalized: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = normalized.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = normalized[(i, j)] * self.scales[i] + self.means[i];
            }
        }
        result
    }

    /// Storage cost in bytes (FP16 per value).
    pub fn size_bytes(&self) -> usize {
        self.rows * 4 // 2 bytes mean + 2 bytes scale per row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_denormalize_roundtrip() {
        let w = DMatrix::from_row_slice(
            3,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0,
            ],
        );

        let (norm, normalized) = ChannelNorm::normalize(&w);
        let recovered = norm.denormalize(&normalized);

        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (w[(i, j)] - recovered[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn normalized_rows_have_unit_variance() {
        let w = DMatrix::from_row_slice(2, 4, &[10.0, 20.0, 30.0, 40.0, -5.0, -10.0, -15.0, -20.0]);

        let (_norm, normalized) = ChannelNorm::normalize(&w);

        for i in 0..2 {
            let row = normalized.row(i);
            let mean: f64 = row.iter().sum::<f64>() / 4.0;
            let var: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4.0;
            assert!(
                (mean).abs() < 1e-10,
                "row {i} mean should be ~0, got {mean}"
            );
            assert!(
                (var - 1.0).abs() < 1e-10,
                "row {i} var should be ~1, got {var}"
            );
        }
    }
}
