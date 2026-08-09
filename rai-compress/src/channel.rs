use nalgebra::DMatrix;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChannelError {
    #[error("invalid channel input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid channel representation: {0}")]
    InvalidRepresentation(&'static str),
    #[error("channel dimensions overflow")]
    SizeOverflow,
    #[error("channel operation produced non-finite values")]
    NumericalFailure,
}

/// Per-channel (per-row) normalization.
///
/// Stage 0 of the pipeline: remove per-row mean and scale.
/// This captures the coarsest structure — some rows have larger
/// magnitudes than others, and removing this before SVD lets the
/// SVD focus on the actual correlation structure.
///
/// Cost: 2 values per row (mean + scale) at FP16 = 32 bits/row = negligible.
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
    pub fn normalize(weights: &DMatrix<f64>) -> Result<(Self, DMatrix<f64>), ChannelError> {
        let rows = weights.nrows();
        let cols = weights.ncols();
        if rows == 0 || cols == 0 {
            return Err(ChannelError::InvalidInput("weights must not be empty"));
        }
        rows.checked_mul(cols).ok_or(ChannelError::SizeOverflow)?;
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(ChannelError::InvalidInput("weights must be finite"));
        }
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

        Ok((
            Self {
                means,
                scales,
                rows,
                cols,
            },
            normalized,
        ))
    }

    /// Denormalize: restore original scale.
    pub fn denormalize(&self, normalized: &DMatrix<f64>) -> Result<DMatrix<f64>, ChannelError> {
        self.validate()?;
        if normalized.shape() != (self.rows, self.cols) {
            return Err(ChannelError::InvalidInput(
                "normalized matrix shape does not match channel metadata",
            ));
        }
        if normalized.iter().any(|value| !value.is_finite()) {
            return Err(ChannelError::InvalidInput(
                "normalized matrix must be finite",
            ));
        }
        let mut result = normalized.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = normalized[(i, j)] * self.scales[i] + self.means[i];
            }
        }
        if result.iter().any(|value| !value.is_finite()) {
            return Err(ChannelError::NumericalFailure);
        }
        Ok(result)
    }

    /// Storage cost in bytes (FP16 per value).
    pub fn size_bytes(&self) -> Result<usize, ChannelError> {
        self.validate()?;
        self.rows.checked_mul(4).ok_or(ChannelError::SizeOverflow)
    }

    /// Check internal consistency of the normalization metadata (shape,
    /// parameter lengths, finiteness, positive scales).
    pub(crate) fn validate(&self) -> Result<(), ChannelError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(ChannelError::InvalidRepresentation(
                "matrix shape must be non-zero",
            ));
        }
        self.rows
            .checked_mul(self.cols)
            .ok_or(ChannelError::SizeOverflow)?;
        if self.means.len() != self.rows || self.scales.len() != self.rows {
            return Err(ChannelError::InvalidRepresentation(
                "parameter lengths do not match row count",
            ));
        }
        if self.means.iter().any(|value| !value.is_finite())
            || self
                .scales
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ChannelError::InvalidRepresentation(
                "parameters must be finite with positive scales",
            ));
        }
        Ok(())
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

        let (norm, normalized) = ChannelNorm::normalize(&w).unwrap();
        let recovered = norm.denormalize(&normalized).unwrap();

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

        let (_norm, normalized) = ChannelNorm::normalize(&w).unwrap();

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

    #[test]
    fn malformed_channel_inputs_are_rejected() {
        assert!(ChannelNorm::normalize(&DMatrix::zeros(0, 0)).is_err());
        let malformed = ChannelNorm {
            means: vec![],
            scales: vec![1.0],
            rows: 1,
            cols: 1,
        };
        assert!(matches!(
            malformed.denormalize(&DMatrix::zeros(1, 1)),
            Err(ChannelError::InvalidRepresentation(_))
        ));
    }
}
