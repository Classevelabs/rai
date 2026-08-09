use nalgebra::{DMatrix, DVector, Dyn, SVD};

/// SVD with an explicit iteration cap that surfaces non-convergence as an
/// error. nalgebra's `svd()` unwraps its own `try_svd` internally (panicking
/// on failure), and its `max_niter = 0` mode iterates without bound, so a
/// finite cap is the only way to turn a non-converging decomposition into a
/// recoverable error instead of a panic or a hang. Convergence typically
/// takes a small multiple of the number of singular values; the cap below is
/// an order of magnitude beyond that, with a generous floor for tiny inputs.
fn checked_svd(weights: &DMatrix<f64>) -> Result<SVD<f64, Dyn, Dyn>, PriorError> {
    let min_dim = weights.nrows().min(weights.ncols());
    let max_niter = min_dim.saturating_mul(30).max(1024);
    weights
        .clone()
        .try_svd(true, true, f64::EPSILON, max_niter)
        .ok_or(PriorError::NumericalFailure(
            "SVD did not converge within the iteration limit",
        ))
}

/// Errors returned for invalid low-rank-prior inputs or representations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PriorError {
    #[error("invalid prior input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid prior representation: {0}")]
    InvalidRepresentation(&'static str),
    #[error("prior dimensions overflow")]
    SizeOverflow,
    #[error("prior numerical failure: {0}")]
    NumericalFailure(&'static str),
}

/// Low-rank prior that captures weight matrix structure.
///
/// W ≈ U_k × diag(S_k) × V_k^T
///
/// The key insight from REM: structured data has small residuals relative
/// to a good prior. LLM weight matrices have massive low-rank structure
/// (attention matrices especially), so the prior captures most of the
/// information and the residual is small → fewer bits needed.
#[derive(Debug, Clone)]
pub struct WeightPrior {
    /// Left singular vectors: (rows, rank)
    pub u: DMatrix<f64>,
    /// Singular values: (rank,)
    pub s: DVector<f64>,
    /// Right singular vectors: (cols, rank)
    pub v: DMatrix<f64>,
    /// Rank of approximation.
    pub rank: usize,
    /// Original matrix shape.
    pub rows: usize,
    pub cols: usize,
}

impl WeightPrior {
    /// Learn a low-rank prior from a weight matrix via truncated SVD.
    ///
    /// `rank` controls the tradeoff:
    /// - Higher rank = better prior = smaller residuals = fewer bits needed
    /// - But the prior itself takes storage: rank × (rows + cols + 1) × 16 bits
    ///
    /// Optimal rank minimizes: prior_size + residual_bits
    pub fn from_weights(weights: &DMatrix<f64>, rank: usize) -> Result<Self, PriorError> {
        let rows = weights.nrows();
        let cols = weights.ncols();
        if rows == 0 || cols == 0 {
            return Err(PriorError::InvalidInput("weights must not be empty"));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::InvalidInput("weights must be finite"));
        }
        let max_rank = rows.min(cols);
        if rank == 0 || rank > max_rank {
            return Err(PriorError::InvalidInput(
                "rank must be between one and the smaller matrix dimension",
            ));
        }
        rows.checked_mul(cols).ok_or(PriorError::SizeOverflow)?;

        let svd = checked_svd(weights)?;
        let u_full = svd
            .u
            .ok_or(PriorError::NumericalFailure("SVD did not return U"))?;
        let s_full = svd.singular_values;
        let v_full = svd
            .v_t
            .ok_or(PriorError::NumericalFailure("SVD did not return V^T"))?
            .transpose();
        if s_full.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::NumericalFailure(
                "SVD returned non-finite singular values",
            ));
        }

        // Truncate to requested rank
        let u = u_full.columns(0, rank).into_owned();
        let s = s_full.rows(0, rank).into_owned();
        let v = v_full.columns(0, rank).into_owned();

        Ok(Self {
            u,
            s,
            v,
            rank,
            rows,
            cols,
        })
    }

    /// Predict (reconstruct) the weight matrix from the prior.
    pub fn predict(&self) -> Result<DMatrix<f64>, PriorError> {
        self.validate()?;
        let s_diag = DMatrix::from_diagonal(&self.s);
        let predicted = &self.u * s_diag * self.v.transpose();
        if predicted.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::NumericalFailure(
                "reconstruction produced non-finite values",
            ));
        }
        Ok(predicted)
    }

    /// Compute the residual: W - Prior(W).
    pub fn residual(&self, weights: &DMatrix<f64>) -> Result<DMatrix<f64>, PriorError> {
        if weights.shape() != (self.rows, self.cols) {
            return Err(PriorError::InvalidInput(
                "weight shape does not match the prior",
            ));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::InvalidInput("weights must be finite"));
        }
        let residual = weights - self.predict()?;
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::NumericalFailure(
                "residual contains non-finite values",
            ));
        }
        Ok(residual)
    }

    /// Storage cost of the prior in bytes (FP16 = 2 bytes per value).
    pub fn prior_size_bytes(&self) -> Result<usize, PriorError> {
        self.validate()?;
        let values = self
            .rows
            .checked_mul(self.rank)
            .and_then(|value| value.checked_add(self.rank))
            .and_then(|value| {
                self.cols
                    .checked_mul(self.rank)
                    .and_then(|right| value.checked_add(right))
            })
            .ok_or(PriorError::SizeOverflow)?;
        values.checked_mul(2).ok_or(PriorError::SizeOverflow)
    }

    /// Fraction of total variance captured by the prior.
    pub fn variance_explained(&self, weights: &DMatrix<f64>) -> Result<f64, PriorError> {
        self.validate()?;
        if weights.shape() != (self.rows, self.cols) {
            return Err(PriorError::InvalidInput(
                "weight shape does not match the prior",
            ));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::InvalidInput("weights must be finite"));
        }
        let total_var: f64 = weights.iter().map(|x| x * x).sum();
        let prior_var: f64 = self.s.iter().map(|x| x * x).sum();
        if !total_var.is_finite() || !prior_var.is_finite() {
            return Err(PriorError::NumericalFailure(
                "variance calculation overflowed",
            ));
        }
        if total_var < 1e-15 {
            return Ok(1.0);
        }
        Ok((prior_var / total_var).clamp(0.0, 1.0))
    }

    /// Find optimal rank that minimizes total compressed size.
    ///
    /// Balances: prior_size (grows with rank) vs residual_bits (shrinks with rank).
    pub fn optimal_rank(
        weights: &DMatrix<f64>,
        block_size: usize,
        target_bits: f64,
    ) -> Result<usize, PriorError> {
        let rows = weights.nrows();
        let cols = weights.ncols();
        if rows == 0 || cols == 0 {
            return Err(PriorError::InvalidInput("weights must not be empty"));
        }
        if block_size == 0 {
            return Err(PriorError::InvalidInput("block_size must be non-zero"));
        }
        if !target_bits.is_finite() || target_bits <= 0.0 {
            return Err(PriorError::InvalidInput(
                "target_bits must be finite and positive",
            ));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::InvalidInput("weights must be finite"));
        }
        let max_rank = rows.min(cols);
        let total_weights = rows.checked_mul(cols).ok_or(PriorError::SizeOverflow)?;

        let svd = checked_svd(weights)?;
        let s = svd.singular_values;
        if s.iter().any(|value| !value.is_finite()) {
            return Err(PriorError::NumericalFailure(
                "SVD returned non-finite singular values",
            ));
        }

        let mut best_rank = 1;
        let mut best_total = f64::INFINITY;

        let total_var: f64 = s.iter().map(|x| x * x).sum();

        for rank in 1..=max_rank.min(64) {
            // Prior storage (FP16)
            let prior_values = rows
                .checked_mul(rank)
                .and_then(|value| value.checked_add(rank))
                .and_then(|value| {
                    cols.checked_mul(rank)
                        .and_then(|right| value.checked_add(right))
                })
                .ok_or(PriorError::SizeOverflow)?;
            let prior_bits = prior_values as f64 * 16.0;

            // Residual variance
            let captured: f64 = s.rows(0, rank).iter().map(|x| x * x).sum();
            let residual_var = (total_var - captured).max(0.0);
            let avg_residual_magnitude = (residual_var / total_weights as f64).sqrt();

            // Estimate bits needed for residual
            // Higher residual → more bits; lower → fewer bits
            let residual_bits_per_weight = if avg_residual_magnitude < 1e-6 {
                0.5
            } else {
                // Information-theoretic: bits ≈ log2(range / precision)
                // With prior, range is much smaller
                (avg_residual_magnitude * 10.0)
                    .log2()
                    .max(0.5)
                    .min(target_bits)
            };

            let residual_bits = total_weights as f64 * residual_bits_per_weight;
            let total = prior_bits + residual_bits;

            if total < best_total {
                best_total = total;
                best_rank = rank;
            }
        }

        Ok(best_rank)
    }

    fn validate(&self) -> Result<(), PriorError> {
        if self.rows == 0 || self.cols == 0 || self.rank == 0 {
            return Err(PriorError::InvalidRepresentation(
                "rows, columns, and rank must be non-zero",
            ));
        }
        if self.rank > self.rows.min(self.cols) {
            return Err(PriorError::InvalidRepresentation(
                "rank exceeds a matrix dimension",
            ));
        }
        if self.u.shape() != (self.rows, self.rank)
            || self.s.len() != self.rank
            || self.v.shape() != (self.cols, self.rank)
        {
            return Err(PriorError::InvalidRepresentation(
                "factor dimensions do not match metadata",
            ));
        }
        if self
            .u
            .iter()
            .chain(self.s.iter())
            .chain(self.v.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PriorError::InvalidRepresentation(
                "factors must contain only finite values",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_rank_matrix_has_small_residual() {
        // Create a rank-2 matrix
        let u = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]);
        let s = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.5]));
        let v = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, -0.5]);
        let w = &u * s * v.transpose();

        let prior = WeightPrior::from_weights(&w, 2).unwrap();
        let residual = prior.residual(&w).unwrap();
        let residual_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            residual_norm < 1e-10,
            "rank-2 matrix should have zero residual with rank-2 prior, got {residual_norm}"
        );
    }

    #[test]
    fn variance_explained_increases_with_rank() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0x5EED_0001);
        let w = DMatrix::from_fn(32, 16, |i, j| {
            // Low-rank + noise
            (i as f64 * 0.1).sin() * (j as f64 * 0.2).cos()
                + rand::Rng::gen_range(&mut rng, -0.01..0.01)
        });

        let var1 = WeightPrior::from_weights(&w, 1)
            .unwrap()
            .variance_explained(&w)
            .unwrap();
        let var4 = WeightPrior::from_weights(&w, 4)
            .unwrap()
            .variance_explained(&w)
            .unwrap();

        assert!(
            var4 > var1,
            "more rank should explain more variance: r1={var1}, r4={var4}"
        );
    }

    #[test]
    fn rejects_empty_nonfinite_and_invalid_rank_inputs() {
        assert!(WeightPrior::from_weights(&DMatrix::zeros(0, 0), 1).is_err());
        assert!(WeightPrior::from_weights(&DMatrix::from_element(1, 1, f64::NAN), 1).is_err());
        assert!(WeightPrior::from_weights(&DMatrix::from_element(1, 1, 1.0), 0).is_err());
        assert!(WeightPrior::from_weights(&DMatrix::from_element(1, 1, 1.0), 2).is_err());
    }

    #[test]
    fn rejects_inconsistent_public_representation() {
        let malformed = WeightPrior {
            u: DMatrix::identity(1, 1),
            s: DVector::from_element(1, 1.0),
            v: DMatrix::identity(1, 1),
            rank: 1,
            rows: 2,
            cols: 1,
        };
        assert!(matches!(
            malformed.predict(),
            Err(PriorError::InvalidRepresentation(_))
        ));
    }
}
