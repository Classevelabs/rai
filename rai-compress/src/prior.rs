use nalgebra::{DMatrix, DVector};

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
    pub fn from_weights(weights: &DMatrix<f64>, rank: usize) -> Self {
        let rows = weights.nrows();
        let cols = weights.ncols();
        let effective_rank = rank.min(rows.min(cols));

        let svd = weights.clone().svd(true, true);
        let u_full = svd.u.unwrap();
        let s_full = svd.singular_values;
        let v_full = svd.v_t.unwrap().transpose();

        // Truncate to requested rank
        let u = u_full.columns(0, effective_rank).into_owned();
        let s = s_full.rows(0, effective_rank).into_owned();
        let v = v_full.columns(0, effective_rank).into_owned();

        Self {
            u,
            s,
            v,
            rank: effective_rank,
            rows,
            cols,
        }
    }

    /// Predict (reconstruct) the weight matrix from the prior.
    pub fn predict(&self) -> DMatrix<f64> {
        let s_diag = DMatrix::from_diagonal(&self.s);
        &self.u * s_diag * self.v.transpose()
    }

    /// Compute the residual: W - Prior(W).
    pub fn residual(&self, weights: &DMatrix<f64>) -> DMatrix<f64> {
        weights - self.predict()
    }

    /// Storage cost of the prior in bytes (FP16 = 2 bytes per value).
    pub fn prior_size_bytes(&self) -> usize {
        let values = self.u.nrows() * self.rank  // U
            + self.rank                           // S
            + self.v.nrows() * self.rank; // V
        values * 2 // FP16
    }

    /// Fraction of total variance captured by the prior.
    pub fn variance_explained(&self, weights: &DMatrix<f64>) -> f64 {
        let total_var: f64 = weights.iter().map(|x| x * x).sum();
        let prior_var: f64 = self.s.iter().map(|x| x * x).sum();
        if total_var < 1e-15 {
            return 1.0;
        }
        prior_var / total_var
    }

    /// Find optimal rank that minimizes total compressed size.
    ///
    /// Balances: prior_size (grows with rank) vs residual_bits (shrinks with rank).
    pub fn optimal_rank(weights: &DMatrix<f64>, _block_size: usize, target_bits: f64) -> usize {
        let rows = weights.nrows();
        let cols = weights.ncols();
        let max_rank = rows.min(cols);
        let total_weights = rows * cols;

        let svd = weights.clone().svd(true, true);
        let s = svd.singular_values;

        let mut best_rank = 1;
        let mut best_total = f64::INFINITY;

        let total_var: f64 = s.iter().map(|x| x * x).sum();

        for rank in 1..=max_rank.min(64) {
            // Prior storage (FP16)
            let prior_bits = (rows * rank + rank + cols * rank) as f64 * 16.0;

            // Residual variance
            let captured: f64 = s.rows(0, rank).iter().map(|x| x * x).sum();
            let residual_var = total_var - captured;
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

        best_rank
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

        let prior = WeightPrior::from_weights(&w, 2);
        let residual = prior.residual(&w);
        let residual_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            residual_norm < 1e-10,
            "rank-2 matrix should have zero residual with rank-2 prior, got {residual_norm}"
        );
    }

    #[test]
    fn variance_explained_increases_with_rank() {
        let mut rng = rand::thread_rng();
        let w = DMatrix::from_fn(32, 16, |i, j| {
            // Low-rank + noise
            (i as f64 * 0.1).sin() * (j as f64 * 0.2).cos()
                + rand::Rng::gen_range(&mut rng, -0.01..0.01)
        });

        let var1 = WeightPrior::from_weights(&w, 1).variance_explained(&w);
        let var4 = WeightPrior::from_weights(&w, 4).variance_explained(&w);

        assert!(
            var4 > var1,
            "more rank should explain more variance: r1={var1}, r4={var4}"
        );
    }
}
