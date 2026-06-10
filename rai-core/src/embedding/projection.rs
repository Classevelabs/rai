use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use rem_nra::Vec64;
use serde::{Deserialize, Serialize};

/// Projects high-dimensional embeddings to NRA/REM dimensions.
///
/// Uses random Gaussian projection (Johnson-Lindenstrauss lemma guarantees
/// approximate distance preservation with high probability).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    /// Projection matrix: target_dim × source_dim
    matrix: DMatrix<f64>,
    /// Source dimension (embedding dim).
    pub source_dim: usize,
    /// Target dimension (omega/key/value dim).
    pub target_dim: usize,
}

impl Projection {
    /// Create a random Gaussian projection (JL lemma).
    pub fn random_gaussian(source_dim: usize, target_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (target_dim as f64).sqrt();
        let matrix = DMatrix::from_fn(target_dim, source_dim, |_, _| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        Self {
            matrix,
            source_dim,
            target_dim,
        }
    }

    /// Learn a PCA-based projection from a set of embeddings.
    pub fn from_pca(embeddings: &[Vec<f64>], target_dim: usize) -> Self {
        let n = embeddings.len();
        let source_dim = embeddings[0].len();

        if n < target_dim || source_dim < target_dim {
            // Not enough data for PCA, fall back to random
            let mut rng = rand::thread_rng();
            return Self::random_gaussian(source_dim, target_dim, &mut rng);
        }

        // Center the data
        let mut mean = vec![0.0f64; source_dim];
        for emb in embeddings {
            for (i, v) in emb.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        // Build covariance matrix (source_dim × source_dim) via data matrix
        // For efficiency, compute X^T X where X is n × source_dim
        let mut data = DMatrix::zeros(n, source_dim);
        for (i, emb) in embeddings.iter().enumerate() {
            for (j, v) in emb.iter().enumerate() {
                data[(i, j)] = v - mean[j];
            }
        }

        let cov = data.transpose() * &data / (n as f64 - 1.0);

        // Eigendecomposition — use symmetric eigen
        let eigen = cov.symmetric_eigen();

        // Take top target_dim eigenvectors (eigenvalues are in ascending order in nalgebra)
        let mut matrix = DMatrix::zeros(target_dim, source_dim);
        for i in 0..target_dim {
            let col_idx = source_dim - 1 - i; // descending order
            for j in 0..source_dim {
                matrix[(i, j)] = eigen.eigenvectors[(j, col_idx)];
            }
        }

        Self {
            matrix,
            source_dim,
            target_dim,
        }
    }

    /// Project an embedding vector to target dimension.
    pub fn project(&self, embedding: &[f64]) -> Vec64 {
        let v = DVector::from_row_slice(embedding);
        &self.matrix * v
    }

    /// Project an embedding vector, then normalize to unit length.
    pub fn project_normalized(&self, embedding: &[f64]) -> Vec64 {
        let mut result = self.project(embedding);
        let norm = result.norm();
        if norm > 1e-10 {
            result /= norm;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_projection_preserves_dimensions() {
        let mut rng = rand::thread_rng();
        let proj = Projection::random_gaussian(384, 32, &mut rng);
        let embedding = vec![0.5; 384];
        let result = proj.project(&embedding);
        assert_eq!(result.nrows(), 32);
    }

    #[test]
    fn pca_projection_works() {
        let mut rng = rand::thread_rng();
        // Generate some correlated embeddings
        let embeddings: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                (0..64)
                    .map(|j| {
                        (i as f64 * 0.1 + j as f64 * 0.05).sin()
                            + rng.sample::<f64, _>(StandardNormal) * 0.01
                    })
                    .collect()
            })
            .collect();
        let proj = Projection::from_pca(&embeddings, 8);
        assert_eq!(proj.target_dim, 8);
        assert_eq!(proj.source_dim, 64);

        let result = proj.project(&embeddings[0]);
        assert_eq!(result.nrows(), 8);
    }
}
