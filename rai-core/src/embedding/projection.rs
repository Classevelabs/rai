use crate::RaiError;
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

    /// Project an embedding vector to target dimension.
    ///
    /// # Errors
    ///
    /// Returns [`RaiError::EmbeddingError`] when `embedding` is not `source_dim` long. The
    /// matrix product would otherwise panic on the dimension mismatch.
    pub fn project(&self, embedding: &[f64]) -> Result<Vec64, RaiError> {
        if embedding.len() != self.source_dim || self.matrix.ncols() != embedding.len() {
            return Err(RaiError::EmbeddingError(format!(
                "projection expects {} input dimensions, got {}",
                self.source_dim,
                embedding.len()
            )));
        }
        let v = DVector::from_row_slice(embedding);
        Ok(&self.matrix * v)
    }

    /// Project an embedding vector, then normalize to unit length.
    ///
    /// # Errors
    ///
    /// Same as [`Projection::project`].
    pub fn project_normalized(&self, embedding: &[f64]) -> Result<Vec64, RaiError> {
        let mut result = self.project(embedding)?;
        let norm = result.norm();
        if norm > 1e-10 {
            result /= norm;
        }
        Ok(result)
    }

    /// Validate serialized metadata against the actual matrix payload.
    pub(crate) fn validate_shape(&self) -> bool {
        self.source_dim > 0
            && self.target_dim > 0
            && self.matrix.nrows() == self.target_dim
            && self.matrix.ncols() == self.source_dim
            && self
                .matrix
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1.0e100)
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
        let result = proj.project(&embedding).expect("matching dimensions");
        assert_eq!(result.nrows(), 32);
    }

    #[test]
    fn dimension_mismatch_is_an_error_not_a_panic() {
        let mut rng = rand::thread_rng();
        let proj = Projection::random_gaussian(384, 32, &mut rng);
        assert!(proj.project(&[0.5; 383]).is_err());
        assert!(proj.project_normalized(&[0.5; 385]).is_err());
    }
}
