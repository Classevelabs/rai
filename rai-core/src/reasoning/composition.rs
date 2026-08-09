use crate::RaiError;
use rem_nra::Vec64;

/// Experimental compositional addressing over address vectors.
pub struct Compositor;

impl Compositor {
    /// Combine multiple address vectors via normalized averaging.
    ///
    /// The result is a heuristic composite query, not a proven concept intersection.
    ///
    /// # Errors
    ///
    /// Returns [`RaiError::InvalidInput`] when no vectors are supplied or when they do not all
    /// share one dimension. Both were previously panics reachable from the public API.
    pub fn intersect(omegas: &[Vec64]) -> Result<Vec64, RaiError> {
        let Some(first) = omegas.first() else {
            return Err(RaiError::InvalidInput(
                "cannot compose zero address vectors".to_string(),
            ));
        };
        let dimension = first.len();
        if omegas.iter().any(|omega| omega.len() != dimension) {
            return Err(RaiError::InvalidInput(
                "every address vector must share one dimension".to_string(),
            ));
        }
        if omegas.len() == 1 {
            return Ok(first.clone());
        }

        let mut combined = first.clone();
        for omega in &omegas[1..] {
            combined += omega;
        }

        let norm = combined.norm();
        if norm > 1e-10 {
            combined /= norm;
        }

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn intersect_normalizes() {
        let a = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let b = DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let result = Compositor::intersect(&[a, b]).expect("valid composition");
        let norm = result.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn single_omega_returns_same() {
        let a = DVector::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
        let result = Compositor::intersect(std::slice::from_ref(&a)).expect("valid composition");
        assert_eq!(result, a);
    }

    #[test]
    fn empty_and_ragged_input_is_an_error_not_a_panic() {
        assert!(Compositor::intersect(&[]).is_err());
        assert!(Compositor::intersect(&[
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
        ])
        .is_err());
    }
}
