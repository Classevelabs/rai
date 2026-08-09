use rem_nra::Vec64;

/// Experimental compositional addressing over omega vectors.
pub struct Compositor;

impl Compositor {
    /// Combine multiple omega vectors via normalized averaging.
    /// The result is a heuristic composite query, not a proven concept intersection.
    pub fn intersect(omegas: &[Vec64]) -> Vec64 {
        if omegas.is_empty() {
            panic!("cannot compose zero omega vectors");
        }
        if omegas.len() == 1 {
            return omegas[0].clone();
        }

        let mut combined = omegas[0].clone();
        for omega in &omegas[1..] {
            combined += omega;
        }

        let norm = combined.norm();
        if norm > 1e-10 {
            combined /= norm;
        }

        combined
    }

    /// Weighted combination of omega vectors.
    pub fn weighted_intersect(omegas: &[Vec64], weights: &[f64]) -> Vec64 {
        assert_eq!(omegas.len(), weights.len(), "omegas and weights must match");
        assert!(!omegas.is_empty(), "cannot compose zero omega vectors");

        let mut combined = &omegas[0] * weights[0];
        for (omega, &w) in omegas[1..].iter().zip(&weights[1..]) {
            combined += omega * w;
        }

        let norm = combined.norm();
        if norm > 1e-10 {
            combined /= norm;
        }

        combined
    }

    /// Difference vector: query for "A but not B".
    pub fn difference(positive: &Vec64, negative: &Vec64) -> Vec64 {
        let mut result = positive - negative;
        let norm = result.norm();
        if norm > 1e-10 {
            result /= norm;
        }
        result
    }

    /// Analogy: "A is to B as C is to ?"
    /// Returns omega_C + (omega_B - omega_A).
    pub fn analogy(a: &Vec64, b: &Vec64, c: &Vec64) -> Vec64 {
        let mut result = c + &(b - a);
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
    use nalgebra::DVector;

    #[test]
    fn intersect_normalizes() {
        let a = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let b = DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let result = Compositor::intersect(&[a, b]);
        let norm = result.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn single_omega_returns_same() {
        let a = DVector::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
        let result = Compositor::intersect(std::slice::from_ref(&a));
        assert_eq!(result, a);
    }
}
