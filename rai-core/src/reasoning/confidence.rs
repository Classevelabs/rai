use crate::types::ConfidenceLevel;

/// Energy thresholds for confidence gating.
pub struct ConfidenceGate {
    /// Below this energy = HIGH confidence.
    pub high_threshold: f64,
    /// Below this energy = MEDIUM confidence.
    pub medium_threshold: f64,
    /// Gradient norm multiplier for no-match detection.
    pub no_match_grad_factor: f64,
    /// Base ODE tolerance (from NRA config).
    pub ode_tol: f64,
}

impl Default for ConfidenceGate {
    fn default() -> Self {
        Self {
            high_threshold: -3.0,
            medium_threshold: -1.0,
            no_match_grad_factor: 100.0,
            ode_tol: 1e-7,
        }
    }
}

impl ConfidenceGate {
    /// Determine confidence from energy and gradient norm.
    pub fn classify(&self, energy: f64, grad_norm: f64) -> ConfidenceLevel {
        // No match: gradient didn't converge
        if grad_norm > self.ode_tol * self.no_match_grad_factor {
            return ConfidenceLevel::NoMatch;
        }

        if energy < self.high_threshold {
            ConfidenceLevel::High
        } else if energy < self.medium_threshold {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Generate a human-readable explanation.
    pub fn explain(&self, energy: f64, grad_norm: f64, confidence: ConfidenceLevel) -> String {
        match confidence {
            ConfidenceLevel::High => format!(
                "HIGH confidence: energy={energy:.3} is well below threshold {:.1}. \
                 Strong attractor found — this memory is firmly established.",
                self.high_threshold
            ),
            ConfidenceLevel::Medium => format!(
                "MEDIUM confidence: energy={energy:.3} is between {:.1} and {:.1}. \
                 Moderate attractor — memory exists but is not as firmly anchored.",
                self.high_threshold, self.medium_threshold
            ),
            ConfidenceLevel::Low => format!(
                "LOW confidence: energy={energy:.3} is above {:.1}. \
                 Weak or no attractor — this retrieval may be unreliable.",
                self.medium_threshold
            ),
            ConfidenceLevel::NoMatch => format!(
                "NO MATCH: gradient norm={grad_norm:.2e} far exceeds convergence tolerance. \
                 The query is in completely novel territory with no stored memory.",
            ),
            ConfidenceLevel::Ambiguous => format!(
                "AMBIGUOUS: multiple attractors found from perturbed starts. \
                 The query sits near a basin boundary between distinct memories. \
                 Energy={energy:.3}, grad_norm={grad_norm:.2e}.",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_classification() {
        let gate = ConfidenceGate::default();

        assert_eq!(gate.classify(-5.0, 1e-10), ConfidenceLevel::High);
        assert_eq!(gate.classify(-2.0, 1e-10), ConfidenceLevel::Medium);
        assert_eq!(gate.classify(0.0, 1e-10), ConfidenceLevel::Low);
        assert_eq!(gate.classify(-5.0, 1e-3), ConfidenceLevel::NoMatch);
    }
}
