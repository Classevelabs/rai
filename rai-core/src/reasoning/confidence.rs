use crate::types::ConfidenceLevel;

/// Thresholds for an experimental, uncalibrated retrieval-score tier.
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
        if !energy.is_finite() || !grad_norm.is_finite() {
            return ConfidenceLevel::Low;
        }
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
        if !energy.is_finite() || !grad_norm.is_finite() {
            return "Confidence diagnostics are unavailable because the retrieval scores were non-finite."
                .to_string();
        }
        match confidence {
            ConfidenceLevel::High => format!(
                "HIGH heuristic score tier: energy={energy:.3} is below threshold {:.1}. \
                 This experimental label is not a calibrated confidence probability.",
                self.high_threshold
            ),
            ConfidenceLevel::Medium => format!(
                "MEDIUM heuristic score tier: energy={energy:.3} is between {:.1} and {:.1}. \
                 This experimental label is not calibrated.",
                self.high_threshold, self.medium_threshold
            ),
            ConfidenceLevel::Low => format!(
                "LOW heuristic score tier: energy={energy:.3} is above {:.1}. \
                 Treat the retrieval as unverified.",
                self.medium_threshold
            ),
            ConfidenceLevel::NoMatch => format!(
                "NO-MATCH heuristic: gradient diagnostic={grad_norm:.2e} exceeded the configured threshold. \
                 This does not prove that no relevant memory exists.",
            ),
            ConfidenceLevel::Ambiguous => format!(
                "AMBIGUOUS experimental score: the diagnostic reported multiple candidate states. \
                 This is not a proven basin-boundary classification. \
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
        assert_eq!(gate.classify(f64::NAN, 0.0), ConfidenceLevel::Low);
    }
}
