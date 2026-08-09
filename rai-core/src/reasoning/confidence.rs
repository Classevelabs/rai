use crate::types::ConfidenceLevel;

/// Thresholds for an experimental, uncalibrated retrieval-score tier.
///
/// The only input is the retrieval score (`-5 · max(cosine, 0)` against the best-matching stored
/// address), so the tiers are a relabelling of cosine similarity and nothing else.
pub struct ConfidenceGate {
    /// Below this score = HIGH tier (cosine above ~0.6 at the default value).
    pub high_threshold: f64,
    /// Below this score = MEDIUM tier (cosine above ~0.2 at the default value).
    pub medium_threshold: f64,
}

impl Default for ConfidenceGate {
    fn default() -> Self {
        Self {
            high_threshold: -3.0,
            medium_threshold: -1.0,
        }
    }
}

impl ConfidenceGate {
    /// Determine the score tier from a retrieval score.
    pub fn classify(&self, energy: f64) -> ConfidenceLevel {
        if !energy.is_finite() {
            return ConfidenceLevel::Low;
        }
        // A score of zero means no stored address had positive similarity with the query,
        // including the case where nothing is stored at all.
        if energy >= 0.0 {
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
    pub fn explain(&self, energy: f64, confidence: ConfidenceLevel) -> String {
        if !energy.is_finite() {
            return "Confidence diagnostics are unavailable because the retrieval score was non-finite."
                .to_string();
        }
        match confidence {
            ConfidenceLevel::High => format!(
                "HIGH heuristic score tier: score={energy:.3} is below threshold {:.1}. \
                 This experimental label is not a calibrated confidence probability.",
                self.high_threshold
            ),
            ConfidenceLevel::Medium => format!(
                "MEDIUM heuristic score tier: score={energy:.3} is between {:.1} and {:.1}. \
                 This experimental label is not calibrated.",
                self.high_threshold, self.medium_threshold
            ),
            ConfidenceLevel::Low => format!(
                "LOW heuristic score tier: score={energy:.3} is above {:.1}. \
                 Treat the retrieval as unverified.",
                self.medium_threshold
            ),
            ConfidenceLevel::NoMatch => {
                "NO-MATCH heuristic: no stored memory scored above zero cosine similarity with \
                 this query. This does not prove that no relevant memory exists."
                    .to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_classification() {
        let gate = ConfidenceGate::default();

        assert_eq!(gate.classify(-5.0), ConfidenceLevel::High);
        assert_eq!(gate.classify(-2.0), ConfidenceLevel::Medium);
        assert_eq!(gate.classify(-0.5), ConfidenceLevel::Low);
        assert_eq!(gate.classify(f64::NAN), ConfidenceLevel::Low);
    }

    #[test]
    fn a_zero_score_is_reported_as_no_match() {
        // An empty store and a query orthogonal to everything both produce a score of zero.
        let gate = ConfidenceGate::default();
        assert_eq!(gate.classify(0.0), ConfidenceLevel::NoMatch);
        assert!(gate
            .explain(0.0, ConfidenceLevel::NoMatch)
            .contains("no stored memory"));
    }
}
