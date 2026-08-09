use crate::types::SurpriseResult;

/// Experimental residual-score heuristic using the current REM nearest-key prediction.
///
/// When REM stores a value, the residual = value - prior.predict(key).
/// High residual norm means the prior couldn't predict this value — it's surprising/novel.
pub struct SurpriseDetector {
    /// Above this residual norm, content is considered novel.
    pub novelty_threshold: f64,
}

impl Default for SurpriseDetector {
    fn default() -> Self {
        Self {
            novelty_threshold: 1.0,
        }
    }
}

impl SurpriseDetector {
    /// Compute surprise score from a residual norm.
    pub fn score(&self, residual_norm: f64) -> SurpriseResult {
        if !residual_norm.is_finite() || residual_norm < 0.0 {
            return SurpriseResult {
                score: 0.0,
                is_novel: false,
                explanation: "Residual heuristic is unavailable because its input was invalid."
                    .to_string(),
            };
        }
        let is_novel = residual_norm > self.novelty_threshold;
        let explanation = if is_novel {
            format!(
                "HIGH residual heuristic: norm {residual_norm:.3} exceeds threshold {:.1}. \
                 This experimental score is not a calibrated novelty judgment.",
                self.novelty_threshold
            )
        } else {
            format!(
                "LOW residual heuristic: norm {residual_norm:.3} is within threshold {:.1}. \
                 This experimental score does not establish that the content is already known.",
                self.novelty_threshold
            )
        };

        SurpriseResult {
            score: residual_norm,
            is_novel,
            explanation,
        }
    }

    /// Compute surprise for a specific key-value pair against a prior prediction.
    pub fn compute(
        &self,
        value: &rem_nra::Vec64,
        prior_prediction: &rem_nra::Vec64,
    ) -> SurpriseResult {
        let residual_norm = (value - prior_prediction).norm();
        self.score(residual_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_residual_is_not_reported_as_novel() {
        let result = SurpriseDetector::default().score(f64::NAN);
        assert!(!result.is_novel);
        assert!(result.score.is_finite());
        assert!(result.explanation.contains("unavailable"));
    }
}
