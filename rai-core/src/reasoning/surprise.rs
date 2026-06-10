use crate::types::SurpriseResult;

/// Novelty detection via REM prior residual norm.
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
        let is_novel = residual_norm > self.novelty_threshold;
        let explanation = if is_novel {
            format!(
                "NOVEL: residual norm {residual_norm:.3} exceeds threshold {:.1}. \
                 The prior model could not predict this — it contains genuinely new information.",
                self.novelty_threshold
            )
        } else {
            format!(
                "EXPECTED: residual norm {residual_norm:.3} is within threshold {:.1}. \
                 The prior model already captures this pattern — this is consistent with existing knowledge.",
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
