use crate::types::{AffectedItem, InterferenceReport, InterferenceSeverity};
use rem_nra::Vec64;

/// Thresholds for interference detection.
pub struct InterferenceDetector {
    /// Energy change above this = minor interference.
    pub minor_threshold: f64,
    /// Energy change above this = major interference.
    pub major_threshold: f64,
    /// Energy change above this = critical (attractor destroyed).
    pub critical_threshold: f64,
}

impl Default for InterferenceDetector {
    fn default() -> Self {
        Self {
            minor_threshold: 0.5,
            major_threshold: 2.0,
            critical_threshold: 5.0,
        }
    }
}

impl InterferenceDetector {
    /// Compare energy snapshots before and after a store operation.
    ///
    /// `before`: (omega, energy) pairs before storing.
    /// `after`: (omega, energy) pairs after storing.
    /// `texts`: corresponding text labels for each item.
    pub fn detect(
        &self,
        before: &[(Vec64, f64)],
        after: &[(Vec64, f64)],
        texts: &[String],
    ) -> InterferenceReport {
        let mut affected_items = Vec::new();
        let mut max_delta = 0.0f64;

        for (i, ((_, e_before), (_, e_after))) in before.iter().zip(after.iter()).enumerate() {
            let delta = e_after - e_before;
            if delta.abs() > self.minor_threshold {
                let text = texts.get(i).cloned().unwrap_or_else(|| format!("item_{i}"));
                affected_items.push(AffectedItem {
                    content: text,
                    energy_before: *e_before,
                    energy_after: *e_after,
                    delta,
                });
                max_delta = max_delta.max(delta);
            }
        }

        let severity = if max_delta > self.critical_threshold {
            InterferenceSeverity::Critical
        } else if max_delta > self.major_threshold {
            InterferenceSeverity::Major
        } else if max_delta > self.minor_threshold {
            InterferenceSeverity::Minor
        } else {
            InterferenceSeverity::None
        };

        InterferenceReport {
            has_interference: !affected_items.is_empty(),
            affected_items,
            severity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn no_interference_on_stable() {
        let detector = InterferenceDetector::default();
        let omega = DVector::from_vec(vec![1.0, 0.0]);
        let before = vec![(omega.clone(), -3.0)];
        let after = vec![(omega, -3.0)];
        let texts = vec!["test".to_string()];

        let report = detector.detect(&before, &after, &texts);
        assert!(!report.has_interference);
        assert_eq!(report.severity, InterferenceSeverity::None);
    }

    #[test]
    fn detects_major_interference() {
        let detector = InterferenceDetector::default();
        let omega = DVector::from_vec(vec![1.0, 0.0]);
        let before = vec![(omega.clone(), -3.0)];
        let after = vec![(omega, 0.0)]; // +3.0 delta
        let texts = vec!["test".to_string()];

        let report = detector.detect(&before, &after, &texts);
        assert!(report.has_interference);
        assert_eq!(report.severity, InterferenceSeverity::Major);
    }
}
