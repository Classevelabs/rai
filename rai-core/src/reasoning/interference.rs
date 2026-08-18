use crate::types::{AffectedItem, InterferenceReport, InterferenceSeverity};
use rem_nra::Vec64;

/// Thresholds for reporting crowding created in the stored address space.
///
/// Each stored item is scored `-5 · cosine` against its nearest *other*
/// neighbour, so scores live in `[-5, 0]` and a *drop* means the item's
/// address got more crowded. Storing a fact can only add a neighbour, so the
/// signal a store produces is exactly that drop — `before - after`, in
/// `[0, 5]`. The tiers are magnitudes of that drop, and `critical` sits under
/// the 5.0 ceiling of the scale on purpose: a threshold at or above the
/// ceiling can never fire, which is a detector that does not exist.
pub struct InterferenceDetector {
    /// Crowding increase above this enters the minor tier.
    pub minor_threshold: f64,
    /// Crowding increase above this enters the major tier.
    pub major_threshold: f64,
    /// Crowding increase above this enters the critical tier.
    pub critical_threshold: f64,
}

impl Default for InterferenceDetector {
    fn default() -> Self {
        Self {
            minor_threshold: 0.5,
            major_threshold: 2.0,
            critical_threshold: 4.0,
        }
    }
}

impl InterferenceDetector {
    /// Compare crowding snapshots before and after a candidate store.
    ///
    /// `before`: (address, crowding score) pairs before storing.
    /// `after`: (address, crowding score) pairs after storing.
    /// `texts`: corresponding text labels for each item.
    ///
    /// An item is affected when its score *dropped* — its nearest neighbour
    /// got closer, meaning the candidate landed near enough to crowd its
    /// address and make the two harder to tell apart at recall time. Score
    /// improvements are ignored: an item whose neighbourhood emptied out did
    /// not lose anything. This is still address-space geometry, not
    /// semantics — two facts can contradict from far-apart addresses.
    pub fn detect(
        &self,
        before: &[(Vec64, f64)],
        after: &[(Vec64, f64)],
        texts: &[String],
    ) -> InterferenceReport {
        let mut affected_items = Vec::new();
        let mut max_crowding = 0.0f64;

        for (i, ((_, e_before), (_, e_after))) in before.iter().zip(after.iter()).enumerate() {
            let crowding = e_before - e_after;
            if crowding > self.minor_threshold {
                let text = texts.get(i).cloned().unwrap_or_else(|| format!("item_{i}"));
                affected_items.push(AffectedItem {
                    content: text,
                    energy_before: *e_before,
                    energy_after: *e_after,
                    delta: e_after - e_before,
                });
                max_crowding = max_crowding.max(crowding);
            }
        }

        let severity = if max_crowding > self.critical_threshold {
            InterferenceSeverity::Critical
        } else if max_crowding > self.major_threshold {
            InterferenceSeverity::Major
        } else if max_crowding > self.minor_threshold {
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

    /// The direction a store can actually move a score: a new neighbour makes
    /// an existing address more crowded, and the score drops.
    #[test]
    fn detects_major_interference() {
        let detector = InterferenceDetector::default();
        let omega = DVector::from_vec(vec![1.0, 0.0]);
        let before = vec![(omega.clone(), 0.0)];
        let after = vec![(omega, -3.0)]; // crowding grew by 3.0
        let texts = vec!["test".to_string()];

        let report = detector.detect(&before, &after, &texts);
        assert!(report.has_interference);
        assert_eq!(report.severity, InterferenceSeverity::Major);
        // The reported delta is the raw score move, so consumers see the drop.
        assert!((report.affected_items[0].delta + 3.0).abs() < 1e-12);
    }

    /// A near-duplicate address is the loudest case the scale can express:
    /// the neighbour similarity jumps most of the way to 1.0.
    #[test]
    fn near_duplicate_addresses_reach_the_critical_tier() {
        let detector = InterferenceDetector::default();
        let omega = DVector::from_vec(vec![1.0, 0.0]);
        let before = vec![(omega.clone(), -0.2)];
        let after = vec![(omega, -4.8)];
        let texts = vec!["test".to_string()];

        let report = detector.detect(&before, &after, &texts);
        assert_eq!(report.severity, InterferenceSeverity::Critical);
    }

    #[test]
    fn score_improvements_are_not_labeled_interference() {
        let detector = InterferenceDetector::default();
        let omega = DVector::from_vec(vec![1.0, 0.0]);
        let report = detector.detect(
            &[(omega.clone(), -10.0)],
            &[(omega, 0.0)],
            &["test".to_string()],
        );
        assert!(!report.has_interference);
        assert_eq!(report.severity, InterferenceSeverity::None);
    }
}
