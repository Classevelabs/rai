use serde::{Deserialize, Serialize};

/// Experimental, uncalibrated tier derived from the current retrieval score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Retrieval score below the configured high-tier threshold.
    High,
    /// Retrieval score in the configured medium tier.
    Medium,
    /// Retrieval score in the low tier; treat retrieval as unverified.
    Low,
    /// No stored memory scored above zero cosine similarity with the query.
    NoMatch,
}

/// Result of a retrieval operation with its score diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The retrieved text content.
    pub content: String,
    /// Experimental score tier; not a calibrated probability.
    pub confidence: ConfidenceLevel,
    /// `-5 · max(cosine, 0)` between the query address and the best-matching stored address.
    /// `0.0` means nothing matched; `-5.0` is an exact match.
    pub energy: f64,
    /// Human-readable explanation of the score tier.
    pub explanation: String,
}

/// Report of how a candidate fact changes crowding in the stored address space.
///
/// This measures address-space crowding only: each stored item is scored against its nearest
/// *other* neighbour, and the report compares those scores before and after. Appending an item
/// can only bring a neighbour closer, so a store never raises another item's score — under the
/// current semantics this report cannot detect a semantic contradiction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceReport {
    /// Whether any stored item's crowding score moved into a reported tier.
    pub has_interference: bool,
    /// Items whose crowding score changed significantly.
    pub affected_items: Vec<AffectedItem>,
    /// Largest reported crowding-change tier.
    pub severity: InterferenceSeverity,
}

/// An item whose crowding score changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedItem {
    /// The text content of the affected memory.
    pub content: String,
    /// Crowding score before the comparison point.
    pub energy_before: f64,
    /// Crowding score after the comparison point.
    pub energy_after: f64,
    /// Raw score move, `energy_after - energy_before`. An item is reported
    /// only when its score *dropped* — the candidate crowded its address —
    /// so every reported delta is negative, at least the detector's minor
    /// threshold in magnitude.
    pub delta: f64,
}

/// Severity of a reported crowding change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterferenceSeverity {
    /// No significant score changes.
    None,
    /// Minor score shift.
    Minor,
    /// Significant score shift; not proof of contradiction.
    Major,
    /// Largest configured score-change tier; not proof of logical contradiction.
    Critical,
}

/// Result of an intersection query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionResult {
    /// The retrieved content nearest the combined concept address.
    pub content: String,
    /// Experimental score tier at the combined address.
    pub confidence: ConfidenceLevel,
    /// Retrieval score at the combined address; see [`RetrievalResult::energy`].
    pub energy: f64,
    /// The individual concept names used.
    pub concepts: Vec<String>,
}

/// Surprise/novelty score from the nearest-key residual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseResult {
    /// Novelty score (residual norm against the nearest stored key's value).
    pub score: f64,
    /// Whether the residual exceeded the configured novelty threshold.
    pub is_novel: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

/// System health diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Number of stored memories.
    pub num_memories: usize,
    /// Mean residual norm across every stored item, or `None` when nothing is stored.
    pub mean_residual_norm: Option<f64>,
    /// Stored items divided by the configured store capacity.
    pub nra_capacity_ratio: f64,
}

/// Detailed score diagnostics for a retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceExplanation {
    /// The score tier.
    pub confidence: ConfidenceLevel,
    /// Retrieval score; see [`RetrievalResult::energy`].
    pub energy: f64,
    /// Detailed explanation.
    pub explanation: String,
}
