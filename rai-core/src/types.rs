use rem_nra::Vec64;
use serde::{Deserialize, Serialize};

/// Experimental, uncalibrated tier derived from the current retrieval score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Experimental score below the configured high-tier threshold.
    High,
    /// Experimental score in the configured medium tier.
    Medium,
    /// Experimental score in the low tier; treat retrieval as unverified.
    Low,
    /// Experimental gradient diagnostic exceeded its configured threshold.
    NoMatch,
    /// Experimental perturbation diagnostic reported multiple candidate states.
    Ambiguous,
}

/// Result of a retrieval operation with full diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The retrieved text content.
    pub content: String,
    /// Confidence level based on energy analysis.
    pub confidence: ConfidenceLevel,
    /// Raw energy at the attractor.
    pub energy: f64,
    /// Number of ODE integration steps.
    pub steps: usize,
    /// Final gradient norm.
    pub grad_norm: f64,
    /// Human-readable explanation of confidence.
    pub explanation: String,
}

/// Report of interference when storing a new fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceReport {
    /// Whether any existing memories were destabilized.
    pub has_interference: bool,
    /// Items whose energy changed significantly.
    pub affected_items: Vec<AffectedItem>,
    /// Overall interference severity.
    pub severity: InterferenceSeverity,
}

/// An item affected by storing a new fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedItem {
    /// The text content of the affected memory.
    pub content: String,
    /// Energy before the new store.
    pub energy_before: f64,
    /// Energy after the new store.
    pub energy_after: f64,
    /// Energy delta (positive means destabilized).
    pub delta: f64,
}

/// Severity of interference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterferenceSeverity {
    /// No significant energy changes.
    None,
    /// Minor perturbation, memories still stable.
    Minor,
    /// Significant score shift; not proof of contradiction.
    Major,
    /// Largest configured score-change tier; not proof of logical contradiction.
    Critical,
}

/// Result of an intersection query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionResult {
    /// The retrieved content at the concept intersection.
    pub content: String,
    /// Confidence at the intersection point.
    pub confidence: ConfidenceLevel,
    /// Energy at the combined omega.
    pub energy: f64,
    /// The individual concept names used.
    pub concepts: Vec<String>,
}

/// Surprise/novelty score from REM prior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseResult {
    /// Novelty score (REM prior residual norm).
    pub score: f64,
    /// Whether this is genuinely novel (high residual).
    pub is_novel: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

/// System health diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Number of stored memories.
    pub num_memories: usize,
    /// NRA mean squared error.
    pub nra_mse: Option<f64>,
    /// REM mean squared error.
    pub rem_mse: Option<f64>,
    /// REM mean residual norm (prior quality).
    pub rem_residual_norm: Option<f64>,
    /// NRA capacity utilization estimate.
    pub nra_capacity_ratio: f64,
    /// Whether the system needs retraining.
    pub needs_training: bool,
}

/// A stored memory entry with text and vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique ID for this memory.
    pub id: usize,
    /// Original text content.
    pub content: String,
    /// Omega vector (NRA address).
    #[serde(skip)]
    pub omega: Option<Vec64>,
    /// Key vector (REM key).
    #[serde(skip)]
    pub key: Option<Vec64>,
    /// Value vector (shared).
    #[serde(skip)]
    pub value: Option<Vec64>,
}

/// Confidence explanation with energy landscape details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceExplanation {
    /// The confidence level.
    pub confidence: ConfidenceLevel,
    /// Energy at attractor.
    pub energy: f64,
    /// Gradient norm at attractor.
    pub grad_norm: f64,
    /// Number of attractors found from perturbed starts.
    pub num_attractors: usize,
    /// Basin analysis: spread of attractor energies from perturbed starts.
    pub basin_spread: f64,
    /// Detailed explanation.
    pub explanation: String,
}
