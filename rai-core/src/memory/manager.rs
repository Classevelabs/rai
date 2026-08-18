use crate::embedding::bridge::{EmbeddingBridge, TextIndex};
use crate::memory::persistence::{
    MemorySnapshot, CURRENT_SNAPSHOT_VERSION, MAX_STORED_ITEMS, MAX_VECTOR_DIMENSION,
};
use crate::reasoning::composition::Compositor;
use crate::reasoning::confidence::ConfidenceGate;
use crate::reasoning::interference::InterferenceDetector;
use crate::reasoning::surprise::SurpriseDetector;
use crate::types::*;
use crate::RaiError;
use rem_nra::nra::{NRAConfig, NonlinearResonanceMemory};
use rem_nra::rem::{REMConfig, ResidualEquilibriumMemory};
use rem_nra::Vec64;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Largest text accepted by any memory operation, in bytes.
///
/// This is the single limit for the whole stack. Transports (the REST handlers and the MCP
/// server) reject oversized input up front so a client gets a clear error before an embedding
/// request is made, and this module enforces the same bound as the last line of defence. The
/// snapshot validator tolerates a larger historical bound so older files still load.
pub const MAX_TEXT_BYTES: usize = 16 * 1024;

/// Largest number of concepts accepted by [`MemoryManager::intersect`].
pub const MAX_INTERSECTION_CONCEPTS: usize = 32;

/// Every structure a mutation has to publish together.
///
/// One lock covers all of them: a reader that saw `nra` updated but `texts` stale would report
/// the wrong label for a memory. The bridge's text index is published inside the same critical
/// section.
struct Inner {
    /// Address/value store.
    nra: NonlinearResonanceMemory,
    /// Key/value store.
    rem: ResidualEquilibriumMemory,
    /// Text labels for stored memories (parallel to NRA items).
    texts: Vec<String>,
}

/// Central memory manager that orchestrates the two stores, embedding, and reasoning.
///
/// Reads (recall, health, snapshot, len) take a shared guard and run concurrently; every
/// mutation takes the exclusive guard, so the service is a single-writer store.
pub struct MemoryManager {
    /// Single lock over every structure a mutation publishes together.
    inner: Arc<RwLock<Inner>>,
    /// Embedding bridge for text <-> vector conversion.
    bridge: Arc<EmbeddingBridge>,
    /// Retrieval score tiering.
    confidence_gate: ConfidenceGate,
    /// Address-space crowding reporter.
    interference_detector: InterferenceDetector,
    /// Residual-based novelty heuristic.
    surprise_detector: SurpriseDetector,
}

impl MemoryManager {
    /// Create a new MemoryManager with validated default configurations.
    pub fn try_new(bridge: Arc<EmbeddingBridge>) -> Result<Self, RaiError> {
        Self::with_configs(bridge, NRAConfig::default(), REMConfig::default())
    }

    /// [`try_new`](Self::try_new) with a caller-chosen memory capacity.
    ///
    /// Everything else stays at the validated defaults. The bounds are
    /// checked here by name — the generic config validation would fold an
    /// out-of-range capacity into an error that names no field and no limit.
    pub fn try_new_with_capacity(
        bridge: Arc<EmbeddingBridge>,
        capacity: usize,
    ) -> Result<Self, RaiError> {
        if capacity == 0 || capacity > MAX_STORED_ITEMS {
            return Err(RaiError::InvalidInput(format!(
                "capacity must be between 1 and {MAX_STORED_ITEMS}"
            )));
        }
        let nra_config = NRAConfig {
            num_units: capacity,
            ..NRAConfig::default()
        };
        Self::with_configs(bridge, nra_config, REMConfig::default())
    }

    /// Create with custom store configs.
    pub fn with_configs(
        bridge: Arc<EmbeddingBridge>,
        nra_config: NRAConfig,
        rem_config: REMConfig,
    ) -> Result<Self, RaiError> {
        validate_memory_configs(&bridge, &nra_config, &rem_config)?;
        let mut rng = rand::thread_rng();
        let nra = NonlinearResonanceMemory::new(nra_config, &mut rng);
        let rem = ResidualEquilibriumMemory::new(rem_config);

        Ok(Self::from_parts(
            Inner {
                nra,
                rem,
                texts: Vec::new(),
            },
            bridge,
        ))
    }

    fn from_parts(inner: Inner, bridge: Arc<EmbeddingBridge>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(inner)),
            bridge,
            confidence_gate: ConfidenceGate::default(),
            interference_detector: InterferenceDetector::default(),
            surprise_detector: SurpriseDetector::default(),
        }
    }

    /// Store a fact and return a crowding report for the existing memories.
    pub async fn store(&self, content: &str) -> Result<InterferenceReport, RaiError> {
        self.store_with_optional_snapshot(content, None).await
    }

    /// Store a fact and atomically persist the resulting state before publishing it in memory.
    ///
    /// If the snapshot write fails, none of the staged memory structures are changed.
    pub async fn store_and_save(
        &self,
        content: &str,
        path: &Path,
    ) -> Result<InterferenceReport, RaiError> {
        self.store_with_optional_snapshot(content, Some(path)).await
    }

    async fn store_with_optional_snapshot(
        &self,
        content: &str,
        snapshot_path: Option<&Path>,
    ) -> Result<InterferenceReport, RaiError> {
        validate_text("content", content)?;

        let (omega, key, value, embedding) = self.bridge.project_text(content).await?;
        let mut published = Arc::clone(&self.inner).write_owned().await;

        // Stage every parallel structure. Durable callers write this staged state first,
        // so a failed disk operation cannot leak a supposedly failed mutation into memory.
        let mut staged_nra = published.nra.clone();
        if staged_nra.len() >= staged_nra.config.num_units {
            return Err(RaiError::CapacityExhausted {
                limit: staged_nra.config.num_units,
            });
        }
        let mut staged_rem = published.rem.clone();
        let mut staged_texts = published.texts.clone();
        let mut staged_text_index = self.bridge.text_index().await;
        let energy_before = staged_nra.energy_snapshot();
        staged_nra
            .store(&omega, &value)
            .map_err(|e| RaiError::MemoryError(format!("NRA store: {e}")))?;
        staged_rem
            .store(&key, &value)
            .map_err(|e| RaiError::MemoryError(format!("REM store: {e}")))?;
        let energy_after = staged_nra.energy_snapshot();

        let len = energy_before.len();
        let report = self.interference_detector.detect(
            &energy_before,
            &energy_after[..len],
            &staged_texts[..len],
        );

        staged_texts.push(content.to_string());
        staged_text_index.insert(content.to_string(), embedding, value);

        if let Some(path) = snapshot_path {
            let snapshot = self.snapshot_from_parts(
                &staged_nra,
                &staged_rem,
                staged_text_index.clone(),
                staged_texts.clone(),
            );
            let path = path.to_path_buf();
            let bridge = Arc::clone(&self.bridge);
            return tokio::task::spawn_blocking(move || {
                snapshot.save(&path)?;
                published.nra = staged_nra;
                published.rem = staged_rem;
                published.texts = staged_texts;
                bridge.restore_text_index_blocking(staged_text_index);
                drop(published);
                Ok(report)
            })
            .await
            .map_err(|error| {
                RaiError::PersistenceError(format!("snapshot transaction failed: {error}"))
            })?;
        }

        *self.bridge.text_index_for_update().await = staged_text_index;
        published.nra = staged_nra;
        published.rem = staged_rem;
        published.texts = staged_texts;
        drop(published);

        Ok(report)
    }

    /// Remove the memory whose stored text is exactly `content`.
    ///
    /// Returns `false` when nothing matches. Exact match is deliberate: the
    /// caller got the text from `recall` or stored it themselves, and a
    /// fuzzy delete that guesses would be a data-loss feature.
    pub async fn forget(&self, content: &str) -> Result<bool, RaiError> {
        self.forget_with_optional_snapshot(content, None).await
    }

    /// [`forget`](Self::forget), atomically persisting the shrunken state
    /// before publishing it in memory — the same transaction discipline as
    /// [`store_and_save`](Self::store_and_save).
    pub async fn forget_and_save(&self, content: &str, path: &Path) -> Result<bool, RaiError> {
        self.forget_with_optional_snapshot(content, Some(path))
            .await
    }

    async fn forget_with_optional_snapshot(
        &self,
        content: &str,
        snapshot_path: Option<&Path>,
    ) -> Result<bool, RaiError> {
        validate_text("content", content)?;
        let mut published = Arc::clone(&self.inner).write_owned().await;

        // `texts` is index-aligned with the item stores by construction:
        // store appends to all of them in one transaction, so the text's
        // position is the item's position everywhere.
        let Some(index) = published.texts.iter().position(|text| text == content) else {
            return Ok(false);
        };

        let mut staged_nra = published.nra.clone();
        let mut staged_rem = published.rem.clone();
        let mut staged_texts = published.texts.clone();
        let mut staged_text_index = self.bridge.text_index().await;
        staged_nra
            .remove(index)
            .map_err(|e| RaiError::MemoryError(format!("NRA remove: {e}")))?;
        staged_rem
            .remove(index)
            .map_err(|e| RaiError::MemoryError(format!("REM remove: {e}")))?;
        staged_texts.remove(index);
        // The text index deduplicates while `texts` does not, so when the same
        // text was stored twice this removes only one item — the decode entry
        // has to survive for the copies still stored.
        if !staged_texts.iter().any(|text| text == content) {
            staged_text_index.remove(content);
        }

        if let Some(path) = snapshot_path {
            let snapshot = self.snapshot_from_parts(
                &staged_nra,
                &staged_rem,
                staged_text_index.clone(),
                staged_texts.clone(),
            );
            let path = path.to_path_buf();
            let bridge = Arc::clone(&self.bridge);
            return tokio::task::spawn_blocking(move || {
                snapshot.save(&path)?;
                published.nra = staged_nra;
                published.rem = staged_rem;
                published.texts = staged_texts;
                bridge.restore_text_index_blocking(staged_text_index);
                drop(published);
                Ok(true)
            })
            .await
            .map_err(|error| {
                RaiError::PersistenceError(format!("snapshot transaction failed: {error}"))
            })?;
        }

        *self.bridge.text_index_for_update().await = staged_text_index;
        published.nra = staged_nra;
        published.rem = staged_rem;
        published.texts = staged_texts;
        drop(published);

        Ok(true)
    }

    /// Recall a memory with its score diagnostics.
    pub async fn recall(&self, query: &str) -> Result<RetrievalResult, RaiError> {
        validate_text("query", query)?;
        let omega = self.bridge.text_to_omega(query).await?;

        let inner = self.inner.read().await;
        let diagnostics = inner
            .nra
            .retrieve_with_diagnostics(&omega)
            .map_err(|e| RaiError::MemoryError(format!("NRA retrieve: {e}")))?;

        let confidence = self.confidence_gate.classify(diagnostics.energy);
        let explanation = self.confidence_gate.explain(diagnostics.energy, confidence);

        let content = self
            .bridge
            .nearest_text(&diagnostics.value)
            .await?
            .unwrap_or_else(|| "(no matching text found)".to_string());

        Ok(RetrievalResult {
            content,
            confidence,
            energy: diagnostics.energy,
            explanation,
        })
    }

    /// Query at a composed concept address.
    pub async fn intersect(&self, concepts: &[String]) -> Result<IntersectionResult, RaiError> {
        if !(2..=MAX_INTERSECTION_CONCEPTS).contains(&concepts.len()) {
            return Err(RaiError::InvalidInput(format!(
                "concept count must be between 2 and {MAX_INTERSECTION_CONCEPTS}"
            )));
        }

        let mut omegas = Vec::with_capacity(concepts.len());
        for concept in concepts {
            validate_text("concept", concept)?;
            let omega = self.bridge.text_to_omega(concept).await?;
            omegas.push(omega);
        }
        let combined = Compositor::intersect(&omegas)?;

        let inner = self.inner.read().await;
        let diagnostics = inner
            .nra
            .retrieve_with_diagnostics(&combined)
            .map_err(|e| RaiError::MemoryError(format!("NRA intersect: {e}")))?;

        let confidence = self.confidence_gate.classify(diagnostics.energy);

        let content = self
            .bridge
            .nearest_text(&diagnostics.value)
            .await?
            .unwrap_or_else(|| "(no matching text at intersection)".to_string());

        Ok(IntersectionResult {
            content,
            confidence,
            energy: diagnostics.energy,
            concepts: concepts.to_vec(),
        })
    }

    /// Report which stored memories the candidate fact would crowd, without storing it.
    ///
    /// This is address-space geometry only: an item is flagged when the candidate lands close
    /// enough that recall could confuse the two. Two facts can contradict from far-apart
    /// addresses, so an empty report is not evidence of consistency.
    pub async fn check_contradiction(&self, fact: &str) -> Result<InterferenceReport, RaiError> {
        validate_text("fact", fact)?;
        let (omega, _key, value, _embedding) = self.bridge.project_text(fact).await?;

        let inner = self.inner.read().await;
        let energy_before = inner.nra.energy_snapshot();
        let mut staged_nra = inner.nra.clone();
        staged_nra
            .store(&omega, &value)
            .map_err(|e| RaiError::MemoryError(format!("NRA contradict: {e}")))?;
        let energy_after = staged_nra.energy_snapshot();

        let len = energy_before.len();
        let report = self.interference_detector.detect(
            &energy_before,
            &energy_after[..len],
            &inner.texts[..len],
        );

        Ok(report)
    }

    /// Measure novelty/surprise of a fact against the nearest stored key.
    pub async fn measure_surprise(&self, content: &str) -> Result<SurpriseResult, RaiError> {
        validate_text("content", content)?;
        let (_omega, key, value, _embedding) = self.bridge.project_text(content).await?;

        let inner = self.inner.read().await;
        let prediction = inner.rem.predict(&key);
        Ok(self.surprise_detector.compute(&value, &prediction))
    }

    /// Explain the score tier of a retrieval in detail.
    pub async fn explain_confidence(&self, query: &str) -> Result<ConfidenceExplanation, RaiError> {
        validate_text("query", query)?;
        let omega = self.bridge.text_to_omega(query).await?;

        let inner = self.inner.read().await;
        let diagnostics = inner
            .nra
            .retrieve_with_diagnostics(&omega)
            .map_err(|e| RaiError::MemoryError(format!("NRA explain: {e}")))?;

        let confidence = self.confidence_gate.classify(diagnostics.energy);
        let explanation = self.confidence_gate.explain(diagnostics.energy, confidence);

        Ok(ConfidenceExplanation {
            confidence,
            energy: diagnostics.energy,
            explanation,
        })
    }

    /// Get system health diagnostics.
    pub async fn health(&self) -> Result<HealthReport, RaiError> {
        let inner = self.inner.read().await;

        let mean_residual_norm = if inner.rem.is_empty() {
            None
        } else {
            Some(inner.rem.mean_residual_norm())
        };

        let num_memories = inner.nra.len();
        let nra_capacity_ratio = num_memories as f64 / inner.nra.config.num_units as f64;

        Ok(HealthReport {
            num_memories,
            mean_residual_norm,
            nra_capacity_ratio,
        })
    }

    /// Save full state to disk.
    pub async fn save(&self, path: &Path) -> Result<(), RaiError> {
        let inner = Arc::clone(&self.inner).read_owned().await;
        let snapshot = self.snapshot_from_parts(
            &inner.nra,
            &inner.rem,
            self.bridge.text_index().await,
            inner.texts.clone(),
        );
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let result = snapshot.save(&path);
            drop(inner);
            result
        })
        .await
        .map_err(|error| RaiError::PersistenceError(format!("snapshot writer failed: {error}")))?
    }

    fn snapshot_from_parts(
        &self,
        nra: &NonlinearResonanceMemory,
        rem: &ResidualEquilibriumMemory,
        text_index: TextIndex,
        texts: Vec<String>,
    ) -> MemorySnapshot {
        MemorySnapshot {
            version: CURRENT_SNAPSHOT_VERSION,
            nra_params: nra.params.clone(),
            nra_config: nra.config.clone(),
            nra_items: nra.items().to_vec(),
            rem_config: rem.config.clone(),
            rem_items: rem.items().to_vec(),
            rem_residual_norm: rem.mean_residual_norm(),
            text_index,
            texts,
            omega_proj: self.bridge.omega_proj.clone(),
            key_proj: self.bridge.key_proj.clone(),
            value_proj: self.bridge.value_proj.clone(),
            total_items: nra.len(),
        }
    }

    /// Load state from disk. Returns a new MemoryManager.
    pub async fn load(path: &Path, bridge: Arc<EmbeddingBridge>) -> Result<Self, RaiError> {
        Self::load_with_capacity(path, bridge, None).await
    }

    /// [`load`](Self::load), overriding the snapshot's stored capacity.
    ///
    /// A snapshot written at the old 512-unit default would otherwise pin the
    /// store to it forever. The override must still hold every persisted item
    /// — `from_snapshot` refuses a capacity below the item count.
    pub async fn load_with_capacity(
        path: &Path,
        bridge: Arc<EmbeddingBridge>,
        capacity: Option<usize>,
    ) -> Result<Self, RaiError> {
        let mut snapshot = MemorySnapshot::load(path)?;
        if let Some(capacity) = capacity {
            // The snapshot's own num_units was validated at load; an override
            // must clear the same ceiling, and `from_snapshot` still refuses
            // one below the persisted item count.
            if capacity == 0 || capacity > MAX_STORED_ITEMS {
                return Err(RaiError::InvalidInput(format!(
                    "capacity must be between 1 and {MAX_STORED_ITEMS}"
                )));
            }
            snapshot.nra_config.num_units = capacity;
        }
        // Version 0 files predate the parallel `texts` field, so labels can only be rebuilt from
        // the text index — which deduplicates. When the same text was stored twice the missing
        // labels are unrecoverable, so say so instead of failing later as an opaque count
        // mismatch that reads like corruption.
        let texts = if snapshot.texts.is_empty() {
            snapshot
                .text_index
                .entries
                .iter()
                .map(|entry| entry.text.clone())
                .collect::<Vec<_>>()
        } else {
            snapshot.texts.clone()
        };
        let item_count = snapshot.nra_items.len();
        if snapshot.version < CURRENT_SNAPSHOT_VERSION && texts.len() != item_count {
            return Err(RaiError::PersistenceError(
                "snapshot predates label persistence (version 0); re-create it with this release \
                 or restore from a version-1 snapshot"
                    .to_string(),
            ));
        }
        if snapshot.total_items != item_count
            || snapshot.rem_items.len() != item_count
            || texts.len() != item_count
        {
            return Err(RaiError::PersistenceError(format!(
                "incoherent snapshot counts: total={}, nra={}, rem={}, texts={}",
                snapshot.total_items,
                item_count,
                snapshot.rem_items.len(),
                texts.len()
            )));
        }

        let embedder = bridge.embedder();
        let embedding_dim = embedder.embedding_dim();
        let projections = [
            ("omega", &snapshot.omega_proj),
            ("key", &snapshot.key_proj),
            ("value", &snapshot.value_proj),
        ];
        for (name, projection) in projections {
            if projection.source_dim != embedding_dim {
                return Err(RaiError::PersistenceError(format!(
                    "{name} projection source dimension {} does not match embedding provider dimension {embedding_dim}",
                    projection.source_dim
                )));
            }
        }
        if snapshot.omega_proj.target_dim != snapshot.nra_config.dim_omega
            || snapshot.key_proj.target_dim != snapshot.rem_config.dim_key
            || snapshot.value_proj.target_dim != snapshot.nra_config.dim_value
            || snapshot.value_proj.target_dim != snapshot.rem_config.dim_value
        {
            return Err(RaiError::PersistenceError(
                "projection and memory dimensions do not match".to_string(),
            ));
        }
        if snapshot
            .text_index
            .entries
            .iter()
            .any(|entry| entry.embedding.len() != embedding_dim)
        {
            return Err(RaiError::PersistenceError(
                "text index embedding dimension does not match provider".to_string(),
            ));
        }

        // Projections are part of the persisted memory coordinate system. Reusing a newly
        // randomised bridge here would make future queries incompatible with stored vectors.
        let restored_bridge = Arc::new(EmbeddingBridge::with_projections(
            embedder,
            snapshot.omega_proj.clone(),
            snapshot.key_proj.clone(),
            snapshot.value_proj.clone(),
        ));

        // Both stores are restored through their validating constructors: the snapshot is
        // untrusted input, and replaying items through `store` would recompute derived state.
        let nra = NonlinearResonanceMemory::from_snapshot(
            snapshot.nra_config,
            snapshot.nra_params,
            snapshot.nra_items,
        )
        .map_err(|e| RaiError::MemoryError(format!("restore NRA: {e}")))?;

        let rem = ResidualEquilibriumMemory::from_snapshot(
            snapshot.rem_config,
            snapshot.rem_items,
            snapshot.rem_residual_norm,
        )
        .map_err(|e| RaiError::MemoryError(format!("restore REM: {e}")))?;

        // Cached value projections are derived state and are not persisted; rebuild them here.
        restored_bridge
            .restore_text_index(snapshot.text_index.clone())
            .await?;

        Ok(Self::from_parts(Inner { nra, rem, texts }, restored_bridge))
    }

    /// Get number of stored memories.
    pub async fn len(&self) -> usize {
        self.inner.read().await.nra.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.nra.is_empty()
    }

    /// Take a crowding snapshot of the stored address space for external use.
    pub async fn energy_snapshot(&self) -> Vec<(Vec64, f64)> {
        self.inner.read().await.nra.energy_snapshot()
    }
}

fn validate_text(field: &str, value: &str) -> Result<(), RaiError> {
    if value.trim().is_empty() {
        return Err(RaiError::InvalidInput(format!("{field} must not be empty")));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(RaiError::InvalidInput(format!(
            "{field} exceeds the {MAX_TEXT_BYTES}-byte limit"
        )));
    }
    Ok(())
}

fn validate_memory_configs(
    bridge: &EmbeddingBridge,
    nra: &NRAConfig,
    rem: &REMConfig,
) -> Result<(), RaiError> {
    let dimensions = [
        nra.dim_state,
        nra.dim_omega,
        nra.dim_value,
        rem.dim_memory,
        rem.dim_key,
        rem.dim_value,
    ];
    if dimensions
        .iter()
        .any(|dimension| *dimension == 0 || *dimension > MAX_VECTOR_DIMENSION)
    {
        return Err(RaiError::InvalidInput(
            "memory dimensions are outside supported bounds".into(),
        ));
    }
    if nra.dim_value != rem.dim_value || nra.num_units == 0 || nra.num_units > MAX_STORED_ITEMS {
        return Err(RaiError::InvalidInput(
            "memory configuration is outside supported bounds".into(),
        ));
    }
    bridge.validate_memory_dimensions(nra.dim_omega, rem.dim_key, nra.dim_value)
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::embedding::{Embedder, MockEmbedder};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Twenty distinct facts used by the recall regression test.
    const DISTINCT_FACTS: [&str; 20] = [
        "the kettle boils at ninety eight degrees in this flat",
        "marta keeps the spare key under the third plant pot",
        "the quarterly review moved to the second thursday",
        "our staging cluster runs in the frankfurt region",
        "the fire drill alarm is tested every first monday",
        "dmitri is allergic to shellfish but not to iodine",
        "the archive tapes live in the basement cage seven",
        "invoice numbering restarts at one every fiscal year",
        "the roof access code was changed to four one nine two",
        "priya owns the deployment runbook for the billing service",
        "the coffee grinder needs descaling every forty days",
        "our vpn certificate expires on the ninth of november",
        "the loading bay is closed for resurfacing until spring",
        "sam prefers written agendas circulated the night before",
        "the backup generator holds fuel for eleven hours",
        "customer refunds above two thousand need dual approval",
        "the conference room projector uses the older hdmi cable",
        "yusuf handles all correspondence with the auditors",
        "the parcel locker pin rotates on the first of each month",
        "our incident retrospectives are written within three days",
    ];

    /// Mock provider whose embeddings share a large common component, the way production
    /// providers do.
    ///
    /// [`MockEmbedder`] is effectively zero-mean and near-orthogonal, which masks a retrieval
    /// that returns the corpus centroid instead of the matching item.
    struct MeanBiasedEmbedder {
        dim: usize,
    }

    impl MeanBiasedEmbedder {
        /// Weight of the direction every embedding shares.
        const COMMON_WEIGHT: f64 = 0.8;
        /// Weight of the text-specific direction.
        const SPECIFIC_WEIGHT: f64 = 0.2;
    }

    #[async_trait::async_trait]
    impl Embedder for MeanBiasedEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f64>, RaiError> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut specific = vec![0.0f64; self.dim];
            for (index, component) in specific.iter_mut().enumerate() {
                let mut hasher = DefaultHasher::new();
                text.hash(&mut hasher);
                index.hash(&mut hasher);
                *component = (hasher.finish() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            }
            normalize(&mut specific);

            let common = 1.0 / (self.dim as f64).sqrt();
            let mut embedding: Vec<f64> = specific
                .iter()
                .map(|component| Self::COMMON_WEIGHT * common + Self::SPECIFIC_WEIGHT * component)
                .collect();
            normalize(&mut embedding);
            Ok(embedding)
        }

        fn embedding_dim(&self) -> usize {
            self.dim
        }
    }

    fn normalize(vector: &mut [f64]) {
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in vector {
                *value /= norm;
            }
        }
    }

    /// Every stored fact must be recallable by its own text, even when the embedding space has a
    /// dominant shared direction. Blending stored values by softmax weight returns the corpus
    /// centroid here, so this fails unless retrieval picks the single best match.
    #[tokio::test]
    async fn recall_returns_the_stored_fact_under_a_mean_biased_embedder() {
        let embedder = Arc::new(MeanBiasedEmbedder { dim: 384 });
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();

        for fact in DISTINCT_FACTS {
            manager.store(fact).await.unwrap();
        }
        assert_eq!(manager.len().await, DISTINCT_FACTS.len());

        for fact in DISTINCT_FACTS {
            let result = manager.recall(fact).await.unwrap();
            assert_eq!(
                result.content, fact,
                "recall returned a different memory than the one queried"
            );
        }
    }

    /// Recall has to keep working after a reload, which is the path that rebuilds the cached
    /// value projections the decode scan compares against.
    #[tokio::test]
    async fn recall_survives_a_snapshot_round_trip() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MeanBiasedEmbedder { dim: 384 }),
            32,
            32,
            64,
        ));
        let manager = MemoryManager::try_new(bridge).unwrap();
        for fact in &DISTINCT_FACTS[..5] {
            manager.store(fact).await.unwrap();
        }
        let path = unique_temp_path("recall-round-trip.json");
        manager.save(&path).await.unwrap();

        let fresh_bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MeanBiasedEmbedder { dim: 384 }),
            32,
            32,
            64,
        ));
        let restored = MemoryManager::load(&path, fresh_bridge).await.unwrap();
        for fact in &DISTINCT_FACTS[..5] {
            assert_eq!(restored.recall(fact).await.unwrap().content, *fact);
        }

        std::fs::remove_file(path).unwrap();
    }

    /// Leave-one-out crowding scores have to move when neighbours arrive, and their magnitudes
    /// have to cross the detector's tiers in the direction a store actually
    /// produces: a new neighbour drops an existing item's score, and the
    /// detector reports the drop.
    #[tokio::test]
    async fn leave_one_out_energy_varies_and_reaches_the_interference_detector() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        for fact in ["alpha fact", "beta fact", "gamma fact"] {
            manager.store(fact).await.unwrap();
        }
        let sparse = manager.energy_snapshot().await;

        // Storing the same text again gives "alpha fact" an exactly coincident neighbour.
        manager.store("alpha fact").await.unwrap();
        let crowded = manager.energy_snapshot().await;

        assert!(
            crowded[0].1 < -4.9,
            "a coincident neighbour must reach the score floor: {crowded:?}"
        );
        assert!(
            crowded.iter().any(|(_, energy)| *energy != crowded[0].1),
            "scores must vary across items rather than stay constant: {crowded:?}"
        );
        assert!(
            sparse[0].1 - crowded[0].1 > 0.5,
            "the score must respond to a new neighbour: {} -> {}",
            sparse[0].1,
            crowded[0].1
        );

        let texts = manager.inner.read().await.texts[..sparse.len()].to_vec();
        let report =
            InterferenceDetector::default().detect(&sparse, &crowded[..sparse.len()], &texts);
        assert!(report.has_interference);
        assert_ne!(report.severity, InterferenceSeverity::None);
        assert!(report
            .affected_items
            .iter()
            .any(|item| item.content == "alpha fact"));
    }

    /// The report a store returns has to fire on the case it exists for: a
    /// candidate landing on an existing address. A coincident duplicate takes
    /// its neighbour's similarity from 0 to 1 — the full width of the scale —
    /// so it must reach the critical tier through both `store` and
    /// `/v1/contradict`'s manager path.
    #[tokio::test]
    async fn storing_a_coincident_duplicate_reports_critical_interference() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            32,
            32,
            64,
        ));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("the sky is blue").await.unwrap();

        let preview = manager
            .check_contradiction("the sky is blue")
            .await
            .unwrap();
        assert!(preview.has_interference);
        assert_eq!(preview.severity, InterferenceSeverity::Critical);
        // The preview stages nothing: the store still holds one memory.
        assert_eq!(manager.len().await, 1);

        let report = manager.store("the sky is blue").await.unwrap();
        assert!(report.has_interference);
        assert_eq!(report.severity, InterferenceSeverity::Critical);
        assert_eq!(report.affected_items[0].content, "the sky is blue");
    }

    #[tokio::test]
    async fn forget_removes_exactly_the_named_fact() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MeanBiasedEmbedder { dim: 384 }),
            32,
            32,
            64,
        ));
        let manager = MemoryManager::try_new(bridge).unwrap();
        for fact in &DISTINCT_FACTS[..5] {
            manager.store(fact).await.unwrap();
        }

        assert!(!manager.forget("never stored").await.unwrap());
        assert_eq!(manager.len().await, 5);

        assert!(manager.forget(DISTINCT_FACTS[2]).await.unwrap());
        assert_eq!(manager.len().await, 4);
        // Forgetting is idempotent in effect: the second call removes nothing.
        assert!(!manager.forget(DISTINCT_FACTS[2]).await.unwrap());

        // The survivors — including the ones whose indices shifted — still
        // recall themselves, which exercises the text-index reindexing.
        for fact in DISTINCT_FACTS[..5]
            .iter()
            .filter(|f| **f != DISTINCT_FACTS[2])
        {
            assert_eq!(manager.recall(fact).await.unwrap().content, *fact);
        }
    }

    /// The text index deduplicates while `texts` does not, so forgetting one
    /// copy of a twice-stored text must leave the survivor decodable.
    #[tokio::test]
    async fn forgetting_one_copy_of_a_duplicate_keeps_the_other_recallable() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MeanBiasedEmbedder { dim: 384 }),
            32,
            32,
            64,
        ));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store(DISTINCT_FACTS[0]).await.unwrap();
        manager.store(DISTINCT_FACTS[0]).await.unwrap();

        assert!(manager.forget(DISTINCT_FACTS[0]).await.unwrap());
        assert_eq!(manager.len().await, 1);
        assert_eq!(
            manager.recall(DISTINCT_FACTS[0]).await.unwrap().content,
            DISTINCT_FACTS[0]
        );

        assert!(manager.forget(DISTINCT_FACTS[0]).await.unwrap());
        assert_eq!(manager.len().await, 0);
    }

    /// The 512-unit default was a dead end: full meant full forever. Both
    /// exits have to work — deleting frees a slot, and a configured capacity
    /// raises the ceiling.
    #[tokio::test]
    async fn forget_frees_capacity_and_capacity_is_configurable() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MeanBiasedEmbedder { dim: 384 }),
            32,
            32,
            64,
        ));
        let manager = MemoryManager::try_new_with_capacity(bridge, 1).unwrap();
        manager.store(DISTINCT_FACTS[0]).await.unwrap();

        let error = manager.store(DISTINCT_FACTS[1]).await.unwrap_err();
        assert!(matches!(error, RaiError::CapacityExhausted { limit: 1 }));

        assert!(manager.forget(DISTINCT_FACTS[0]).await.unwrap());
        manager.store(DISTINCT_FACTS[1]).await.unwrap();
        assert_eq!(
            manager.recall(DISTINCT_FACTS[1]).await.unwrap().content,
            DISTINCT_FACTS[1]
        );
    }

    #[test]
    fn checked_default_constructor_rejects_mismatched_bridge_dimensions() {
        let bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            8,
            8,
            8,
        ));

        let error = match MemoryManager::try_new(bridge) {
            Err(error) => error,
            Ok(_) => panic!("mismatched default bridge unexpectedly accepted"),
        };
        assert!(error.to_string().contains("dimensions do not match"));
    }

    #[tokio::test]
    async fn load_restores_saved_projection_coordinate_system() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let probe: Vec<f64> = (0..384).map(|index| index as f64 / 384.0).collect();
        let expected_omega = bridge.omega_proj.project(&probe).unwrap();
        let expected_key = bridge.key_proj.project(&probe).unwrap();
        let expected_value = bridge.value_proj.project(&probe).unwrap();
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("persistent memory").await.unwrap();

        let path = unique_temp_path("projection-round-trip.json");
        manager.save(&path).await.unwrap();
        manager.store("second persistent memory").await.unwrap();
        manager.store("persistent memory").await.unwrap();
        let expected_residual_norm = manager.inner.read().await.rem.mean_residual_norm();
        manager.save(&path).await.unwrap();

        let fresh_embedder = Arc::new(MockEmbedder::new(384));
        let fresh_bridge = Arc::new(EmbeddingBridge::new(fresh_embedder, 32, 32, 64));
        let restored = MemoryManager::load(&path, fresh_bridge).await.unwrap();

        assert_eq!(restored.len().await, 3);
        assert_eq!(restored.inner.read().await.texts.len(), 3);
        assert_vectors_close(
            restored
                .bridge
                .omega_proj
                .project(&probe)
                .unwrap()
                .as_slice(),
            expected_omega.as_slice(),
        );
        assert_vectors_close(
            restored.bridge.key_proj.project(&probe).unwrap().as_slice(),
            expected_key.as_slice(),
        );
        assert_vectors_close(
            restored
                .bridge
                .value_proj
                .project(&probe)
                .unwrap()
                .as_slice(),
            expected_value.as_slice(),
        );
        let restored_residual = restored.inner.read().await.rem.mean_residual_norm();
        assert!((restored_residual - expected_residual_norm).abs() < 1e-12);

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn contradiction_check_does_not_mutate_memory() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("the sky is blue").await.unwrap();
        let before = manager.len().await;

        manager
            .check_contradiction("the sky is green")
            .await
            .unwrap();

        assert_eq!(manager.len().await, before);
    }

    #[tokio::test]
    async fn surprise_is_query_specific_and_does_not_mutate_memory() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("known fact").await.unwrap();
        let before_len = manager.len().await;
        let before_residual = manager.inner.read().await.rem.mean_residual_norm();

        let known = manager.measure_surprise("known fact").await.unwrap();
        let novel = manager.measure_surprise("different fact").await.unwrap();

        assert!(known.score < 1e-12);
        assert!(novel.score > known.score);
        assert_eq!(manager.len().await, before_len);
        assert_eq!(
            manager.inner.read().await.rem.mean_residual_norm(),
            before_residual
        );
    }

    #[tokio::test]
    async fn concurrent_stores_commit_all_parallel_structures() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = Arc::new(MemoryManager::try_new(bridge).unwrap());
        let mut tasks = Vec::new();
        for index in 0..8 {
            let manager = Arc::clone(&manager);
            tasks.push(tokio::spawn(async move {
                manager.store(&format!("fact {index}")).await.unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        assert_eq!(manager.len().await, 8);
        let inner = manager.inner.read().await;
        assert_eq!(inner.rem.items().len(), 8);
        assert_eq!(inner.texts.len(), 8);
        drop(inner);
        assert_eq!(manager.bridge.text_index().await.entries.len(), 8);
    }

    #[tokio::test]
    async fn failed_durable_store_does_not_publish_in_memory() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        let path = unique_temp_path("missing-parent").join("state.json");

        assert!(manager
            .store_and_save("must remain absent", &path)
            .await
            .is_err());
        assert_eq!(manager.len().await, 0);
        assert!(manager.inner.read().await.texts.is_empty());
        assert!(manager.bridge.text_index().await.entries.is_empty());
    }

    #[tokio::test]
    async fn durable_store_publishes_only_after_the_snapshot_is_written() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        let path = unique_temp_path("durable-store.json");

        manager
            .store_and_save("durable memory", &path)
            .await
            .unwrap();

        assert_eq!(manager.len().await, 1);
        assert_eq!(manager.inner.read().await.texts, vec!["durable memory"]);
        let persisted = MemorySnapshot::load(&path).unwrap();
        assert_eq!(persisted.total_items, 1);
        assert_eq!(persisted.texts, vec!["durable memory"]);

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn manager_enforces_content_concept_and_capacity_limits() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let nra_config = NRAConfig {
            num_units: 1,
            ..Default::default()
        };
        let manager = MemoryManager::with_configs(bridge, nra_config, REMConfig::default())
            .expect("valid bounded manager");

        assert!(manager
            .store(&"x".repeat(MAX_TEXT_BYTES + 1))
            .await
            .is_err());
        let too_many_concepts = vec!["concept".to_string(); MAX_INTERSECTION_CONCEPTS + 1];
        assert!(manager.intersect(&too_many_concepts).await.is_err());

        manager.store("first").await.unwrap();
        let error = manager.store("second").await.unwrap_err();
        assert!(
            matches!(error, RaiError::CapacityExhausted { limit } if limit == 1),
            "capacity exhaustion must be a structured client error, got: {error}"
        );
        assert_eq!(manager.len().await, 1);
    }

    #[tokio::test]
    async fn load_rejects_incoherent_snapshot_before_reconstruction() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("persistent memory").await.unwrap();
        let valid_path = unique_temp_path("valid.json");
        let corrupt_path = unique_temp_path("corrupt.json");
        manager.save(&valid_path).await.unwrap();

        let mut snapshot = MemorySnapshot::load(&valid_path).unwrap();
        snapshot.total_items += 1;
        std::fs::write(&corrupt_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();
        let fresh_bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            32,
            32,
            64,
        ));
        assert!(MemoryManager::load(&corrupt_path, fresh_bridge)
            .await
            .is_err());

        std::fs::remove_file(valid_path).unwrap();
        std::fs::remove_file(corrupt_path).unwrap();
    }

    #[tokio::test]
    async fn current_snapshot_rejects_substituted_text_index_labels() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("original memory").await.unwrap();
        let valid_path = unique_temp_path("valid-text-index.json");
        let corrupt_path = unique_temp_path("corrupt-text-index.json");
        manager.save(&valid_path).await.unwrap();

        let mut snapshot = MemorySnapshot::load(&valid_path).unwrap();
        let (replacement, id) = {
            let entry = &mut snapshot.text_index.entries[0];
            entry.text = "substituted memory".to_string();
            (entry.text.clone(), entry.id)
        };
        snapshot.text_index.text_to_id.clear();
        snapshot.text_index.text_to_id.insert(replacement, id);
        std::fs::write(&corrupt_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

        let error = match MemorySnapshot::load(&corrupt_path) {
            Err(error) => error,
            Ok(_) => panic!("incoherent current text index unexpectedly loaded"),
        };
        assert!(error.to_string().contains("text labels do not match"));

        std::fs::remove_file(valid_path).unwrap();
        std::fs::remove_file(corrupt_path).unwrap();
    }

    /// A snapshot written before the schema shrank still carries the REM encoder/decoder biases,
    /// the rolling memory state, the training loss, and the NRA value basis. Those keys are no
    /// longer part of the schema and must be ignored rather than fail the load.
    #[tokio::test]
    async fn snapshots_carrying_retired_fields_still_load() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("legacy shaped memory").await.unwrap();
        let valid_path = unique_temp_path("valid-retired-fields.json");
        let legacy_path = unique_temp_path("retired-fields.json");
        manager.save(&valid_path).await.unwrap();

        let mut document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&valid_path).unwrap()).unwrap();
        let object = document.as_object_mut().unwrap();
        object.insert(
            "rem_encoder".to_string(),
            serde_json::json!({ "bias": vec![0.0f64; 256] }),
        );
        object.insert(
            "rem_decoder".to_string(),
            serde_json::json!({ "bias": vec![0.0f64; 64] }),
        );
        object.insert(
            "rem_memory_state".to_string(),
            serde_json::json!(vec![0.25f64; 256]),
        );
        object.insert("rem_last_loss".to_string(), serde_json::json!(1.5));
        object["nra_params"]["value_basis"] = serde_json::json!({
            "nrows": 64, "ncols": 64, "data": vec![0.0f64; 64 * 64]
        });
        std::fs::write(&legacy_path, serde_json::to_vec(&document).unwrap()).unwrap();

        let fresh_bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            32,
            32,
            64,
        ));
        let restored = MemoryManager::load(&legacy_path, fresh_bridge)
            .await
            .unwrap();
        assert_eq!(restored.len().await, 1);
        assert_eq!(
            restored
                .recall("legacy shaped memory")
                .await
                .unwrap()
                .content,
            "legacy shaped memory"
        );

        // What this release writes back no longer contains those keys.
        let rewritten_path = unique_temp_path("rewritten.json");
        restored.save(&rewritten_path).await.unwrap();
        let rewritten: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&rewritten_path).unwrap()).unwrap();
        for retired in [
            "rem_encoder",
            "rem_decoder",
            "rem_memory_state",
            "rem_last_loss",
        ] {
            assert!(rewritten.get(retired).is_none(), "{retired} was rewritten");
        }
        assert!(rewritten["nra_params"].get("value_basis").is_none());

        for path in [valid_path, legacy_path, rewritten_path] {
            std::fs::remove_file(path).unwrap();
        }
    }

    /// Rebuilding labels from the deduplicating text index is only sound when every stored text
    /// was unique, so that remains the one version-0 shape this release still accepts.
    #[tokio::test]
    async fn legacy_snapshot_without_duplicate_labels_remains_loadable() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("legacy memory").await.unwrap();
        manager.store("another legacy memory").await.unwrap();
        let valid_path = unique_temp_path("valid-legacy.json");
        let legacy_path = unique_temp_path("legacy-v0.json");
        manager.save(&valid_path).await.unwrap();

        let mut snapshot = MemorySnapshot::load(&valid_path).unwrap();
        snapshot.version = 0;
        snapshot.texts.clear();
        std::fs::write(&legacy_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

        let fresh_bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            32,
            32,
            64,
        ));
        let restored = MemoryManager::load(&legacy_path, fresh_bridge)
            .await
            .unwrap();
        assert_eq!(
            restored.inner.read().await.texts,
            vec![
                "legacy memory".to_string(),
                "another legacy memory".to_string()
            ]
        );

        std::fs::remove_file(valid_path).unwrap();
        std::fs::remove_file(legacy_path).unwrap();
    }

    #[tokio::test]
    async fn legacy_snapshot_with_duplicate_labels_is_rejected_with_an_actionable_error() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        // The text index deduplicates, so two stores of the same text leave a version-0 file
        // with one index entry for two items: the labels are genuinely unrecoverable.
        manager.store("repeated memory").await.unwrap();
        manager.store("repeated memory").await.unwrap();
        let valid_path = unique_temp_path("valid-duplicate.json");
        let legacy_path = unique_temp_path("legacy-duplicate-v0.json");
        manager.save(&valid_path).await.unwrap();

        let mut snapshot = MemorySnapshot::load(&valid_path).unwrap();
        assert_eq!(snapshot.total_items, 2);
        assert_eq!(snapshot.text_index.entries.len(), 1);
        snapshot.version = 0;
        snapshot.texts.clear();
        std::fs::write(&legacy_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

        let fresh_bridge = Arc::new(EmbeddingBridge::new(
            Arc::new(MockEmbedder::new(384)),
            32,
            32,
            64,
        ));
        let error = match MemoryManager::load(&legacy_path, fresh_bridge).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("unrecoverable version-0 snapshot unexpectedly loaded"),
        };
        assert!(
            error.contains("predates label persistence"),
            "expected an actionable version-0 error, got: {error}"
        );
        assert!(
            !error.contains("incoherent snapshot counts"),
            "the count mismatch must not surface as corruption: {error}"
        );

        std::fs::remove_file(valid_path).unwrap();
        std::fs::remove_file(legacy_path).unwrap();
    }

    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rai-memory-{}-{unique}-{label}",
            std::process::id()
        ))
    }

    fn assert_vectors_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() < 1e-12,
                "projection differed at {index}: {left} vs {right}"
            );
        }
    }
}
