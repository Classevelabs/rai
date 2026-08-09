use crate::embedding::bridge::EmbeddingBridge;
use crate::memory::persistence::{
    MemorySnapshot, CURRENT_SNAPSHOT_VERSION, MAX_STORED_ITEMS, MAX_TRAIN_EPOCHS,
    MAX_VECTOR_DIMENSION,
};
use crate::memory::training::TrainingOrchestrator;
use crate::reasoning::basins::BasinAnalyzer;
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
use tokio::sync::Mutex;

const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_INTERSECTION_CONCEPTS: usize = 32;

/// Central memory manager that orchestrates NRA, REM, embedding, and reasoning.
pub struct MemoryManager {
    /// NRA memory for nonlinear addressing.
    nra: Arc<Mutex<NonlinearResonanceMemory>>,
    /// REM memory for structure-aware storage.
    rem: Arc<Mutex<ResidualEquilibriumMemory>>,
    /// Embedding bridge for text <-> vector conversion.
    bridge: Arc<EmbeddingBridge>,
    /// Confidence gating module.
    confidence_gate: ConfidenceGate,
    /// Interference detector.
    interference_detector: InterferenceDetector,
    /// Basin analyzer.
    basin_analyzer: BasinAnalyzer,
    /// Surprise detector.
    surprise_detector: SurpriseDetector,
    /// Training orchestrator.
    training: Arc<Mutex<TrainingOrchestrator>>,
    /// Text labels for stored memories (parallel to NRA items).
    texts: Arc<Mutex<Vec<String>>>,
    /// Next memory ID.
    next_id: Arc<Mutex<usize>>,
    /// Serialises multi-structure mutations and coherent snapshots.
    mutation_lock: Arc<Mutex<()>>,
}

impl MemoryManager {
    /// Create a new MemoryManager with default configurations.
    ///
    /// This infallible constructor is retained for API compatibility and requires bridge
    /// projection targets of 32 (omega), 32 (key), and 64 (value). New callers should use
    /// [`MemoryManager::try_new`] so a mismatched or malformed bridge is rejected at the
    /// construction boundary. Call [`MemoryManager::with_configs`] for non-default dimensions.
    pub fn new(bridge: Arc<EmbeddingBridge>) -> Self {
        let nra_config = NRAConfig {
            dim_state: 64,
            dim_omega: 32,
            dim_value: 64,
            num_units: 512,
            train_epochs: 300,
            ..Default::default()
        };
        let rem_config = REMConfig {
            dim_memory: 256,
            dim_key: 32,
            dim_value: 64,
            ..Default::default()
        };

        let mut rng = rand::thread_rng();
        let nra = NonlinearResonanceMemory::new(nra_config, &mut rng);
        let rem = ResidualEquilibriumMemory::new(rem_config, &mut rng);

        Self {
            nra: Arc::new(Mutex::new(nra)),
            rem: Arc::new(Mutex::new(rem)),
            bridge,
            confidence_gate: ConfidenceGate::default(),
            interference_detector: InterferenceDetector::default(),
            basin_analyzer: BasinAnalyzer::default(),
            surprise_detector: SurpriseDetector::default(),
            training: Arc::new(Mutex::new(TrainingOrchestrator::default())),
            texts: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(Mutex::new(0)),
            mutation_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Create a new MemoryManager with validated default configurations.
    pub fn try_new(bridge: Arc<EmbeddingBridge>) -> Result<Self, RaiError> {
        let nra_config = NRAConfig {
            dim_state: 64,
            dim_omega: 32,
            dim_value: 64,
            num_units: 512,
            train_epochs: 300,
            ..Default::default()
        };
        let rem_config = REMConfig {
            dim_memory: 256,
            dim_key: 32,
            dim_value: 64,
            ..Default::default()
        };
        Self::with_configs(bridge, nra_config, rem_config)
    }

    /// Create with custom NRA/REM configs.
    pub fn with_configs(
        bridge: Arc<EmbeddingBridge>,
        nra_config: NRAConfig,
        rem_config: REMConfig,
    ) -> Result<Self, RaiError> {
        validate_memory_configs(&bridge, &nra_config, &rem_config)?;
        let mut rng = rand::thread_rng();
        let nra = NonlinearResonanceMemory::new(nra_config, &mut rng);
        let rem = ResidualEquilibriumMemory::new(rem_config, &mut rng);

        Ok(Self {
            nra: Arc::new(Mutex::new(nra)),
            rem: Arc::new(Mutex::new(rem)),
            bridge,
            confidence_gate: ConfidenceGate::default(),
            interference_detector: InterferenceDetector::default(),
            basin_analyzer: BasinAnalyzer::default(),
            surprise_detector: SurpriseDetector::default(),
            training: Arc::new(Mutex::new(TrainingOrchestrator::default())),
            texts: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(Mutex::new(0)),
            mutation_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Store a fact and return an interference report.
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
        let mutation = Arc::clone(&self.mutation_lock).lock_owned().await;

        // Stage every parallel structure. Durable callers write this staged state first,
        // so a failed disk operation cannot leak a supposedly failed mutation into memory.
        let mut staged_nra = self.nra.lock().await.clone();
        if staged_nra.len() >= staged_nra.config.num_units {
            return Err(RaiError::MemoryError(format!(
                "memory capacity of {} items has been reached",
                staged_nra.config.num_units
            )));
        }
        let mut staged_rem = self.rem.lock().await.clone();
        let mut staged_texts = self.texts.lock().await.clone();
        let mut staged_text_index = self.bridge.text_index().await;
        let mut staged_training = self.training.lock().await.clone();
        let staged_next_id = self
            .next_id
            .lock()
            .await
            .checked_add(1)
            .ok_or_else(|| RaiError::MemoryError("memory ID space exhausted".into()))?;
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
        staged_training.item_stored();
        staged_text_index.insert(content.to_string(), embedding);

        if let Some(path) = snapshot_path {
            let snapshot = self.snapshot_from_parts(
                &staged_nra,
                &staged_rem,
                staged_text_index.clone(),
                staged_texts.clone(),
            );
            let path = path.to_path_buf();
            let nra = Arc::clone(&self.nra);
            let rem = Arc::clone(&self.rem);
            let texts = Arc::clone(&self.texts);
            let next_id = Arc::clone(&self.next_id);
            let training = Arc::clone(&self.training);
            let bridge = Arc::clone(&self.bridge);
            return tokio::task::spawn_blocking(move || {
                let _mutation = mutation;
                snapshot.save(&path)?;
                *nra.blocking_lock() = staged_nra;
                *rem.blocking_lock() = staged_rem;
                *texts.blocking_lock() = staged_texts;
                *next_id.blocking_lock() = staged_next_id;
                *training.blocking_lock() = staged_training;
                bridge.restore_text_index_blocking(staged_text_index);
                Ok(report)
            })
            .await
            .map_err(|error| {
                RaiError::PersistenceError(format!("snapshot transaction failed: {error}"))
            })?;
        }

        let mut nra = self.nra.lock().await;
        let mut rem = self.rem.lock().await;
        let mut texts = self.texts.lock().await;
        let mut next_id = self.next_id.lock().await;
        let mut training = self.training.lock().await;
        let mut text_index = self.bridge.text_index_for_update().await;
        *nra = staged_nra;
        *rem = staged_rem;
        *texts = staged_texts;
        *next_id = staged_next_id;
        *training = staged_training;
        *text_index = staged_text_index;
        drop(mutation);

        Ok(report)
    }

    /// Recall a memory with confidence diagnostics.
    pub async fn recall(&self, query: &str) -> Result<RetrievalResult, RaiError> {
        validate_text("query", query)?;
        let omega = self.bridge.text_to_omega(query).await?;
        let _mutation = self.mutation_lock.lock().await;

        let diagnostics = {
            let nra = self.nra.lock().await;
            nra.retrieve_with_diagnostics(&omega)
                .map_err(|e| RaiError::MemoryError(format!("NRA retrieve: {e}")))?
        };

        let confidence = self
            .confidence_gate
            .classify(diagnostics.energy, diagnostics.grad_norm);
        let explanation =
            self.confidence_gate
                .explain(diagnostics.energy, diagnostics.grad_norm, confidence);

        // Find nearest text
        let content = self
            .bridge
            .nearest_text(&diagnostics.value)
            .await
            .unwrap_or_else(|| "(no matching text found)".to_string());

        Ok(RetrievalResult {
            content,
            confidence,
            energy: diagnostics.energy,
            steps: diagnostics.steps,
            grad_norm: diagnostics.grad_norm,
            explanation,
        })
    }

    /// Query at concept intersection using compositional addressing.
    pub async fn intersect(&self, concepts: &[String]) -> Result<IntersectionResult, RaiError> {
        if !(2..=MAX_INTERSECTION_CONCEPTS).contains(&concepts.len()) {
            return Err(RaiError::InvalidInput(format!(
                "concept count must be between 2 and {MAX_INTERSECTION_CONCEPTS}"
            )));
        }

        // Embed each concept to omega
        let mut omegas = Vec::with_capacity(concepts.len());
        for concept in concepts {
            validate_text("concept", concept)?;
            let omega = self.bridge.text_to_omega(concept).await?;
            omegas.push(omega);
        }
        let _mutation = self.mutation_lock.lock().await;

        // Compose omegas
        let combined = Compositor::intersect(&omegas);

        // Retrieve at intersection
        let diagnostics = {
            let nra = self.nra.lock().await;
            nra.retrieve_with_diagnostics(&combined)
                .map_err(|e| RaiError::MemoryError(format!("NRA intersect: {e}")))?
        };

        let confidence = self
            .confidence_gate
            .classify(diagnostics.energy, diagnostics.grad_norm);

        let content = self
            .bridge
            .nearest_text(&diagnostics.value)
            .await
            .unwrap_or_else(|| "(no matching text at intersection)".to_string());

        Ok(IntersectionResult {
            content,
            confidence,
            energy: diagnostics.energy,
            concepts: concepts.to_vec(),
        })
    }

    /// Check if a new fact contradicts existing memory.
    pub async fn check_contradiction(&self, fact: &str) -> Result<InterferenceReport, RaiError> {
        validate_text("fact", fact)?;
        let (omega, _key, value, _embedding) = self.bridge.project_text(fact).await?;
        let _mutation = self.mutation_lock.lock().await;

        let (energy_before, mut staged_nra) = {
            let nra = self.nra.lock().await;
            (nra.energy_snapshot(), nra.clone())
        };
        staged_nra
            .store(&omega, &value)
            .map_err(|e| RaiError::MemoryError(format!("NRA contradict: {e}")))?;
        let energy_after = staged_nra.energy_snapshot();

        let texts = self.texts.lock().await;
        let len = energy_before.len();
        let report =
            self.interference_detector
                .detect(&energy_before, &energy_after[..len], &texts[..len]);

        Ok(report)
    }

    /// Measure novelty/surprise of a fact using REM prior.
    pub async fn measure_surprise(&self, content: &str) -> Result<SurpriseResult, RaiError> {
        validate_text("content", content)?;
        let (_omega, key, value, _embedding) = self.bridge.project_text(content).await?;
        let _mutation = self.mutation_lock.lock().await;

        let rem = self.rem.lock().await;
        let prediction = rem.predict(&key);
        Ok(self.surprise_detector.compute(&value, &prediction))
    }

    /// Explain the confidence of a retrieval in detail.
    pub async fn explain_confidence(&self, query: &str) -> Result<ConfidenceExplanation, RaiError> {
        validate_text("query", query)?;
        let omega = self.bridge.text_to_omega(query).await?;
        let _mutation = self.mutation_lock.lock().await;

        let nra = self.nra.lock().await;
        let diagnostics = nra
            .retrieve_with_diagnostics(&omega)
            .map_err(|e| RaiError::MemoryError(format!("NRA explain: {e}")))?;

        let mut confidence = self
            .confidence_gate
            .classify(diagnostics.energy, diagnostics.grad_norm);

        // Basin analysis
        let mut rng = rand::thread_rng();
        let basin_result = self
            .basin_analyzer
            .analyze(&nra.params, &omega, &nra.config, &mut rng);

        if basin_result.is_ambiguous {
            confidence = ConfidenceLevel::Ambiguous;
        }

        let explanation =
            self.confidence_gate
                .explain(diagnostics.energy, diagnostics.grad_norm, confidence);

        Ok(ConfidenceExplanation {
            confidence,
            energy: diagnostics.energy,
            grad_norm: diagnostics.grad_norm,
            num_attractors: basin_result.attractors.len(),
            basin_spread: basin_result.energy_spread,
            explanation,
        })
    }

    /// Get system health diagnostics.
    pub async fn health(&self) -> Result<HealthReport, RaiError> {
        let _mutation = self.mutation_lock.lock().await;
        let nra = self.nra.lock().await;
        let rem = self.rem.lock().await;

        let nra_mse = nra.mse().ok();
        let rem_mse = rem.mse().ok();
        let rem_residual_norm = if rem.is_empty() {
            None
        } else {
            Some(rem.mean_residual_norm())
        };

        let num_memories = nra.len();
        let nra_capacity_ratio = num_memories as f64 / nra.config.num_units as f64;

        let training = self.training.lock().await;
        let needs_training = training.needs_nra_retrain() || training.needs_rem_retrain();

        Ok(HealthReport {
            num_memories,
            nra_mse,
            rem_mse,
            rem_residual_norm,
            nra_capacity_ratio,
            needs_training,
        })
    }

    /// Trigger NRA retraining.
    pub async fn train_nra(&self) -> Result<Vec<f64>, RaiError> {
        Err(training_not_implemented())
    }

    /// Retrain NRA and publish the trained state only after its snapshot is durable.
    pub async fn train_nra_and_save(&self, path: &Path) -> Result<Vec<f64>, RaiError> {
        let _ = path;
        Err(training_not_implemented())
    }

    /// Trigger REM retraining.
    pub async fn train_rem(&self) -> Result<Vec<f64>, RaiError> {
        Err(training_not_implemented())
    }

    /// Retrain REM and publish the trained state only after its snapshot is durable.
    pub async fn train_rem_and_save(&self, path: &Path) -> Result<Vec<f64>, RaiError> {
        let _ = path;
        Err(training_not_implemented())
    }

    /// Save full state to disk.
    pub async fn save(&self, path: &Path) -> Result<(), RaiError> {
        let mutation = Arc::clone(&self.mutation_lock).lock_owned().await;
        let nra = self.nra.lock().await.clone();
        let rem = self.rem.lock().await.clone();
        let snapshot = self.snapshot_from_parts(
            &nra,
            &rem,
            self.bridge.text_index().await,
            self.texts.lock().await.clone(),
        );
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let _mutation = mutation;
            snapshot.save(&path)
        })
        .await
        .map_err(|error| RaiError::PersistenceError(format!("snapshot writer failed: {error}")))?
    }

    fn snapshot_from_parts(
        &self,
        nra: &NonlinearResonanceMemory,
        rem: &ResidualEquilibriumMemory,
        text_index: crate::embedding::bridge::TextIndex,
        texts: Vec<String>,
    ) -> MemorySnapshot {
        MemorySnapshot {
            version: CURRENT_SNAPSHOT_VERSION,
            nra_params: nra.params.clone(),
            nra_config: nra.config.clone(),
            nra_items: nra.items().to_vec(),
            rem_config: rem.config.clone(),
            rem_encoder: rem.encoder.clone(),
            rem_decoder: rem.decoder.clone(),
            rem_memory_state: rem.memory_state.clone(),
            rem_items: rem.items().to_vec(),
            rem_residual_norm: rem.mean_residual_norm(),
            rem_last_loss: rem.last_loss(),
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
        let snapshot = MemorySnapshot::load(path)?;
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
        if snapshot.nra_params.omega_basis.nrows() != snapshot.nra_config.dim_state
            || snapshot.nra_params.omega_basis.ncols() != snapshot.nra_config.dim_omega
            || snapshot.nra_params.value_basis.nrows() != snapshot.nra_config.dim_value
            || snapshot.nra_params.value_basis.ncols() != snapshot.nra_config.dim_state
        {
            return Err(RaiError::PersistenceError(
                "NRA parameter matrices do not match snapshot configuration".to_string(),
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

        // Reconstruct NRA
        let mut nra =
            NonlinearResonanceMemory::from_params(snapshot.nra_params, snapshot.nra_config);
        for (omega, value) in &snapshot.nra_items {
            nra.store(omega, value)
                .map_err(|e| RaiError::MemoryError(format!("restore NRA: {e}")))?;
        }

        // Restore REM directly. Replaying items through `store` would update the persisted
        // rolling memory state a second time and silently corrupt the snapshot.
        let rem = ResidualEquilibriumMemory::from_snapshot(
            snapshot.rem_config,
            snapshot.rem_encoder,
            snapshot.rem_decoder,
            snapshot.rem_memory_state,
            snapshot.rem_items,
            snapshot.rem_residual_norm,
            snapshot.rem_last_loss,
        )
        .map_err(|e| RaiError::MemoryError(format!("restore REM: {e}")))?;

        // Restore text index
        restored_bridge
            .restore_text_index(snapshot.text_index.clone())
            .await;

        Ok(Self {
            nra: Arc::new(Mutex::new(nra)),
            rem: Arc::new(Mutex::new(rem)),
            bridge: restored_bridge,
            confidence_gate: ConfidenceGate::default(),
            interference_detector: InterferenceDetector::default(),
            basin_analyzer: BasinAnalyzer::default(),
            surprise_detector: SurpriseDetector::default(),
            training: Arc::new(Mutex::new(TrainingOrchestrator::default())),
            texts: Arc::new(Mutex::new(texts)),
            next_id: Arc::new(Mutex::new(snapshot.total_items)),
            mutation_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Get number of stored memories.
    pub async fn len(&self) -> usize {
        let _mutation = self.mutation_lock.lock().await;
        let nra = self.nra.lock().await;
        nra.len()
    }

    pub async fn is_empty(&self) -> bool {
        let _mutation = self.mutation_lock.lock().await;
        let nra = self.nra.lock().await;
        nra.is_empty()
    }

    /// Take an NRA energy snapshot for external use.
    pub async fn energy_snapshot(&self) -> Vec<(Vec64, f64)> {
        let _mutation = self.mutation_lock.lock().await;
        let nra = self.nra.lock().await;
        nra.energy_snapshot()
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
    if nra.dim_value != rem.dim_value
        || nra.num_units == 0
        || nra.num_units > MAX_STORED_ITEMS
        || nra.train_epochs > MAX_TRAIN_EPOCHS
        || rem.train_epochs > MAX_TRAIN_EPOCHS
        || !nra.ode_tol.is_finite()
        || nra.ode_tol <= 0.0
        || nra.ode_tol > 1.0
    {
        return Err(RaiError::InvalidInput(
            "memory configuration is outside supported bounds".into(),
        ));
    }
    bridge.validate_memory_dimensions(nra.dim_omega, rem.dim_key, nra.dim_value)
}

fn training_not_implemented() -> RaiError {
    RaiError::TrainingError(
        "training is unavailable because this build does not implement parameter optimization"
            .into(),
    )
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::embedding::MockEmbedder;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let expected_omega = bridge.omega_proj.project(&probe);
        let expected_key = bridge.key_proj.project(&probe);
        let expected_value = bridge.value_proj.project(&probe);
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("persistent memory").await.unwrap();

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rai-memory-{}-{unique}.json", std::process::id()));
        manager.save(&path).await.unwrap();
        manager.store("second persistent memory").await.unwrap();
        manager.store("persistent memory").await.unwrap();
        let (expected_rem_state, expected_residual_norm) = {
            let rem = manager.rem.lock().await;
            (rem.memory_state.clone(), rem.mean_residual_norm())
        };
        manager.save(&path).await.unwrap();

        let fresh_embedder = Arc::new(MockEmbedder::new(384));
        let fresh_bridge = Arc::new(EmbeddingBridge::new(fresh_embedder, 32, 32, 64));
        let restored = MemoryManager::load(&path, fresh_bridge).await.unwrap();

        assert_eq!(restored.len().await, 3);
        assert_eq!(restored.texts.lock().await.len(), 3);
        assert_vectors_close(
            restored.bridge.omega_proj.project(&probe).as_slice(),
            expected_omega.as_slice(),
        );
        assert_vectors_close(
            restored.bridge.key_proj.project(&probe).as_slice(),
            expected_key.as_slice(),
        );
        assert_vectors_close(
            restored.bridge.value_proj.project(&probe).as_slice(),
            expected_value.as_slice(),
        );
        let restored_rem = restored.rem.lock().await;
        assert_vectors_close(
            restored_rem.memory_state.as_slice(),
            expected_rem_state.as_slice(),
        );
        assert!((restored_rem.mean_residual_norm() - expected_residual_norm).abs() < 1e-12);
        drop(restored_rem);

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
        let before_residual = manager.rem.lock().await.mean_residual_norm();

        let known = manager.measure_surprise("known fact").await.unwrap();
        let novel = manager.measure_surprise("different fact").await.unwrap();

        assert!(known.score < 1e-12);
        assert!(novel.score > known.score);
        assert_eq!(manager.len().await, before_len);
        assert_eq!(
            manager.rem.lock().await.mean_residual_norm(),
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
        assert_eq!(manager.rem.lock().await.items().len(), 8);
        assert_eq!(manager.texts.lock().await.len(), 8);
        assert_eq!(*manager.next_id.lock().await, 8);
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
        assert!(manager.texts.lock().await.is_empty());
        assert!(manager.bridge.text_index().await.entries.is_empty());
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
        assert!(manager.store("second").await.is_err());
        assert_eq!(manager.len().await, 1);
    }

    #[tokio::test]
    async fn training_does_not_report_false_success_or_clear_need() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        for index in 0..5 {
            manager.store(&format!("fact {index}")).await.unwrap();
        }
        assert!(manager.health().await.unwrap().needs_training);

        let error = manager.train_nra().await.unwrap_err();
        assert!(error.to_string().contains("does not implement"));
        assert!(manager.health().await.unwrap().needs_training);
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

    #[tokio::test]
    async fn legacy_snapshot_without_parallel_texts_remains_loadable() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        let manager = MemoryManager::try_new(bridge).unwrap();
        manager.store("legacy memory").await.unwrap();
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
        let restored_texts = restored.texts.lock().await;
        assert_eq!(restored_texts.len(), 1);
        assert_eq!(restored_texts[0], "legacy memory");

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
