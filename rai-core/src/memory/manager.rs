use crate::embedding::bridge::EmbeddingBridge;
use crate::memory::persistence::MemorySnapshot;
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
}

impl MemoryManager {
    /// Create a new MemoryManager with default configurations.
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
        }
    }

    /// Create with custom NRA/REM configs.
    pub fn with_configs(
        bridge: Arc<EmbeddingBridge>,
        nra_config: NRAConfig,
        rem_config: REMConfig,
    ) -> Self {
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
        }
    }

    /// Store a fact and return an interference report.
    pub async fn store(&self, content: &str) -> Result<InterferenceReport, RaiError> {
        let (omega, key, value) = self.bridge.embed_text(content).await?;

        // Take energy snapshot before
        let energy_before = {
            let nra = self.nra.lock().await;
            nra.energy_snapshot()
        };

        // Store in NRA
        {
            let mut nra = self.nra.lock().await;
            nra.store(&omega, &value)
                .map_err(|e| RaiError::MemoryError(format!("NRA store: {e}")))?;
        }

        // Store in REM
        {
            let mut rem = self.rem.lock().await;
            rem.store(&key, &value)
                .map_err(|e| RaiError::MemoryError(format!("REM store: {e}")))?;
        }

        // Record text
        {
            let mut texts = self.texts.lock().await;
            texts.push(content.to_string());
        }

        // Increment ID
        {
            let mut id = self.next_id.lock().await;
            *id += 1;
        }

        // Notify training orchestrator
        {
            let mut training = self.training.lock().await;
            training.item_stored();
        }

        // Take energy snapshot after
        let energy_after = {
            let nra = self.nra.lock().await;
            nra.energy_snapshot()
        };

        // Detect interference
        let texts = self.texts.lock().await;
        // The before snapshot has one fewer item, so only compare overlapping items
        let len = energy_before.len();
        let report =
            self.interference_detector
                .detect(&energy_before, &energy_after[..len], &texts[..len]);

        Ok(report)
    }

    /// Recall a memory with confidence diagnostics.
    pub async fn recall(&self, query: &str) -> Result<RetrievalResult, RaiError> {
        let omega = self.bridge.text_to_omega(query).await?;

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
        if concepts.is_empty() {
            return Err(RaiError::InvalidInput("no concepts provided".into()));
        }

        // Embed each concept to omega
        let mut omegas = Vec::with_capacity(concepts.len());
        for concept in concepts {
            let omega = self.bridge.text_to_omega(concept).await?;
            omegas.push(omega);
        }

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
        let (omega, _key, value) = self.bridge.embed_text(fact).await?;

        // Snapshot before
        let energy_before = {
            let nra = self.nra.lock().await;
            nra.energy_snapshot()
        };

        // Temporarily store (we'll revert if needed)
        {
            let mut nra = self.nra.lock().await;
            nra.store(&omega, &value)
                .map_err(|e| RaiError::MemoryError(format!("NRA contradict: {e}")))?;
        }

        // Snapshot after
        let energy_after = {
            let nra = self.nra.lock().await;
            nra.energy_snapshot()
        };

        let texts = self.texts.lock().await;
        let len = energy_before.len();
        let report =
            self.interference_detector
                .detect(&energy_before, &energy_after[..len], &texts[..len]);

        Ok(report)
    }

    /// Measure novelty/surprise of a fact using REM prior.
    pub async fn measure_surprise(&self, content: &str) -> Result<SurpriseResult, RaiError> {
        let (_omega, _key, _value) = self.bridge.embed_text(content).await?;

        // Get REM prior prediction
        let rem = self.rem.lock().await;
        let residual_norm = rem.mean_residual_norm();

        Ok(self.surprise_detector.score(residual_norm))
    }

    /// Explain the confidence of a retrieval in detail.
    pub async fn explain_confidence(&self, query: &str) -> Result<ConfidenceExplanation, RaiError> {
        let omega = self.bridge.text_to_omega(query).await?;

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
        let nra = self.nra.clone();
        let handle = TrainingOrchestrator::spawn_nra_retrain(nra);
        let result = handle
            .await
            .map_err(|e| RaiError::TrainingError(format!("join: {e}")))?;
        {
            let mut training = self.training.lock().await;
            training.nra_trained();
        }
        result
    }

    /// Trigger REM retraining.
    pub async fn train_rem(&self) -> Result<Vec<f64>, RaiError> {
        let rem = self.rem.clone();
        let handle = TrainingOrchestrator::spawn_rem_retrain(rem);
        let result = handle
            .await
            .map_err(|e| RaiError::TrainingError(format!("join: {e}")))?;
        {
            let mut training = self.training.lock().await;
            training.rem_trained();
        }
        result
    }

    /// Save full state to disk.
    pub async fn save(&self, path: &Path) -> Result<(), RaiError> {
        let nra = self.nra.lock().await;
        let rem = self.rem.lock().await;

        let snapshot = MemorySnapshot {
            nra_params: nra.params.clone(),
            nra_config: nra.config.clone(),
            nra_items: nra.items().to_vec(),
            rem_config: rem.config.clone(),
            rem_encoder: rem.encoder.clone(),
            rem_decoder: rem.decoder.clone(),
            rem_memory_state: rem.memory_state.clone(),
            rem_items: rem.items().to_vec(),
            text_index: self.bridge.text_index().await,
            omega_proj: self.bridge.omega_proj.clone(),
            key_proj: self.bridge.key_proj.clone(),
            value_proj: self.bridge.value_proj.clone(),
            total_items: nra.len(),
        };

        snapshot.save(path)
    }

    /// Load state from disk. Returns a new MemoryManager.
    pub async fn load(path: &Path, bridge: Arc<EmbeddingBridge>) -> Result<Self, RaiError> {
        let snapshot = MemorySnapshot::load(path)?;

        // Reconstruct NRA
        let mut nra =
            NonlinearResonanceMemory::from_params(snapshot.nra_params, snapshot.nra_config);
        for (omega, value) in &snapshot.nra_items {
            nra.store(omega, value)
                .map_err(|e| RaiError::MemoryError(format!("restore NRA: {e}")))?;
        }

        // Reconstruct REM
        let mut rng = rand::thread_rng();
        let mut rem = ResidualEquilibriumMemory::new(snapshot.rem_config, &mut rng);
        // Restore encoder/decoder params
        rem.encoder = snapshot.rem_encoder;
        rem.decoder = snapshot.rem_decoder;
        rem.memory_state = snapshot.rem_memory_state;
        for (key, value) in &snapshot.rem_items {
            // We only want to restore items, not re-encode (since we have the memory state)
            rem.store(key, value)
                .map_err(|e| RaiError::MemoryError(format!("restore REM: {e}")))?;
        }

        // Restore text index
        bridge.restore_text_index(snapshot.text_index.clone()).await;

        let texts: Vec<String> = snapshot
            .text_index
            .entries
            .iter()
            .map(|e| e.text.clone())
            .collect();

        Ok(Self {
            nra: Arc::new(Mutex::new(nra)),
            rem: Arc::new(Mutex::new(rem)),
            bridge,
            confidence_gate: ConfidenceGate::default(),
            interference_detector: InterferenceDetector::default(),
            basin_analyzer: BasinAnalyzer::default(),
            surprise_detector: SurpriseDetector::default(),
            training: Arc::new(Mutex::new(TrainingOrchestrator::default())),
            texts: Arc::new(Mutex::new(texts)),
            next_id: Arc::new(Mutex::new(snapshot.total_items)),
        })
    }

    /// Get number of stored memories.
    pub async fn len(&self) -> usize {
        let nra = self.nra.lock().await;
        nra.len()
    }

    /// Return true when no memories are stored.
    pub async fn is_empty(&self) -> bool {
        let nra = self.nra.lock().await;
        nra.is_empty()
    }

    /// Take an NRA energy snapshot for external use.
    pub async fn energy_snapshot(&self) -> Vec<(Vec64, f64)> {
        let nra = self.nra.lock().await;
        nra.energy_snapshot()
    }
}
