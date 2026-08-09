use crate::RaiError;
use rem_nra::nra::NonlinearResonanceMemory;
use rem_nra::rem::ResidualEquilibriumMemory;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Orchestrates background retraining of NRA and REM.
#[derive(Clone)]
pub struct TrainingOrchestrator {
    /// Minimum items before triggering NRA retrain.
    pub nra_retrain_threshold: usize,
    /// Minimum items before triggering REM retrain.
    pub rem_retrain_threshold: usize,
    /// Items stored since last NRA retrain.
    nra_items_since_train: usize,
    /// Items stored since last REM retrain.
    rem_items_since_train: usize,
}

impl Default for TrainingOrchestrator {
    fn default() -> Self {
        Self {
            nra_retrain_threshold: 5,
            rem_retrain_threshold: 5,
            nra_items_since_train: 0,
            rem_items_since_train: 0,
        }
    }
}

impl TrainingOrchestrator {
    /// Notify that a new item was stored.
    pub fn item_stored(&mut self) {
        self.nra_items_since_train += 1;
        self.rem_items_since_train += 1;
    }

    /// Check if NRA needs retraining.
    pub fn needs_nra_retrain(&self) -> bool {
        self.nra_items_since_train >= self.nra_retrain_threshold
    }

    /// Check if REM needs retraining.
    pub fn needs_rem_retrain(&self) -> bool {
        self.rem_items_since_train >= self.rem_retrain_threshold
    }

    /// Reset NRA retrain counter.
    pub fn nra_trained(&mut self) {
        self.nra_items_since_train = 0;
    }

    /// Reset REM retrain counter.
    pub fn rem_trained(&mut self) {
        self.rem_items_since_train = 0;
    }

    /// Spawn a background NRA retrain task.
    pub fn spawn_nra_retrain(
        nra: Arc<Mutex<NonlinearResonanceMemory>>,
    ) -> tokio::task::JoinHandle<Result<Vec<f64>, RaiError>> {
        tokio::task::spawn_blocking(move || {
            let mut nra = nra.blocking_lock();
            nra.train_two_phase()
                .map_err(|e| RaiError::TrainingError(format!("NRA train: {e}")))
        })
    }

    /// Spawn a background REM retrain task.
    pub fn spawn_rem_retrain(
        rem: Arc<Mutex<ResidualEquilibriumMemory>>,
    ) -> tokio::task::JoinHandle<Result<Vec<f64>, RaiError>> {
        tokio::task::spawn_blocking(move || {
            let mut rem = rem.blocking_lock();
            rem.train()
                .map_err(|e| RaiError::TrainingError(format!("REM train: {e}")))
        })
    }
}
