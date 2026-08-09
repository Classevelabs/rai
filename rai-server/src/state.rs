use rai_core::{InterferenceReport, MemoryManager, RaiError};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard};

/// Shared application state for REST and MCP transports.
#[derive(Clone)]
pub struct AppState {
    pub manager: Arc<MemoryManager>,
    persistence_path: Option<Arc<PathBuf>>,
    persistence_lock: Arc<Mutex<()>>,
    training_lock: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(manager: Arc<MemoryManager>, persistence_path: Option<PathBuf>) -> Self {
        Self {
            manager,
            persistence_path: persistence_path.map(Arc::new),
            persistence_lock: Arc::new(Mutex::new(())),
            training_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Store through the durable mutation boundary when persistence is enabled.
    pub async fn store(&self, content: &str) -> Result<InterferenceReport, RaiError> {
        let Some(path) = &self.persistence_path else {
            return self.manager.store(content).await;
        };

        let _guard = self.persistence_lock.lock().await;
        sanitize_persistence_error(self.manager.store_and_save(content, path.as_ref()).await)
    }

    /// Retrain NRA without publishing an undurable trained state.
    pub async fn train_nra(&self) -> Result<Vec<f64>, RaiError> {
        let Some(path) = &self.persistence_path else {
            return self.manager.train_nra().await;
        };

        let _guard = self.persistence_lock.lock().await;
        sanitize_persistence_error(self.manager.train_nra_and_save(path.as_ref()).await)
    }

    /// Return a guard only when no other training request is active.
    pub fn try_training_lock(&self) -> Option<OwnedMutexGuard<()>> {
        Arc::clone(&self.training_lock).try_lock_owned().ok()
    }
}

fn sanitize_persistence_error<T>(result: Result<T, RaiError>) -> Result<T, RaiError> {
    match result {
        Err(RaiError::PersistenceError(details)) => {
            log::error!("durable state update failed: {details}");
            Err(RaiError::PersistenceError(
                "durable state update failed".to_string(),
            ))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_details_are_not_returned_to_callers() {
        let error = sanitize_persistence_error::<()>(Err(RaiError::PersistenceError(
            "C:\\private\\user\\memory.json: access denied".to_string(),
        )))
        .unwrap_err();
        let public_message = error.to_string();
        assert!(public_message.contains("durable state update failed"));
        assert!(!public_message.contains("private"));
        assert!(!public_message.contains("access denied"));
    }
}
