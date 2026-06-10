pub mod embedding;
pub mod memory;
pub mod reasoning;
pub mod types;

pub use memory::MemoryManager;
pub use types::*;

/// RAI error type.
#[derive(Debug, thiserror::Error)]
pub enum RaiError {
    #[error("embedding error: {0}")]
    EmbeddingError(String),

    #[error("memory error: {0}")]
    MemoryError(String),

    #[error("training error: {0}")]
    TrainingError(String),

    #[error("persistence error: {0}")]
    PersistenceError(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}
