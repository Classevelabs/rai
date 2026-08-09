pub mod nra;
pub mod rem;

pub type Vec64 = nalgebra::DVector<f64>;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("memory is empty")]
    Empty,
    #[error(
        "training is unavailable because this build does not implement parameter optimization"
    )]
    TrainingUnavailable,
    #[error("invalid memory data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
