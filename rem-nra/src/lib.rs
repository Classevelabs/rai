pub mod nra;
pub mod rem;

pub type Vec64 = nalgebra::DVector<f64>;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("memory is empty")]
    Empty,
}

pub type Result<T> = std::result::Result<T, MemoryError>;
