//! Local associative-memory backend for the RAI workspace.
//!
//! Both stores are exact nearest-neighbour tables over cosine similarity: `store` appends a
//! (address, value) pair and retrieval scans every stored address for the best match. There is
//! no optimizer, no dynamical system, and no iterative solver in this crate.
#![forbid(unsafe_code)]

pub mod nra;
pub mod rem;

pub type Vec64 = nalgebra::DVector<f64>;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid memory data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
