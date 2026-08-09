//! Memory, embedding, and reasoning primitives for the RAI memory service.
//!
//! Text is embedded by a provider, projected into fixed-dimension address/key/value vectors, and
//! stored in the `rem-nra` nearest-neighbour tables. Every retrieval, intersection, contradiction,
//! and confidence output in this crate is derived from cosine similarity over those vectors.

pub mod embedding;
pub mod memory;
pub mod reasoning;
pub mod types;

pub use memory::manager::{MAX_INTERSECTION_CONCEPTS, MAX_TEXT_BYTES};
pub use memory::MemoryManager;
pub use types::*;

/// RAI error type.
#[derive(Debug, thiserror::Error)]
pub enum RaiError {
    #[error("embedding error: {0}")]
    EmbeddingError(String),

    #[error("memory error: {0}")]
    MemoryError(String),

    /// The store is full. This is a client-visible condition, not an internal fault: the caller
    /// has to remove memories or raise the configured capacity before storing again.
    #[error("memory is full: the {limit}-item store capacity has been reached")]
    CapacityExhausted { limit: usize },

    #[error("persistence error: {0}")]
    PersistenceError(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}
