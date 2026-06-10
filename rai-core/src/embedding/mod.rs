pub mod bridge;
pub mod projection;
pub mod provider;

pub use bridge::EmbeddingBridge;
pub use provider::{Embedder, MockEmbedder, OpenAIEmbedder};
