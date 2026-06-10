use crate::embedding::projection::Projection;
use crate::embedding::provider::Embedder;
use crate::RaiError;
use rem_nra::Vec64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bridges text to NRA/REM vector space via external embeddings + projection.
pub struct EmbeddingBridge {
    /// External embedding provider (OpenAI, local, mock).
    embedder: Arc<dyn Embedder>,
    /// Projection from embedding space to omega (NRA address).
    pub omega_proj: Projection,
    /// Projection from embedding space to key (REM key).
    pub key_proj: Projection,
    /// Projection from embedding space to value (shared value space).
    pub value_proj: Projection,
    /// Text index: maps text to its embedding for decoding back.
    text_index: Arc<RwLock<TextIndex>>,
}

/// Maps between text and vector representations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextIndex {
    /// All stored texts, indexed by ID.
    pub entries: Vec<TextEntry>,
    /// Text -> ID lookup.
    pub text_to_id: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEntry {
    pub id: usize,
    pub text: String,
    pub embedding: Vec<f64>,
}

impl TextIndex {
    pub fn insert(&mut self, text: String, embedding: Vec<f64>) -> usize {
        if let Some(&id) = self.text_to_id.get(&text) {
            return id;
        }
        let id = self.entries.len();
        self.text_to_id.insert(text.clone(), id);
        self.entries.push(TextEntry {
            id,
            text,
            embedding,
        });
        id
    }

    pub fn find_nearest(&self, embedding: &[f64]) -> Option<&TextEntry> {
        self.entries.iter().max_by(|a, b| {
            let sim_a = cosine_similarity(&a.embedding, embedding);
            let sim_b = cosine_similarity(&b.embedding, embedding);
            sim_a
                .partial_cmp(&sim_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn get_by_id(&self, id: usize) -> Option<&TextEntry> {
        self.entries.get(id)
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl EmbeddingBridge {
    /// Create a new bridge with random projections.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        dim_omega: usize,
        dim_key: usize,
        dim_value: usize,
    ) -> Self {
        let embed_dim = embedder.embedding_dim();
        let mut rng = rand::thread_rng();
        Self {
            embedder,
            omega_proj: Projection::random_gaussian(embed_dim, dim_omega, &mut rng),
            key_proj: Projection::random_gaussian(embed_dim, dim_key, &mut rng),
            value_proj: Projection::random_gaussian(embed_dim, dim_value, &mut rng),
            text_index: Arc::new(RwLock::new(TextIndex::default())),
        }
    }

    /// Create with custom projections (e.g., PCA-learned).
    pub fn with_projections(
        embedder: Arc<dyn Embedder>,
        omega_proj: Projection,
        key_proj: Projection,
        value_proj: Projection,
    ) -> Self {
        Self {
            embedder,
            omega_proj,
            key_proj,
            value_proj,
            text_index: Arc::new(RwLock::new(TextIndex::default())),
        }
    }

    /// Convert text to omega vector (NRA address).
    pub async fn text_to_omega(&self, text: &str) -> Result<Vec64, RaiError> {
        let embedding = self.embedder.embed(text).await?;
        let omega = self.omega_proj.project_normalized(&embedding);
        // Store in text index
        let mut index = self.text_index.write().await;
        index.insert(text.to_string(), embedding);
        Ok(omega)
    }

    /// Convert text to key vector (REM key).
    pub async fn text_to_key(&self, text: &str) -> Result<Vec64, RaiError> {
        let embedding = self.embedder.embed(text).await?;
        let key = self.key_proj.project_normalized(&embedding);
        let mut index = self.text_index.write().await;
        index.insert(text.to_string(), embedding);
        Ok(key)
    }

    /// Convert text to value vector.
    pub async fn text_to_value(&self, text: &str) -> Result<Vec64, RaiError> {
        let embedding = self.embedder.embed(text).await?;
        let value = self.value_proj.project(&embedding);
        let mut index = self.text_index.write().await;
        index.insert(text.to_string(), embedding);
        Ok(value)
    }

    /// Embed text and return all three projections (omega, key, value).
    pub async fn embed_text(&self, text: &str) -> Result<(Vec64, Vec64, Vec64), RaiError> {
        let embedding = self.embedder.embed(text).await?;
        let omega = self.omega_proj.project_normalized(&embedding);
        let key = self.key_proj.project_normalized(&embedding);
        let value = self.value_proj.project(&embedding);
        let mut index = self.text_index.write().await;
        index.insert(text.to_string(), embedding);
        Ok((omega, key, value))
    }

    /// Find the nearest stored text to a retrieved value vector.
    pub async fn nearest_text(&self, value: &Vec64) -> Option<String> {
        let index = self.text_index.read().await;
        // Project the value back to approximate embedding space
        // This is lossy but gives a nearest-neighbor hint
        index
            .entries
            .iter()
            .max_by(|a, b| {
                let proj_a = self.value_proj.project(&a.embedding);
                let proj_b = self.value_proj.project(&b.embedding);
                let sim_a = value.dot(&proj_a) / (value.norm() * proj_a.norm() + 1e-10);
                let sim_b = value.dot(&proj_b) / (value.norm() * proj_b.norm() + 1e-10);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.text.clone())
    }

    /// Get the text index (for persistence).
    pub async fn text_index(&self) -> TextIndex {
        self.text_index.read().await.clone()
    }

    /// Restore text index from persistence.
    pub async fn restore_text_index(&self, index: TextIndex) {
        *self.text_index.write().await = index;
    }
}
