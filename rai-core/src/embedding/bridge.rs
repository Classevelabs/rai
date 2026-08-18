use crate::embedding::projection::Projection;
use crate::embedding::provider::Embedder;
use crate::RaiError;
use rem_nra::Vec64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, RwLockWriteGuard};

/// Bridges text to memory vector space via external embeddings + projection.
pub struct EmbeddingBridge {
    /// External embedding provider (OpenAI, local, mock).
    embedder: Arc<dyn Embedder>,
    /// Projection from embedding space to the memory address space.
    pub omega_proj: Projection,
    /// Projection from embedding space to key space.
    pub key_proj: Projection,
    /// Projection from embedding space to the shared value space.
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
    /// `value_proj` applied to `embedding`, cached so a decode scan does not re-project every
    /// stored embedding. Derived state: rebuilt from `embedding` on load, never persisted.
    #[serde(skip, default = "uncached_projection")]
    value_projection: Vec64,
}

/// Placeholder for a projection that has not been rebuilt yet.
fn uncached_projection() -> Vec64 {
    Vec64::zeros(0)
}

impl TextIndex {
    /// Insert a text with its embedding and the matching value-space projection.
    ///
    /// The caller already computes the value projection when it projects the text, so passing it
    /// in keeps the decode scan a plain cosine comparison over cached vectors.
    pub fn insert(&mut self, text: String, embedding: Vec<f64>, value_projection: Vec64) -> usize {
        if let Some(&id) = self.text_to_id.get(&text) {
            return id;
        }
        let id = self.entries.len();
        self.text_to_id.insert(text.clone(), id);
        self.entries.push(TextEntry {
            id,
            text,
            embedding,
            value_projection,
        });
        id
    }

    /// Remove a text, if present.
    ///
    /// `insert` assigns `id = entries.len()`, so ids equal positions; the
    /// entries behind the removed one shift down and both their `id` fields
    /// and the lookup map are rewritten to keep that invariant.
    pub fn remove(&mut self, text: &str) -> bool {
        let Some(id) = self.text_to_id.remove(text) else {
            return false;
        };
        self.entries.remove(id);
        for (position, entry) in self.entries.iter_mut().enumerate().skip(id) {
            entry.id = position;
            self.text_to_id.insert(entry.text.clone(), position);
        }
        true
    }

    /// Recompute every cached value projection after a load.
    fn rebuild_value_projections(&mut self, projection: &Projection) -> Result<(), RaiError> {
        for entry in &mut self.entries {
            entry.value_projection = projection.project(&entry.embedding)?;
        }
        Ok(())
    }
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

    /// Create with custom projections (e.g., restored from a persisted snapshot).
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

    /// Clone the provider so persisted bridges can be reconstructed with saved projections.
    pub(crate) fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.embedder)
    }

    pub(crate) fn validate_memory_dimensions(
        &self,
        dim_omega: usize,
        dim_key: usize,
        dim_value: usize,
    ) -> Result<(), RaiError> {
        self.validate_projection("omega", &self.omega_proj)?;
        self.validate_projection("key", &self.key_proj)?;
        self.validate_projection("value", &self.value_proj)?;
        if self.omega_proj.target_dim != dim_omega
            || self.key_proj.target_dim != dim_key
            || self.value_proj.target_dim != dim_value
        {
            return Err(RaiError::InvalidInput(
                "embedding projection and memory dimensions do not match".into(),
            ));
        }
        Ok(())
    }

    /// Convert text to its memory address vector.
    pub async fn text_to_omega(&self, text: &str) -> Result<Vec64, RaiError> {
        self.validate_projection("omega", &self.omega_proj)?;
        let embedding = self.embed_validated(text).await?;
        let omega = self.omega_proj.project_normalized(&embedding)?;
        validate_projected("omega", &omega)?;
        Ok(omega)
    }

    /// Project text without mutating the durable text index.
    pub(crate) async fn project_text(
        &self,
        text: &str,
    ) -> Result<(Vec64, Vec64, Vec64, Vec<f64>), RaiError> {
        self.validate_projection("omega", &self.omega_proj)?;
        self.validate_projection("key", &self.key_proj)?;
        self.validate_projection("value", &self.value_proj)?;
        let embedding = self.embed_validated(text).await?;
        let omega = self.omega_proj.project_normalized(&embedding)?;
        let key = self.key_proj.project_normalized(&embedding)?;
        let value = self.value_proj.project(&embedding)?;
        validate_projected("omega", &omega)?;
        validate_projected("key", &key)?;
        validate_projected("value", &value)?;
        Ok((omega, key, value, embedding))
    }

    /// Find the nearest stored text to a retrieved value vector.
    ///
    /// The comparison runs against each entry's cached value projection, so this is a cosine
    /// scan rather than a re-projection of every stored embedding.
    pub async fn nearest_text(&self, value: &Vec64) -> Result<Option<String>, RaiError> {
        self.validate_projection("value", &self.value_proj)?;
        if value.len() != self.value_proj.target_dim {
            return Err(RaiError::EmbeddingError(format!(
                "value vector has {} dimensions; expected {}",
                value.len(),
                self.value_proj.target_dim
            )));
        }
        let index = self.text_index.read().await;
        Ok(index
            .entries
            .iter()
            .filter(|entry| entry.value_projection.len() == value.len())
            .max_by(|a, b| {
                let sim_a = cosine(value, &a.value_projection);
                let sim_b = cosine(value, &b.value_projection);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.text.clone()))
    }

    /// Get the text index (for persistence).
    pub(crate) async fn text_index(&self) -> TextIndex {
        self.text_index.read().await.clone()
    }

    /// Restore a text index loaded from disk, rebuilding the cached value projections.
    pub(crate) async fn restore_text_index(&self, mut index: TextIndex) -> Result<(), RaiError> {
        index.rebuild_value_projections(&self.value_proj)?;
        *self.text_index.write().await = index;
        Ok(())
    }

    pub(crate) async fn text_index_for_update(&self) -> RwLockWriteGuard<'_, TextIndex> {
        self.text_index.write().await
    }

    /// Publish an index staged in memory. The staged entries already carry their cached
    /// projections, so nothing is recomputed on this path.
    pub(crate) fn restore_text_index_blocking(&self, index: TextIndex) {
        *self.text_index.blocking_write() = index;
    }

    fn validate_projection(&self, name: &str, projection: &Projection) -> Result<(), RaiError> {
        if projection.source_dim != self.embedder.embedding_dim() || !projection.validate_shape() {
            return Err(RaiError::EmbeddingError(format!(
                "invalid {name} projection dimensions"
            )));
        }
        Ok(())
    }

    async fn embed_validated(&self, text: &str) -> Result<Vec<f64>, RaiError> {
        let embedding = self.embedder.embed(text).await?;
        let expected = self.embedder.embedding_dim();
        if embedding.len() != expected {
            return Err(RaiError::EmbeddingError(format!(
                "provider returned dimension {}; expected {expected}",
                embedding.len()
            )));
        }
        if embedding
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1.0e100)
        {
            return Err(RaiError::EmbeddingError(
                "provider returned non-finite embedding values".into(),
            ));
        }
        Ok(embedding)
    }
}

fn cosine(value: &Vec64, projected: &Vec64) -> f64 {
    value.dot(projected) / (value.norm() * projected.norm() + 1e-10)
}

fn validate_projected(name: &str, vector: &Vec64) -> Result<(), RaiError> {
    if vector
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 1.0e100)
    {
        return Err(RaiError::EmbeddingError(format!(
            "{name} projection produced non-finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedder;

    #[tokio::test]
    async fn cached_projections_match_a_fresh_projection_after_a_reload() {
        let bridge = EmbeddingBridge::new(Arc::new(MockEmbedder::new(64)), 32, 32, 16);
        let (_, _, value, embedding) = bridge.project_text("cached entry").await.unwrap();
        {
            let mut index = bridge.text_index_for_update().await;
            index.insert("cached entry".to_string(), embedding.clone(), value.clone());
        }
        assert_eq!(
            bridge.nearest_text(&value).await.unwrap().as_deref(),
            Some("cached entry")
        );

        // A snapshot round-trip drops the cached vectors; the restore path has to rebuild them
        // or the decode scan would silently match nothing.
        let serialized = serde_json::to_string(&bridge.text_index().await).unwrap();
        let reloaded: TextIndex = serde_json::from_str(&serialized).unwrap();
        assert_eq!(reloaded.entries[0].value_projection.len(), 0);

        bridge.restore_text_index(reloaded).await.unwrap();
        assert_eq!(
            bridge.nearest_text(&value).await.unwrap().as_deref(),
            Some("cached entry")
        );
    }

    #[tokio::test]
    async fn nearest_text_rejects_a_wrongly_sized_value_vector() {
        let bridge = EmbeddingBridge::new(Arc::new(MockEmbedder::new(64)), 32, 32, 16);
        assert!(bridge.nearest_text(&Vec64::zeros(15)).await.is_err());
        assert!(bridge.nearest_text(&Vec64::zeros(16)).await.is_ok());
    }
}
