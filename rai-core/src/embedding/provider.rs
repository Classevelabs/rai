use crate::RaiError;

/// Trait for embedding providers.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f64>, RaiError>;

    /// Embed multiple texts (batch).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, RaiError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// The output dimension of embeddings.
    fn embedding_dim(&self) -> usize;
}

/// OpenAI-compatible embedding provider.
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    dim: usize,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: "text-embedding-3-small".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            dim: 1536,
        }
    }

    pub fn with_model(mut self, model: &str, dim: usize) -> Self {
        self.model = model.to_string();
        self.dim = dim;
        self
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest {
    input: Vec<String>,
    model: String,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(serde::Deserialize)]
struct EmbedData {
    embedding: Vec<f64>,
}

#[async_trait::async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, RaiError> {
        let batch = self.embed_batch(&[text]).await?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| RaiError::EmbeddingError("empty response".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, RaiError> {
        let request = EmbedRequest {
            input: texts.iter().map(|t| t.to_string()).collect(),
            model: self.model.clone(),
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| RaiError::EmbeddingError(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(RaiError::EmbeddingError(format!(
                "API error {status}: {body}"
            )));
        }

        let resp: EmbedResponse = response
            .json()
            .await
            .map_err(|e| RaiError::EmbeddingError(format!("parse error: {e}")))?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// Mock embedder for testing — uses simple hash-based vectors.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait::async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, RaiError> {
        // Deterministic pseudo-embedding from text hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f64; self.dim];
        for (i, chunk) in embedding.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Map to [-1, 1] range
            *chunk = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }

        // Normalize
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}
