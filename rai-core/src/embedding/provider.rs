use crate::RaiError;
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_EMBEDDING_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_BATCH_ITEMS: usize = 128;
const MAX_INPUT_BYTES: usize = 64 * 1024;

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
    pub fn new(api_key: String) -> Result<Self, RaiError> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| {
                RaiError::EmbeddingError(format!("failed to create HTTP client: {error}"))
            })?;
        Ok(Self {
            client,
            api_key,
            model: "text-embedding-3-small".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            dim: 1536,
        })
    }

    pub fn with_model(mut self, model: &str, dim: usize) -> Result<Self, RaiError> {
        if model.trim().is_empty() || dim == 0 || dim > 16_384 {
            return Err(RaiError::InvalidInput(
                "embedding model and dimension are invalid".into(),
            ));
        }
        self.model = model.to_string();
        self.dim = dim;
        Ok(self)
    }

    pub fn with_base_url(mut self, url: &str) -> Result<Self, RaiError> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|_| RaiError::InvalidInput("embedding base URL is invalid".into()))?;
        let loopback_http = parsed.scheme() == "http"
            && parsed.host_str().is_some_and(|host| {
                host.eq_ignore_ascii_case("localhost")
                    || host
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|address| address.is_loopback())
            });
        if (parsed.scheme() != "https" && !loopback_http)
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(RaiError::InvalidInput(
                "embedding base URL must use HTTPS (or loopback HTTP) without credentials, query, or fragment"
                    .into(),
            ));
        }
        self.base_url = url.trim_end_matches('/').to_string();
        Ok(self)
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
    index: usize,
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
        if texts.len() > MAX_BATCH_ITEMS {
            return Err(RaiError::InvalidInput(format!(
                "embedding batch exceeds the {MAX_BATCH_ITEMS}-item limit"
            )));
        }
        if texts
            .iter()
            .any(|text| text.is_empty() || text.len() > MAX_INPUT_BYTES)
        {
            return Err(RaiError::InvalidInput(format!(
                "embedding inputs must contain 1..={MAX_INPUT_BYTES} bytes"
            )));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = EmbedRequest {
            input: texts.iter().map(|t| t.to_string()).collect(),
            model: self.model.clone(),
        };

        let mut response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| RaiError::EmbeddingError(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(RaiError::EmbeddingError(format!("API returned {status}")));
        }

        if response
            .content_length()
            .is_some_and(|length| length > MAX_EMBEDDING_RESPONSE_BYTES as u64)
        {
            return Err(RaiError::EmbeddingError(
                "embedding response exceeds size limit".into(),
            ));
        }

        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| RaiError::EmbeddingError(format!("response read failed: {error}")))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_EMBEDDING_RESPONSE_BYTES {
                return Err(RaiError::EmbeddingError(
                    "embedding response exceeds size limit".into(),
                ));
            }
            body.extend_from_slice(&chunk);
        }

        decode_embedding_response(&body, texts.len(), self.dim)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

fn decode_embedding_response(
    body: &[u8],
    expected_count: usize,
    expected_dim: usize,
) -> Result<Vec<Vec<f64>>, RaiError> {
    let response: EmbedResponse = serde_json::from_slice(body)
        .map_err(|error| RaiError::EmbeddingError(format!("parse error: {error}")))?;
    if response.data.len() != expected_count {
        return Err(RaiError::EmbeddingError(format!(
            "provider returned {} embeddings; expected {expected_count}",
            response.data.len()
        )));
    }

    let mut ordered = vec![None; expected_count];
    for item in response.data {
        if item.index >= expected_count || ordered[item.index].is_some() {
            return Err(RaiError::EmbeddingError(
                "provider returned invalid embedding indexes".into(),
            ));
        }
        if item.embedding.len() != expected_dim {
            return Err(RaiError::EmbeddingError(format!(
                "provider returned dimension {}; expected {expected_dim}",
                item.embedding.len()
            )));
        }
        if item
            .embedding
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1.0e100)
        {
            return Err(RaiError::EmbeddingError(
                "provider returned non-finite embedding values".into(),
            ));
        }
        ordered[item.index] = Some(item.embedding);
    }

    ordered
        .into_iter()
        .map(|embedding| {
            embedding.ok_or_else(|| {
                RaiError::EmbeddingError("provider omitted an embedding index".into())
            })
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_response_is_reordered_by_index() {
        let body =
            br#"{"data":[{"index":1,"embedding":[3.0,4.0]},{"index":0,"embedding":[1.0,2.0]}]}"#;
        let decoded = decode_embedding_response(body, 2, 2).unwrap();
        assert_eq!(decoded, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn embedding_response_rejects_wrong_dimensions_and_indexes() {
        let wrong_dimension = br#"{"data":[{"index":0,"embedding":[1.0]}]}"#;
        assert!(decode_embedding_response(wrong_dimension, 1, 2).is_err());

        let duplicate_index =
            br#"{"data":[{"index":0,"embedding":[1.0,2.0]},{"index":0,"embedding":[3.0,4.0]}]}"#;
        assert!(decode_embedding_response(duplicate_index, 2, 2).is_err());
    }

    #[test]
    fn custom_embedding_endpoint_requires_secure_transport() {
        assert!(OpenAIEmbedder::new("secret".into())
            .unwrap()
            .with_base_url("http://api.example.test/v1")
            .is_err());
        assert!(OpenAIEmbedder::new("secret".into())
            .unwrap()
            .with_base_url("http://127.0.0.1:8080/v1")
            .is_ok());
        assert!(OpenAIEmbedder::new("secret".into())
            .unwrap()
            .with_base_url("https://api.example.test/v1")
            .is_ok());
    }
}
