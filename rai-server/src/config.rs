use serde::{Deserialize, Serialize};

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP port.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Embedding provider: "openai" or "mock".
    pub embedding_provider: String,
    /// OpenAI API key (if using OpenAI).
    pub openai_api_key: Option<String>,
    /// NRA state dimension.
    pub dim_state: usize,
    /// NRA omega dimension.
    pub dim_omega: usize,
    /// NRA/REM value dimension.
    pub dim_value: usize,
    /// REM memory dimension.
    pub dim_memory: usize,
    /// REM key dimension.
    pub dim_key: usize,
    /// NRA number of nonlinear units.
    pub num_units: usize,
    /// Path for persistence.
    pub data_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            host: "0.0.0.0".to_string(),
            embedding_provider: "mock".to_string(),
            openai_api_key: None,
            dim_state: 64,
            dim_omega: 32,
            dim_value: 64,
            dim_memory: 256,
            dim_key: 32,
            num_units: 512,
            data_path: None,
        }
    }
}

impl ServerConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(port) = std::env::var("RAI_PORT") {
            if let Ok(p) = port.parse() {
                config.port = p;
            }
        }
        if let Ok(host) = std::env::var("RAI_HOST") {
            config.host = host;
        }
        if let Ok(provider) = std::env::var("RAI_EMBEDDING_PROVIDER") {
            config.embedding_provider = provider;
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.openai_api_key = Some(key);
        }
        if let Ok(path) = std::env::var("RAI_DATA_PATH") {
            config.data_path = Some(path);
        }

        config
    }
}
