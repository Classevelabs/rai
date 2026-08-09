use std::fmt;

const MIN_API_TOKEN_BYTES: usize = 32;

/// Invalid server configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Server configuration.
#[derive(Clone)]
pub struct ServerConfig {
    /// HTTP port.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Embedding provider: "openai" or "mock".
    pub embedding_provider: String,
    /// OpenAI API key (if using OpenAI).
    pub openai_api_key: Option<String>,
    /// Bearer token required by the REST API when configured.
    pub api_token: Option<String>,
    /// Whether MCP clients may invoke state-changing tools.
    pub mcp_mutations_enabled: bool,
    /// NRA omega dimension.
    pub dim_omega: usize,
    /// NRA/REM value dimension.
    pub dim_value: usize,
    /// REM key dimension.
    pub dim_key: usize,
    /// Path for persistence.
    pub data_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            host: "127.0.0.1".to_string(),
            embedding_provider: "mock".to_string(),
            openai_api_key: None,
            api_token: None,
            mcp_mutations_enabled: false,
            dim_omega: 32,
            dim_value: 64,
            dim_key: 32,
            data_path: None,
        }
    }
}

impl ServerConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Self::default();

        if let Ok(port) = std::env::var("RAI_PORT") {
            config.port = port.parse::<u16>().map_err(|_| {
                ConfigError::new(format!("invalid RAI_PORT '{port}'; expected 1..=65535"))
            })?;
            if config.port == 0 {
                return Err(ConfigError::new("RAI_PORT must be greater than zero"));
            }
        }
        if let Ok(host) = std::env::var("RAI_HOST") {
            if host.trim().is_empty() {
                return Err(ConfigError::new("RAI_HOST must not be empty"));
            }
            config.host = host;
        }
        if let Ok(provider) = std::env::var("RAI_EMBEDDING_PROVIDER") {
            config.embedding_provider = provider;
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.openai_api_key = Some(key);
        }
        if let Ok(token) = std::env::var("RAI_API_TOKEN") {
            config.api_token = Some(token);
        }
        if let Ok(enabled) = std::env::var("RAI_MCP_MUTATIONS_ENABLED") {
            config.mcp_mutations_enabled =
                parse_strict_bool("RAI_MCP_MUTATIONS_ENABLED", &enabled)?;
        }
        if let Ok(path) = std::env::var("RAI_DATA_PATH") {
            config.data_path = Some(path);
        }

        config.validate()?;
        Ok(config)
    }

    /// Validate provider and credential settings used in both REST and MCP modes.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.embedding_provider.as_str() {
            "mock" => {}
            "openai" => {
                if self
                    .openai_api_key
                    .as_deref()
                    .is_none_or(|key| key.trim().is_empty())
                {
                    return Err(ConfigError::new(
                        "OPENAI_API_KEY must be set when RAI_EMBEDDING_PROVIDER=openai",
                    ));
                }
            }
            provider => {
                return Err(ConfigError::new(format!(
                    "unsupported RAI_EMBEDDING_PROVIDER '{provider}'; expected 'mock' or 'openai'"
                )));
            }
        }

        if let Some(token) = &self.api_token {
            if token.len() < MIN_API_TOKEN_BYTES {
                return Err(ConfigError::new(format!(
                    "RAI_API_TOKEN must be at least {MIN_API_TOKEN_BYTES} bytes"
                )));
            }
        }

        if self
            .data_path
            .as_deref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(ConfigError::new("RAI_DATA_PATH must not be empty"));
        }

        Ok(())
    }

    /// Refuse network exposure because the built-in listener does not terminate TLS.
    pub fn validate_rest_security(&self) -> Result<(), ConfigError> {
        if !is_loopback_host(&self.host) {
            return Err(ConfigError::new(
                "RAI serves HTTP without TLS and refuses non-loopback RAI_HOST values; keep it on loopback behind a TLS reverse proxy",
            ));
        }
        Ok(())
    }
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let unbracketed = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    unbracketed
        .parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_loopback())
}

fn parse_strict_bool(name: &str, value: &str) -> Result<bool, ConfigError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ConfigError::new(format!(
            "invalid {name} '{value}'; expected 'true' or 'false'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_loopback_only() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.validate_rest_security().is_ok());
        let ipv6 = ServerConfig {
            host: "::1".to_string(),
            ..ServerConfig::default()
        };
        assert!(ipv6.validate_rest_security().is_ok());
    }

    #[test]
    fn rejects_unknown_embedding_provider() {
        let config = ServerConfig {
            embedding_provider: "typo".to_string(),
            ..ServerConfig::default()
        };

        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn external_bind_is_refused_even_with_a_token() {
        let mut config = ServerConfig {
            host: "0.0.0.0".to_string(),
            ..ServerConfig::default()
        };
        assert!(config.validate_rest_security().is_err());

        config.api_token = Some("too-short".to_string());
        assert!(config.validate().is_err());

        config.api_token = Some("a".repeat(MIN_API_TOKEN_BYTES));
        assert!(config.validate().is_ok());
        assert!(config.validate_rest_security().is_err());
    }

    #[test]
    fn mcp_mutations_require_strict_explicit_opt_in() {
        assert!(parse_strict_bool("flag", "true").unwrap());
        assert!(!parse_strict_bool("flag", "false").unwrap());
        assert!(parse_strict_bool("flag", "TRUE").is_err());
        assert!(parse_strict_bool("flag", "1").is_err());
    }
}
