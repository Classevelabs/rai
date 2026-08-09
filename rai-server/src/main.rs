mod api;
mod config;
mod mcp;
mod state;

use config::ServerConfig;
use rai_core::embedding::{EmbeddingBridge, MockEmbedder, OpenAIEmbedder};
use rai_core::MemoryManager;
use state::AppState;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mode = std::env::args().nth(1).unwrap_or_default();
    if !matches!(mode.as_str(), "" | "rest" | "mcp") {
        return Err(config::ConfigError::new(format!(
            "unsupported mode '{mode}'; expected 'rest' or 'mcp'"
        ))
        .into());
    }
    let config = ServerConfig::from_env()?;
    if mode != "mcp" {
        config.validate_rest_security()?;
    }

    // Create embedding provider
    let embedder: Arc<dyn rai_core::embedding::Embedder> = match config.embedding_provider.as_str()
    {
        "openai" => {
            let api_key = config
                .openai_api_key
                .clone()
                .ok_or_else(|| config::ConfigError::new("OPENAI_API_KEY is missing"))?;
            Arc::new(OpenAIEmbedder::new(api_key)?)
        }
        "mock" => {
            eprintln!(
                "WARNING: RAI is using deterministic mock embeddings; this mode is for tests and demos, not semantic retrieval"
            );
            Arc::new(MockEmbedder::new(384))
        }
        provider => {
            return Err(config::ConfigError::new(format!(
                "unsupported embedding provider '{provider}'"
            ))
            .into());
        }
    };

    // Create embedding bridge
    let bridge = Arc::new(EmbeddingBridge::new(
        embedder,
        config.dim_omega,
        config.dim_key,
        config.dim_value,
    ));

    let data_path = config.data_path.as_deref().map(PathBuf::from);
    if data_path.is_none() {
        eprintln!(
            "WARNING: RAI_DATA_PATH is not set; all stored memories are ephemeral and will be lost on shutdown"
        );
    }
    let manager = Arc::new(load_or_create_manager(data_path.as_deref(), bridge).await?);
    let state = AppState::new(manager, data_path);

    if mode == "mcp" {
        // MCP stdio mode for MCP clients (e.g. Claude Desktop, Claude Code)
        log::info!("Starting RAI MCP server on stdio");
        if config.mcp_mutations_enabled {
            eprintln!(
                "WARNING: MCP mutation tools are enabled and inherit the connected client's OS permissions"
            );
        }
        mcp::server::run_mcp_stdio(state, config.mcp_mutations_enabled).await;
    } else {
        // REST API mode
        let addr = if config.host.contains(':') && !config.host.starts_with('[') {
            format!("[{}]:{}", config.host, config.port)
        } else {
            format!("{}:{}", config.host, config.port)
        };
        log::info!("Starting RAI REST server on {addr}");

        let router = api::routes::build_router(state, config.api_token.clone(), config.port);

        let listener = tokio::net::TcpListener::bind(&addr).await?;

        println!("RAI server listening on {addr}");
        println!("  POST /v1/store       - Store a fact");
        println!("  POST /v1/recall      - Retrieve with confidence");
        println!("  POST /v1/intersect   - Concept intersection query");
        println!("  POST /v1/contradict  - Check for contradictions");
        println!("  POST /v1/surprise    - Measure novelty");
        println!("  POST /v1/confidence  - Explain confidence");
        println!("  POST /v1/train       - Unavailable (optimization not implemented)");
        println!("  POST /v1/snapshot    - Energy snapshot");
        println!("  GET  /v1/health      - System diagnostics");

        axum::serve(listener, router).await?;
    }

    Ok(())
}

async fn load_or_create_manager(
    data_path: Option<&Path>,
    bridge: Arc<EmbeddingBridge>,
) -> Result<MemoryManager, Box<dyn std::error::Error>> {
    let Some(path) = data_path else {
        return Ok(MemoryManager::try_new(bridge)?);
    };

    if path.is_dir() {
        return Err(config::ConfigError::new(format!(
            "RAI_DATA_PATH must name a file, not a directory: {}",
            path.display()
        ))
        .into());
    }

    if path.exists() {
        log::info!("Loading persisted state from {}", path.display());
        return Ok(MemoryManager::load(path, bridge).await?);
    }

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    log::info!(
        "Persistence enabled; a snapshot will be created at {} after the first mutation",
        path.display()
    );
    Ok(MemoryManager::try_new(bridge)?)
}
