mod api;
mod config;
mod mcp;
mod state;
mod validate;

use config::ServerConfig;
use rai_core::embedding::{EmbeddingBridge, MockEmbedder, OpenAIEmbedder};
use rai_core::MemoryManager;
use state::AppState;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const USAGE: &str = "\
rai-server — a local memory service for AI tools. Stores facts, recalls them by
meaning, and reports how confident it is. Runs on your machine; nothing leaves
it unless you configure an external embedding provider.

Usage: rai-server [rest|mcp]

Modes:
  rest    Local REST API on RAI_HOST:RAI_PORT (default, 127.0.0.1:3000)
  mcp     Model Context Protocol server on stdio, for MCP clients

Options:
  -h, --help       Print this message and exit
  -V, --version    Print the version and exit

Getting started:
  rai-server rest                      serve the REST API on 127.0.0.1:3000
  RAI_DATA_PATH=./rai.json rai-server rest    keep memories across restarts

Configuration is read from the environment: RAI_HOST, RAI_PORT, RAI_API_TOKEN,
RAI_EMBEDDING_PROVIDER, OPENAI_API_KEY, RAI_DATA_PATH, RAI_CAPACITY,
RAI_MCP_MUTATIONS_ENABLED. See the README for defaults and accepted values.

Two defaults worth knowing: without RAI_DATA_PATH every memory is lost on exit,
and MCP write tools stay disabled until RAI_MCP_MUTATIONS_ENABLED=true.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            return Ok(());
        }
        "-V" | "--version" | "version" => {
            println!("rai-server {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        "" | "rest" | "mcp" => {}
        other => {
            return Err(config::ConfigError::new(format!(
                "unsupported mode '{other}'; expected 'rest' or 'mcp'. Run 'rai-server --help'."
            ))
            .into());
        }
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
    let manager =
        Arc::new(load_or_create_manager(data_path.as_deref(), bridge, config.capacity).await?);
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
        println!("  POST /v1/forget      - Remove a stored fact by its exact text");
        println!("  POST /v1/recall      - Retrieve with a score tier");
        println!("  POST /v1/intersect   - Concept intersection query");
        println!("  POST /v1/contradict  - Report address-space crowding for a candidate fact");
        println!("  POST /v1/surprise    - Measure novelty");
        println!("  POST /v1/confidence  - Explain the retrieval score");
        println!("  POST /v1/snapshot    - Crowding snapshot");
        println!("  GET  /v1/health      - System diagnostics");
        println!("Press Ctrl-C to stop; in-flight requests finish first.");

        // A durable store publishes its state only after the snapshot is on disk, so an abrupt
        // exit during that window would drop an acknowledged write. Draining first avoids it.
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }

    Ok(())
}

/// Resolve when the process is asked to stop.
async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => log::info!("shutdown requested; draining in-flight requests"),
        Err(error) => {
            // Without a working handler there is nothing to wait for; never returning would
            // block shutdown entirely, so stay pending and let the runtime exit normally.
            log::error!("could not install the Ctrl-C handler: {error}");
            std::future::pending::<()>().await;
        }
    }
}

async fn load_or_create_manager(
    data_path: Option<&Path>,
    bridge: Arc<EmbeddingBridge>,
    capacity: Option<usize>,
) -> Result<MemoryManager, Box<dyn std::error::Error>> {
    let fresh = |bridge| match capacity {
        Some(capacity) => MemoryManager::try_new_with_capacity(bridge, capacity),
        None => MemoryManager::try_new(bridge),
    };
    let Some(path) = data_path else {
        return Ok(fresh(bridge)?);
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
        return Ok(MemoryManager::load_with_capacity(path, bridge, capacity).await?);
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
    Ok(fresh(bridge)?)
}
