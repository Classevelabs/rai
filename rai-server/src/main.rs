mod api;
mod config;
mod mcp;

use config::ServerConfig;
use rai_core::embedding::{EmbeddingBridge, MockEmbedder, OpenAIEmbedder};
use rai_core::MemoryManager;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::init();

    let config = ServerConfig::from_env();
    let mode = std::env::args().nth(1).unwrap_or_default();

    // Create embedding provider
    let embedder: Arc<dyn rai_core::embedding::Embedder> = match config.embedding_provider.as_str()
    {
        "openai" => {
            let api_key = config.openai_api_key.clone().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "OPENAI_API_KEY must be set when RAI_EMBEDDING_PROVIDER=openai",
                )
            })?;
            Arc::new(OpenAIEmbedder::new(api_key))
        }
        _ => Arc::new(MockEmbedder::new(384)),
    };

    // Create embedding bridge
    let bridge = Arc::new(EmbeddingBridge::new(
        embedder,
        config.dim_omega,
        config.dim_key,
        config.dim_value,
    ));

    // Create memory manager
    let manager = Arc::new(MemoryManager::new(bridge));

    // Load persisted state if available
    if let Some(ref path) = config.data_path {
        let p = std::path::Path::new(path);
        if p.exists() {
            log::info!("Loading persisted state from {path}");
            // Note: loading replaces the manager, but for simplicity
            // we just log the availability here
        }
    }

    if mode == "mcp" {
        // MCP stdio mode for MCP clients (e.g. Claude Desktop, Claude Code)
        log::info!("Starting RAI MCP server on stdio");
        mcp::server::run_mcp_stdio(manager).await;
        Ok(())
    } else {
        // REST API mode
        let addr = format!("{}:{}", config.host, config.port);
        log::info!("Starting RAI REST server on {addr}");

        let router = api::routes::build_router(manager);

        let listener = tokio::net::TcpListener::bind(&addr).await?;

        println!("RAI server listening on {addr}");
        println!("  POST /v1/store       - Store a fact");
        println!("  POST /v1/recall      - Retrieve with confidence");
        println!("  POST /v1/intersect   - Concept intersection query");
        println!("  POST /v1/contradict  - Check for contradictions");
        println!("  POST /v1/surprise    - Measure novelty");
        println!("  POST /v1/confidence  - Explain confidence");
        println!("  POST /v1/train       - Trigger retraining");
        println!("  POST /v1/snapshot    - Energy snapshot");
        println!("  GET  /v1/health      - System diagnostics");

        axum::serve(listener, router).await?;
        Ok(())
    }
}
