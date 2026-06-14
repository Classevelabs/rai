use crate::mcp::schema::{JsonRpcRequest, JsonRpcResponse, ToolCallResult};
use crate::mcp::tools::tool_definitions;
use rai_core::MemoryManager;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Run the MCP server on stdio so MCP clients (such as Claude Desktop or
/// Claude Code) can use RAI memory as a set of tools.
pub async fn run_mcp_stdio(manager: Arc<MemoryManager>) {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                write_response(&mut stdout, &resp).await;
                continue;
            }
        };

        let response = handle_request(&manager, request).await;
        write_response(&mut stdout, &response).await;
    }
}

async fn handle_request(manager: &Arc<MemoryManager>, request: JsonRpcRequest) -> JsonRpcResponse {
    if request.jsonrpc != "2.0" {
        return JsonRpcResponse::error(
            request.id,
            -32600,
            "Invalid JSON-RPC version; expected 2.0".to_string(),
        );
    }

    match request.method.as_str() {
        "initialize" => JsonRpcResponse::success(
            request.id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "rai-server",
                    "version": "0.1.0"
                }
            }),
        ),

        "notifications/initialized" => {
            // No response needed for notifications, but we still return success
            JsonRpcResponse::success(request.id, json!({}))
        }

        "tools/list" => {
            let tools = tool_definitions();
            JsonRpcResponse::success(request.id, json!({ "tools": tools }))
        }

        "tools/call" => {
            let tool_name = request
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(json!({}));

            let result = handle_tool_call(manager, tool_name, arguments).await;

            match serde_json::to_value(result) {
                Ok(value) => JsonRpcResponse::success(request.id, value),
                Err(e) => JsonRpcResponse::error(
                    request.id,
                    -32603,
                    format!("Failed to serialize tool result: {e}"),
                ),
            }
        }

        _ => JsonRpcResponse::error(
            request.id,
            -32601,
            format!("Method not found: {}", request.method),
        ),
    }
}

async fn write_response(stdout: &mut tokio::io::Stdout, response: &JsonRpcResponse) {
    let out = match serde_json::to_string(response) {
        Ok(out) => out,
        Err(e) => {
            log::error!("failed to serialize MCP response: {e}");
            return;
        }
    };

    if let Err(e) = stdout.write_all(out.as_bytes()).await {
        log::error!("failed to write MCP response: {e}");
        return;
    }
    if let Err(e) = stdout.write_all(b"\n").await {
        log::error!("failed to terminate MCP response: {e}");
        return;
    }
    if let Err(e) = stdout.flush().await {
        log::error!("failed to flush MCP response: {e}");
    }
}

fn pretty_json<T: Serialize>(value: &T) -> Result<String, ToolCallResult> {
    serde_json::to_string_pretty(value)
        .map_err(|e| ToolCallResult::error(format!("Failed to serialize response: {e}")))
}

async fn handle_tool_call(
    manager: &Arc<MemoryManager>,
    tool_name: &str,
    args: Value,
) -> ToolCallResult {
    match tool_name {
        "rai_store" => {
            let content = match args.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => return ToolCallResult::error("Missing 'content' parameter".into()),
            };
            match manager.store(content).await {
                Ok(report) => {
                    let text = match pretty_json(&report) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(format!(
                        "Stored successfully.\n\nInterference report:\n{text}"
                    ))
                }
                Err(e) => ToolCallResult::error(format!("Store failed: {e}")),
            }
        }

        "rai_recall" => {
            let query = match args.get("query").and_then(|v| v.as_str()) {
                Some(q) => q,
                None => return ToolCallResult::error("Missing 'query' parameter".into()),
            };
            match manager.recall(query).await {
                Ok(result) => {
                    let text = match pretty_json(&result) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(text)
                }
                Err(e) => ToolCallResult::error(format!("Recall failed: {e}")),
            }
        }

        "rai_intersect" => {
            let concepts: Vec<String> = match args.get("concepts").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                None => return ToolCallResult::error("Missing 'concepts' parameter".into()),
            };
            match manager.intersect(&concepts).await {
                Ok(result) => {
                    let text = match pretty_json(&result) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(text)
                }
                Err(e) => ToolCallResult::error(format!("Intersect failed: {e}")),
            }
        }

        "rai_contradict" => {
            let fact = match args.get("fact").and_then(|v| v.as_str()) {
                Some(f) => f,
                None => return ToolCallResult::error("Missing 'fact' parameter".into()),
            };
            match manager.check_contradiction(fact).await {
                Ok(report) => {
                    let text = match pretty_json(&report) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(text)
                }
                Err(e) => ToolCallResult::error(format!("Contradiction check failed: {e}")),
            }
        }

        "rai_surprise" => {
            let content = match args.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => return ToolCallResult::error("Missing 'content' parameter".into()),
            };
            match manager.measure_surprise(content).await {
                Ok(result) => {
                    let text = match pretty_json(&result) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(text)
                }
                Err(e) => ToolCallResult::error(format!("Surprise measurement failed: {e}")),
            }
        }

        "rai_explain_confidence" => {
            let query = match args.get("query").and_then(|v| v.as_str()) {
                Some(q) => q,
                None => return ToolCallResult::error("Missing 'query' parameter".into()),
            };
            match manager.explain_confidence(query).await {
                Ok(result) => {
                    let text = match pretty_json(&result) {
                        Ok(text) => text,
                        Err(result) => return result,
                    };
                    ToolCallResult::text(text)
                }
                Err(e) => ToolCallResult::error(format!("Confidence explanation failed: {e}")),
            }
        }

        "rai_memory_health" => match manager.health().await {
            Ok(report) => {
                let text = match pretty_json(&report) {
                    Ok(text) => text,
                    Err(result) => return result,
                };
                ToolCallResult::text(text)
            }
            Err(e) => ToolCallResult::error(format!("Health check failed: {e}")),
        },

        _ => ToolCallResult::error(format!("Unknown tool: {tool_name}")),
    }
}
