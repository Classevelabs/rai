use crate::mcp::schema::{JsonRpcRequest, JsonRpcResponse, ToolCallResult};
use crate::mcp::tools::tool_definitions;
use crate::state::AppState;
use serde::Serialize;
use serde_json::{json, Value};
use std::io;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

const MAX_MCP_FRAME_BYTES: usize = 64 * 1024;
const MAX_TEXT_CHARS: usize = 16_384;
const MAX_INTERSECT_CONCEPTS: usize = 32;
const SUPPORTED_PROTOCOL_VERSION: &str = "2024-11-05";

enum Frame {
    Message(Vec<u8>),
    TooLarge,
}

/// Run the MCP server on stdio so MCP clients (such as Claude Desktop or
/// Claude Code) can use RAI memory as a set of tools.
pub async fn run_mcp_stdio(state: AppState, mutations_enabled: bool) {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);

    loop {
        let frame = match read_frame(&mut reader).await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(error) => {
                log::error!("MCP stdin read failed: {error}");
                break;
            }
        };

        let response = match frame {
            Frame::Message(bytes) if bytes.iter().all(u8::is_ascii_whitespace) => None,
            Frame::Message(bytes) => process_message(&state, &bytes, mutations_enabled).await,
            Frame::TooLarge => Some(JsonRpcResponse::error(
                Value::Null,
                -32600,
                format!("Request exceeds the {MAX_MCP_FRAME_BYTES}-byte limit"),
            )),
        };

        if let Some(response) = response {
            if let Err(error) = write_response(&mut stdout, &response).await {
                log::error!("MCP stdout write failed: {error}");
                break;
            }
        }
    }
}

async fn read_frame<R>(reader: &mut R) -> io::Result<Option<Frame>>
where
    R: AsyncBufRead + Unpin,
{
    let mut frame = Vec::new();
    let mut too_large = false;

    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return if frame.is_empty() && !too_large {
                Ok(None)
            } else if too_large {
                Ok(Some(Frame::TooLarge))
            } else {
                Ok(Some(Frame::Message(frame)))
            };
        }

        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(available.len());

        if !too_large {
            if frame.len().saturating_add(take) > MAX_MCP_FRAME_BYTES {
                frame.clear();
                too_large = true;
            } else {
                frame.extend_from_slice(&available[..take]);
            }
        }

        let consumed = take + usize::from(newline.is_some());
        reader.consume(consumed);

        if newline.is_some() {
            if frame.last() == Some(&b'\r') {
                frame.pop();
            }
            return Ok(Some(if too_large {
                Frame::TooLarge
            } else {
                Frame::Message(frame)
            }));
        }
    }
}

async fn write_response<W>(writer: &mut W, response: &JsonRpcResponse) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let output = serde_json::to_vec(response)
        .map_err(|error| io::Error::other(format!("serializing JSON-RPC response: {error}")))?;
    writer.write_all(&output).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

async fn process_message(
    state: &AppState,
    bytes: &[u8],
    mutations_enabled: bool,
) -> Option<JsonRpcResponse> {
    let value: Value = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(_) => {
            return Some(JsonRpcResponse::error(
                Value::Null,
                -32700,
                "Parse error".to_string(),
            ));
        }
    };
    let object = match value.as_object() {
        Some(object) => object,
        None => {
            return Some(JsonRpcResponse::error(
                Value::Null,
                -32600,
                "Invalid JSON-RPC request".to_string(),
            ));
        }
    };
    let has_id = object.contains_key("id");
    if object
        .get("id")
        .is_some_and(|id| !(id.is_null() || id.is_string() || id.is_number()))
        || object
            .get("params")
            .is_some_and(|params| !(params.is_object() || params.is_array()))
    {
        return Some(JsonRpcResponse::error(
            Value::Null,
            -32600,
            "Invalid JSON-RPC request".to_string(),
        ));
    }

    let request: JsonRpcRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(_) => {
            return Some(JsonRpcResponse::error(
                Value::Null,
                -32600,
                "Invalid JSON-RPC request".to_string(),
            ));
        }
    };

    handle_request(state, request, has_id, mutations_enabled).await
}

async fn handle_request(
    state: &AppState,
    request: JsonRpcRequest,
    has_id: bool,
    mutations_enabled: bool,
) -> Option<JsonRpcResponse> {
    // JSON-RPC notifications never receive a response, including unknown
    // notification methods and notifications/initialized. An explicit null
    // ID is still a request under JSON-RPC, not a notification.
    if !has_id {
        return None;
    }
    let id = request.id.unwrap_or(Value::Null);

    let response = match request.method.as_str() {
        "initialize" => JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": negotiate_protocol_version(&request.params),
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    }
                },
                "serverInfo": {
                    "name": "rai-server",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ),

        "tools/list" => {
            let tools = tool_definitions()
                .into_iter()
                .filter(|tool| mutations_enabled || tool.name != "rai_store")
                .collect::<Vec<_>>();
            JsonRpcResponse::success(id, json!({ "tools": tools }))
        }

        "tools/call" => {
            let tool_name = request
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            let result = handle_tool_call(state, tool_name, arguments, mutations_enabled).await;
            match serde_json::to_value(result) {
                Ok(value) => JsonRpcResponse::success(id, value),
                Err(error) => {
                    log::error!("serializing MCP tool result failed: {error}");
                    JsonRpcResponse::error(id, -32603, "Internal error".to_string())
                }
            }
        }

        _ => JsonRpcResponse::error(id, -32601, "Method not found".to_string()),
    };

    Some(response)
}

fn negotiate_protocol_version(params: &Value) -> &'static str {
    // RAI currently implements the 2024-11-05 core tool protocol. Returning
    // the supported version is the MCP fallback when a client proposes a
    // newer version; the client can then accept it or close the connection.
    let _client_version = params.get("protocolVersion").and_then(Value::as_str);
    SUPPORTED_PROTOCOL_VERSION
}

async fn handle_tool_call(
    state: &AppState,
    tool_name: &str,
    args: Value,
    mutations_enabled: bool,
) -> ToolCallResult {
    match tool_name {
        "rai_store" => {
            if !mutations_enabled {
                return ToolCallResult::error(
                    "rai_store is disabled; set RAI_MCP_MUTATIONS_ENABLED=true to opt in".into(),
                );
            }
            let content = match required_text(&args, "content") {
                Ok(content) => content,
                Err(error) => return error,
            };
            match state.store(content).await {
                Ok(report) => {
                    serialize_tool_result(&report, "Stored successfully.\n\nInterference report:\n")
                }
                Err(error) => internal_tool_error("store", &error),
            }
        }

        "rai_recall" => {
            let query = match required_text(&args, "query") {
                Ok(query) => query,
                Err(error) => return error,
            };
            match state.manager.recall(query).await {
                Ok(result) => serialize_tool_result(&result, ""),
                Err(error) => internal_tool_error("recall", &error),
            }
        }

        "rai_intersect" => {
            let concepts = match required_concepts(&args) {
                Ok(concepts) => concepts,
                Err(error) => return error,
            };
            match state.manager.intersect(&concepts).await {
                Ok(result) => serialize_tool_result(&result, ""),
                Err(error) => internal_tool_error("intersect", &error),
            }
        }

        "rai_contradict" => {
            let fact = match required_text(&args, "fact") {
                Ok(fact) => fact,
                Err(error) => return error,
            };
            match state.manager.check_contradiction(fact).await {
                Ok(report) => serialize_tool_result(&report, ""),
                Err(error) => internal_tool_error("contradiction check", &error),
            }
        }

        "rai_surprise" => {
            let content = match required_text(&args, "content") {
                Ok(content) => content,
                Err(error) => return error,
            };
            match state.manager.measure_surprise(content).await {
                Ok(result) => serialize_tool_result(&result, ""),
                Err(error) => internal_tool_error("surprise measurement", &error),
            }
        }

        "rai_explain_confidence" => {
            let query = match required_text(&args, "query") {
                Ok(query) => query,
                Err(error) => return error,
            };
            match state.manager.explain_confidence(query).await {
                Ok(result) => serialize_tool_result(&result, ""),
                Err(error) => internal_tool_error("confidence explanation", &error),
            }
        }

        "rai_memory_health" => match state.manager.health().await {
            Ok(report) => serialize_tool_result(&report, ""),
            Err(error) => internal_tool_error("memory health", &error),
        },

        _ => ToolCallResult::error("Unknown tool".to_string()),
    }
}

fn required_text<'a>(args: &'a Value, field: &str) -> Result<&'a str, ToolCallResult> {
    let Some(object) = args.as_object() else {
        return Err(ToolCallResult::error("Arguments must be an object".into()));
    };
    if object.len() != 1 || !object.contains_key(field) {
        return Err(ToolCallResult::error(format!(
            "Expected exactly the '{field}' parameter"
        )));
    }
    let Some(value) = object.get(field).and_then(Value::as_str) else {
        return Err(ToolCallResult::error(format!("'{field}' must be a string")));
    };
    if value.trim().is_empty() {
        return Err(ToolCallResult::error(format!(
            "'{field}' must not be empty"
        )));
    }
    if value.chars().count() > MAX_TEXT_CHARS {
        return Err(ToolCallResult::error(format!(
            "'{field}' exceeds the {MAX_TEXT_CHARS}-character limit"
        )));
    }
    Ok(value)
}

fn required_concepts(args: &Value) -> Result<Vec<String>, ToolCallResult> {
    let Some(object) = args.as_object() else {
        return Err(ToolCallResult::error("Arguments must be an object".into()));
    };
    if object.len() != 1 || !object.contains_key("concepts") {
        return Err(ToolCallResult::error(
            "Expected exactly the 'concepts' parameter".into(),
        ));
    }
    let Some(values) = object.get("concepts").and_then(Value::as_array) else {
        return Err(ToolCallResult::error("'concepts' must be an array".into()));
    };
    if !(2..=MAX_INTERSECT_CONCEPTS).contains(&values.len()) {
        return Err(ToolCallResult::error(format!(
            "'concepts' must contain 2..={MAX_INTERSECT_CONCEPTS} strings"
        )));
    }

    values
        .iter()
        .map(|value| {
            let Some(concept) = value.as_str() else {
                return Err(ToolCallResult::error(
                    "every concept must be a string".into(),
                ));
            };
            if concept.trim().is_empty() || concept.chars().count() > MAX_TEXT_CHARS {
                return Err(ToolCallResult::error(format!(
                    "each concept must contain 1..={MAX_TEXT_CHARS} characters"
                )));
            }
            Ok(concept.to_string())
        })
        .collect()
}

fn serialize_tool_result<T: Serialize>(value: &T, prefix: &str) -> ToolCallResult {
    match serde_json::to_string_pretty(value) {
        Ok(text) => ToolCallResult::text(format!("{prefix}{text}")),
        Err(error) => {
            log::error!("serializing MCP tool output failed: {error}");
            ToolCallResult::error("Operation succeeded but its result could not be encoded".into())
        }
    }
}

fn internal_tool_error(operation: &str, error: &impl std::fmt::Display) -> ToolCallResult {
    log::error!("MCP {operation} failed: {error}");
    ToolCallResult::error(format!("{operation} failed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rai_core::embedding::{EmbeddingBridge, MockEmbedder};
    use rai_core::MemoryManager;
    use std::sync::Arc;

    fn test_state() -> AppState {
        let embedder = Arc::new(MockEmbedder::new(16));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        AppState::new(
            Arc::new(MemoryManager::try_new(bridge).expect("valid test manager")),
            None,
        )
    }

    #[tokio::test]
    async fn bounded_reader_drains_an_oversized_frame_and_recovers() {
        let mut input = vec![b'x'; MAX_MCP_FRAME_BYTES + 1];
        input.extend_from_slice(b"\n{}\n");
        let mut reader = BufReader::new(input.as_slice());

        assert!(matches!(
            read_frame(&mut reader).await.unwrap(),
            Some(Frame::TooLarge)
        ));
        match read_frame(&mut reader).await.unwrap() {
            Some(Frame::Message(bytes)) => assert_eq!(bytes, b"{}"),
            _ => panic!("expected the next bounded frame"),
        }
    }

    #[tokio::test]
    async fn notifications_never_receive_a_response() {
        let response = process_message(
            &test_state(),
            br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            false,
        )
        .await;
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn an_explicit_null_id_still_receives_a_response() {
        let response = process_message(
            &test_state(),
            br#"{"jsonrpc":"2.0","id":null,"method":"tools/list"}"#,
            false,
        )
        .await
        .expect("an explicit null ID is a request");
        assert_eq!(response.id, Value::Null);
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn store_is_hidden_and_denied_without_explicit_opt_in() {
        let state = test_state();
        let list = process_message(
            &state,
            br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            false,
        )
        .await
        .unwrap();
        let tools = list.result.unwrap()["tools"].as_array().unwrap().clone();
        assert!(tools.iter().all(|tool| tool["name"] != "rai_store"));

        let call = process_message(
            &state,
            br#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rai_store","arguments":{"content":"fact"}}}"#,
            false,
        )
        .await
        .unwrap();
        assert_eq!(call.result.unwrap()["isError"], true);
    }

    #[tokio::test]
    async fn invalid_jsonrpc_is_sanitized() {
        let response = process_message(
            &test_state(),
            br#"{"jsonrpc":"1.0","id":1,"method":"tools/list"}"#,
            false,
        )
        .await
        .unwrap();
        let error = response.error.unwrap();
        assert_eq!(error.code, -32600);
        assert_eq!(error.message, "Invalid JSON-RPC request");
    }

    #[tokio::test]
    async fn invalid_id_and_params_shapes_are_rejected() {
        for request in [
            br#"{"jsonrpc":"2.0","id":{},"method":"tools/list"}"#.as_slice(),
            br#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":true}"#.as_slice(),
        ] {
            let response = process_message(&test_state(), request, false)
                .await
                .expect("invalid requests receive an error");
            assert_eq!(response.id, Value::Null);
            assert_eq!(response.error.expect("error response").code, -32600);
        }
    }
}
