use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// JSON-RPC request for MCP.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(rename = "jsonrpc", deserialize_with = "deserialize_jsonrpc_version")]
    pub _jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC response for MCP.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// MCP tool definition.
#[derive(Debug, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    pub annotations: ToolAnnotations,
}

/// Behavioral hints advertised to MCP clients. These are hints rather than an
/// authorization mechanism, but explicit values prevent clients from assuming unsafe defaults.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ToolAnnotations {
    #[serde(rename = "readOnlyHint")]
    pub read_only_hint: bool,
    #[serde(rename = "destructiveHint")]
    pub destructive_hint: bool,
    #[serde(rename = "idempotentHint")]
    pub idempotent_hint: bool,
    #[serde(rename = "openWorldHint")]
    pub open_world_hint: bool,
}

/// MCP tool result content.
#[derive(Debug, Serialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// MCP tool call result.
#[derive(Debug, Serialize)]
pub struct ToolCallResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

impl ToolCallResult {
    pub fn text(text: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }],
            is_error: None,
        }
    }

    pub fn error(text: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }],
            is_error: Some(true),
        }
    }
}

fn deserialize_jsonrpc_version<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let version = String::deserialize(deserializer)?;
    if version == "2.0" {
        Ok(version)
    } else {
        Err(de::Error::custom("jsonrpc must be exactly \"2.0\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn notification_without_id_deserializes() {
        let request: JsonRpcRequest = serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }))
        .unwrap();
        assert!(request.id.is_none());
        assert_eq!(request.method, "notifications/initialized");
    }

    #[test]
    fn invalid_jsonrpc_version_is_rejected() {
        let result = serde_json::from_value::<JsonRpcRequest>(json!({
            "jsonrpc": "1.0",
            "id": 1,
            "method": "tools/list"
        }));
        assert!(result.is_err());
    }

    #[test]
    fn request_id_is_preserved() {
        let request: JsonRpcRequest = serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "tools/list"
        }))
        .unwrap();
        assert_eq!(request.id, Some(json!("request-1")));
    }
}
