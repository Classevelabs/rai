use crate::mcp::schema::ToolDefinition;
use serde_json::json;

/// Return the 7 MCP tool definitions.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "rai_store".to_string(),
            description: "Store a fact in RAI memory. Returns an interference report \
                          showing if the new fact contradicts existing memories."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or knowledge to store"
                    }
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: "rai_recall".to_string(),
            description: "Retrieve knowledge from RAI memory with energy-based confidence. \
                          Returns the most relevant memory along with a mathematical confidence \
                          score derived from the NRA energy landscape. Unlike RAG, this knows \
                          when it doesn't know — LOW confidence means the retrieval is unreliable."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to recall knowledge about"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "rai_intersect".to_string(),
            description: "Query at the intersection of multiple concepts using compositional \
                          omega addressing. This is unique to NRA — it combines address vectors \
                          to find knowledge that lives at the intersection of concepts."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of concepts to intersect (2 or more)"
                    }
                },
                "required": ["concepts"]
            }),
        },
        ToolDefinition {
            name: "rai_contradict".to_string(),
            description: "Check if a new fact contradicts existing memory by measuring \
                          energy landscape disturbance. Returns an interference report with \
                          severity levels: None, Minor, Major, or Critical."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to check for contradictions"
                    }
                },
                "required": ["fact"]
            }),
        },
        ToolDefinition {
            name: "rai_surprise".to_string(),
            description: "Measure the novelty/surprise of a fact using the REM prior \
                          residual norm. High surprise means the prior model couldn't predict \
                          this — it contains genuinely new information."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to measure surprise for"
                    }
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: "rai_explain_confidence".to_string(),
            description: "Explain why a retrieval has a particular confidence level. \
                          Provides energy landscape analysis, basin boundary detection, \
                          and attractor diagnostics."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to explain confidence for"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "rai_memory_health".to_string(),
            description: "Get system diagnostics: number of memories, NRA/REM MSE, \
                          prior quality, capacity utilization, and training status."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
    ]
}
