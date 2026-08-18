use crate::mcp::schema::{ToolAnnotations, ToolDefinition};
use crate::validate::{MAX_INTERSECTION_CONCEPTS, MAX_TEXT_BYTES};
use serde_json::json;

/// Return the 8 MCP tool definitions.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "rai_store".to_string(),
            description: "Store a fact in RAI memory. Returns a report of how crowded the \
                          stored address space became; it does not check the fact against \
                          existing memories for contradiction."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The fact or knowledge to store"
                    }
                },
                "required": ["content"],
                "additionalProperties": false
            }),
            annotations: mutating_annotations(),
        },
        ToolDefinition {
            name: "rai_forget".to_string(),
            description: "Remove a stored memory whose text exactly matches `content`. \
                          Returns whether anything was removed. Exact match only — this \
                          tool never guesses at a nearby memory."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The exact stored text to remove"
                    }
                },
                "required": ["content"],
                "additionalProperties": false
            }),
            annotations: forgetting_annotations(),
        },
        ToolDefinition {
            name: "rai_recall".to_string(),
            description: "Retrieve the stored memory whose address has the highest cosine \
                          similarity to the query. Confidence labels are experimental score \
                          tiers, not calibrated probabilities."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The query to recall knowledge about"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
        ToolDefinition {
            name: "rai_intersect".to_string(),
            description: "Experimental concept composition: normalize the average of the \
                          concept address vectors, then return the nearest stored memory."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_TEXT_BYTES
                        },
                        "minItems": 2,
                        "maxItems": MAX_INTERSECTION_CONCEPTS,
                        "description": "List of concepts to intersect (2 or more)"
                    }
                },
                "required": ["concepts"],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
        ToolDefinition {
            name: "rai_contradict".to_string(),
            description: "Report which stored memories the candidate fact would crowd — \
                          existing items whose addresses it lands close enough to that recall \
                          could confuse them. This measures address-space geometry only, not \
                          meaning: two facts can contradict from far-apart addresses. Do not \
                          treat an empty report as evidence that the fact is consistent with \
                          memory."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The fact to check for contradictions"
                    }
                },
                "required": ["fact"],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
        ToolDefinition {
            name: "rai_surprise".to_string(),
            description: "Return the residual between a candidate value and the value of the \
                          nearest stored key. The score is a heuristic, not a calibrated \
                          novelty measurement."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The content to measure surprise for"
                    }
                },
                "required": ["content"],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
        ToolDefinition {
            name: "rai_explain_confidence".to_string(),
            description: "Return the retrieval score for a query and the tier it falls in. \
                          These values are heuristic and should not be treated as calibrated \
                          confidence."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES,
                        "description": "The query to explain confidence for"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
        ToolDefinition {
            name: "rai_memory_health".to_string(),
            description: "Get system diagnostics: number of stored memories, mean residual \
                          norm, and capacity utilization."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            annotations: read_only_annotations(),
        },
    ]
}

fn mutating_annotations() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: false,
        destructive_hint: false,
        idempotent_hint: false,
        open_world_hint: false,
    }
}

/// Forgetting deletes data, and forgetting the same text twice removes it
/// once: destructive and idempotent, unlike a store.
fn forgetting_annotations() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: false,
        destructive_hint: true,
        idempotent_hint: true,
        open_world_hint: false,
    }
}

fn read_only_annotations() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: true,
        destructive_hint: false,
        idempotent_hint: true,
        open_world_hint: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_advertise_explicit_safety_annotations() {
        let tools = tool_definitions();
        let store = tools.iter().find(|tool| tool.name == "rai_store").unwrap();
        assert!(!store.annotations.read_only_hint);
        assert!(!store.annotations.idempotent_hint);

        let forget = tools.iter().find(|tool| tool.name == "rai_forget").unwrap();
        assert!(!forget.annotations.read_only_hint);
        assert!(forget.annotations.destructive_hint);
        assert!(forget.annotations.idempotent_hint);

        for tool in tools
            .iter()
            .filter(|tool| tool.name != "rai_store" && tool.name != "rai_forget")
        {
            assert!(tool.annotations.read_only_hint, "{}", tool.name);
            assert!(tool.annotations.idempotent_hint, "{}", tool.name);
            assert!(!tool.annotations.destructive_hint, "{}", tool.name);
        }
    }

    #[test]
    fn tool_schemas_reject_unknown_properties_and_bound_arrays() {
        let tools = tool_definitions();
        for tool in &tools {
            assert_eq!(
                tool.input_schema["additionalProperties"], false,
                "{}",
                tool.name
            );
        }

        let intersect = tools
            .iter()
            .find(|tool| tool.name == "rai_intersect")
            .unwrap();
        assert_eq!(
            intersect.input_schema["properties"]["concepts"]["maxItems"],
            32
        );
        assert_eq!(
            intersect.input_schema["properties"]["concepts"]["minItems"],
            2
        );
    }

    #[test]
    fn annotations_use_mcp_camel_case_fields() {
        let value = serde_json::to_value(tool_definitions()).unwrap();
        let annotations = &value[0]["annotations"];
        assert!(annotations.get("readOnlyHint").is_some());
        assert!(annotations.get("destructiveHint").is_some());
        assert!(annotations.get("idempotentHint").is_some());
        assert!(annotations.get("openWorldHint").is_some());
    }
}
