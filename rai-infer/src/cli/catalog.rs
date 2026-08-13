//! What is on this machine, for `rai serve`'s JSON API.
//!
//! Two questions a model-manager UI has to answer before anything else:
//!
//! * *What can I run?* — [`list_directory`] describes every `.raimodel` in a
//!   directory from its **header alone** ([`RaiModelFile::read_summary`]), so
//!   pointing the UI at a folder of 8 GB files costs a few hundred bytes of
//!   I/O each rather than tens of gigabytes.
//! * *Would this checkpoint even convert?* — [`inspect`] runs the converter's
//!   own preflight ([`crate::convert::preflight`]) against a `config.json`,
//!   with no weights required, so a user learns a 14 GB download is
//!   unsupported before starting it.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::cli::{format_bytes, MODEL_EXTENSION, TOKENIZER_FILENAME};
use crate::convert::{preflight, PreflightReport};
use crate::format::{ModelSummary, RaiModelFile};
use crate::layers::{Activation, RopeScaling};

/// Name of the shard index a sharded checkpoint carries.
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
const SAFETENSORS_SINGLE: &str = "model.safetensors";

/// Every `*.raimodel` in `dir`, described from its header.
///
/// The `Err` case is the *directory* being unreadable. A single unreadable
/// model is reported as an entry with `readable: false` and its error, never
/// hidden: a model that silently vanishes from a picker is worse than one the
/// user can see is broken.
pub fn list_directory(dir: &Path, loaded: Option<&Path>) -> std::io::Result<Vec<Value>> {
    let mut entries: Vec<(PathBuf, u64)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_none_or(|ext| !ext.eq_ignore_ascii_case(MODEL_EXTENSION))
        {
            continue;
        }
        let size = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        entries.push((path, size));
    }
    entries.sort_by(|left, right| left.0.file_name().cmp(&right.0.file_name()));

    Ok(entries
        .into_iter()
        .map(|(path, size)| describe_model(&path, size, loaded))
        .collect())
}

fn describe_model(path: &Path, size_bytes: u64, loaded: Option<&Path>) -> Value {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let tokenizer = dir.join(TOKENIZER_FILENAME);
    let is_loaded = loaded.is_some_and(|other| same_file(path, other));

    let mut entry = json!({
        "name": name,
        "path": path.display().to_string(),
        "dir": dir.display().to_string(),
        "size_bytes": size_bytes,
        "size_human": format_bytes(size_bytes),
        "loaded": is_loaded,
        "tokenizer_present": tokenizer.is_file(),
        "tokenizer_path": tokenizer.display().to_string(),
    });

    // Header only: never `open()`, which would map and validate the whole file.
    match RaiModelFile::read_summary(path) {
        Ok(summary) => {
            entry["readable"] = json!(true);
            entry["error"] = Value::Null;
            entry["header"] = header_json(&summary);
        }
        Err(error) => {
            entry["readable"] = json!(false);
            entry["error"] = json!(format!("{error:#}"));
            entry["header"] = Value::Null;
        }
    }
    entry
}

/// The header summary, with the v2 fields always present.
///
/// A v1 file has no bytes for `activation`, `rope`, biases or `embed_scale`;
/// the reader materializes the v1 defaults for them, and so does this, so a UI
/// never has to branch on `format_version` to render a row. `format_version`
/// is there for the cases where the distinction matters.
pub fn header_json(summary: &ModelSummary) -> Value {
    let config = &summary.config;
    json!({
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_context": config.max_context,
        "rope_theta": config.rope_theta,
        "norm_eps": config.norm_eps,
        "bits": config.bits,
        "group_size": config.group_size,
        "embed_bits": config.embed_bits,
        "embed_group_size": config.embed_group_size,
        "format_version": config.version,
        "num_sections": summary.num_sections,
        "tied_lm_head": summary.tied_embeddings(),
        "activation": activation_name(config.activation),
        "rope_type": rope_type_name(config.rope_scaling),
        "rope_scaling": rope_scaling_json(config.rope_scaling),
        "has_biases": config.bias_mask != 0,
        "bias_mask": config.bias_mask,
        "biased_projections": biased_projections(config.bias_mask),
        "embed_scale": config.embed_scale,
    })
}

pub fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::Silu => "silu",
        Activation::GeluTanh => "gelu_tanh",
    }
}

pub fn rope_type_name(scaling: RopeScaling) -> &'static str {
    match scaling {
        RopeScaling::None => "none",
        RopeScaling::Llama3 { .. } => "llama3",
    }
}

pub fn rope_scaling_json(scaling: RopeScaling) -> Value {
    match scaling {
        RopeScaling::None => json!({ "type": "none" }),
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } => json!({
            "type": "llama3",
            "factor": factor,
            "low_freq_factor": low_freq_factor,
            "high_freq_factor": high_freq_factor,
            "original_max_position": original_max_position,
        }),
    }
}

pub fn biased_projections(bias_mask: u8) -> Vec<&'static str> {
    crate::format::PROJECTION_NAMES
        .iter()
        .enumerate()
        .filter(|(index, _)| bias_mask & (1 << index) != 0)
        .map(|(_, name)| *name)
        .collect()
}

/// Two paths that name the same file, as far as the filesystem will say.
fn same_file(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

/// What kind of thing `source` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// A local checkpoint directory, weights included.
    LocalCheckpoint,
    /// A local directory (or a `config.json`) with no weights beside it.
    LocalConfigOnly,
    /// Looks like a HuggingFace repo id, e.g. `Qwen/Qwen2.5-0.5B-Instruct`.
    HuggingFaceId,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::LocalCheckpoint => "local-checkpoint",
            SourceKind::LocalConfigOnly => "local-config-only",
            SourceKind::HuggingFaceId => "huggingface-id",
        }
    }
}

/// Classify an `/api/inspect` source without touching the network.
pub fn classify(source: &str) -> SourceKind {
    let path = Path::new(source);
    if path.is_dir() {
        if has_weights(path) {
            return SourceKind::LocalCheckpoint;
        }
        return SourceKind::LocalConfigOnly;
    }
    if path.is_file() {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        if has_weights(dir) {
            return SourceKind::LocalCheckpoint;
        }
        return SourceKind::LocalConfigOnly;
    }
    SourceKind::HuggingFaceId
}

pub fn has_weights(dir: &Path) -> bool {
    dir.join(SAFETENSORS_INDEX).is_file() || dir.join(SAFETENSORS_SINGLE).is_file()
}

/// Where a source's `config.json` and (optionally) its weights live.
pub struct ResolvedSource {
    pub config_path: PathBuf,
    /// `None` when there are no weights to check against.
    pub weights_dir: Option<PathBuf>,
}

/// Resolve a local source to its `config.json`.
pub fn resolve_local(source: &str) -> Result<ResolvedSource, String> {
    let path = Path::new(source);
    let (config_path, dir) = if path.is_dir() {
        (path.join("config.json"), path.to_path_buf())
    } else if path.is_file() {
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        (path.to_path_buf(), dir)
    } else {
        return Err(format!("{source} does not exist on this machine"));
    };
    if !config_path.is_file() {
        return Err(format!(
            "{} has no config.json; a HuggingFace checkpoint directory always does",
            dir.display()
        ));
    }
    Ok(ResolvedSource {
        weights_dir: has_weights(&dir).then_some(dir),
        config_path,
    })
}

/// Run the converter's preflight over a parsed `config.json` and render it.
pub fn inspect(
    source: &str,
    kind: SourceKind,
    hf: &Value,
    weights_dir: Option<&Path>,
    group_size: u32,
    embed_group_size: u32,
    max_context: u32,
) -> Value {
    match preflight(hf, weights_dir, group_size, embed_group_size, max_context) {
        Ok(report) => report_json(source, kind, &report, max_context),
        // The config parsed as JSON but is not a transformer config — no shape
        // can be reported, and the answer is still a definite "no".
        Err(error) => json!({
            "source": source,
            "kind": kind.as_str(),
            "supported": false,
            "reason": format!("{error:#}"),
            "weights_checked": weights_dir.is_some(),
            "model_type": hf.get("model_type").and_then(Value::as_str),
            "architectures": hf.get("architectures").cloned().unwrap_or(Value::Null),
            "shape": Value::Null,
            "container": Value::Null,
        }),
    }
}

fn report_json(
    source: &str,
    kind: SourceKind,
    report: &PreflightReport,
    max_context: u32,
) -> Value {
    json!({
        "source": source,
        "kind": kind.as_str(),
        "supported": report.supported,
        "reason": report.reason,
        "weights_checked": report.weights_checked,
        "model_type": report.model_type,
        "architectures": report.architectures,
        "checked_with": {
            "max_context": max_context,
        },
        "shape": {
            "hidden_size": report.hidden_size,
            "num_layers": report.num_layers,
            "num_heads": report.num_heads,
            "num_kv_heads": report.num_kv_heads,
            "head_dim": report.head_dim,
            "attention_dim": report.num_heads as u64 * report.head_dim as u64,
            "kv_dim": report.num_kv_heads as u64 * report.head_dim as u64,
            "intermediate_size": report.intermediate_size,
            "vocab_size": report.vocab_size,
            "max_position_embeddings": report.max_position_embeddings,
            "sliding_window": report.sliding_window,
            "rope_theta": report.rope_theta,
            "norm_eps": report.norm_eps,
            "tied_embeddings": report.tied_embeddings,
            "parameters": report.parameters,
            "parameters_human": human_parameters(report.parameters),
        },
        "container": report.container.as_ref().map(|container| json!({
            "format_version": container.version,
            "activation": activation_name(container.activation),
            "rope_type": rope_type_name(container.rope_scaling),
            "rope_scaling": rope_scaling_json(container.rope_scaling),
            "bias_mask": container.bias_mask,
            "has_biases": container.bias_mask != 0,
            "biased_projections": biased_projections(container.bias_mask),
            "embed_scale": container.embed_scale,
            "tied_lm_head": container.tied_embeddings,
            "num_sections": container.num_sections,
            "output_bytes": container.output_bytes,
            "output_human": format_bytes(container.output_bytes),
        })),
    })
}

/// `494.0M`, `1.1B` — the unit a model is actually named after.
fn human_parameters(parameters: u64) -> String {
    let value = parameters as f64;
    if value >= 1e9 {
        format!("{:.1}B", value / 1e9)
    } else if value >= 1e6 {
        format!("{:.1}M", value / 1e6)
    } else {
        format!("{parameters}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rai-catalog-{label}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_corrupt_model_is_listed_with_its_error_rather_than_dropped() {
        let dir = scratch("broken");
        std::fs::write(dir.join("broken.raimodel"), vec![0u8; 200]).unwrap();
        std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();

        let entries = list_directory(&dir, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "broken.raimodel");
        assert_eq!(entries[0]["readable"], false);
        assert_eq!(entries[0]["size_bytes"], 200);
        assert!(entries[0]["error"].is_string());
        assert!(entries[0]["header"].is_null());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unsupported_architecture_is_refused_from_config_alone() {
        // This example has been rewritten twice as the container grew:
        // Gemma2 gained softcapping and sandwich norms, then Gemma3 gained
        // per-layer RoPE bases. What is left is a refusal no capability bit
        // can lift, because it is about the arithmetic of a scheme the
        // container does not implement rather than a weight it cannot hold.
        let dir = scratch("yarn-rope");
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0},
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

        assert_eq!(classify(dir.to_str().unwrap()), SourceKind::LocalConfigOnly);
        let resolved = resolve_local(dir.to_str().unwrap()).unwrap();
        assert!(resolved.weights_dir.is_none());

        let value = inspect(
            dir.to_str().unwrap(),
            SourceKind::LocalConfigOnly,
            &config,
            None,
            128,
            64,
            2048,
        );
        assert_eq!(value["supported"], false);
        let reason = value["reason"].as_str().unwrap();
        assert!(reason.contains("yarn"), "{reason}");
        assert!(value["container"].is_null());
        // The shape is still reported: the user wants to know what it *is*.
        assert_eq!(value["shape"]["num_layers"], 16);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_supported_architecture_reports_the_container_it_would_produce() {
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
        });
        let value = inspect(
            "x",
            SourceKind::LocalConfigOnly,
            &config,
            None,
            128,
            64,
            2048,
        );
        assert_eq!(value["supported"], true);
        assert_eq!(value["container"]["format_version"], 1);
        assert_eq!(value["container"]["activation"], "silu");
        assert_eq!(value["container"]["tied_lm_head"], false);
        assert!(value["container"]["output_bytes"].as_u64().unwrap() > 0);
        assert_eq!(value["shape"]["parameters_human"], "1.1B");
        // Nothing was checked against weights, and the report says so.
        assert_eq!(value["weights_checked"], false);
    }

    #[test]
    fn a_config_that_is_not_a_transformer_is_refused_without_a_shape() {
        let config = serde_json::json!({"model_type": "bert", "hidden_size": 768});
        let value = inspect(
            "x",
            SourceKind::LocalConfigOnly,
            &config,
            None,
            128,
            64,
            2048,
        );
        assert_eq!(value["supported"], false);
        assert!(value["shape"].is_null());
        assert!(value["reason"]
            .as_str()
            .unwrap()
            .contains("num_hidden_layers"));
    }

    #[test]
    fn a_repo_id_is_not_mistaken_for_a_path() {
        assert_eq!(
            classify("Qwen/Qwen2.5-0.5B-Instruct"),
            SourceKind::HuggingFaceId
        );
        assert!(resolve_local("Qwen/Qwen2.5-0.5B-Instruct").is_err());
    }
}
