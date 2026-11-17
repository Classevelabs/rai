//! `rai check` — will `rai convert` take this checkpoint?
//!
//! Gives the converter's own verdict on a checkpoint without writing anything.
//! A `config.json` is enough, so `rai-get-model` asks before it downloads
//! gigabytes of weights; with the weights beside it, every tensor the
//! conversion would read is checked too.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::cli::format_bytes;
use crate::convert::{preflight, ConvertOptions, FOLLOW_MODEL_CONTEXT};

#[derive(clap::Args, Debug, Clone)]
pub struct CheckArgs {
    /// A checkpoint folder (its config.json, with or without the weights), or
    /// a config.json file
    #[arg(value_name = "PATH")]
    pub path: PathBuf,
    /// Context length to check for, in tokens (default: the model's own)
    #[arg(long)]
    pub max_context: Option<u32>,
}

fn has_weights(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
    })
}

pub fn run(args: &CheckArgs) -> Result<()> {
    let (config_path, dir) = if args.path.is_dir() {
        (args.path.join("config.json"), Some(args.path.as_path()))
    } else {
        (args.path.clone(), args.path.parent())
    };
    let raw = std::fs::read(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let hf: serde_json::Value = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    // Before a download only the configuration is there; the weights are
    // checked when they are.
    let weights = dir.filter(|dir| has_weights(dir));
    let defaults = ConvertOptions::default();
    let report = preflight(
        &hf,
        weights,
        defaults.group_size,
        defaults.embed_group_size,
        args.max_context.unwrap_or(FOLLOW_MODEL_CONTEXT),
    )?;

    let architecture = report
        .architectures
        .first()
        .cloned()
        .unwrap_or_else(|| report.model_type.clone());
    println!(
        "{architecture} ({}), {:.2} billion parameters",
        report.model_type,
        report.parameters as f64 / 1e9
    );
    if let (Some(tokens), Some(source)) = (report.max_context, report.context_source) {
        println!("Context: {tokens} tokens ({})", source.describe());
    }
    if let Some(container) = &report.container {
        println!("Converted size: {}", format_bytes(container.output_bytes));
    }
    if !report.supported {
        bail!("RAI cannot convert this model: {}", report.reason);
    }
    println!(
        "RAI can convert this model{}.",
        if report.weights_checked {
            ", and every tensor it needs is present"
        } else {
            " (its weights were not checked; they are not in this folder)"
        }
    );
    Ok(())
}
