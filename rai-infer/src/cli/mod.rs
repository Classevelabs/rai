//! The `rai` command-line surface.
//!
//! One binary, four verbs: `convert`, `run`, `serve`, `models`. Every verb's
//! implementation lives here rather than in `src/bin/`, for two reasons:
//!
//! * a single binary can host all of them, so users learn one command name
//!   instead of four; and
//! * argument validation and the model/tokenizer resolution rules become unit
//!   testable, which they are not inside a `fn main`.
//!
//! The pre-`rai` binaries (`rai-convert`, `rai-generate`, `rai-chat`) are kept
//! as thin wrappers over these same entry points so documentation and scripts
//! that name them keep working.
//!
//! Gated on the `cli` feature: `--no-default-features` still builds the lean
//! inference library with none of this compiled in.

pub mod convert;
pub mod models;
pub mod run;
pub mod serve;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Extension every model file carries, without the dot.
pub const MODEL_EXTENSION: &str = "raimodel";

/// The tokenizer file `rai convert` writes beside every model it produces.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Resolve the tokenizer for `model`, defaulting to `tokenizer.json` beside it.
///
/// Conversion writes the tokenizer next to the model, so the default is right
/// for anything this tool produced; passing `--tokenizer` remains the escape
/// hatch for a model that was moved away from its tokenizer.
pub fn resolve_tokenizer(model: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        anyhow::ensure!(path.is_file(), "tokenizer not found: {}", path.display());
        return Ok(path.to_path_buf());
    }

    let beside = model
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(TOKENIZER_FILENAME);
    if beside.is_file() {
        return Ok(beside);
    }
    anyhow::bail!(
        "no tokenizer found at {}.\n\
         `rai convert` writes {TOKENIZER_FILENAME} beside the model it produces; if this model \
         was moved, point at its tokenizer with --tokenizer <path>.",
        beside.display()
    )
}

/// Human-readable byte count, e.g. `591.0 MB`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Load a tokenizer, mapping the crate's boxed error into `anyhow`.
pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .with_context(|| format!("loading tokenizer {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_formatting_switches_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(619_538_088), "590.8 MB");
        assert_eq!(format_bytes(3_917_728_120), "3.6 GB");
    }

    #[test]
    fn tokenizer_defaults_to_the_file_beside_the_model() {
        let dir = std::env::temp_dir().join(format!("rai-cli-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.raimodel");
        std::fs::write(&model, b"x").unwrap();

        // Nothing beside it yet: the error must say where it looked and what
        // to do, not just "not found".
        let error = resolve_tokenizer(&model, None).unwrap_err().to_string();
        assert!(error.contains("tokenizer.json"), "{error}");
        assert!(error.contains("--tokenizer"), "{error}");

        let beside = dir.join(TOKENIZER_FILENAME);
        std::fs::write(&beside, b"{}").unwrap();
        assert_eq!(resolve_tokenizer(&model, None).unwrap(), beside);

        // An explicit path that does not exist is an error, not a silent
        // fallback to the neighbour file.
        assert!(resolve_tokenizer(&model, Some(Path::new("nope.json"))).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
