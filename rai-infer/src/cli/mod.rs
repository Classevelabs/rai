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

pub mod catalog;
pub mod convert;
pub mod jobs;
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

/// How far a model's vocabulary may exceed its tokenizer's before the two are
/// treated as a mismatched pair rather than a padded one.
///
/// Checkpoints routinely round the embedding matrix up to a multiple of 64 or
/// 128 — Qwen2.5 is 151643/151936, Llama-3 is 128000/128256 — so a small
/// shortfall is normal. A large one is two different models' files sitting in
/// the same directory.
const MAX_VOCAB_PADDING: usize = 1024;

/// Load a tokenizer and refuse it if it does not belong to this model.
///
/// `resolve_tokenizer` finds a file by name, which says nothing about whether
/// it was produced by the same conversion. Pairing a model with a stranger's
/// tokenizer does not fail: every id decodes to *some* piece, so generation
/// runs to completion and returns fluent nonsense with a zero exit status.
/// Comparing vocabularies is the check that turns that into a refusal, and it
/// costs one integer compare.
pub fn load_tokenizer_for_model(path: &Path, model_vocab: usize) -> Result<tokenizers::Tokenizer> {
    let tokenizer = load_tokenizer(path)?;
    let tokenizer_vocab = tokenizer.get_vocab_size(true);

    anyhow::ensure!(
        tokenizer_vocab <= model_vocab,
        "{} has a {tokenizer_vocab}-token vocabulary but this model's is {model_vocab}, so it \
         can emit ids the model has no embedding for. It belongs to a different model — pass \
         the right one with --tokenizer <path>.",
        path.display()
    );
    anyhow::ensure!(
        model_vocab - tokenizer_vocab <= MAX_VOCAB_PADDING,
        "{} has a {tokenizer_vocab}-token vocabulary and this model's is {model_vocab}. That gap \
         is too large to be embedding padding, so this is another model's tokenizer; using it \
         would produce fluent nonsense rather than an error. Pass the matching one with \
         --tokenizer <path>.",
        path.display()
    );

    Ok(tokenizer)
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
