//! `rai models` — list the `.raimodel` files in a directory.
//!
//! Reads each file's header only (see [`RaiModelFile::read_summary`]), so
//! listing a directory of 7B checkpoints costs a few hundred bytes of I/O
//! rather than the tens of gigabytes a full load would.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::{format_bytes, MODEL_EXTENSION, TOKENIZER_FILENAME};
use crate::format::{ModelSummary, RaiModelFile};

#[derive(clap::Args, Debug, Clone)]
pub struct ModelsArgs {
    /// Directory to scan (default: the current directory)
    #[arg(value_name = "DIR", default_value = ".")]
    pub dir: PathBuf,
}

/// One row of the listing: a readable file with an outcome attached.
struct Entry {
    path: PathBuf,
    file_bytes: u64,
    /// `Err` for a file that is named `.raimodel` but does not parse — listed
    /// rather than hidden, because a silently missing model is worse than a
    /// named broken one.
    summary: Result<ModelSummary>,
}

/// Collect every `*.raimodel` in `dir`, sorted by name.
fn collect(dir: &Path) -> Result<Vec<Entry>> {
    anyhow::ensure!(dir.is_dir(), "{} is not a directory", dir.display());
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry.with_context(|| format!("reading an entry of {}", dir.display()))?;
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
        let file_bytes = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        entries.push(Entry {
            summary: RaiModelFile::read_summary(&path),
            path,
            file_bytes,
        });
    }
    entries.sort_by(|left, right| left.path.file_name().cmp(&right.path.file_name()));
    Ok(entries)
}

/// The architecture line printed under a model's name.
fn architecture_line(summary: &ModelSummary) -> String {
    let config = &summary.config;
    format!(
        "hidden {}  layers {}  heads {}/{} kv  inter {}  vocab {}  ctx {}  {}-bit/g{}  {}  v{}",
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size,
        config.max_context,
        config.bits,
        config.group_size,
        if summary.tied_embeddings() {
            "tied lm_head"
        } else {
            "untied lm_head"
        },
        config.version,
    )
}

pub fn run(args: &ModelsArgs) -> Result<()> {
    let entries = collect(&args.dir)?;
    if entries.is_empty() {
        println!("No .raimodel files in {}.", args.dir.display());
        println!("\nConvert a HuggingFace checkpoint to make one:");
        println!("  rai convert <model-dir> -o my-model.raimodel");
        return Ok(());
    }

    let plural = if entries.len() == 1 { "" } else { "s" };
    println!(
        "{} model{plural} in {}\n",
        entries.len(),
        args.dir.display()
    );

    for entry in &entries {
        let name = entry
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| entry.path.display().to_string());
        match &entry.summary {
            Ok(summary) => {
                println!("{name}  ({})", format_bytes(summary.file_bytes));
                println!("  {}", architecture_line(summary));
                // A model whose tokenizer went missing runs nowhere, and the
                // failure surfaces much later; say so while the user is looking.
                let tokenizer = entry
                    .path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(TOKENIZER_FILENAME);
                if !tokenizer.is_file() {
                    println!("  no {TOKENIZER_FILENAME} beside it — pass --tokenizer to run it");
                }
            }
            Err(error) => {
                println!("{name}  ({})", format_bytes(entry.file_bytes));
                println!("  unreadable: {error:#}");
            }
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_with_no_models_is_not_an_error() {
        let dir = std::env::temp_dir().join(format!("rai-models-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();
        assert!(collect(&dir).unwrap().is_empty());
        assert!(run(&ModelsArgs { dir: dir.clone() }).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_model_is_listed_rather_than_hidden() {
        let dir = std::env::temp_dir().join(format!("rai-models-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.raimodel"), vec![0u8; 200]).unwrap();
        let entries = collect(&dir).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].summary.is_err());
        // Still a successful listing: the command reports the file's state.
        assert!(run(&ModelsArgs { dir: dir.clone() }).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        assert!(collect(Path::new("no-such-directory-here")).is_err());
    }
}
