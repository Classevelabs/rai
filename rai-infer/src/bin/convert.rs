//! `rai-convert` — deprecated alias for `rai convert`.
//!
//! Kept so existing documentation, scripts and CI keep working. The flags are
//! unchanged (`--model` stays a flag here; `rai convert` takes the directory as
//! a positional). Both call the same library entry point.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use rai_infer::cli::convert::{ConvertArgs, ConvertTuning};

#[derive(Parser, Debug)]
#[command(
    name = "rai-convert",
    version,
    about = "Convert a HuggingFace checkpoint to .raimodel (round-to-nearest 4-bit)",
    after_help = "DEPRECATED: use `rai convert <model-dir>` instead. This binary is a wrapper \
                  kept for compatibility and will be removed in a future release."
)]
struct Args {
    /// HuggingFace checkpoint directory (config.json + .safetensors + tokenizer.json).
    #[arg(long)]
    model: PathBuf,
    #[command(flatten)]
    tuning: ConvertTuning,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rai_infer::cli::convert::run(&ConvertArgs {
        model_dir: args.model,
        tuning: args.tuning,
    })
}
