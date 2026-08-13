//! `rai convert` — HuggingFace checkpoint to `.raimodel`, without Python.
//!
//! The conversion itself lives in [`crate::convert`]; this is only the
//! command-line shape around it.

use std::path::PathBuf;

use anyhow::Result;

use crate::cli::format_bytes;
use crate::convert::{convert, ConvertOptions};

/// The tuning knobs, shared with the deprecated `rai-convert` binary.
#[derive(clap::Args, Debug, Clone)]
pub struct ConvertTuning {
    /// Output file; defaults to <model-dir-name>-q4.raimodel
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Columns per quantization group for the 4-bit linears
    #[arg(long, default_value_t = 128, value_name = "N")]
    pub group_size: u32,
    /// Columns per quantization group for the 8-bit embedding
    #[arg(long, default_value_t = 64, value_name = "N")]
    pub embed_group_size: u32,
    /// Context length the model is built for (sizes the RoPE table)
    #[arg(long, default_value_t = 2048, value_name = "TOKENS")]
    pub max_context: u32,
    /// Where to copy tokenizer.json; defaults to next to the output file
    #[arg(long, value_name = "FILE")]
    pub tokenizer_out: Option<PathBuf>,
    /// Suppress progress output
    #[arg(long, default_value_t = false)]
    pub quiet: bool,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ConvertArgs {
    /// HuggingFace checkpoint directory (config.json + .safetensors + tokenizer.json)
    #[arg(value_name = "MODEL_DIR")]
    pub model_dir: PathBuf,
    #[command(flatten)]
    pub tuning: ConvertTuning,
}

impl ConvertArgs {
    fn options(&self) -> ConvertOptions {
        ConvertOptions {
            model_dir: self.model_dir.clone(),
            output: self.tuning.output.clone(),
            group_size: self.tuning.group_size,
            embed_group_size: self.tuning.embed_group_size,
            max_context: self.tuning.max_context,
            tokenizer_out: self.tuning.tokenizer_out.clone(),
            quiet: self.tuning.quiet,
        }
    }
}

pub fn run(args: &ConvertArgs) -> Result<()> {
    // `convert` already narrates its own progress and final byte count; the
    // only thing missing at the end of a conversion is what to do next.
    let summary = convert(&args.options())?;
    if !args.tuning.quiet {
        println!(
            "\n{} is ready ({}). Run it with:\n  rai run {} --prompt \"...\"",
            summary.output_path.display(),
            format_bytes(summary.bytes_written),
            summary.output_path.display()
        );
    }
    Ok(())
}
