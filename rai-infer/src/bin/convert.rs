//! rai-convert: HuggingFace checkpoint -> `.raimodel`, without Python.
//!
//! Reads `.safetensors` directly and streams one row block at a time, so peak
//! memory stays near a single block instead of the whole model. Output is
//! byte-identical to `scripts/export_rtn.py` for the same inputs.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use rai_infer::convert::{convert, ConvertOptions};

#[derive(Parser, Debug)]
#[command(
    name = "rai-convert",
    version,
    about = "Convert a HuggingFace checkpoint to .raimodel (round-to-nearest 4-bit)"
)]
struct Args {
    /// HuggingFace checkpoint directory (config.json + .safetensors + tokenizer.json).
    #[arg(long)]
    model: PathBuf,
    /// Output file; defaults to <model-dir-name lowercased>-q4.raimodel.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Columns per quantization group for the 4-bit linears.
    #[arg(long, default_value_t = 128)]
    group_size: u32,
    /// Columns per quantization group for the 8-bit embedding.
    #[arg(long, default_value_t = 64)]
    embed_group_size: u32,
    /// Context length the model is built for (sizes the RoPE table).
    #[arg(long, default_value_t = 2048)]
    max_context: u32,
    /// Where to copy tokenizer.json; defaults to next to the output file.
    #[arg(long)]
    tokenizer_out: Option<PathBuf>,
    /// Suppress progress output.
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    convert(&ConvertOptions {
        model_dir: args.model,
        output: args.output,
        group_size: args.group_size,
        embed_group_size: args.embed_group_size,
        max_context: args.max_context,
        tokenizer_out: args.tokenizer_out,
        quiet: args.quiet,
    })?;
    Ok(())
}
