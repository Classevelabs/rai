//! `rai-generate` — deprecated alias for `rai run`.
//!
//! Kept so existing documentation, scripts and CI keep working. The flags are
//! unchanged: `--model` and `--tokenizer` are both required flags here, where
//! `rai run` takes the model as a positional and defaults the tokenizer to the
//! one beside it. Both call the same library entry point.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use rai_infer::cli::run::{GenerationArgs, RunArgs};

#[derive(Parser, Debug)]
#[command(
    name = "rai-generate",
    version,
    about = "Generate text from a .raimodel on the CPU",
    long_about = concat!(
        "Generate text from a .raimodel file using CPU-only 4-bit inference. ",
        "No GPU and no Python runtime. Convert a HuggingFace checkpoint first ",
        "with rai convert."
    ),
    after_help = concat!(
        "DEPRECATED: use `rai run <model.raimodel> --prompt \"...\"` instead. This\n",
        "binary is a wrapper kept for compatibility and will be removed in a\n",
        "future release; `rai run` also finds tokenizer.json beside the model,\n",
        "so it needs no --tokenizer.\n",
        "\n",
        "EXAMPLE:\n",
        "  rai-generate --model tinyllama-q4.raimodel --tokenizer tokenizer.json \\\n",
        "    --chat-template zephyr --prompt \"Explain photosynthesis.\" --max-tokens 80\n",
        "\n",
        "Instruction-tuned models need --chat-template. Without it they usually\n",
        "emit end-of-sequence immediately and print nothing."
    )
)]
struct Args {
    /// Path to the .raimodel file to run
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json, written beside the model at conversion time
    #[arg(long)]
    tokenizer: PathBuf,
    #[command(flatten)]
    generation: GenerationArgs,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rai_infer::cli::run::run(&RunArgs {
        model: args.model,
        tokenizer: Some(args.tokenizer),
        generation: args.generation,
    })
}
