//! `rai-chat` — deprecated alias for `rai serve`.
//!
//! Kept so existing documentation, scripts and CI keep working. The flags are
//! unchanged: `--model` and `--tokenizer` are both required flags here, where
//! `rai serve` takes the model as a positional and defaults the tokenizer to
//! the one beside it. Both call the same library entry point.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use rai_infer::cli::serve::{ServeArgs, ServeOptions};

#[derive(Parser, Debug)]
#[command(
    name = "rai-chat",
    version,
    about = "Chat with any .raimodel — edge inference with pondering",
    after_help = "DEPRECATED: use `rai serve <model.raimodel>` instead. This binary is a wrapper \
                  kept for compatibility and will be removed in a future release."
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[command(flatten)]
    options: ServeOptions,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rai_infer::cli::serve::run(&ServeArgs {
        model: args.model,
        tokenizer: Some(args.tokenizer),
        options: args.options,
    })
}
