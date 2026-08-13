//! `rai` — the single entry point to the RAI engine.
//!
//! Parse, dispatch, done. Every subcommand's implementation lives in
//! `rai_infer::cli`.

use anyhow::Result;
use clap::{Parser, Subcommand};

use rai_infer::cli;

#[derive(Parser, Debug)]
#[command(
    name = "rai",
    version,
    about = "Run large language models on the CPU — no GPU, no Python",
    long_about = concat!(
        "rai runs 4-bit quantized language models on an ordinary CPU.\n",
        "Convert a HuggingFace checkpoint once, then run or serve the\n",
        ".raimodel file it produces. No GPU and no Python runtime."
    ),
    after_help = concat!(
        "EXAMPLE:\n",
        "  rai convert ./TinyLlama-1.1B-Chat -o tinyllama.raimodel\n",
        "  rai run tinyllama.raimodel --prompt \"The capital of France is\"\n",
        "  rai serve tinyllama.raimodel        # chat UI on http://localhost:8090\n",
        "  rai models                          # what is in this directory\n",
        "\n",
        "The tokenizer is written beside the model at conversion time and is\n",
        "picked up automatically; --tokenizer overrides it.\n",
        "\n",
        "Instruction-tuned models need a matching --chat-template. Without one\n",
        "they usually emit end-of-sequence immediately and print nothing."
    ),
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert a HuggingFace checkpoint to a .raimodel file
    #[command(
        after_help = "EXAMPLE:\n  rai convert ./TinyLlama-1.1B-Chat -o tinyllama.raimodel\n\nReads .safetensors directly — no Python, no GPU. Writes tokenizer.json\nbeside the output so `rai run` and `rai serve` find it without a flag."
    )]
    Convert(cli::convert::ConvertArgs),

    /// Generate text from a .raimodel
    #[command(
        after_help = "EXAMPLE:\n  rai run tinyllama.raimodel --prompt \"The capital of France is\" --max-tokens 40\n  rai run zephyr-7b.raimodel --chat-template zephyr --prompt \"Explain photosynthesis.\"\n\n--tokenizer defaults to tokenizer.json beside the model.\nInstruction-tuned models need the matching --chat-template; without it\nthey usually emit end-of-sequence immediately and print nothing."
    )]
    Run(cli::run::RunArgs),

    /// Serve a .raimodel over a local web chat UI
    #[command(
        after_help = "EXAMPLE:\n  rai serve tinyllama.raimodel --port 8090\n\nBinds 127.0.0.1 only and rejects requests from other hosts."
    )]
    Serve(cli::serve::ServeArgs),

    /// List the .raimodel files in a directory
    #[command(after_help = "EXAMPLE:\n  rai models\n  rai models ~/models")]
    Models(cli::models::ModelsArgs),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Convert(args) => cli::convert::run(&args),
        Command::Run(args) => cli::run::run(&args),
        Command::Serve(args) => cli::serve::run(&args),
        Command::Models(args) => cli::models::run(&args),
    }
}
