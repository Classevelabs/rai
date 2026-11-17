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
        "  rai-get-model qwen2.5-1.5b          # download and convert a starter model\n",
        "  rai run model.raimodel --prompt \"What is the capital of France?\"\n",
        "  rai serve model.raimodel            # chat in your browser at http://localhost:8090\n",
        "  rai check ./SomeModel               # will this checkpoint convert?\n",
        "  rai convert ./SomeModel -o some-model.raimodel\n",
        "  rai models                          # the models here and in ~/RAI/models\n",
        "\n",
        "The tokenizer and the model's chat template (<name>.chat.json) are\n",
        "written beside the model when it is converted and found automatically.\n",
        "`rai run` and `rai serve` use the model's own chat template; pass\n",
        "--chat-template none to send a prompt as plain text."
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
        after_help = "EXAMPLE:\n  rai convert ./TinyLlama-1.1B-Chat -o tinyllama.raimodel\n\nReads .safetensors directly — no Python, no GPU. Writes tokenizer.json and\nthe model's chat template (<name>.chat.json) beside the output so `rai run`\nand `rai serve` find them without a flag."
    )]
    Convert(cli::convert::ConvertArgs),

    /// Generate text from a .raimodel
    #[command(
        after_help = "EXAMPLE:\n  rai run model.raimodel --prompt \"Explain photosynthesis in two sentences.\"\n  rai run base-model.raimodel --chat-template none --prompt \"Once upon a time\"\n\n--tokenizer defaults to tokenizer.json beside the model. The model's own\nchat template, recorded when it was converted, is used unless you pass\n--chat-template; a model without one gets the prompt as plain text."
    )]
    Run(cli::run::RunArgs),

    /// Serve a .raimodel over a local web chat UI
    #[command(
        after_help = "EXAMPLE:\n  rai serve tinyllama.raimodel --port 8090\n\nBinds loopback only (127.0.0.1, or [::1] on a machine without IPv4
loopback) and rejects requests from other hosts."
    )]
    Serve(cli::serve::ServeArgs),

    /// Check whether a Hugging Face checkpoint can be converted
    #[command(
        after_help = "EXAMPLE:\n  rai check ./Qwen2.5-1.5B-Instruct\n  rai check ./downloads/config.json\n\nOnly config.json is needed, so you can check a model before downloading its\nweights. With the weights in the folder, every tensor is checked as well.\nExits 0 when the model converts, 1 when it does not, with the reason."
    )]
    Check(cli::check::CheckArgs),

    /// List the models in a folder (default: here and ~/RAI/models)
    #[command(after_help = "EXAMPLE:\n  rai models\n  rai models ~/models")]
    Models(cli::models::ModelsArgs),

    /// Measure perplexity on a text file — what quantization cost this model
    #[command(
        after_help = "EXAMPLE:\n  rai perplexity qwen2.5-1.5b-q4.raimodel --text wiki.test.raw\n\nSame method as llama-perplexity: non-overlapping chunks of --context\ntokens with the second half of each scored, so the two numbers are\ndirectly comparable. Compare a model against its own fp16 reference on\nthe same text and tokenizer; perplexity across different tokenizers is\nnot a comparison."
    )]
    Perplexity(cli::perplexity::PerplexityArgs),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Convert(args) => cli::convert::run(&args),
        Command::Run(args) => cli::run::run(&args),
        Command::Serve(args) => cli::serve::run(&args),
        Command::Check(args) => cli::check::run(&args),
        Command::Models(args) => cli::models::run(&args),
        Command::Perplexity(args) => cli::perplexity::run(&args),
    }
}
