# Installation

RAI is distributed as a Rust workspace. The public GitHub repository is the
source of truth for the initial release.

## Requirements

- Stable Rust toolchain.
- x86-64 CPU with AVX2, FMA, and F16C for optimized inference.
- Python only when exporting HuggingFace models into `.raimodel`.

The repository builds with `target-cpu=native` through `.cargo/config.toml`.
This gives best local performance. Remove that flag before building portable
binary artifacts for machines with different CPU capabilities.

## Install From GitHub

Install inference binaries:

```bash
cargo install --git https://github.com/classeve-public/rai \
  --package classeve-rai-infer \
  --locked
```

Install the REST/MCP server:

```bash
cargo install --git https://github.com/classeve-public/rai \
  --package classeve-rai-server \
  --locked
```

## Install From A Local Checkout

```bash
git clone https://github.com/classeve-public/rai.git
cd rai
cargo install --path rai-infer --locked
cargo install --path rai-server --locked
```

## Verify The Checkout

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

## Model Files

RAI does not commit model weights. Use the scripts under `rai-infer/scripts/`
to export a compatible `.raimodel` file and `tokenizer.json` from a supported
HuggingFace model.
