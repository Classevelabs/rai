# Installation

RAI is distributed through its public source repository and five crates.io
packages. The currently published v0.1.0 artifacts predate the unreleased
hardening work in this checkout; do not describe an untagged checkout as the
published crate release.

## Requirements

- Rust 1.87 or newer. This repository pins Rust 1.95.0 for repeatable checks.
- x86-64 with AVX2, FMA, and F16C for the optimized inference path. Scalar
  fallbacks exist, but ARM acceleration is not implemented.
- Python only for model export or draft-model training. The scripts import
  `torch`, `transformers`, `datasets`, and `numpy`; their versions and model or
  dataset revisions are not yet locked, so record `pip freeze` and every
  HuggingFace revision used.

## Source checkout

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --workspace --release --locked
```

The repository's `.cargo/config.toml` uses `target-cpu=native`; binaries built
that way are for the build machine. Remove or override that flag before moving
a binary to a different CPU.

Install the two end-user binary crates from a checkout:

```bash
cargo install --path rai-infer --locked
cargo install --path rai-server --locked
```

The public v0.1.0 crates can be installed explicitly with:

```bash
cargo install classeve-rai-infer --version 0.1.0 --locked
cargo install classeve-rai-server --version 0.1.0 --locked
```

## Container

The container is an amd64/x86-64-v3 MCP stdio image and runs as a non-root
user. It does not expose a REST port because the built-in REST listener is
intentionally loopback-only.

```bash
docker build -t rai-server .
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run --rm -i rai-server mcp
```

Persisted memory must be mounted at a path writable by UID 10001 and explicitly
configured with `RAI_DATA_PATH`. Set `RAI_MCP_MUTATIONS_ENABLED=true` only when
the launching MCP client should be allowed to modify that state.

## Verification

See [RELEASE_READINESS.md](./RELEASE_READINESS.md) for the complete candidate
gate. A successful local build alone is not release approval.
