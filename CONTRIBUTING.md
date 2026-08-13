# Contributing

Thank you for improving RAI. This repository is maintained by Classeve and is
intended to stay readable, reproducible, and useful for CPU-first inference
work.

## Development setup

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --workspace
```

`rust-toolchain.toml` pins Rust 1.95.0, which `rustup` installs automatically.
The workspace declares `rust-version = "1.87"` as its minimum supported
compiler, so avoid language or standard-library features newer than that.

Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

For release confidence, also run:

```bash
cargo build --workspace --release --locked
```

Documentation is part of the build. If a change alters a command, a flag, a
supported architecture, or a measured number, update `README.md`, `INSTALL.md`,
`docs/`, and `CHANGELOG.md` in the same pull request. Every performance figure
in the documentation must trace to `BENCHMARKS.md`; if you add one, add the
measurement behind it.

## Pull request expectations

- Keep changes scoped and explain the user-visible behavior.
- Include tests for parser, compression, sampling, server, or reasoning changes
  when behavior changes.
- Do not commit model weights, `.raimodel` files, tokens, API keys, cache
  folders, or benchmark artifacts.
- Be careful around SIMD and unsafe code. Keep safety assumptions local and
  documented.
- Benchmark performance-sensitive changes where practical and include the
  hardware details.

## Release philosophy

RAI is pre-1.0. Public APIs may evolve, but breaking changes should be called
out clearly in release notes and should not be hidden inside unrelated cleanup.
