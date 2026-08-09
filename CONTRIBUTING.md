# Contributing

Thank you for improving RAI. This repository is maintained by Classeve and is
intended to stay readable, reproducible, and useful for CPU-first inference
work.

## Development Setup

```bash
git clone https://github.com/classeve-public/rai.git
cd rai
cargo build --workspace
```

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

## Pull Request Expectations

- Keep changes scoped and explain the user-visible behavior.
- Include tests for parser, compression, sampling, server, or reasoning changes
  when behavior changes.
- Do not commit model weights, `.raimodel` files, tokens, API keys, cache
  folders, or benchmark artifacts.
- Be careful around SIMD and unsafe code. Keep safety assumptions local and
  documented.
- Benchmark performance-sensitive changes where practical and include the
  hardware details.

## Release Philosophy

RAI is pre-1.0. Public APIs may evolve, but breaking changes should be called
out clearly in release notes and should not be hidden inside unrelated cleanup.
