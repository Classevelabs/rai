# Release Procedure

This document is the maintainer checklist for Classeve RAI releases.

## Local Gate

Run the full quality gate from a clean checkout:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

Do not tag a release until CI is green on `main`.

## Versioning

RAI is pre-1.0. Use semantic version tags such as `v0.1.0`. Breaking changes
are allowed before `1.0`, but release notes must call them out.

## Tagging

```bash
git tag -a v0.1.0 -m "RAI v0.1.0"
git push origin v0.1.0
```

## GitHub Release Notes

Release notes should include:

- What changed.
- Installation commands.
- Verification status.
- Known limitations.
- Benchmark notes when performance claims changed.

## Publishing Crates

The crates are named with the `classeve-rai-*` namespace for crates.io
compatibility. Crates.io publication is separate from the GitHub release and
requires maintainer credentials plus a publish-order check for internal path
dependencies.
