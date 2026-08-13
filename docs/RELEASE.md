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

The tag and `[workspace.package] version` in `Cargo.toml` must agree. The
release workflow refuses to build otherwise, rather than shipping archives whose
names disagree with their contents.

## Tagging

```bash
git tag -a v0.1.0 -m "RAI v0.1.0"
git push origin v0.1.0
```

## Binaries

Pushing a `v*` tag runs [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which needs no manual step:

1. Verifies the tag against the workspace version and opens a **draft** release.
2. Builds `rai-convert`, `rai-generate`, `rai-chat`, and `rai-server` on
   `ubuntu-24.04`, `windows-2025`, and `macos-15-intel` with
   `RUSTFLAGS="-C target-cpu=x86-64-v2"`, packaging each platform as
   `rai-<version>-<target>.{tar.gz,zip}` with LICENSE, NOTICE, README.md, and
   RUNNING.txt.
3. Downloads the three archives back from the release, writes `SHA256SUMS` from
   what the release actually serves, attaches it, and only then clears the draft
   flag.

`workflow_dispatch` accepts an existing tag and re-runs the same path; asset
uploads use `--clobber`, so a re-run replaces rather than duplicates.

**Never let this workflow inherit `target-cpu=native`.** The repository's
`.cargo/config.toml` sets it, which is correct for local builds and fatal for
distributed ones: those binaries die with SIGILL on any CPU older than the
runner that built them, after download, with no diagnostic. The build job
asserts the override took effect. `x86-64-v2` is the right floor rather than
`v3` because the AVX2/FMA/F16C kernels are selected at runtime by
`has_avx2()` in `rai-infer/src/gemm.rs`, not by the compile-time baseline.

Releasing an unsigned binary is deliberate: macOS users clear the quarantine
attribute themselves and Windows shows a SmartScreen warning. `SHA256SUMS` is
the integrity story, so it must never be missing from a published release.

## GitHub Release Notes

The workflow writes a default body covering checksum verification and the CPU
baseline. Add to it:

- What changed.
- Verification status.
- Known limitations.
- Benchmark notes when performance claims changed.

## Publishing Crates

The crates are named with the `classeve-rai-*` namespace for crates.io
compatibility. Crates.io publication is separate from the GitHub release and
requires maintainer credentials plus a publish-order check for internal path
dependencies.
