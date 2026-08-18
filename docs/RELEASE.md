# Release procedure

The maintainer checklist for RAI releases. Nothing here is needed to use RAI —
see [INSTALL.md](../INSTALL.md) for that.

## Gate

Run every one of these on the exact clean candidate commit. Any failure is a
no-go; a target date is not approval.

```bash
cargo metadata --locked --format-version 1 --no-deps
cargo +1.87.0 check --workspace --all-targets --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
cargo build --workspace --release --locked
cargo install cargo-audit --version 0.22.2 --locked --no-default-features
cargo audit --deny warnings
cargo package -p classeve-rai-infer -p classeve-rai-compress --locked
python -m compileall -q rai-infer/scripts
docker build -t rai-release-candidate .
```

`cargo +1.87.0 check` is what keeps `rust-version = "1.87"` honest; the
repository otherwise builds on the pinned 1.95.0 toolchain.

Then smoke-test the packaged binaries, REST authentication and mutation routes,
MCP initialization / read-only behaviour / mutation opt-in, persistence
recovery, malformed model files, and the container. Also require:

- a candidate version newer than every published crate and GitHub tag;
- a changelog and release notes that identify security changes and
  compatibility breaks;
- green CI on Linux, Windows, Apple Silicon macOS, and Intel macOS for the
  exact commit — the Apple Silicon job is also the only one that executes the
  scalar fallback kernels, since it is the only runner without AVX2;
- pinned model/dataset revisions, Python dependencies, hardware identity,
  commands, commit SHA, and retained raw output behind any new benchmark claim;
- crate archive checksums, RustSec output, CI URL, and smoke evidence recorded
  in the release notes.

Existing benchmark tables are a record, not a gate.

## Versioning

RAI is pre-1.0. Use semantic version tags such as `v0.2.0`. Breaking changes
are allowed before `1.0`, but release notes must call them out.

The tag and `[workspace.package] version` in `Cargo.toml` must agree. The
release workflow refuses to build otherwise, rather than shipping archives whose
names disagree with their contents.

## Tagging

```bash
git tag -a v0.2.0 -m "RAI v0.2.0"
git push origin v0.2.0
```

## Binaries

Pushing a `v*` tag runs [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which needs no manual step:

1. Verifies the tag against the workspace version and opens a **draft** release.
2. Builds `rai` and `rai-server`, plus the deprecated `rai-convert`,
   `rai-generate`, and `rai-chat` wrappers, on `ubuntu-24.04`, `windows-2025`,
   `macos-15-intel`, and `macos-15` (Apple Silicon) — the x86 targets with
   `RUSTFLAGS="-C target-cpu=x86-64-v2"` — then **executes every binary** and
   requires `--version` to answer with the release's version before anything
   is packaged, and packages each platform as
   `rai-<version>-<target>.{tar.gz,zip}` with LICENSE, NOTICE, README.md,
   INSTALL.md, the `launchers/` scripts, and RUNNING.txt.
3. Downloads the four archives back from the release, asserts the count,
   writes `SHA256SUMS` from what the release actually serves, attaches it, and
   only then clears the draft flag.

`workflow_dispatch` accepts an existing tag and re-runs the same path,
addressing the draft by release id rather than by tag so a draft re-run can
find it.

**Never let this workflow build with `target-cpu=native`.** Such binaries die
with SIGILL on any CPU older than the runner that built them, after download,
with no diagnostic. `.cargo/config.toml` pins the same `x86-64-v2` floor this
workflow sets, so nothing has to be overridden — but the build job still
asserts `RUSTFLAGS` explicitly, because that config file is one edit away from
being changed back and a published release is the wrong place to discover it.
`x86-64-v2` is the right floor rather than `v3` because the AVX2/FMA/F16C
kernels are selected at runtime by `has_avx2()` in `rai-infer/src/gemm.rs`, not
by the compile-time baseline.

Releasing an unsigned binary is deliberate: macOS users clear the quarantine
attribute themselves and Windows shows a SmartScreen warning. `SHA256SUMS` is
the integrity story, so it must never be missing from a published release.

## Release notes

The workflow writes a default body covering checksum verification and the CPU
baseline. Add what changed, verification status, known limitations, and
benchmark notes when performance claims changed.

## Publishing crates

Two crates publish, under the `classeve-rai-*` namespace:
`classeve-rai-infer` and `classeve-rai-compress`. They have no path
dependencies on each other, so there is no ordering. The other three crates
are `publish = false` and stay that way; their 0.1.0 releases are yanked.

Publication is separate from the GitHub release: dispatch the `Publish`
workflow by hand. It authenticates through crates.io trusted publishing (an
OIDC exchange scoped to this repository and workflow file — no long-lived
token exists anywhere), refuses to republish a version the index already
carries, and publishes with `--locked`.

## Rollback

Published artifacts are immutable in practice. Do not promise that a bad crate
or Git tag can be replaced under the same version.

Before releasing, record the candidate commit, crate archive checksums,
container digest if one is published, CI URL, and the last known-good versions.
Keep a pre-upgrade snapshot backup and its checksum.

**If a candidate fails before publication:** stop the rollout, keep the previous
binary active, restore the backup only if the candidate wrote state, and open a
tracked incident with the failed gate and logs. Fix forward on a new candidate
commit and rerun every gate.

**If a published release is unsafe:**

1. Halt promotion and publish an advisory naming the affected versions.
2. Yank the affected crates.io versions when installation should be
   discouraged. Yanking does not delete already-downloaded code or break
   existing lockfiles.
3. Mark the GitHub release as affected. Do not move or recreate its tag to hide
   the original artifact.
4. Direct operators to the last known-good version and snapshot backup. Verify
   recovery on a copy before replacing live state.
5. Publish a higher patch version with the fix and complete evidence. Never
   reuse the affected version number.

If a container is published, deploy by immutable digest and roll back to the
recorded last known-good digest. Removing a mutable tag is not sufficient,
because cached images and pulled digests remain accessible.
