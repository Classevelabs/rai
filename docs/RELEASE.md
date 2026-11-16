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
cargo install cargo-deny --version 0.18.2 --locked
cargo deny --all-features check licenses bans sources
cargo install cargo-about --version 0.7.1 --locked
cargo about generate --manifest-path rai-infer/Cargo.toml \
    -c about.toml about.hbs -o /tmp/notices.md
diff -u THIRD-PARTY-NOTICES.md /tmp/notices.md
python -m compileall -q rai-infer/scripts
python rai-infer/scripts/test_raimodel.py
cargo bench --package classeve-rai-infer --bench gemm --locked
docker build -t rai-release-candidate .
```

The benchmark is a record, not a gate — throughput depends on the machine, and
a release must not be blocked by a busy laptop. Read its `iqr%` column first:
above about 5 the machine moved more than any kernel change is worth, and the
medians beside it settle nothing. Compare against the previous release's run on
the *same* machine, ideally alternating between the two builds.

`cargo about` reaches `api.clearlydefined.io` for crates whose own files do not
settle their license, and that request is flaky. If it warns
`failed to request license information`, the file it wrote is missing license
sections and the `diff` above means nothing — retry until it does not.
`.github/workflows/ci.yml` retries five times and then fails the job rather
than skipping the check: a skipped check once let a release ship notices naming
the previous version. Do not reach for `--offline` to make it
quiet: it substitutes a default license file and drops copyright lines that only
the upstream repository carries, which is the information this file exists to
reproduce.

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
  scalar fallback kernels, since it is the only runner without AVX2. The
  release workflow enforces this one: it waits for CI on the tagged commit and
  refuses to build a commit CI has not passed. A run cancelled by a later push
  counts as not passed; start CI on the tag itself
  (`gh workflow run ci.yml --ref v0.2.0`) and run the release again;
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

1. Resolves the tag to one commit (from the tag namespace only, so a branch
   of the same name cannot stand in for it), verifies it against the
   workspace version, waits for green CI on that commit, and opens a
   **draft** release. Every later job builds that commit and checks that it
   did, so a tag moved between a failed job and its re-run cannot mix two
   trees in one release.
2. Builds `rai` plus the deprecated `rai-convert`,
   `rai-generate`, and `rai-chat` wrappers — `--package classeve-rai-infer`
   only, so no non-product crate can reach an archive — on `ubuntu-24.04`, `windows-2025`,
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
find it. It resumes a **draft** only: a published release is refused, never
rebuilt, because a rebuild replaced every archive and rewrote `SHA256SUMS`
under hashes people had already pinned. A bad published release is fixed
forward with a new version.

Each archive also gets a signed build-provenance attestation
(`actions/attest-build-provenance`), which ties it to this workflow and the
tagged commit. Users verify one with
`gh attestation verify <archive> --repo Classevelabs/rai`; the release notes
say so.

**Never let this workflow build with `target-cpu=native`.** Such binaries die
with SIGILL on any CPU older than the runner that built them, after download,
with no diagnostic. `.cargo/config.toml` pins the same `x86-64-v2` floor this
workflow sets, so nothing has to be overridden — but the build job still
asserts `RUSTFLAGS` explicitly, because that config file is one edit away from
being changed back and a published release is the wrong place to discover it.
`x86-64-v2` is the right floor rather than `v3` because the AVX2/FMA/F16C
kernels are selected at runtime by `has_avx2()` in `rai-infer/src/gemm.rs`, not
by the compile-time baseline.

Releasing a binary without platform code signing is deliberate: macOS users
clear the quarantine attribute themselves and Windows shows a SmartScreen
warning. `SHA256SUMS` and the provenance attestations are the integrity story,
so neither may be missing from a published release.

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
workflow by hand with the release tag. It publishes that tag's commit, never
the branch it was dispatched from, requires green CI on that commit, packages
both crates before publishing either, and authenticates through crates.io
trusted publishing (an OIDC exchange scoped to this repository and workflow
file — no long-lived token exists anywhere). A crate already at that version
is skipped rather than failing the run, so a run that published one crate and
then failed on the other is finished by running it again.

The job runs in the `crates-io` environment. Give that environment required
reviewers (Settings → Environments) so that publishing needs a second person's
approval, and name the environment in each crate's trusted-publisher entry on
crates.io so that no other workflow job can mint the token.

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
