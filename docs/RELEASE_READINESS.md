# Release Readiness

RAI distinguishes technical publication from a marketing launch. The GitHub
repository, v0.1.0 GitHub release, project webpage, and five v0.1.0 crates.io
packages are already public. A marketing campaign does not make those existing
artifacts newly private or newly published.

Run these gates on the exact clean candidate commit:

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
cargo package --workspace --locked
python -m compileall -q rai-infer/scripts
docker build -t rai-release-candidate .
```

Smoke-test both packaged binaries, REST authentication and mutation routes,
MCP initialization/read-only behavior/mutation opt-in, persistence recovery,
malformed model files, and the built container. Also require:

- a candidate version newer than every published crate and GitHub tag;
- a complete changelog and release notes that identify security changes and
  compatibility breaks;
- green CI on Linux, Windows, and Intel macOS for the exact commit;
- reconciliation with the current public `main` branch before tagging;
- pinned model/dataset revisions, Python dependencies, hardware identity,
  commands, commit SHA, and retained raw output for any benchmark claim used in
  marketing;
- an approved rollback owner and verified recovery procedure; and
- container digest, crate archive checksums, RustSec output, CI URL, and smoke
  evidence recorded in the release notes.

Use the [operations](./OPERATIONS.md) and [rollback](./ROLLBACK.md) runbooks;
replace their generic owner placeholders with named people before release.

Existing benchmark tables are historical author-reported measurements, not a
release gate. A target date is not approval: any failed or unverified item is a
no-go.
