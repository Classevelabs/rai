# Changelog

All notable changes to RAI are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning for its pre-1.0 releases.

## [Unreleased]

### Security

- Hardened REST and MCP request boundaries, resource limits, authentication,
  persistence, and error handling.
- Made MCP persistence mutations an explicit opt-in and bounded stdio frames.
- Added stricter `.raimodel` validation and allocation/shape checks.
- Pinned CI actions, Rust toolchain, and container base-image digests.

### Changed

- Narrowed runtime dependency features and removed the default native
  Oniguruma/C++ features from the tokenizer dependency.
- Added locked cross-platform CI, RustSec, crate-package, Python syntax, and
  container smoke gates.
- Qualified historical benchmark and experimental compression claims.

### Fixed

- Corrected JSON-RPC notification handling and MCP tool annotations.
- Made durable memory mutations atomic at the application boundary.

## [0.1.0] - 2026-06-14

- First public GitHub release. The five `classeve-rai-*` crates were initially
  published to crates.io on 2026-06-11.
