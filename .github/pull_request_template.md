## Summary

Describe the change and the user-visible impact.

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo test --workspace --locked`
- [ ] `cargo build --workspace --release --locked`

## Risk

Call out parser, unsafe/SIMD, server, model-format, or performance risks.

## Notes

Mention benchmarks, follow-up work, or breaking changes when relevant.
