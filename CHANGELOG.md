# Changelog

All notable changes to RAI are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning for its pre-1.0 releases.

## [0.2.0] - Unreleased

### Added
- **One `rai` binary with four verbs — `rai convert`, `rai run`, `rai serve`,
  `rai models`** — replacing the four separate commands. `rai run` and
  `rai serve` take the model as a positional argument and default `--tokenizer`
  to the `tokenizer.json` beside it. `rai models` lists a directory's
  `.raimodel` files from their headers alone. `rai-convert`, `rai-generate`,
  and `rai-chat` remain as deprecated wrappers over the same entry points and
  will be removed in a future release.
- **A Rust-native streaming converter** behind `rai convert`. It reads
  `.safetensors` directly, needs no Python or torch, and writes each section in
  place, so peak memory is one row block rather than the whole model — a 7B
  checkpoint converts on a 16 GB machine. Output is byte-identical to
  `export_rtn.py` for the same checkpoint and options, pinned by
  `tests/convert_matches_python.rs`.
- **Container format v2**, adding Qwen2/Qwen2.5, Llama-3.1/3.2, and Gemma
  support. The header grows to 128 bytes and gains an activation code (GeGLU
  alongside SwiGLU), the `llama3` RoPE rescaling parameters, a per-projection
  bias mask, and an embedding scale; layer sections gain optional f32 bias
  vectors. Gemma's `(1 + w)` RMSNorm folds into the stored norm weights at
  conversion. A converter emits v1 whenever none of the new fields is needed,
  so pre-v2 checkpoints still produce identical bytes. Decoupled `head_dim` is
  now accepted. The Python exporters remain v1-only and still refuse all three
  families.
- **Prompt-lookup speculative decoding** (`--lookup-k`, `--lookup-ngram`,
  `--lookup-min-ngram`): the draft is copied from the context, so there is no
  draft model and no draft forward pass. Off by default — it is a gain on
  context-quoting workloads and a loss on original prose, both measured in
  BENCHMARKS.md.
- Converter preflight that refuses checkpoints the `.raimodel` format cannot
  represent — per-head QK norms (Qwen3, OLMo2, Gemma3), mixture-of-experts
  routing, logit softcapping (Gemma2), unsupported `rope_type` and activation
  values, and non-Llama module trees. These previously exported "successfully"
  and generated nonsense.
- Chat templates `chatml` (`<|im_start|>`) and `zephyr` (`<|user|>`), with
  auto-detection for ChatML. Zephyr-style models (TinyLlama-Chat) need the
  template passed explicitly because their markers are plain text, not
  vocabulary tokens.
- `rai-infer/scripts/requirements-lock.txt`: exact dependency versions verified
  to convert a real model end to end.
- `rai run` explains itself when a model emits end-of-sequence immediately
  instead of printing nothing at all.
- Double-click launchers in `launchers/` that start the local chat UI without a
  terminal on Windows, macOS, and Linux.
- rai-infer test coverage: parallel-path GEMM reference tests (fused QKV,
  fused gate/up, and the tied LM head against dequantized references above the
  rayon threshold), chat-template unit tests with an in-memory tokenizer stub,
  and a seeded speculative-decoding statistical smoke test
  (`tests/speculative_equivalence.rs`, tied fixture drafting for the untied
  fixture).
- `# Safety` documentation on every unsafe AVX2 kernel and `# Panics`
  documentation on rai-infer's asserting public APIs.
- rai-server request timeout: any REST request still running after 30 seconds is
  abandoned with HTTP 503 and a JSON error, releasing its concurrency permit.
- rai-server graceful shutdown on Ctrl-C, so in-flight durable stores finish
  before the process exits.
- `rai-server --help`/`-h` and `--version`/`-V`.
- A portless loopback `Host` header (`Host: localhost`) is accepted, and a
  rejected `Host` now names the required form instead of returning a bare 403.
- `#![forbid(unsafe_code)]` in rem-nra, plus crate-level documentation for
  rem-nra and rai-core.

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
- Renamed rai-infer's GEMM entry points from `w4a32_*` to `w4a8_*`
  (`w4a8_matvec`, `w4a8_fused_qkv`, `w4a8_fused_gate_up`, `w4a8_matmul`): the
  kernels quantize activations to int8, so the old names claimed a precision
  the hot path does not use.
- Made the rai-infer library lean: `clap`, `tokenizers`, `tiny_http`, `serde`,
  `serde_json`, and `memmap2` now sit behind a default-on `cli` feature (all
  binaries require it); `--no-default-features` builds the inference library
  with only `half`, `rayon`, `anyhow`, and `rand`. The tokenizer-aware
  `ChatTemplate::auto_detect`/`from_str_arg` helpers are gated accordingly.
- Speculative and self-speculative `step()` no longer take a
  generation-history argument: config validation requires
  `repetition_penalty == 1.0` for exact verification, so the in-loop penalty
  application (and its per-token O(context) clones) was dead by construction
  and has been removed.
- The `profile-fwd` and `bw-bench` dev tools take `--model <path>` instead of
  a hardcoded repo-relative model path (`bw-bench` skips its mmap section when
  no model is given), and their remaining `unwrap()`s on file access became
  contextual errors.
- Deduplicated coupled constants: `format.rs` now imports `gemm::MAX_GROUPS`
  and `layers::MAX_ROPE_TABLE_BYTES` instead of redefining them, keeping
  `.raimodel` validation the single gate for kernel capacity.
- **Breaking (rai-compress):** the encoding entry points no longer panic on
  caller input — `compress`, `compress_uniform_4bit`, `compare`,
  `hrc_compress`, `sac_compress`, `full_compare`, `quantize_uniform`,
  `choose_bits`, and `BitPacker::pack` now return `Result` with the crate's
  error types, matching the decoders.
- **Breaking (rai-compress):** the seven hand-rolled error enums are now
  `thiserror` derives with `#[from]`/`source` chaining; `CompressionError`
  wraps every stage error (including `GptqError` and `BitPackError`) and
  gained an `InvalidInput` variant; `BitPackError` gained `ValueOutOfRange`.
- **Breaking (rai-compress):** renamed `HRCStats::psnr_db` and
  `SACStats::psnr_db` to `snr_db` — the computed metric is a mean-signal-power
  SNR, not a peak-based PSNR — and `SACStats::original_bytes`/`ratio` now use
  the same FP64 baseline as the RC/HRC stats (previously FP16).
- rai-compress sparse outlier extraction uses an inclusive threshold so
  magnitude ties at the cutoff are extracted and the configured `fraction` is
  honored exactly; `hessian_weighted_mse` computes the same trace through
  nalgebra matrix products instead of a scalar triple loop; the HRC/SAC decode
  paths share one validated helper; the crate forbids `unsafe`, documents its
  research status (RC/HRC/SAC do not produce `.raimodel` files), and
  re-exports all types reachable through public fields.
- rai-compress tests and benches now use seeded RNGs (`StdRng`) and smaller
  matrices with verified assertion margins: the crate's test suite runs in
  about 3 seconds instead of about 8 minutes.
- **Breaking (rai-core):** `Compositor::intersect` and
  `Projection::project`/`project_normalized` return `Result` instead of
  panicking on empty, ragged, or wrongly-sized input, and
  `EmbeddingBridge::nearest_text` returns `Result<Option<String>, RaiError>`
  after validating its input like every other bridge method.
- **Breaking (rem-nra):** `NonlinearResonanceMemory::from_snapshot` replaces
  `from_params`, and `ResidualEquilibriumMemory::new`/`from_snapshot` lost their
  now-meaningless parameters. Both validate untrusted persisted state, and the
  cosine helpers treat a ragged vector as zero similarity instead of panicking.
- **Breaking (rai-core):** `HealthReport::rem_residual_norm` is renamed
  `mean_residual_norm`.
- **Breaking (rai-core):** a full store returns the new
  `RaiError::CapacityExhausted { limit }` instead of a generic memory error;
  rai-server maps it to HTTP 409 (was 500) and to a named MCP tool error.
- **Breaking (snapshot schema):** `rem_encoder`, `rem_decoder`,
  `rem_memory_state`, `rem_last_loss`, and `nra_params.value_basis` are no
  longer written, and the strict validator no longer checks their shapes. The
  schema version stays at 1 and existing version-0 and version-1 snapshots still
  load — the retired keys are ignored. Files written by this release cannot be
  read by 0.1.0.
- **Breaking (rai-server):** one text limit for the whole memory stack —
  16 KiB measured in *bytes*, shared by the REST handlers, the MCP server, and
  the library. The MCP transport previously counted 16,384 *characters*, which
  allowed up to 64 KiB. The snapshot validator still tolerates the larger
  historical bound so older files load.
- `/v1/contradict`, `rai_contradict`, and the interference report returned by
  `/v1/store` are documented for what they measure: address-space crowding. A
  store can only bring neighbours closer, so they cannot detect a semantic
  contradiction, and an empty report is not evidence of consistency. Doc
  comments and tool descriptions no longer use ODE, attractor, or basin
  language.

### Fixed
- GPTQ calibration no longer dies on `datasets` 5.x: the hardcoded bare
  `wikitext` repo id is rejected by the namespaced-id rule, so calibration is
  now `Salesforce/wikitext` and overridable with `--calibration-dataset` /
  `--calibration-config`.
- The export scripts no longer require the optional `accelerate` package: they
  load weights on CPU without `device_map`, and pass the dtype keyword under the
  name the installed transformers expects (`dtype` from 5.0, `torch_dtype`
  before it). A clean environment previously failed at the first step.

- Corrected JSON-RPC notification handling and MCP tool annotations.
- Made durable memory mutations atomic at the application boundary.
- Non-Unicode environment variables fail rai-server startup instead of being
  silently ignored. A mangled `RAI_API_TOKEN` previously started the REST API
  with authentication disabled.
- The memory manager's six inner mutexes and its global mutation lock collapsed
  into one `RwLock`: reads now run concurrently instead of serializing behind
  the writer lock. The staged-commit durable-store transaction is unchanged —
  stage, write to disk, publish only on success.
- Recall no longer re-projects every stored embedding inside the lock. Each
  entry's value projection is cached at insert time and rebuilt on load, turning
  an O(n·d²) scan inside a comparator into a cosine scan over cached vectors.

### Removed

- Roughly 1,000 lines of dead rai-infer code: three compiled-out AVX2 kernel
  blocks in `gemm.rs`, the unused `w4a32_matmul_preq`/`QuantizedBatchInput`
  batched API, `silu_inplace`/`rms_norm_inplace`, unreachable
  zero-draft/bounds branches and no-op repetition-penalty plumbing in the
  speculative decoders, the drifted duplicate examples (`bench_gemm`,
  `profile_forward`), and an uncontended mutex in the deliberately
  single-threaded chat server.
- rai-compress dead dependencies: `serde` and `rand_distr` are gone, `rand`
  moved to dev-dependencies, and nalgebra's `serde-serialize` feature was
  dropped.
- **Breaking:** `POST /v1/train` and its single-flight training lock. No build
  has an optimizer, and a permanent HTTP 501 was worse than no endpoint.
  `RaiError::TrainingError`, `MemoryManager::train_nra`/`train_nra_and_save`/
  `train_rem`/`train_rem_and_save`, the whole `memory::training` module
  (`TrainingOrchestrator`), `NonlinearResonanceMemory::train_two_phase`,
  `ResidualEquilibriumMemory::train`, and `MemoryError::TrainingUnavailable` go
  with it. No MCP training tool existed.
- **Breaking:** response fields `RetrievalResult::steps`,
  `RetrievalResult::grad_norm`, and
  `ConfidenceExplanation::grad_norm`/`num_attractors`/`basin_spread`. They
  reported ODE-integration diagnostics no integrator produced. `energy` stays —
  it is a real leave-one-out crowding score.
- **Breaking:** `HealthReport::needs_training`, `nra_mse`, and `rem_mse`. The
  two "MSE" figures were the same formula over the same vectors — mean value
  magnitude, not error — and were always equal. The `mse`/`last_loss`
  accessors that computed them are gone too.
- **Breaking:** the `ConfidenceLevel::Ambiguous` tier and the `BasinAnalyzer`
  perturbation diagnostic that was its only producer. `NoMatch` now means what
  it says: no stored memory scored above zero cosine similarity.
- **Breaking:** `ConfidenceGate::with_ode_tol`/`ode_tol`/`no_match_grad_factor`
  and `NRAConfig::ode_tol`. With the gradient diagnostic gone there is no ODE
  tolerance to configure; `classify` and `explain` take only the retrieval score.
- **Breaking:** `NRAConfig::train_epochs`, `REMConfig::train_epochs`,
  `NRAParams::value_basis`, `rem_nra::nra::find_attractor`,
  `rem_nra::nra::energy`, `AttractorResult`, `MemoryError::Empty`, and
  `ResidualEquilibriumMemory`'s encoder/decoder biases and rolling
  `memory_state` — none of which fed a live computation.
- Dead code: `MemoryManager::new` (unvalidated, zero callers, enabled a
  dimension-mismatch panic), `MemoryEntry`, `TextIndex::find_nearest`/
  `get_by_id`, `EmbeddingBridge::text_to_key`/`text_to_value`/`embed_text`,
  `Compositor::weighted_intersect`/`difference`/`analogy`, and the never-read
  `next_id` counter.
- Dead dependencies: `log` from rai-core and `tower-http` from rai-server.

## [0.1.0] - 2026-06-14

- First public GitHub release. The five `classeve-rai-*` crates were initially
  published to crates.io on 2026-06-11.
