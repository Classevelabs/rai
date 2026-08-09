# RAI

**A CPU-only LLM inference workspace written in Rust.** RAI runs 4-bit quantized
language models with hand-written AVX2 kernels — no GPU, no Python runtime, no
PyTorch, no GGML, no BLAS. Load a `.raimodel` file and generate text on any
supported x86-64 machine. Python is needed only for optional model conversion
and offline draft-model preparation.

Built by [ClassEve](https://classeve.com). Licensed under Apache-2.0.

## Why

Most LLM inference stacks assume a GPU, a CUDA toolchain, or a heavyweight ML
framework. RAI takes the opposite bet: a small, auditable Rust workspace whose
inference matrix kernels are hand-written and whose runtime dependency tree is
captured in `Cargo.lock`.

- **CPU-only by design.** AVX2 + FMA + F16C accelerate inference on compatible
  x86-64 CPUs; scalar fallbacks exist for unsupported instruction sets.
- **4-bit weights, dequantized in registers.** Weights stay packed in memory;
  unpacking happens on-the-fly inside the GEMM inner loop. No FP32 weight copy
  ever exists in RAM.
- **One flat model file.** The `.raimodel` format is a single binary blob with
  a 64-byte header. The loader validates its structure after one heap read and
  then exposes borrowed views over the in-memory sections.
- **Speculative decoding.** Draft-model and self-speculative (first-N-layers)
  modes with experimental target-model acceptance and verification logic.
- **Local serving.** An HTTP chat server with a built-in web UI, plus a REST +
  MCP server so agentic tools (e.g. Claude Desktop, Claude Code) can use RAI
  as a tool backend.

## Workspace layout

| Crate | Purpose |
| --- | --- |
| `rai-infer` | The inference engine: `.raimodel` loader, AVX2 W4A8/W4A32 GEMM kernels, transformer layers (RMSNorm, RoPE, GQA, SwiGLU), KV cache, sampling, speculative decoding, CLI + HTTP chat binaries |
| `rai-compress` | Experimental quantization/compression algorithms. GPTQ export emits `.raimodel`; RC/HRC/SAC accounting is prototype-only and does not yet serialize an artifact. |
| `rai-server` | REST + MCP server exposing the RAI memory/reasoning layer to HTTP clients and MCP-capable agents |
| `rai-core` | Memory, embedding, and reasoning primitives used by `rai-server` |
| `rem-nra` | Resonance-memory backend used by `rai-core` |

The inference and memory-service paths are separate. `rai-generate` and
`rai-chat` load `.raimodel` files through `rai-infer`; `rai-server` does not run
those models. Instead, its REST/MCP adapters call `rai-core`, which obtains an
embedding from the configured provider and stores/queries state through
`rem-nra`. `AppState` serializes REST stores and opted-in MCP stores to
`RAI_DATA_PATH`.

## Requirements

| Requirement | Details |
| --- | --- |
| Rust | 1.87+; the repository pins 1.95.0, edition 2021 |
| CPU | x86-64 with AVX2, FMA, and F16C for optimized paths; scalar fallbacks otherwise |
| OS | Linux, Windows, or macOS (the release candidate must pass CI on each) |
| GPU at runtime | **Not required** |
| Python | Only for optional model export and draft-model preparation |

> `.cargo/config.toml` builds with `target-cpu=native` so the kernels use
> everything your CPU offers. Remove that flag if you need portable binaries.

## Build

```bash
cargo build --workspace --release --locked
```

The repository and v0.1.0 release are already public, and all five
`classeve-rai-*` v0.1.0 crates are on crates.io. This checkout contains
unreleased changes; building it does not change the published v0.1.0 artifacts.
See [installation](./docs/INSTALL.md) and the
[release-readiness gate](./docs/RELEASE_READINESS.md). No container image is
currently published; the Dockerfile builds an MCP stdio image from source.

Development checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
```

## Get a model

RAI runs models in its own `.raimodel` format. The export scripts under
`rai-infer/scripts/` convert any LLaMA/Mistral-family HuggingFace model
(SmolLM, Mistral-7B-Instruct, LLaMA-2/3, …). They need Python with `torch`,
`transformers`, `datasets`, and `numpy`; a CUDA GPU speeds up calibration but
is not required.

```bash
# GPTQ 4-bit export (calibrated)
python3 rai-infer/scripts/export_raimodel.py \
  --model HuggingFaceTB/SmolLM-135M \
  --output smollm-135m-q4.raimodel

# Round-to-nearest export (no calibration)
python3 rai-infer/scripts/export_rtn.py \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

Both write the model file plus a `tokenizer.json` alongside it. Exporter Python
dependencies and HuggingFace model/dataset revisions are not yet locked; record
the exact revisions, arguments, and `pip freeze` output for reproducible work.

## Quickstart

### Generate text

```bash
./target/release/rai-generate \
  --model smollm-135m-q4.raimodel \
  --tokenizer tokenizer.json \
  --prompt "The future of computing is" \
  --max-tokens 64 \
  --temperature 0.7
```

Sampling controls: `--temperature`, `--top-k`, `--top-p`,
`--repetition-penalty`, `--seed`. Test-time compute ("pondering") strategies:
`--ponder-strategy cfg|ensemble|cfg-ensemble|adaptive` with
`--guidance-scale`, `--ensemble-n`, `--noise-sigma`, `--entropy-threshold`.

### Speculative decoding

```bash
# Draft-model speculation: a small model proposes, the big model verifies
./target/release/rai-generate \
  --model mistral-7b-q4.raimodel \
  --tokenizer tokenizer.json \
  --draft mistral-draft-100m-q4.raimodel \
  --draft-k 6 \
  --prompt "Explain speculative decoding in one paragraph."

# Self-speculative: the model's own first N layers act as the draft
./target/release/rai-generate \
  --model mistral-7b-q4.raimodel \
  --tokenizer tokenizer.json \
  --self-spec-layers 8 --self-spec-k 8 \
  --prompt "Hello"
```

Draft and target must share a tokenizer. The verification algorithm is intended
to preserve the target distribution, but exact equivalence is not a marketing
claim until retained golden outputs and statistical tests demonstrate it.
`rai-infer/scripts/train_draft.py` is an experimental distillation helper for
compatible Mistral-family teachers; its throughput projections are not release
benchmarks.

### Chat over HTTP

```bash
./target/release/rai-chat \
  --model smollm-135m-q4.raimodel \
  --tokenizer tokenizer.json \
  --port 8090
```

Open `http://localhost:8090` for the built-in web UI, or POST to `/api/chat`
(JSON) for programmatic access. `--chat-template auto|mistral|llama3|few-shot`
selects prompt formatting. The chat server binds to `127.0.0.1` only and limits
request bodies to 64 KiB.

### REST + MCP server

`rai-server` exposes an experimental memory/reasoning prototype:

```bash
# REST mode (default 127.0.0.1:3000; configure with RAI_HOST / RAI_PORT)
./target/release/rai-server

# MCP mode on stdio — for MCP clients such as Claude Desktop or Claude Code
./target/release/rai-server mcp
```

REST endpoints: `POST /v1/store`, `/v1/recall`, `/v1/intersect`,
`/v1/contradict`, `/v1/surprise`, `/v1/confidence`, `/v1/train`,
`/v1/snapshot`, and `GET /v1/health`.

`POST /v1/train` is reserved but returns HTTP 501 in this build; functional
training is not implemented and the service does not report a no-op as trained.

The current backend is not a validated resonance-training system: training is
not yet an optimizer, retrieval is nearest-vector/cosine-like, and
contradiction, confidence, attractor, and interference outputs do not yet have
evaluation evidence sufficient for product claims. Treat these endpoints as an
experimental API until their algorithms and tests are completed.

REST request bodies are limited to 64 KiB, concurrent work is bounded, and a
global request ceiling protects the local process from accidental overload.
Local loopback usage needs no token. `rai-server` serves plain HTTP and therefore
refuses every non-loopback `RAI_HOST`, even when a token is configured. For
remote access, keep RAI on loopback behind a TLS-terminating reverse proxy and
have the proxy send the internal `Host` value (`127.0.0.1:<RAI_PORT>`). Host and
Origin checks reject DNS-rebinding and browser cross-origin requests.

Set `RAI_API_TOKEN` to a random value of at least 32 bytes to require bearer
authentication on every `/v1/*` request, including loopback requests:

```bash
curl -H "Authorization: Bearer $RAI_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"example"}' http://127.0.0.1:3000/v1/recall
```

Set `RAI_DATA_PATH=/path/to/rai-memory.json` to load memory on startup and save
it after successful REST or opted-in MCP stores. The parent directory is created
when needed. Snapshots use write-then-rename replacement; Unix files are created
with mode `0600`. On Windows, place the snapshot in a user-private directory
because file access inherits that directory's ACL. An unreadable or invalid
snapshot fails startup instead of silently starting with empty memory.

In MCP mode the same operations are exposed as tools (`rai_store`,
`rai_recall`, `rai_intersect`, `rai_contradict`, `rai_surprise`,
`rai_explain_confidence`, `rai_memory_health`). Example MCP client
configuration:

```json
{
  "mcpServers": {
    "rai": {
      "command": "/path/to/rai-server",
      "args": ["mcp"]
    }
  }
}
```

MCP is a trusted local stdio transport and inherits the launching client's OS
permissions. It has no independent network-authentication boundary. `rai_store`
is hidden and denied by default; set `RAI_MCP_MUTATIONS_ENABLED=true` only when
that MCP client should be allowed to modify and persist memory.

Embeddings default to a deterministic built-in mock provider intended only for
tests and demonstrations; startup prints a warning whenever it is active. Set
`RAI_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use an
OpenAI embedding API instead. Any other provider value is rejected
at startup rather than silently falling back to mock embeddings.

### Server configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAI_HOST` | `127.0.0.1` | REST bind host; only loopback names/addresses are accepted |
| `RAI_PORT` | `3000` | REST port, from 1 through 65535 |
| `RAI_API_TOKEN` | unset | Optional REST bearer token; at least 32 bytes when set |
| `RAI_EMBEDDING_PROVIDER` | `mock` | `mock` for demonstrations or `openai` |
| `OPENAI_API_KEY` | unset | Required when the provider is `openai` |
| `RAI_DATA_PATH` | unset | Snapshot file; state is ephemeral when unset |
| `RAI_MCP_MUTATIONS_ENABLED` | `false` | Exact `true`/`false`; exposes mutating MCP tools when true |
| `RAYON_NUM_THREADS` | Rayon default | Inference worker count |

Common startup failures are deliberate safeguards: use a loopback
`RAI_HOST`, provide a 32-byte-or-longer token, set an OpenAI key with the
`openai` provider, and make `RAI_DATA_PATH` a writable file path rather than a
directory. A model that fails to load should be regenerated and treated as
untrusted until its header, dimensions, offsets, and file length are verified.
See [installation](./docs/INSTALL.md), [operations](./docs/OPERATIONS.md), and
[release readiness](./docs/RELEASE_READINESS.md) for the full runbooks.

## The `.raimodel` format

A single flat, little-endian binary file:

```text
┌─────────────────────────────────────────────┐
│ Header (64 bytes)                           │  magic "RAIM", version,
│                                             │  architecture hyperparameters,
│                                             │  quantization config
├─────────────────────────────────────────────┤
│ Section index table (16 bytes per section)  │  offset + size per section
├─────────────────────────────────────────────┤
│ Section 0: embedding table (8-bit)          │
│ Sections 1..N: transformer layers (4-bit)   │  per linear: dims, f16
│ Section N+1: final RMSNorm (f32)            │  scale/zero per group,
└─────────────────────────────────────────────┘  nibble-packed codes
```

Linear weights are GPTQ-quantized to 4-bit codes with per-group (128-column)
f16 scale/zero parameters, packed two codes per byte in the layout the AVX2
kernels consume. The embedding table is 8-bit quantized. Scale/zero parameters
are round-tripped through f16 at export time so the Rust reader and Python
exporter use the same stored values.

The loader performs one read of the whole file into heap memory, validates the
header and section bounds, and then hands out borrowed slices into that buffer.
This avoids additional copies between parsed sections, but loading still copies
the file from storage into the process heap. No retained benchmark establishes
a general performance advantage over `mmap`.

## Performance

One historical author-run measurement reported **~195 tokens/s** for
SmolLM-135M (83 MB at 4-bit) on a described 4-core/8-thread laptop-class x86-64
CPU. Raw output and the complete environment were not retained, so this is not
an independently reproduced release result.

See [BENCHMARKS.md](./BENCHMARKS.md) for methodology, the full numbers, and
compression-quality measurements.

## Project status

RAI is young software, released because it is useful and readable — not
because it is finished. Interfaces may change. Current state:

- Small-model and Mistral-7B-family export/inference paths exist, but this
  repository does not retain a release qualification matrix or model fixtures.
- Historical quantization experiments used Hessian-weighted output error
  against an FP16 reference (see BENCHMARKS.md); reproducible perplexity and
  quality sweeps are still needed.
- x86-64 only for the optimized paths. There is a scalar fallback, but ARM
  NEON kernels are future work.
- The memory/reasoning service and RC/HRC/SAC compression paths are research
  prototypes; do not rely on their training, confidence, contradiction, or
  modeled-size outputs as validated product guarantees.

Issues and PRs are welcome.

## About

Built and maintained by [ClassEve](https://classeve.com) — engineering for AI agents and developer tooling. Project page: [classeve.com/public/rai](https://classeve.com/public/rai).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
Copyright 2025-2026 ClassEve.
