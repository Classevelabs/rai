# RAI

[![Latest release](https://img.shields.io/github/v/release/Classevelabs/rai)](https://github.com/Classevelabs/rai/releases/latest) [![License](https://img.shields.io/github/license/Classevelabs/rai)](LICENSE)

**A CPU-only LLM inference engine written in Rust.** RAI runs 4-bit quantized
language models with hand-written AVX2 kernels — no GPU, no CUDA, no Python
runtime, no PyTorch, no GGML, no BLAS. Load a `.raimodel` file and generate
text on any supported x86-64 machine.

Built by [ClassEve](https://classeve.com). Licensed under Apache-2.0.

> **Official repository.** This is the only official repository for RAI.
> ClassEve's complete list of official accounts is at [classeve.com/official](https://classeve.com/official).
> The GitHub account `github.com/ClassEve` is an unrelated third party, not affiliated with ClassEve.

## Measured performance

TinyLlama-1.1B-Chat, converted from the fp16 checkpoint and decoded greedily
against HuggingFace `transformers` on the same machine — an Intel i5-10300H
(4 cores / 8 threads), 2026-08-09, RAI 0.2.0:

| | RAI 0.2.0 (4-bit) | transformers 5.15 fp32 (CPU) |
| --- | --- | --- |
| Decode, 91 tokens | **21.8 tok/s** | 4.3 tok/s |
| Relative speed | **5.1×** | 1× |
| Peak process RSS | **629 MB** | fp32 weights alone are ~4.4 GB |
| Model on disk | **619.5 MB** | 2,200 MB |
| Load time | **0.33 s** | ~2 s |

Conversion of the whole 1.1B model took **47.0 s**. The environment is pinned
in `rai-infer/scripts/requirements-lock.txt`. Method, per-tensor quantization
error, output-quality comparison, and the measured (negative) result for
self-speculative decoding are in [BENCHMARKS.md](./BENCHMARKS.md).

## Quickstart

```bash
# 1. Build (or take a release archive — see INSTALL.md)
cargo build --workspace --release --locked

# 2. Convert a checkpoint — no Python required
./target/release/rai-convert \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-q4.raimodel

# 3. Generate
./target/release/rai-generate \
  --model tinyllama-1.1b-q4.raimodel \
  --tokenizer tokenizer.json \
  --chat-template zephyr \
  --prompt "Explain photosynthesis in simple terms." \
  --max-tokens 64
```

Conversion writes `tokenizer.json` next to the model file. Instruction-tuned
models need the chat template they were trained on, or they emit
end-of-sequence immediately; TinyLlama-Chat uses `zephyr`. Prebuilt binaries
are in [INSTALL.md](./INSTALL.md); full conversion options, including the
calibrated GPTQ path, are in
[docs/INSTALL.md](./docs/INSTALL.md#converting-a-model).

## Which models work

RAI runs one architecture: the plain Llama/Mistral decoder — RMSNorm, rotary
embeddings from a single theta, grouped-query attention, SwiGLU, no bias
vectors. Everything else is refused at conversion time, by name, before a file
is written.

| | |
| --- | --- |
| **Works** | TinyLlama-1.1B, SmolLM / SmolLM2 (135M / 360M / 1.7B), Llama-2 7B and 13B, Mistral-7B v0.1–v0.3, and fine-tunes of those (Zephyr-7B, OpenHermes-Mistral, Vicuna) |
| **Refused** | Qwen2 / 2.5 (attention bias), Qwen3 and OLMo2 (per-head QK norm), Llama-3.1 / 3.2 (RoPE scaling), every Gemma (GeGLU and a different RMSNorm), Mixtral and other MoE (router + expert weights) |

Check a checkpoint's `config.json` before downloading its weights, and read the
one-line reason for every family, in **[docs/MODELS.md](./docs/MODELS.md)**.
That page also costs out what supporting each refused family would take.

## Why

Most LLM inference stacks assume a GPU, a CUDA toolchain, or a heavyweight ML
framework. RAI takes the opposite bet: an auditable Rust workspace whose
inference matrix kernels are hand-written and whose runtime dependency tree is
captured in `Cargo.lock`.

- **CPU-only by design.** AVX2 + FMA + F16C accelerate inference on compatible
  x86-64 CPUs; scalar fallbacks exist for other instruction sets.
- **4-bit weights, dequantized in registers.** Weights stay packed in memory;
  unpacking happens on the fly inside the GEMM inner loop. No fp32 weight copy
  ever exists in RAM.
- **One flat model file.** The `.raimodel` format is a single binary blob with
  a 64-byte header. The loader validates its structure after one heap read and
  then exposes borrowed views over the in-memory sections.
- **A lean library.** `--no-default-features` builds the inference library —
  format reader, kernels, model, sampling, speculative decoding — against
  `half`, `rayon`, `anyhow`, and `rand` alone. The CLI, tokenizer, and chat
  server sit behind the default-on `cli` feature.
- **Speculative decoding.** Draft-model and self-speculative (first-N-layers)
  modes with experimental target-model acceptance and verification logic.
- **Local serving.** An HTTP chat server with a built-in web UI, plus a REST +
  MCP server so agentic tools (e.g. Claude Desktop, Claude Code) can use RAI
  as a tool backend.

## Workspace layout

| Crate | Purpose |
| --- | --- |
| `rai-infer` | The inference engine: `.raimodel` loader, AVX2 W4A8 GEMM kernels, transformer layers (RMSNorm, RoPE, GQA, SwiGLU), KV cache, sampling, speculative decoding, CLI + HTTP chat binaries (behind the default-on `cli` feature; `--no-default-features` builds the lean library) |
| `rai-compress` | Quantization and compression research toolkit. Its Rust GPTQ implementation is independent of the Python `.raimodel` export pipeline; RC/HRC/SAC report modeled sizes and serialize no artifact. Nothing here is on the inference path. |
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
| Python | Calibrated (GPTQ) export and draft-model preparation. `rai-convert` does round-to-nearest conversion without it. |

> `.cargo/config.toml` builds with `target-cpu=native` so the kernels use
> everything your CPU offers. That binary is for the machine that built it: run
> it on anything older and it dies with SIGILL on startup. Override the flag
> before you copy one anywhere —
> `RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release --locked` — which
> is exactly what the release archives are built with, at no measurable cost,
> because the AVX2 kernels are selected at runtime rather than at compile time.

## Build

```bash
cargo build --workspace --release --locked
```

The repository and v0.1.0 release are already public, and all five
`classeve-rai-*` v0.1.0 crates are on crates.io. This checkout contains
unreleased changes; building it does not change the published v0.1.0 artifacts.
See [installation](./docs/INSTALL.md) and the
[release-readiness gate](./docs/RELEASE_READINESS.md). No container image is
currently published; the `Dockerfile` builds an MCP stdio image from source by
default, and `docker build --target cli .` builds an inference-CLI image
carrying `rai-convert` and `rai-generate` instead.

Development checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
```

## Converting a model

RAI runs models in its own `.raimodel` format. Two paths produce it:

```bash
# Round-to-nearest — no Python, no torch
./target/release/rai-convert \
  --model /path/to/Mistral-7B-Instruct-v0.3 \
  --max-context 4096

# GPTQ, calibrated — needs the Python environment and a calibration corpus
python3 rai-infer/scripts/export_raimodel.py \
  --model /path/to/SmolLM-135M \
  --output smollm-135m-q4.raimodel
```

Both write the model file plus a `tokenizer.json` alongside it, and both refuse
architectures the format cannot represent. Check
[docs/MODELS.md](./docs/MODELS.md) before downloading a checkpoint;
[docs/INSTALL.md](./docs/INSTALL.md#converting-a-model) has every flag.
HuggingFace model and dataset revisions are not pinned by the exporters, so
record the exact revisions and arguments for reproducible work.

## Running

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

Set `RAYON_NUM_THREADS` to cap the inference worker count used by `rai-generate`
and `rai-chat`; it defaults to Rayon's own choice and does not affect
`rai-server`.

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

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/store` | Store a fact; returns an address-space crowding report |
| `POST /v1/recall` | Return the stored memory with the highest cosine similarity to the query |
| `POST /v1/intersect` | Retrieve at the normalized average of several concept addresses |
| `POST /v1/contradict` | Report how a candidate fact would change address-space crowding |
| `POST /v1/surprise` | Residual against the nearest stored key's value |
| `POST /v1/confidence` | The retrieval score and the tier it falls in |
| `POST /v1/snapshot` | Per-item crowding scores |
| `GET /v1/health` | Stored count, mean residual norm, capacity ratio |

The current backend is a cosine nearest-neighbour store, not a validated
resonance-training system. There is no training: no endpoint, no tool, and no
optimizer. Retrieval is nearest-vector by cosine similarity, and the confidence
tiers are a relabelling of that similarity rather than calibrated probabilities.

`/v1/contradict` (and the `rai_contradict` tool) reports **address-space
crowding**, not semantic contradiction. Each stored item is scored against its
nearest other neighbour; the endpoint compares those scores with and without the
candidate fact. Because appending a memory can only bring a neighbour closer, a
store can never raise another item's score — so under the current semantics this
cannot detect a contradiction, and an empty report is not evidence that a fact
agrees with memory. The same caveat applies to the interference report returned
by `/v1/store`.

The store holds **512 items by default** (`num_units`); a store beyond that
returns HTTP 409 with the limit in the message rather than a generic failure.
The service is **single-writer**: reads run concurrently, every mutation takes
an exclusive lock, and a durable store publishes in memory only after its
snapshot is on disk.

REST request bodies are limited to 64 KiB and individual text fields to 16 KiB
(bytes, not characters — the REST and MCP transports share one limit); concurrent
work is bounded, a global request ceiling protects the local process from
accidental overload, and any request still running after 30 seconds is abandoned
with HTTP 503. Ctrl-C shuts down gracefully so in-flight durable stores finish.
Local loopback usage needs no token. `rai-server` serves plain HTTP and therefore
refuses every non-loopback `RAI_HOST`, even when a token is configured. For
remote access, keep RAI on loopback behind a TLS-terminating reverse proxy and
have the proxy send a loopback `Host` value (`127.0.0.1:<RAI_PORT>`, or a
portless `localhost`; a port, when present, must match `RAI_PORT`). Host and
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

A variable whose value is not valid Unicode fails startup rather than being
ignored. `rai-server --help` prints the mode and variable summary;
`rai-server --version` prints the version.

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
│ Section N+2: lm_head (4-bit, when untied)   │  nibble-packed codes
└─────────────────────────────────────────────┘
```

Each layer section holds seven 4-bit projections — `q`, `k`, `v`, `o`, `gate`,
`up`, `down` — followed by two f32 RMSNorm weight vectors. Linear weights carry
per-group (128-column by default) f16 scale/zero parameters and are packed two
codes per byte in the layout the AVX2 kernels consume. The embedding table is
8-bit quantized. Scale/zero parameters are round-tripped through f16 at export
time so the reader and the exporter use the same stored values. The header
carries the architecture dimensions plus `rope_theta` and `norm_eps`; there is
no field for anything else, which is what makes the compatibility list in
[docs/MODELS.md](./docs/MODELS.md) what it is.

The loader performs one read of the whole file into heap memory, validates the
header and section bounds, and then hands out borrowed slices into that buffer.
This avoids additional copies between parsed sections, but loading still copies
the file from storage into the process heap. No retained benchmark establishes
a general performance advantage over `mmap`.

## More measurements

The headline table is at the top of this file. Beyond it, BENCHMARKS.md records
what shorter and longer runs cost on the same machine: 40–45-token generations
measured 29.4–33.0 tok/s, and a 301-token prompt prefilled in 6.33 s
(47.6 tok/s), which is the time-to-first-token to expect for long prompts on
four cores. Per-tensor quantization MSE ranged from 6.4e-06 (`q_proj`) to
6.2e-09 (embedding).

Self-speculative decoding was measured and is not a speedup on this model:
`--self-spec-layers 11 --self-spec-k 4` achieved a 2.2% draft acceptance rate
and ran at 5.6 tok/s against the 21.8 tok/s baseline. The feature is correct
and available; it is not presented as a win.

An older author-run measurement reported ~195 tokens/s for SmolLM-135M, but its
raw output and environment were not retained; it is kept in BENCHMARKS.md as
historical context, not as release evidence.

See [BENCHMARKS.md](./BENCHMARKS.md) for the full method and the
quantization-quality comparison.

## Project status

RAI is young software, released because it is useful and readable — not
because it is finished. Interfaces may change. Current state:

- One measured end-to-end conversion and decode run exists (TinyLlama-1.1B,
  BENCHMARKS.md). This repository does not retain a release qualification
  matrix or model fixtures across the rest of the supported list.
- The architecture coverage is deliberately narrow and enforced at conversion
  time. [docs/MODELS.md](./docs/MODELS.md) states what is out and what each
  addition would cost.
- Historical quantization experiments used Hessian-weighted output error
  against an FP16 reference (see BENCHMARKS.md); reproducible perplexity and
  quality sweeps are still needed.
- The optimized paths are x86-64. A scalar fallback exists; ARM NEON kernels
  are future work.
- The memory/reasoning service and RC/HRC/SAC compression paths are research
  prototypes; do not rely on their confidence, crowding, or modeled-size outputs
  as validated product guarantees. The memory service has no training of any
  kind, and its crowding report cannot detect semantic contradiction.

Issues and PRs are welcome. See [SECURITY.md](./SECURITY.md),
[SUPPORT.md](./SUPPORT.md), and [CONTRIBUTING.md](./CONTRIBUTING.md).

## About

Built and maintained by [ClassEve](https://classeve.com) — engineering for AI agents and developer tooling. Project page: [classeve.com/public/rai](https://classeve.com/public/rai).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
Copyright 2025-2026 ClassEve.
