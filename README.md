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

Measured on an Intel i5-10300H (4 cores / 8 threads), 2026-08-09, RAI 0.2.0.
Full method, roofline, and the results that came out negative are in
[BENCHMARKS.md](./BENCHMARKS.md).

**Converting a checkpoint** — `rai convert` streams `.safetensors`, so peak
memory does not grow with the model:

| | `export_rtn.py` (PyTorch) | **`rai convert`** |
| --- | --- | --- |
| TinyLlama-1.1B | 188.8 s, 4,981 MB RAM | **7.6 s, 22.9 MB RAM** |
| Zephyr-7B | needs ~29 GB — will not run | **82.6 s, 26.3 MB RAM** |

Both produce byte-identical output. A 7B model converts on a 16 GB laptop.

**Running a model** — greedy decoding, same machine:

| | TinyLlama-1.1B (619 MB) | Zephyr-7B (3.9 GB) |
| --- | --- | --- |
| Decode | **21.8 tok/s** | **2.96 tok/s** |
| Peak RSS | 629 MB | ~4.0 GB |
| Load (warm) | 0.33 s | — |

HuggingFace `transformers` fp32 runs the same TinyLlama checkpoint at 4.3 tok/s
on this machine, so RAI is **5.1× faster in 1/7th the memory**. The batched-GEMM
rewrite made prefill **~1.3× faster**, and prompt-lookup decoding adds
**1.12–1.20×** on context-quoting workloads (off by default; it is slower on
original prose, and BENCHMARKS.md says by how much).

## Quickstart

```bash
# 1. Build (or take a release archive — see INSTALL.md)
cargo build --workspace --release --locked

# 2. Convert a checkpoint — no Python required
rai convert /path/to/Qwen2.5-0.5B-Instruct

# 3. Generate
rai run qwen2.5-0.5b-instruct-q4.raimodel \
  --chat-template chatml \
  --prompt "Explain photosynthesis in simple terms." \
  --max-tokens 64

# 4. Or open the local chat interface
rai serve qwen2.5-0.5b-instruct-q4.raimodel

# What is on this machine?
rai models .
```

Conversion writes `tokenizer.json` beside the model, and `rai run` picks it up
automatically. Instruction-tuned models need the chat template they were
trained on, or they emit end-of-sequence immediately and print nothing:
`chatml` for Qwen, `llama3` for Llama-3, `zephyr` for TinyLlama-Chat.

Prebuilt binaries are in [INSTALL.md](./INSTALL.md); full conversion options,
including the calibrated GPTQ path, are in
[docs/INSTALL.md](./docs/INSTALL.md#converting-a-model). The double-click
launchers in `launchers/` start the local interface without a terminal.

## Which models work

RAI runs the Llama-family decoder and the capabilities layered on top of it.
Anything it cannot represent is refused at conversion time, by name, before a
file is written — never silently mis-converted.

<!-- FINALISE: the Landing row is a placeholder. Per-head QK norm (Qwen3,
     OLMo2), logit softcapping (Gemma2) and possibly Gemma3-text are being
     implemented; move each family into Runs or Refused when that work reports.
     The same four are marked in docs/MODELS.md. -->

| | |
| --- | --- |
| **Runs** | **Qwen2 / Qwen2.5**, **Llama-3.1 / 3.2**, **Gemma**, Llama-2, Mistral-7B v0.1–v0.3, TinyLlama, SmolLM / SmolLM2, and fine-tunes of those (Zephyr-7B, OpenHermes-Mistral, Vicuna) |
| **Refused** | Mixtral and other mixture-of-experts models — use dense Mistral-7B instead; Phi / Falcon / GPT-NeoX / MPT / GPT-2, whose module tree is not Llama-shaped — use a Llama or Mistral model of the same size; any `rope_scaling` other than `default` or `llama3` |
| **Landing** | Qwen3, OLMo2, Gemma2, Gemma3 — verdict pending, alternatives listed in docs/MODELS.md |

Point `rai convert` at your folder, or press **Check** in Studio: both run the
same preflight, refuse before writing anything, and name the blocker. What to
use instead of a refused checkpoint, and what has to be in the folder, are in
**[docs/MODELS.md](./docs/MODELS.md)**. Qwen2.5-0.5B, Llama-3.2-1B and
gemma-2b-it were each converted and generated coherent text.

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
- **Speculative decoding.** Draft-model and prompt-lookup modes. Both accept or
  reject each drafted token against the target model's own sampled
  distribution, with a correction sample on rejection; a seeded statistical
  smoke test guards that path.
- **Local serving.** An HTTP chat server with a built-in web UI, plus a REST +
  MCP server so agentic tools (e.g. Claude Desktop, Claude Code) can use RAI
  as a tool backend.

## Workspace layout

| Crate | Purpose |
| --- | --- |
| `rai-infer` | The inference engine: `.raimodel` loader and writer, AVX2 W4A8 GEMM kernels, transformer layers (RMSNorm, RoPE, GQA, SwiGLU/GeGLU), KV cache, sampling, speculative decoding, and the `rai` binary (behind the default-on `cli` feature; `--no-default-features` builds the lean library) |
| `rai-compress` | Quantization and compression research toolkit. Its Rust GPTQ implementation is independent of the Python `.raimodel` export pipeline; RC/HRC/SAC report modeled sizes and serialize no artifact. Nothing here is on the inference path. |

**RAI is `rai-infer` and `rai-compress`.** The three crates below are a separate
memory/reasoning service that only shares this workspace. They are **not part of
the RAI product**, are not published to crates.io (0.1.0 yanked 2026-08-13,
`publish = false` set), and `rai-server` imports `rai-infer` zero times — it
cannot run a model.

| Not part of RAI | Purpose |
| --- | --- |
| `rai-server` | REST + MCP server for the memory/reasoning layer |
| `rai-core` | Memory, embedding and reasoning primitives used by `rai-server` |
| `rem-nra` | Resonance-memory backend used by `rai-core` |

The inference and memory-service paths are separate. `rai run` and `rai serve`
load `.raimodel` files through `rai-infer`; `rai-server` does not run those
models. Instead, its REST/MCP adapters call `rai-core`, which obtains an
embedding from the configured provider and stores/queries state through
`rem-nra`. `AppState` serializes REST stores and opted-in MCP stores to
`RAI_DATA_PATH`.

## Requirements

| Requirement | Details |
| --- | --- |
| Rust | 1.87+; the repository pins 1.95.0, edition 2021 |
| CPU | x86-64 with AVX2, FMA, and F16C for optimized paths; scalar fallbacks otherwise |
| OS | Linux, Windows, or macOS |
| GPU at runtime | **Not required** |
| Python | Calibrated (GPTQ) export and draft-model preparation only. `rai convert` does round-to-nearest conversion without it. |

> `.cargo/config.toml` pins the x86-64 build to the **x86-64-v2** baseline —
> the same floor the release archives use — so a binary you build here runs on
> any x86-64 machine you copy it to. This costs nothing measurable: the AVX2,
> FMA, and F16C kernels are chosen at runtime, not by the compile-time
> baseline. aarch64 (Apple Silicon, ARM servers) is left at the toolchain
> default. To tune a build to the machine in front of you — for a local
> benchmark, never for a binary you hand to anyone else —
> `RUSTFLAGS="-C target-cpu=native" cargo build --release --locked`; that
> binary dies with SIGILL on any older CPU.

## Build

```bash
cargo build --workspace --release --locked
```

That produces `rai` and `rai-server`, plus the deprecated `rai-convert`,
`rai-generate`, and `rai-chat` wrappers. See [installation](./docs/INSTALL.md)
for source installs and the container. No container image is published; the
`Dockerfile` builds an MCP stdio image from source by default, and
`docker build --target cli .` builds an image carrying the `rai` CLI instead.

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
rai convert /path/to/Mistral-7B-Instruct-v0.3 --max-context 4096

# GPTQ, calibrated — needs the Python environment and a calibration corpus
python3 rai-infer/scripts/export_raimodel.py \
  --model /path/to/SmolLM-135M \
  --output smollm-135m-q4.raimodel
```

Both write the model file plus a `tokenizer.json` alongside it, and both refuse
architectures the format cannot represent. They do not cover the same models:
`rai convert` writes container v2 and handles Qwen2/2.5, Llama-3.1/3.2, and
Gemma, while the Python exporters write container v1 and refuse all three.
Check [docs/MODELS.md](./docs/MODELS.md) before downloading a checkpoint;
[docs/INSTALL.md](./docs/INSTALL.md#converting-a-model) has every flag.
HuggingFace model and dataset revisions are not pinned by the Python exporters,
so record the exact revisions and arguments for reproducible work.

## Running

### Generate text

```bash
rai run smollm-135m-q4.raimodel \
  --prompt "The future of computing is" \
  --max-tokens 64 \
  --temperature 0.7
```

`--tokenizer` defaults to the `tokenizer.json` written beside the model at
conversion time; pass it explicitly only for a model that was moved away from
its tokenizer.

Sampling controls: `--temperature`, `--top-k`, `--top-p`,
`--repetition-penalty`, `--seed`. Test-time compute ("pondering") strategies:
`--ponder-strategy cfg|ensemble|cfg-ensemble|adaptive` with
`--guidance-scale`, `--ensemble-n`, `--noise-sigma`, `--entropy-threshold`.

Set `RAYON_NUM_THREADS` to cap the inference worker count used by `rai run` and
`rai serve`; it defaults to Rayon's own choice and does not affect `rai-server`.

### Speculative decoding

Two modes, mutually exclusive. Both verify against the target model and are
gated on exact sampling (`--top-k 0 --top-p 1 --repetition-penalty 1`), so
verification uses the distribution the target actually produced.

```bash
# Draft-model speculation: a small model proposes, the big model verifies
rai run mistral-7b-q4.raimodel \
  --draft mistral-draft-100m-q4.raimodel \
  --draft-k 6 \
  --top-k 0 --top-p 1 --repetition-penalty 1 \
  --prompt "Explain speculative decoding in one paragraph."

# Prompt-lookup: the draft is copied from the context, so there is no draft
# model and no draft forward pass
rai run tinyllama-q4.raimodel \
  --lookup-k 2 \
  --top-k 0 --top-p 1 --repetition-penalty 1 \
  --prompt "Summarise the passage above."
```

Draft and target must share a tokenizer. Prompt-lookup is off by default: it is
a gain only when the output reuses the context, and BENCHMARKS.md records both
the gain and the loss. `rai-infer/scripts/train_draft.py` is an experimental
distillation helper for compatible Mistral-family teachers; its throughput
projections are not release benchmarks.

### Chat over HTTP

```bash
rai serve smollm-135m-q4.raimodel --port 8090
```

Open `http://localhost:8090` for the built-in web UI, or POST to `/api/chat`
(JSON) for programmatic access.
`--chat-template auto|none|few-shot|mistral|llama3|chatml|zephyr` selects prompt
formatting. The chat server binds to `127.0.0.1` only and limits request bodies
to 64 KiB.

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
See [installation](./docs/INSTALL.md) and [operations](./docs/OPERATIONS.md)
for the full runbooks.

## The `.raimodel` format

A single flat, little-endian binary file:

```text
┌─────────────────────────────────────────────┐
│ Header (64 bytes at v1, 128 at v2)          │  magic "RAIM", version,
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
`up`, `down` — then any f32 bias vectors the header's `bias_mask` declares, then
two f32 RMSNorm weight vectors. Linear weights carry per-group (128-column by
default) f16 scale/zero parameters and are packed two codes per byte in the
layout the AVX2 kernels consume. The embedding table is 8-bit quantized.
Scale/zero parameters are round-tripped through f16 at export time so the reader
and the exporter use the same stored values.

Version 1 holds the architecture dimensions, `rope_theta`, and `norm_eps`.
Version 2 extends the header to 128 bytes and adds the activation code, the
`llama3` RoPE rescaling parameters, the bias mask, and the embedding scale — the
four fields that brought Qwen2/2.5, Llama-3.1/3.2, and Gemma inside the
supported set. A converter emits v1 whenever none of them is needed, so files
that converted before v2 still produce the same bytes. There is no field for
anything beyond that, which is what makes the compatibility list in
[docs/MODELS.md](./docs/MODELS.md) what it is.

The loader performs one read of the whole file into heap memory, validates the
header and section bounds, and then hands out borrowed slices into that buffer.
This avoids additional copies between parsed sections, but loading still copies
the file from storage into the process heap. No retained benchmark establishes
a general performance advantage over `mmap`.

## More measurements

The headline table is at the top of this file. Beyond it, BENCHMARKS.md records
the roofline analysis (decode reaches 45% of this machine's 26.4 GB/s memory
ceiling), the batched-GEMM prefill rewrite, and the per-tensor quantization
error, which ranged from 6.4e-06 (`q_proj`) to 6.2e-09 (the 8-bit embedding).

It also records what was measured and rejected, so nobody re-derives it:
self-speculative early exit reached 0.4% draft acceptance and ran roughly 15×
slower than plain decoding, so it is not in the CLI — the library
implementation remains for use with a trained exit head. An older author-run
measurement reported ~195 tokens/s for SmolLM-135M, but its raw output and
environment were not retained; it is kept as historical context, not as release
evidence.

See [BENCHMARKS.md](./BENCHMARKS.md) for the full method and the
quantization-quality comparison.

## Project status

RAI is pre-1.0 and interfaces may change. What that means concretely:

- One end-to-end conversion and decode run is measured (TinyLlama-1.1B,
  BENCHMARKS.md). The rest of the supported list is verified by architecture
  and, for Qwen2.5-0.5B, Llama-3.2-1B, and gemma-2b-it, by a coherent
  generation — not by a retained qualification matrix.
- Architecture coverage is deliberately narrow and enforced at conversion time.
  [docs/MODELS.md](./docs/MODELS.md) states what is out and what each addition
  would cost.
- Quantization quality is measured as Hessian-weighted output error
  (BENCHMARKS.md). No perplexity sweep has been run, so no perplexity claim is
  made.
- The optimized paths are x86-64. A scalar fallback exists; ARM NEON kernels
  are future work.
- The memory/reasoning service and the RC/HRC/SAC compression paths are
  research prototypes. Their confidence, crowding, and modeled-size outputs are
  not validated guarantees.

Issues and PRs are welcome. See [SECURITY.md](./SECURITY.md),
[SUPPORT.md](./SUPPORT.md), and [CONTRIBUTING.md](./CONTRIBUTING.md).

## About

Built and maintained by [ClassEve](https://classeve.com) — engineering for AI agents and developer tooling. Project page: [classeve.com/public/rai](https://classeve.com/public/rai).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
Copyright 2025-2026 ClassEve.
