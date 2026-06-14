# RAI

**A CPU-only LLM inference engine in pure Rust.** RAI runs 4-bit quantized
language models with hand-written AVX2 kernels — no GPU, no Python runtime, no
PyTorch, no GGML, no BLAS. Load a `.raimodel` file and generate text on any
modern x86-64 laptop.

Built by [Classeve](https://classeve.com). Licensed under Apache-2.0.

## Why

Most LLM inference stacks assume a GPU, a CUDA toolchain, or a heavyweight ML
framework. RAI takes the opposite bet: a small, auditable Rust workspace where
every matrix multiply is a hand-written SIMD kernel and the entire runtime
dependency tree fits in a `Cargo.lock`.

- **CPU-only by design.** AVX2 + FMA + F16C — hardware that has shipped in
  every mainstream x86 CPU since ~2013.
- **4-bit weights, dequantized in registers.** Weights stay packed in memory;
  unpacking happens on-the-fly inside the GEMM inner loop. No FP32 weight copy
  ever exists in RAM.
- **One flat model file.** The `.raimodel` format is a single binary blob with
  a 64-byte header — trivially parseable, zero-copy loadable.
- **Speculative decoding.** Draft-model and self-speculative (first-N-layers)
  modes, with mathematically exact verification against the target model.
- **Local serving.** An HTTP chat server with a built-in web UI, plus a REST +
  MCP server so agentic tools (e.g. Claude Desktop, Claude Code) can use RAI
  as a tool backend.

## Workspace layout

| Crate | Purpose |
| --- | --- |
| `rai-infer` | The inference engine: `.raimodel` loader, AVX2 W4A8/W4A32 GEMM kernels, transformer layers (RMSNorm, RoPE, GQA, SwiGLU), KV cache, sampling, speculative decoding, CLI + HTTP chat binaries |
| `rai-compress` | Weight quantization and compression: GPTQ with Hessian-guided error compensation, adaptive bit allocation, bit-packing |
| `rai-server` | REST + MCP server exposing the RAI memory/reasoning layer to HTTP clients and MCP-capable agents |
| `rai-core` | Memory, embedding, and reasoning primitives used by `rai-server` |
| `rem-nra` | Resonance-memory backend used by `rai-core` |

## Documentation

- [Installation](./docs/INSTALL.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Benchmarks](./BENCHMARKS.md)
- [Release procedure](./docs/RELEASE.md)
- [Security policy](./SECURITY.md)

## Requirements

| Requirement | Details |
| --- | --- |
| Rust | Stable toolchain, edition 2021 |
| CPU | x86-64 with AVX2, FMA, and F16C (Intel Haswell 2013+, AMD Zen 2019+) |
| OS | Linux, Windows, or macOS |
| GPU at runtime | **Not required** |
| Python | Only for the optional model-export scripts (one-time conversion) |

> `.cargo/config.toml` builds with `target-cpu=native` so the kernels use
> everything your CPU offers. Remove that flag if you need portable binaries.

## Build

```bash
cargo build --workspace --release
```

## Install

Install the end-user inference binaries (`rai-generate`, `rai-chat`,
`profile-fwd`, `gemm-bench`, and `bw-bench`) directly from GitHub:

```bash
cargo install --git https://github.com/classeve-public/rai \
  --package classeve-rai-infer \
  --locked
```

Install the local REST/MCP server:

```bash
cargo install --git https://github.com/classeve-public/rai \
  --package classeve-rai-server \
  --locked
```

For local development from a checkout:

```bash
cargo install --path rai-infer --locked
cargo install --path rai-server --locked
```

Development checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

## Get a model

RAI runs models in its own `.raimodel` format. The export scripts under
`rai-infer/scripts/` convert any LLaMA/Mistral-family HuggingFace model
(SmolLM, Mistral-7B-Instruct, LLaMA-2/3, …). They need Python with `torch`,
`transformers`, `datasets`, and `numpy`; a CUDA GPU speeds up calibration but
is not required.

```bash
# GPTQ 4-bit export (calibrated, best quality)
python3 rai-infer/scripts/export_raimodel.py \
  --model HuggingFaceTB/SmolLM-135M \
  --output smollm-135m-q4.raimodel

# Round-to-nearest export (no calibration, fastest, slightly lower quality)
python3 rai-infer/scripts/export_rtn.py \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

Both write the model file plus a `tokenizer.json` alongside it.

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

Draft and target must share a tokenizer. Verification is exact: output
quality is identical to standard decoding, only faster.
`rai-infer/scripts/train_draft.py` distills a ~100M-parameter draft model
from any teacher.

### Chat over HTTP

```bash
./target/release/rai-chat \
  --model smollm-135m-q4.raimodel \
  --tokenizer tokenizer.json \
  --port 8090
```

Open `http://localhost:8090` for the built-in web UI, or POST to `/api/chat`
(JSON) for programmatic access. `--chat-template auto|mistral|llama3|few-shot`
selects prompt formatting.

### REST + MCP server

`rai-server` exposes the RAI memory/reasoning layer:

```bash
# REST mode (default port 3000; configure with RAI_HOST / RAI_PORT)
./target/release/rai-server

# MCP mode on stdio — for MCP clients such as Claude Desktop or Claude Code
./target/release/rai-server mcp
```

REST endpoints: `POST /v1/store`, `/v1/recall`, `/v1/intersect`,
`/v1/contradict`, `/v1/surprise`, `/v1/confidence`, `/v1/train`,
`/v1/snapshot`, and `GET /v1/health`.

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

Embeddings default to a built-in mock provider; set
`RAI_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use an
OpenAI-compatible embedding API instead.

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
f16 scale/zero parameters, packed two codes per byte in the exact layout the
AVX2 kernels consume. The embedding table is 8-bit quantized. Scale/zero
parameters are round-tripped through f16 at export time so the Rust
dequantization bit-exactly matches the Python exporter.

The loader reads the whole file into heap memory (requesting transparent huge
pages where the OS supports them) and hands out zero-copy slices — measurably
faster than `mmap` for bandwidth-bound decoding.

## Performance

On a 4-core/8-thread laptop-class x86-64 CPU, SmolLM-135M (83 MB at 4-bit)
decodes at **~195 tokens/s**. Quantized-kernel decoding is memory-bandwidth
bound, and the kernels sustain a large fraction of practical DRAM bandwidth.

See [BENCHMARKS.md](./BENCHMARKS.md) for methodology, the full numbers, and
compression-quality measurements.

## Project status

RAI is an active initial public release from Classeve. The Rust workspace,
CPU inference path, quantization toolkit, REST/MCP server, tests, benchmarks,
and CI gates are public and maintained. Current production-readiness boundaries:

- Small models (135M class) are exercised end to end; Mistral-7B-family export
  and inference paths are available and continue to be hardened.
- Quantization quality is measured by Hessian-weighted output error against an
  FP16 reference (see BENCHMARKS.md). End-to-end perplexity sweeps are still a
  planned benchmark addition, not a claimed result.
- Optimized inference targets x86-64 CPUs with AVX2, FMA, and F16C. Scalar
  fallbacks exist for portability, while dedicated ARM NEON kernels remain on
  the roadmap.
- Public APIs may evolve before `1.0`; release notes will call out breaking
  changes.

Security reports, issues, and pull requests are welcome. See
[SECURITY.md](./SECURITY.md), [SUPPORT.md](./SUPPORT.md), and
[CONTRIBUTING.md](./CONTRIBUTING.md).

## About

Built and maintained by [Classeve](https://classeve.com) — engineering for AI agents and developer tooling. Project page: [classeve.com/public/rai](https://classeve.com/public/rai).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
Copyright 2025-2026 Classeve.
