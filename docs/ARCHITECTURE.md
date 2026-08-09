# Architecture

RAI is a Rust workspace with five crates:

| Crate | Role |
| --- | --- |
| `classeve-rai-infer` | `.raimodel` loader, AVX2/FMA/F16C inference kernels, transformer layers, KV cache, sampling, speculative decoding, and CLI/chat binaries. |
| `classeve-rai-compress` | Quantization and compression tooling, including GPTQ, adaptive residual coding, sparse outlier extraction, and bit-packing. |
| `classeve-rai-server` | Local REST and MCP server for the memory/reasoning layer. |
| `classeve-rai-core` | Embeddings, memory management, reasoning, confidence, interference, and surprise primitives. |
| `classeve-rai-rem-nra` | Resonance memory backend used by `rai-core`. |

## Runtime Flow

1. Export a HuggingFace model into `.raimodel` plus `tokenizer.json`.
2. Load the flat `.raimodel` file through `rai-infer`.
3. Decode with packed 4-bit linears and 8-bit embedding tables.
4. Generate through `rai-generate`, chat through `rai-chat`, or expose
   memory/reasoning through `rai-server`.

## Model Format

The `.raimodel` file is a flat little-endian binary:

- 64-byte header.
- Section table.
- Quantized embedding table.
- One section per transformer layer.
- Final normalization weights.
- Optional untied language-model head.

The loader validates section bounds before returning zero-copy slices into the
owned model buffer.

## Performance-Sensitive Code

The inference crate intentionally uses explicit indices and larger function
signatures in several hot paths. Those choices keep SIMD kernels direct,
allocation-free, and easier to audit against the binary layout. The crate-level
clippy allowances in `rai-infer` are limited to that kernel style; the workspace
still runs clippy with `-D warnings`.

## Server Modes

`rai-server` has two modes:

- REST mode for local HTTP clients.
- MCP stdio mode for agent/tool clients.

Embeddings default to a deterministic mock provider for local use. Set
`RAI_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use the OpenAI-backed
embedding provider.
