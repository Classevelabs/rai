# Architecture

RAI is a Rust workspace with five crates. Two independent paths run through
it: inference (`rai-infer`) and the memory/reasoning service (`rai-server` →
`rai-core` → `rem-nra`). They share no runtime state. `rai-server` does not run
`.raimodel` files.

| Crate | Role |
| --- | --- |
| `classeve-rai-infer` | `.raimodel` loader and writer, AVX2/FMA/F16C W4A8 GEMM kernels, transformer layers, KV cache, sampling, pondering, speculative decoding, and the `rai` binary. |
| `classeve-rai-compress` | Quantization and compression research toolkit: an independent Rust GPTQ implementation, plus RC/HRC/SAC adaptive residual coding, sparse outlier extraction, and bit-packing. Not on the inference path and not part of the `.raimodel` export pipeline. |
| `classeve-rai-server` | Local REST and MCP server for the memory/reasoning layer. |
| `classeve-rai-core` | Embeddings, memory management, composition, confidence, interference, and surprise primitives. |
| `classeve-rai-rem-nra` | Resonance memory backend (NRA address/value store plus REM key/value store) used by `rai-core`. |

## Inference

### Runtime flow

1. Convert a HuggingFace checkpoint into `.raimodel` plus `tokenizer.json` —
   `rai convert` for round-to-nearest, the Python exporters for GPTQ. Their
   architecture preflights differ; see [MODELS.md](./MODELS.md).
2. Load the flat `.raimodel` file through `rai-infer`.
3. Decode with packed 4-bit linears and an 8-bit embedding table.
4. Generate with `rai run`, chat with `rai serve`.

### Model format

The `.raimodel` file is a flat little-endian binary:

- Header: magic `RAIM`, format version, architecture dimensions, `rope_theta`,
  `norm_eps`, and the quantization configuration — 64 bytes at container v1.
  Version 2 extends it to 128 bytes and adds the activation code, the `llama3`
  RoPE rescaling parameters, the per-projection bias mask, and the embedding
  scale. A converter emits v1 whenever none of those is needed, so pre-v2
  checkpoints still produce identical bytes.
- Section index table: 16 bytes per section, offset and size.
- Section 0: 8-bit quantized embedding table.
- Sections 1..N: one per transformer layer, each holding seven 4-bit
  projections (`q`, `k`, `v`, `o`, `gate`, `up`, `down`), then one f32 bias
  vector per bit set in the header's `bias_mask`, then two f32 RMSNorm weight
  vectors.
- Section N+1: final RMSNorm weights (f32).
- Section N+2: 4-bit `lm_head`, present when the head is not tied to the
  embedding table.

There is no field for anything else. That is why the supported-architecture
list is what it is.

The loader reads the whole file into one heap buffer, validates the header,
the section count, section contiguity, per-section sizes against the declared
dimensions, and every quantization scale and norm weight for finiteness. It
then hands out borrowed slices into that buffer. Accepting a file is exactly
the guarantee the kernels rely on: `format.rs` imports `gemm::MAX_GROUPS` and
`layers::MAX_ROPE_TABLE_BYTES` rather than redefining them, so a loaded model
can never exceed the GEMM group capacity or the RoPE table budget.

### Kernels

`layers.rs` implements RMSNorm as `x / rms * w`, rotary position embeddings
built from a single theta with an optional `llama3` frequency rescale, full
causal attention with grouped-query head mapping, and a gated MLP with either a
SiLU (SwiGLU) or a GeLU-tanh (GeGLU) gate. Each has a scalar path and an
AVX2+FMA path selected at run time by `has_avx2()`.

The GEMM entry points are named `w4a8_*` (`w4a8_matvec`, `w4a8_fused_qkv`,
`w4a8_fused_gate_up`, `w4a8_matmul`): they quantize f32 activations to int8
before the integer dot product. They are therefore not bit-identical to full
f32 arithmetic, and no equivalence claim is made without measured tolerances.

### Feature split

`rai-infer` separates the inference library from the command-line surface.
`--no-default-features` compiles the inference library — format reader,
kernels, model, sampling, speculative decoding — against `half`, `rayon`,
`anyhow`, and `rand` alone. The default-on `cli` feature adds `clap`,
`tokenizers`, `tiny_http`, `serde`, `serde_json`, and `memmap2`; every binary
(`rai`, the deprecated `rai-convert`/`rai-generate`/`rai-chat` wrappers, and the
`profile-fwd`, `gemm-bench`, `bw-bench` dev tools) requires it, as do the
checkpoint converter and the tokenizer-aware `ChatTemplate::auto_detect` and
`ChatTemplate::from_str_arg` helpers.

Each `rai` subcommand's implementation lives in `src/cli/`, not in `src/bin/`,
so argument validation and the model/tokenizer resolution rules are unit
testable rather than trapped inside a `fn main`.

### Performance-sensitive code

The inference crate uses explicit indices and wide function signatures in
several hot paths. Those choices keep the SIMD kernels direct, allocation-free,
and auditable against the binary layout. The crate-level clippy allowances in
`rai-infer` are limited to that kernel style; the workspace still runs clippy
with `-D warnings`. Every `unsafe` AVX2 kernel carries a `# Safety` block
naming the invariants its caller asserts.

## Memory service

### Server modes

`rai-server` has two modes:

- REST mode for local HTTP clients, bound to loopback.
- MCP stdio mode for agent/tool clients.

Both expose the same eight operations: `/v1/store`, `/v1/recall`,
`/v1/intersect`, `/v1/contradict`, `/v1/surprise`, `/v1/confidence`,
`/v1/snapshot`, and `/v1/health`. There is no training endpoint and no
optimizer in any build; `POST /v1/train` was removed rather than left as a
permanent 501, and a test asserts the route returns 404.

### Concurrency

`MemoryManager` holds one `Arc<RwLock<Inner>>` covering every structure a
mutation must publish together — the NRA address/value store, the REM
key/value store, and the parallel text labels. Reads (recall, health, snapshot,
length) take the shared guard and run concurrently; every mutation takes the
exclusive guard, making the service single-writer. The embedding bridge's text
index is published inside the same critical section. A durable store stages,
writes its snapshot to disk, and publishes in memory only on success.

### What the reasoning primitives report

The backend is a cosine nearest-neighbour store. Retrieval is nearest-vector by
cosine similarity, and the confidence tiers relabel that similarity rather than
calibrating a probability. `/v1/contradict` and the interference report
returned by `/v1/store` measure **address-space crowding**: each stored item is
scored against its nearest other neighbour, with and without the candidate
fact. Appending a memory can only bring a neighbour closer, so these cannot
detect a semantic contradiction, and an empty report is not evidence of
consistency.

No ODE integrator, attractor search, or basin analysis exists in any build. The
response fields that reported such diagnostics (`RetrievalResult::steps` and
`grad_norm`, `ConfidenceExplanation::grad_norm` / `num_attractors` /
`basin_spread`, the `ConfidenceLevel::Ambiguous` tier) were removed because no
integrator produced them. `energy` remains — it is a real leave-one-out
crowding score.

### Embeddings

Embeddings default to a deterministic mock provider intended for tests and
demonstrations; startup warns whenever it is active. Set
`RAI_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use the OpenAI-backed
provider. Any other value fails startup rather than falling back to mock
embeddings.
