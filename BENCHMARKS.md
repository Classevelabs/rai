# Historical benchmark record

The numbers below are historical, author-reported measurements from development
on a described but not uniquely identified **4-core / 8-thread laptop-class
x86-64 CPU** (AVX2 + FMA + F16C, dual-channel DDR4, 8 MB L3). No raw output,
exact CPU model, operating-system image, commit SHA, model/dataset revisions,
or Python dependency lock was retained in this repository. They therefore must
not be presented as independently reproduced release evidence.

Results will vary with memory bandwidth, core count, compiler flags, and
thermal limits. The commands in [Attempting reproduction](#attempting-reproduction)
exercise the same code paths, but do not recreate the original environment.

Test model: **SmolLM-135M** (30 layers, hidden 576, vocab 49,152), exported to
`.raimodel` with GPTQ 4-bit linears (group size 128) and an 8-bit embedding
table.

## Decode speed

| Metric | Value |
| --- | --- |
| Decode speed (32-token generation) | **195 tok/s** |
| Decode speed (128-token generation) | 139 tok/s |
| Peak measured decode speed | 199.6 tok/s |
| Single-token forward pass (position 32) | 5.2 ms |
| Effective memory bandwidth during decode | 16.5 GB/s |
| Process RSS with model loaded | ~95 MB |
| Weight memory overhead beyond the packed file | 0 bytes |

Per-operation breakdown of one forward pass (30 layers, position 32, average
of 30 iterations):

| Operation | Time | Share |
| --- | --- | --- |
| SwiGLU MLP (gate + up + down, ×30) | 2563 µs | 49% |
| LM head (×1) | 976 µs | 19% |
| QKV projections (×30) | 724 µs | 14% |
| Output projections (×30) | 552 µs | 11% |
| Norms, RoPE, KV store, attention, residuals | 329 µs | 6% |

GEMM is ~93% of the forward pass — single-token decoding is memory-bandwidth
bound, which is exactly where the 4-bit packed format pays off.

### Kernel optimization journey

Cumulative effect of the kernel work, same hardware and model throughout:

| Stage | tok/s |
| --- | --- |
| Naive Rust (scalar loops) | 6 |
| + AVX2 SIMD dequantization | 45 |
| + Factored dequantization (integer dot + group correction) | 70 |
| + Fused QKV / gate-up dispatches | 95 |
| + W4A8 integer GEMM (PMADDUBSW) | 118 |
| + Heap weight store (vs mmap) | 148 |
| + Int8 LM head + software prefetch | 155 |
| + Pre-allocated buffers + allocator/OS tuning (historical author label) | 175 |
| + 256-bit PMADDUBSW inner loop | 184 |
| + Tuned prefetch distance + unchecked parameter reads | 192 |
| + Dynamic chunk sizing | **195** |

The table reports a 32.5× improvement over the naive implementation in the
original development runs. The W4A8 path quantizes f32 activations to i8 and is
not generally bit-identical to W4A32 or scalar arithmetic. No committed golden
log supports the earlier bit-identity claim; numerical equivalence must be
measured with explicit tolerances before it is claimed.

The exact allocator/OS tuning used for the 175 tok/s stage was not retained,
and the current loader does not explicitly request transparent huge pages. The
heap-versus-`mmap` comparison is likewise not backed by retained raw results.

## Model size and compression

SmolLM-135M (134.5M parameters), as reported for the original development
weights:

| Format | Bits/weight | Size | vs FP16 |
| --- | --- | --- | --- |
| FP16 (original) | 16.0 | 269.0 MB | 1.0× |
| INT8 | 8.0 | 134.5 MB | 2.0× |
| Uniform 4-bit (round-to-nearest) | 4.6 | 77.3 MB | 3.5× |
| GPTQ 4-bit | 4.2 | 71.4 MB | 3.8× |
| Shipped `.raimodel` (GPTQ-4b linears + 8-bit embedding + f32 norms) | — | **83 MB** | 3.25× |

The RC/HRC/SAC structures in `rai-compress` are research prototypes, not
serialized model formats. Their size helpers model some prior, channel,
outlier, scale, and zero-point values as FP16 even though the current in-memory
structures retain f64 values and perform no FP16 serialization roundtrip. Some
ratios also compare against an FP64 baseline rather than a typical FP16 source.
Those byte counts, ratios, and MSE values are theoretical estimates, not emitted
artifact measurements; FP16 conversion error is not included.

GPTQ calibration used 262,144 tokens of wikitext-2-raw-v1 (128 chunks × 2048
tokens, ~90 s of calibration). Quantizing all layers of the 135M model takes
roughly 8 minutes single-core; the cost is dominated by the Cholesky inverse
of each layer's Hessian.

## Quantization quality

Quality is measured by **Hessian-weighted output error** — `trace((W−Q)ᵀ(W−Q)H)/n`,
i.e. the error in the layer's *output* under the calibration distribution,
which is the quantity GPTQ optimizes and the one that correlates with model
quality. Compared against uniform 4-bit round-to-nearest on the same weights:

| Layer group | GPTQ-4bit improvement over uniform 4-bit |
| --- | --- |
| All measured layers (10/10 wins) | **2.8× lower output error** (average) |
| Attention projections (Q/K/V/O) | 9.6× average, up to 18.4× |
| MLP projections (gate/up/down) | 2.1× average |

Two honest caveats:

- GPTQ deliberately trades raw weight MSE (~0.8× of uniform, i.e. slightly
  worse) for output accuracy. Raw weight error is a misleading metric for
  quantization quality; output error is what matters.
- We have not yet published end-to-end perplexity sweeps. Quality so far is
  validated through the output-error measurements above plus qualitative
  generation checks. Perplexity benchmarking is planned.
- Raw calibration outputs and pinned model/dataset revisions are not committed,
  so the numeric improvement table remains author-reported rather than a
  release acceptance test.

## Attempting reproduction

Before running, record the exact commit, `rustc -Vv`, OS and CPU identity,
model and dataset revisions, exporter arguments, and `python -m pip freeze`.
The exporters currently resolve unpinned HuggingFace revisions and Python
packages, so results can drift even with the same visible command. Retain the
raw profiler, benchmark, and quality outputs with any new public claim.

```bash
# Export the test model (writes rai-infer/scripts/smollm-135m-q4.raimodel)
python3 rai-infer/scripts/export_raimodel.py \
  --model HuggingFaceTB/SmolLM-135M \
  --output rai-infer/scripts/smollm-135m-q4.raimodel

cargo build --workspace --release

# Per-operation forward-pass profiler (expects the model path above)
./target/release/profile-fwd

# Memory bandwidth benchmark
./target/release/bw-bench

# GEMM microbenchmark
./target/release/gemm-bench

# End-to-end decode timing
./target/release/rai-generate \
  --model rai-infer/scripts/smollm-135m-q4.raimodel \
  --tokenizer rai-infer/scripts/tokenizer.json \
  --prompt "The future of computing is" --max-tokens 128

# Compression kernels (criterion)
cargo bench -p rai-compress
```
