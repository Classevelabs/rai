# Benchmark record

This file has two parts. The **verified run** below was measured end to end on
a named machine on 2026-08-09 with the environment recorded in
`rai-infer/scripts/requirements-lock.txt`, and is the only section that should
be quoted as evidence. Everything after it is the older, author-reported
development record, kept for context and explicitly not reproduced.

## Verified run — TinyLlama-1.1B-Chat, 2026-08-09

**Machine:** Intel Core i5-10300H (4 cores / 8 threads, AVX2 + FMA + F16C),
15.8 GB RAM, Windows 11. **Build:** `cargo build --release` (`target-cpu=native`,
fat LTO), rustc 1.95.0, RAI 0.2.0.
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, hidden 2048, 32 heads
/ 4 KV heads, intermediate 5632, vocab 32,000, untied `lm_head`), exported with
`export_rtn.py` (round-to-nearest 4-bit linears, group size 128, 8-bit embedding).

### Conversion

| Metric | Value |
| --- | --- |
| Conversion time (whole model) | **47.0 s** |
| — quantizing 22 layers | 38.2 s |
| — embedding (8-bit) | 2.9 s |
| — `lm_head` (4-bit) | 2.5 s |
| Source checkpoint (fp16 safetensors) | 2,200 MB |
| Output `.raimodel` | **619.5 MB** (3.55× smaller) |
| Per-tensor quantization MSE | 6.4e-06 … 6.8e-06 (`q_proj`), 3.3e-06 (`down_proj`), 6.2e-09 (embedding) |

The calibrated path was exercised on the same checkpoint:
`export_fast.py --cal-chunks 4 --seq-len 512` took **1,780 s** (328 s
calibration + 1,436 s quantization) and produced a byte-identical-sized
619.5 MB / 25-section file. Both exports generate correct text; GPTQ is far
slower to produce and needs a calibration corpus, which is the trade it makes
for quality. This run used a deliberately small calibration set to bound the
time, so it is a working-path demonstration, not a quality claim against RTN.

### Inference, against HuggingFace on the same machine

Greedy decoding (`--temperature 0 --top-k 0 --top-p 1 --repetition-penalty 1`),
identical prompt, identical generated length (91 tokens).

| Metric | RAI 0.2.0 (4-bit) | transformers 5.15 fp32 (CPU) |
| --- | --- | --- |
| Decode speed, 91 tokens | **21.8 tok/s** (21.55 / 21.84 / 22.06, n=3) | 4.3 tok/s (3.95 / 4.66, n=2) |
| Relative speed | **5.1×** | 1× |
| Peak process RSS | **629 MB** | fp32 weights alone are ~4.4 GB |
| Model load time | 0.33 s | ~2 s |

Shorter generations run faster because attention cost grows with context:
40–45-token runs measured 29.4–33.0 tok/s on the same build. Prefill is
compute-bound rather than bandwidth-bound — a 301-token prompt took 6.33 s
(47.6 tok/s), which is the time-to-first-token to expect for long prompts on
four cores.

### Output quality

Against the fp32 reference on identical greedy prompts, the 4-bit model
reproduced the substantive content: "The capital of France is" → *Paris*;
"Water boils at a temperature of" → *100°C (212°F)*; `def fibonacci(n):` →
the same recursive implementation the fp32 model emits (differing only in
indent width). Continuations diverge after several tokens, which is expected:
under greedy decoding any single differing logit changes the rest of the text.
No perplexity sweep has been run, so no perplexity claim is made.

### Self-speculative decoding: measured, and not currently useful

Early-exit self-speculation (`--self-spec-layers 11 --self-spec-k 4`) on this
model achieved a **2.2% draft acceptance rate**, which made generation
**slower** (5.6 tok/s versus 21.8 tok/s baseline). The first-N-layers draft is
too weak a predictor of the full model without a trained early-exit head. The
feature remains available and correct, but it is not a speedup on this model
and should not be presented as one.

## Historical record (not reproduced)

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
not generally bit-identical to full-f32 or scalar arithmetic. No committed
golden log supports the earlier bit-identity claim; numerical equivalence must
be measured with explicit tolerances before it is claimed. (The GEMM entry
points are named `w4a8_*` accordingly.)

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
