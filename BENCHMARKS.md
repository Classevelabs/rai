# Benchmark record

This file has two parts. The **verified run** below was measured end to end on
a named machine on 2026-08-09 with the environment recorded in
`rai-infer/scripts/requirements-lock.txt`, and is the only section that should
be quoted as evidence. Everything after it is the older, author-reported
development record, kept for context and explicitly not reproduced.

## Reconciling the 0.2.2 numbers, 2026-08-18

The 0.2.2 changelog records TinyLlama decode at 2.0 tok/s before the rayon
dispatch fix and 10.9 tok/s after it — on the same machine that measured
21.8 tok/s below. Those figures sat unreconciled, so the run was repeated on
the 0.2.4 code with the **shipped `x86-64-v2` baseline** (the 2026-08-09 run
used `target-cpu=native`), the same greedy settings, and the same 91-token
generation. Ten runs on a machine whose background load came and went:

```
1.75  2.03  8.36  6.72  17.46  10.53  22.11  23.04  5.14  11.03   tok/s
```

The two runs that landed in a quiet window — **22.11 and 23.04** — reproduce
the 21.8-class figure on the shipped baseline; the rest is the same binary
under background load, and it brackets both 0.2.2 figures. So: the pre-fix 2.0
was the rayon dispatch regression compounding with load, the post-fix 10.9 was
a loaded-machine measurement rather than the machine's capability, and the
quiet-machine headline stands. The honest caveat stands with it: a 4-core
laptop under load can deliver a tenth of its quiet number.

## Verified runs, 2026-08-09

**Machine:** Intel Core i5-10300H (4 cores / 8 threads, AVX2 + FMA + F16C),
15.8 GB RAM, Windows 11. **Build:** `cargo build --release` with
`RUSTFLAGS="-C target-cpu=native"` (fat LTO), rustc 1.95.0, RAI 0.2.0. Python
environment pinned in `rai-infer/scripts/requirements-lock.txt`.

> These runs predate the change that made `x86-64-v2` the repository default,
> so reproducing them needs that `RUSTFLAGS` set explicitly. The ratios below
> are A/B comparisons of one binary against itself and do not depend on the
> baseline; the absolute figures were taken at `native`.

**On measurement noise.** This is a 4-core laptop that was shared with other
work for much of the session. Absolute throughput swung by up to 40% between
runs of identical code under load. Every ratio below is an interleaved A/B of
the same binary in the same minute, which is unaffected; absolute figures state
the conditions they were taken under. Where a number could not be taken on a
quiet machine, it says so.

### Models converted

| Model | Architecture | Source | `.raimodel` | Sections |
| --- | --- | --- | --- | --- |
| TinyLlama-1.1B-Chat | Llama, untied lm_head | 2,200 MB fp16 | **619.5 MB** | 25 |
| SmolLM2-1.7B-Instruct | Llama, tied | 3,422 MB fp16 | **963.0 MB** | 26 |
| Zephyr-7B-beta | Mistral, untied lm_head | 14,000 MB fp16 | **3,917.7 MB** | 35 |

### Conversion: `rai convert` versus the Python exporter

`rai convert` reads `.safetensors` directly and streams tensor by tensor, so
peak memory does not grow with the model.

| | `export_rtn.py` (torch) | **`rai convert`** |
| --- | --- | --- |
| TinyLlama-1.1B wall time | 188.8 s | **7.6 s** (24.8×) |
| TinyLlama-1.1B peak RSS | 4,980.9 MB | **22.9 MB** (217×) |
| Zephyr-7B wall time | cannot run — needs ~29 GB | **82.6 s** |
| Zephyr-7B peak RSS | — | **26.3 MB** |

Output is **byte-identical** between the two implementations: SHA-256
`B3B40DB6…2AF091FC` for TinyLlama-1.1B from both. Reaching that required
matching numpy's tie-break for ±0.0 zero-points in all-zero embedding groups;
1,714 bytes differed before the fix.

Converting a 7B model on a 16 GB machine is possible only because of the
streaming design — the Python path loads the whole checkpoint into RAM.

### Inference

Greedy decoding (`--temperature 0 --top-k 0 --top-p 1 --repetition-penalty 1`).

| | TinyLlama-1.1B | Zephyr-7B |
| --- | --- | --- |
| Decode, quiet machine | **21.8 tok/s** (n=3: 21.55 / 21.84 / 22.06) | not taken quiet |
| Decode, 49% background load | 15.5–16.4 tok/s (best of 5) | **2.96 tok/s** (n=3: 2.92 / 2.93 / 2.96) |
| Peak process RSS | 629 MB | ~4.0 GB |
| Model load, warm | 0.33 s | — |
| Model load, cold (first read from disk) | — | 48.3 s |

For comparison on the same machine, HuggingFace `transformers` 5.15 running the
same TinyLlama checkpoint in fp32 on CPU decoded at **4.3 tok/s** (n=2: 3.95 /
4.66) over an identical 91-token generation — RAI is **5.1× faster** and uses
629 MB against roughly 4.4 GB for the fp32 weights alone.

The 7B figure is bandwidth-dominated, which is why its variance is tight even
under load: 3.9 GB of weights are streamed per token.

### Roofline

Decode streams **549.5 MB per token** for TinyLlama-1.1B — the sum of the
attention projections (110.3 MB), MLP projections (404.4 MB), and the untied
4-bit lm_head (34.8 MB). The 69.6 MB embedding table is not streamed: one row
is read per token. `bw-bench` measured **26.4 GB/s** achievable read bandwidth
on this machine.

```
ceiling = 26.4 GB/s / 549.5 MB = 48.0 tok/s
21.8 tok/s = 45% of ceiling      33.0 tok/s (short runs) = 69%
```

Decode is bandwidth-bound by construction: 87–97% of a decode step is already
inside the weight-streaming GEMMs.

### Batched-GEMM rewrite (prefill)

Profiling attributed 88.5% of prefill to `w4a8_matmul`, which drove a batch
through the single-token kernel once per token and so repeated the per-row f16
scale decode, the prefetch loop, and the 4-bit unpacking once for every token.
The sequential per-token attention that looked like the obvious culprit was
11.1%.

| | Before | After |
| --- | --- | --- |
| Prefill, 308-token prompt, end to end | 1.0× | **~1.3×** (6 interleaved pairs: 1.13–1.48×) |
| `forward_batch` alone, warm | 12,766 / 15,370 ms | **8,222 / 9,297 ms** (1.55–1.65×) |
| Attention share of prefill | 11.1% | **4.3%** |
| Decode attention at position 512 | 26.3 ms | **16.8 ms** (1.57×) |
| Decode throughput | — | unchanged by design (decode never calls the batched path) |

Greedy output is **byte-identical** before and after, and the
sequential-versus-batched logit difference in `tests/model_invariants.rs` is
exactly **0**.

### Prompt-lookup speculative decoding

Drafts tokens by copying what followed the most recent occurrence of the
current suffix n-gram. No draft model, no draft forward pass, no training.
Measured with `--lookup-k 2`, interleaved against baseline at equal load:

| Workload | Baseline | Prompt-lookup | Ratio | Acceptance | Tokens/step |
| --- | --- | --- | --- | --- | --- |
| Repeat a 90-word passage (3 seeds) | 10.9–12.2 tok/s | **13.1–13.7 tok/s** | **1.12–1.20×** | 82–90% | 2.56–2.63 |
| Original creative writing (2 seeds) | 11.5–12.1 tok/s | 9.0–10.1 tok/s | **0.74–0.88×** | 21–24% | — |

It is a real gain when the output reuses context — summarisation, question
answering over a document, RAG, code editing — and a real loss otherwise, so it
ships **off by default**. Before the batched-GEMM rewrite the same benchmark
measured 0.93×: the technique was blocked by verification cost, not by drafting
quality.

### Measured and rejected

Recorded so nobody spends time re-deriving them:

- **Self-speculative early exit: 0.4% acceptance, ~15× slower** than plain
  decoding on TinyLlama-1.1B. No layer count or K wins, because an untrained
  early exit predicts the full model poorly. Removed from the CLI; the library
  implementation remains for use with a trained exit head.
- **Per-call activation allocations in `w4a8_matvec`**: all 177 allocations per
  token cost 19.2 µs, or **0.06% of a decode step**. Not worth removing.
- **f16 KV cache**: KV attention is 1.0% of decode at position 128 and 7.7% at
  512, so halving its bandwidth buys at most 4% for a numerics change.
- **Unconditional parallel decode attention**: 6× *slower* at position 8, where
  rayon's splitting costs more than the work. Enabled only above position 256.

### Output quality

Against an fp32 reference on identical greedy prompts, the 4-bit model
reproduced the substantive content: "The capital of France is" → *Paris*;
"Water boils at a temperature of" → *100°C (212°F)*; `def fibonacci(n):` → the
same recursive implementation, differing only in indent width. Zephyr-7B
explains Rayleigh scattering correctly. Continuations diverge after several
tokens, which is expected under greedy decoding once any single logit differs.
No perplexity sweep has been run, so no perplexity claim is made.

Per-tensor quantization error for TinyLlama-1.1B: 6.4e-06 … 6.8e-06 for
`q_proj`, 3.3e-06 for `down_proj`, 6.2e-09 for the 8-bit embedding.

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

MODEL=rai-infer/scripts/smollm-135m-q4.raimodel

# Per-operation forward-pass profiler
./target/release/profile-fwd --model "$MODEL"

# Memory bandwidth benchmark (--model enables the mmap read section)
./target/release/bw-bench --model "$MODEL"

# GEMM microbenchmark
./target/release/gemm-bench

# End-to-end decode timing
./target/release/rai run "$MODEL" \
  --tokenizer rai-infer/scripts/tokenizer.json \
  --prompt "The future of computing is" --max-tokens 128

# Compression kernels (criterion)
cargo bench -p rai-compress
```
