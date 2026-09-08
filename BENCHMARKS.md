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
A perplexity sweep has since been run; see **Quantization cost, measured** below.

Per-tensor quantization error for TinyLlama-1.1B: 6.4e-06 … 6.8e-06 for
`q_proj`, 3.3e-06 for `down_proj`, 6.2e-09 for the 8-bit embedding.

## Quantization cost, measured (2026-09-08)

Everything else in this file is a speed. Speed is the easy half: a 4-bit model
that generates nonsense is very fast. This section is the other half, and until
it existed this file said — correctly — that no perplexity sweep had been run
and that no perplexity claim was therefore made.

### Method

`rai perplexity`, which implements `llama-perplexity`'s method so the two are a
comparison rather than two numbers:

1. The corpus is tokenized once, no special tokens. It is continuous prose, and
   a BOS inserted at every chunk boundary would score a position the text does
   not contain.
2. Non-overlapping chunks of `--context` tokens.
3. One batched forward pass per chunk, from an empty KV cache.
4. Only the **second half** of each chunk is scored, so every scored token has
   at least `context / 2` tokens of real context behind it.
5. `exp(mean NLL)` over every scored token, accumulated in `f64`.

At `--context 512 --max-chunks 40`: **20,480 tokens evaluated, 10,200 scored.**
Both engines report identical counts, which is the check that they scored the
same positions rather than merely producing similar-looking numbers.

**Every perplexity here carries its standard error.** A perplexity is a sample
mean; over 10,200 tokens the standard error puts about **±2.5%** on any single
figure, so these are quoted to two decimals and not four. Both
`rai perplexity` and `scripts/reference_ppl.py` compute and report it.

**Machine.** A 16-vCPU cloud container with AVX2, FMA and F16C. The platform
does not expose the CPU model, so it is not stated. RAI was built at the
shipped `x86-64-v2` baseline, not `target-cpu=native`: this has to measure what
a user downloads. The fp16 references ran on an NVIDIA L4 — they are a
correctness ceiling, not a speed comparison, and no CPU-versus-GPU claim is
made from them.

**Corpus.** wikitext-2 raw test, materialized once so every engine reads
identical bytes. SHA-256
`bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`, 298,938
tokens under the Qwen2.5 tokenizer.

### Against llama.cpp, at the same file size

This is the comparison that decides anything. A quantizer can always look good
by spending more bits, so a perplexity is only meaningful next to the size of
the file that produced it.

**SmolLM2-1.7B**, wikitext-2, context 512, 40 chunks, every build scoring the
same 10,200 tokens:

| build | file | bits/weight | perplexity |
| --- | ---: | ---: | ---: |
| llama.cpp Q4_K_M | 1,006.7 MiB | 4.94 | **8.52** |
| RAI calibrated, `--group-size 64` | **966.4 MiB** | 4.74 | **8.59** |
| llama.cpp Q4_0 | 944.8 MiB | 4.63 | 8.82 |
| RAI calibrated, `--group-size 128` | **918.4 MiB** | 4.50 | 8.80 |
| RAI `rai convert`, the shipped default | **918.4 MiB** | 4.50 | 10.30 |

**What is and is not resolvable here.** llama.cpp reports its Q4_K_M figure as
8.5191 ± 0.21393, which is ±2.51%; RAI's own figures carry about the same. So
of the differences above, only two are larger than the measurement error: the
shipped round-to-nearest default is genuinely behind everything else (+20.9%),
and RAI's calibrated build at group size 128 is genuinely behind Q4_K_M
(+3.25%) while being 8.8% smaller.

The interesting row — RAI calibrated at group size 64 against Q4_K_M, +0.81% at
4% less disk — **sits inside the error bar, and no claim is made from it in
either direction.** Forty chunks was enough to establish a 30% loss and is not
enough to establish a 1% one. Settling that needs the full wikitext-2 test
split, which is what the literature scores and what this file will carry before
any parity claim is made.

### Calibrated conversion, in the converter (2026-09-08)

`rai convert --calibration-text <file>` quantizes against what each projection
actually receives instead of against the range its weights happen to span. It
runs the checkpoint over the calibration text, records the per-input-channel
activation energy entering each of the four projection groups, and chooses each
group's scale and zero to minimise the error weighted by that energy.

No Python, no GPU, and no change to the container: the calibrated file is the
same format, the same group size and the same size on disk as the
round-to-nearest one, so nothing about loading or decoding it differs.

**SmolLM2-1.7B**, wikitext-2, context 512, 40 chunks, calibrated on 16
sequences of 512 tokens from the **train** split and scored on **test**:

| build | file | perplexity | cost over fp16 | conversion |
| --- | ---: | ---: | ---: | ---: |
| fp16 reference | — | 7.9080 | — | — |
| `rai convert` round-to-nearest | 918.4 MiB | 10.3018 | +30.3% | 20 s |
| **`rai convert --calibration-text`** | **918.4 MiB** | **9.2224** | **+16.6%** | 462 s |
| `scripts/export_raimodel.py` (GPTQ, needs a GPU) | 918.4 MiB | 8.7958 | +11.2% | ~60 min |

**10.5% lower perplexity than the shipped default at a byte-for-byte identical
file size**, recovering 72% of the distance to the GPU-only GPTQ path.

It stops short of that path on purpose. Full GPTQ propagates each column's
quantization error into the columns not yet quantized, which needs the Cholesky
factor of the inverse Hessian — an O(cols³) factorization that for a 7B MLP is
a 14336×14336 matrix, hours of CPU, and 1.6 GB of working set per projection.
The activation energy is the *diagonal* of the same Hessian: it costs one pass
over the activations instead, which is what makes calibration something a
converter can do by default rather than something a GPU has to be found for.

#### What it does to the model's output

A perplexity is an average, and an average can improve while the model still
behaves badly. These are greedy generations, so the two builds differ as models
and not as two random draws.

Asked to continue "The history of the printing press begins", the
round-to-nearest build restarts its own sentence:

> …was a major step in the history of printing and the history of books. The
> invention of the printing press was a major step in the history of printing…

and the calibrated build does not:

> …by Johannes Gutenberg in 1439. The printing press was a major step forward
> in the dissemination of knowledge. The printing press allowed for the mass
> production of books, which in turn allowed for the mass dissemination of
> knowledge.

Asked for the three largest cities in Japan, round-to-nearest answers "Tokyo,
Osaka, and Nagoya" and then restates itself — "The second largest city is
Osaka, which is the second largest city in Japan". The calibrated build answers
"Tokyo, Yokohama and Osaka", which is correct, and does not.

Both builds write correct Python for `def fibonacci(n):`. The repetition and the
one corrected fact are the visible half of the same 10.5%.

#### What it costs and what it refuses

Calibration reads the checkpoint a second time and runs a forward pass over the
calibration text, so conversion goes from seconds to minutes. Peak memory grows
with the calibration size rather than staying flat: the residual stream for
every calibration token is held at once, which is 8,192 × `hidden_size` floats
at the default settings.

It **refuses** rather than approximating. Mixture-of-experts routing,
sandwich-normed and post-norm checkpoints, and per-head QK norms each produce a
named error telling the user to convert without `--calibration-text`. Which
expert sees which token, or how a QK norm reshapes attention, is part of the
model: a pass that skipped it would collect statistics for a model that does not
exist, and the resulting file would load cleanly and be quantized against
nothing. Llama, Mistral, SmolLM2 and Qwen2/2.5 calibrate today; Qwen3 and Gemma3
are refusals, not silent wrong answers.

### The result depends heavily on model size, and the small models mislead

Quantization damage shrinks as models grow, and every figure above is from a
model far smaller than the ones the engine is used on:

| model | fp16 | `rai convert` 4-bit | cost |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12.54 | 15.49 | +23.5% |
| SmolLM2-1.7B | 7.91 | 10.30 | +30.3% |
| Qwen2.5-3B | 7.42 | 8.98 | +21.0% |
| Llama-3.1-8B | 5.95 | 6.80 | +14.4% |
| **Mistral-7B-v0.2** | **5.73** | **5.97** | **+4.25%** |

The shipped default costs 4.25% on Mistral-7B and 30.3% on SmolLM2-1.7B. Both
are true; only the first describes what most users will run. Earlier revisions
of this file reported the small-model figures without that context, which
overstated the cost of the default by roughly seven times for a 7B user.

### What 4-bit costs

`rai convert` is **round-to-nearest**, group size 128, with an 8-bit embedding
at group size 64. It needs no Python and it is what the quickstart produces —
so it is what this table measures. The calibrated GPTQ export is a separate
Python path that writes container v1 and therefore cannot export any checkpoint
with projection biases — which excludes the three Qwen2.5 rows below, though
not the SmolLM2 one. That one is measured through both paths in **The
calibrated quantizer, measured end to end**, further down; it costs less than
half as much.

| Model | fp16 | `rai convert` 4-bit | ΔNLL | cost |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12.54 ±2.8% | **15.49** | +0.211 nats | **+23.5%** |
| Qwen2.5-1.5B | 8.68 ±2.6% | **9.83** | +0.124 nats | **+13.2%** |
| Qwen2.5-3B | 7.42 ±2.5% | **8.98** | +0.191 nats | **+21.0%** |
| SmolLM2-1.7B | 7.91 ±2.5% | **10.30** ±2.6% | +0.264 nats | **+30.3%** |

SmolLM2 is a Llama-architecture model with a 49,152-token vocabulary against
Qwen2.5's 151,936, so it shares neither an architecture family nor a tokenizer
with the three above. It is the largest loss in the table. Whatever this is, it
is not specific to Qwen.

Perplexity is a function of the tokenizer as much as of the model, so these
rows are not comparable to each other, and neither is any of them to a
published number taken on different text. The comparison each row supports is
the one inside it: same model, same tokenizer, same tokens, different weights.

**The degradation is not monotonic in model size, and the reason turned out to
be the model.** Qwen2.5-3B quantizes worse than Qwen2.5-1.5B (+0.191 against
+0.124 nats), which is the opposite of the usual result and far outside the
±2.5% sampling error. Three of this file's measurements agree on it — the
paired log-probability delta, the top-1 agreement rate, and llama.cpp's Q4_K_M,
which also loses more on the 3B (+6.1%) than on the 1.5B (+4.6%). Since a
completely different quantizer shows the same ordering, this is a property of
that checkpoint rather than of RAI's quantizer.

### What 4-bit costs, measured on the same tokens

Perplexity is one number over a whole corpus, and two models can reach the same
one while disagreeing about every individual token. These are paired: the same
tokens at the same positions under both models, from the top-64 records
`--dump-topk` writes, reduced by `scripts/compare_topk.py`.

| Model | top-1 agreement with fp16 | Δ log-prob of the correct token | KL over top-64 | top-64 coverage |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | **74.7%** | −0.2110 nats | 0.244 | 0.907 |
| Qwen2.5-1.5B | **80.6%** | −0.1240 nats | 0.162 | 0.940 |
| Qwen2.5-3B | **79.0%** | −0.1909 nats | 0.246 | 0.951 |
| SmolLM2-1.7B | **75.4%** | −0.2644 nats | 0.264 | 0.939 |

**Top-1 agreement is the number a user feels.** At `--temperature 0` the
round-to-nearest model emits a different token than the fp16 model would have
between one time in four and one time in five, depending on the model. Nothing
about a perplexity figure says that out loud. The calibrated export moves this
to one time in six on the model where both paths could be measured.

The Δ log-prob column is exact and needs no caveat: both engines record the
log-probability they assigned to the token that actually came next, so the
difference is computed with no approximation. It reproduces the ΔNLL in the
table above to four decimals — two independent code paths, the perplexity
accumulator and this comparison, agreeing. That agreement is the reason to
believe either of them.

The KL column is approximate and labelled so: a true KL needs both full
distributions, and a full-vocabulary record over 151,936 entries per token
would be gigabytes. It is the KL restricted to the fp16 model's top-64 support
and renormalized over it; the coverage column is how much of the fp16 model's
probability mass that support holds, so a reader can see how much of the
distribution the number actually accounts for.

### The calibrated quantizer, measured end to end

Everything above measures `rai convert`, the round-to-nearest path the
quickstart takes. `scripts/export_raimodel.py` is the other one: same container,
same 4-bit W4A8 layout, same kernels at run time — a different choice of which
4-bit code each weight gets, made by GPTQ against a Hessian accumulated over
calibration text. Until now this repository measured that path only by
*Hessian-weighted output error*, a per-layer proxy. This is the end-to-end
number, on the same corpus, the same 40 chunks, the same 10,200 tokens.

SmolLM2-1.7B, 128 calibration chunks, exported on an L4 in 60 minutes:

| Path | perplexity | ΔNLL vs fp16 | cost | top-1 agreement | KL over top-64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 reference | 7.91 ±2.5% | — | — | — | — |
| `rai convert` (RTN) | **10.30** ±2.6% | +0.2644 nats | **+30.3%** | 75.4% | 0.264 |
| `export_raimodel.py` (GPTQ) | **8.80** ±2.5% | +0.1064 nats | **+11.2%** | **83.3%** | **0.115** |

**Both files are 962,996,000 bytes.** Identical size, identical container
version, identical decode path. `rai convert` emits v1 rather than v2 whenever
the checkpoint needs nothing v2 adds — `RaiConfig::version` in
`rai-infer/src/convert.rs` — and SmolLM2 needs nothing, so the two paths write
the same container and differ in exactly one thing: which 4-bit code each
weight was assigned. The quality gain therefore costs nothing at run time —
not a byte of memory, not a cycle of decode. It costs an hour of calibration
on a GPU, once, at export.

#### Why this is a paired comparison, and why that matters

Each perplexity above carries about ±2.5%. A reader who treats the two as
independent measurements would propagate roughly **±3.6%** onto the ratio
between them, and that is the wrong model of the error. Nearly all of the
±2.5% is *which text was sampled* — and both engines scored **the same 10,200
tokens in the same order**, so that component is common to both numbers and
cancels in the difference. The paired interval below is **±1.8%** on the ratio,
half the naive figure, and it is the honest one.

Recomputed from the raw `--dump-topk` records — a third code path, independent
of both the engine's perplexity accumulator and `compare_topk.py` — and
resampled with a **block bootstrap over the 40 scoring blocks** (255 tokens
each; bootstrapping over individual tokens would ignore the autocorrelation
inside a block and report an interval several times too narrow), 20,000
resamples:

| Paired difference | ΔNLL | 95% CI | perplexity ratio | 95% CI | blocks with the opposite sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTN − fp16 | +0.2644 | [0.2445, 0.2841] | 1.303 | [1.277, 1.329] | 0 of 40 |
| GPTQ − fp16 | +0.1064 | [0.0959, 0.1171] | 1.112 | [1.101, 1.124] | 0 of 40 |
| **RTN − GPTQ** | **+0.1580** | **[0.1399, 0.1757]** | **1.171** | **[1.150, 1.192]** | **0 of 40** |

RTN is worse than GPTQ on **every one of the 40 blocks** — not 38, not 39 — and
the difference sits 17.4 bootstrap standard errors away from zero. The
per-number ±2.5% is sampling error on *which text was scored*; it cancels when
both engines score the same text.

The block-level NLLs recomputed here — 2.067872, 2.332314, 2.174276 — reproduce
the engines' own reported 2.0679, 2.3323 and 2.1743 exactly at the four
decimals they print. That is the check that the bootstrap is resampling real
per-token data rather than re-deriving the summary it is meant to test.

**GPTQ removes 59.8% of the quantization loss** (95% CI 55.8%–63.5%), measured
as the shrinkage of ΔNLL against fp16.

#### What it changes for someone running the model

| | RTN | GPTQ |
| --- | ---: | ---: |
| Agrees with fp16's greedy token | 75.4% | **83.3%** |
| Tokens where only this path matched fp16 | 634 | **1,443** |
| Tokens closer to fp16 on the correct token | 34.3% | **65.7%** (0 ties) |

At `--temperature 0`, round-to-nearest emits a different token than fp16 would
have about **one time in four**; the calibrated export, about **one time in
six**. McNemar's test on the paired agreement gives χ² = 314.3 on 1 degree of
freedom — the two paths are not the same model wearing different labels.

#### The finding this produces

The better quantizer is in this repository and **most users cannot reach it.**

- `rai convert` — the shipped binary, the quickstart, the only path that needs
  no Python — implements round-to-nearest only. It has no GPTQ mode.
- `scripts/export_raimodel.py` — the path that does — writes **container v1**,
  which stores weights only and so cannot represent projection biases, per-head
  QK norms, RoPE scaling or logit softcapping. `assert_exportable_architecture`
  refuses each of those by name rather than dropping them silently, and that
  refusal covers most of the advertised model list: **Qwen2 and Qwen2.5** carry
  biases on q/k/v, **Qwen3** carries per-head QK norms, and the **Gemma**
  family is denied outright for its (1 + weight) RMSNorm. The three Qwen2.5
  rows in the tables above are round-to-nearest for exactly this reason — the
  calibrated path cannot export them at all. SmolLM2 is measurable here
  precisely because it is a plain Llama-architecture checkpoint.

So the gap against llama.cpp below is not primarily a format limitation. Of the
22.6 percentage points between round-to-nearest and Q4_K_M on this checkpoint,
**19.1 close** with the quantizer that already exists — on the container that
already ships, with no change to the file format and no cost at run time. What
is left after that is 3.5 points, and that is the mixed-precision question.
Nothing new has to be invented for the first 19; the two halves have to meet.

### Where the bits go, against llama.cpp Q4_K_M

RAI's `.raimodel` is smaller than the Q4_K_M GGUF of the same checkpoint, every
time:

| Model | RAI 4-bit | llama.cpp Q4_K_M |
| --- | ---: | ---: |
| Qwen2.5-0.5B | **319.5 MiB** | 373.7 MiB (6.35 bits/weight) |
| Qwen2.5-1.5B | **900.8 MiB** | 934.7 MiB (5.08 bits/weight) |
| Qwen2.5-3B | **1,721.9 MiB** | 1,834.8 MiB (4.99 bits/weight) |
| SmolLM2-1.7B | **918.4 MiB** (4.50 bits/weight) | 1,006.7 MiB (4.93 bits/weight) |

That is not free, and the reason is in llama.cpp's own quantization log:
**Q4_K_M is not uniformly 4-bit.** Re-running `llama-quantize` on all four
checkpoints and counting what each tensor became:

| Model | tensors | `q4_K` | `q6_K` | legacy fallback | what got 6 bits |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B | 169 | 12 | 12 | **132 `q5_0` + 13 `q8_0`** | 12 of 24 `ffn_down` |
| Qwen2.5-1.5B | 197 | 168 | 29 | — | 14 of 28 `attn_v`, 14 of 28 `ffn_down`, `token_embd` |
| Qwen2.5-3B | 253 | 216 | 37 | — | 18 of 36 `attn_v`, 18 of 36 `ffn_down`, `token_embd` |
| SmolLM2-1.7B | 169 | 144 | 25 | — | 12 of 24 `attn_v`, 12 of 24 `ffn_down`, `token_embd` |

Two things fall out of that table, and an earlier revision of this file got
both wrong.

**Q4_K_M promotes half of each sensitive tensor type, not all of it.** The
layers it picks are the first few, the last few, and every third one in between
— on Qwen2.5-3B exactly 0-3, 6, 9, 12, 15, 18, 21, 24, 27 and 30-35. So the
extra bits buy 36 of 252 layer tensors on the 3B and 24 of 168 on SmolLM2. RAI
quantizes all seven projections uniformly at 4 bits, and spends **8** bits on
the embedding where Q4_K_M spends 6.

**Qwen2.5-0.5B's "Q4_K_M" is not a 4-bit file.** 145 of its 169 tensors fall
back to the legacy `q5_0` and `q8_0` formats, and only `ffn_down` is K-quantized
at all. The cause is arithmetic: K-quants need a row length divisible by 256,
and that checkpoint's hidden size is 896. Its `ffn_down` rows are 4,864 long and
qualify; everything else does not. That is why its GGUF measures 6.35
bits/weight against RAI's 4-and-a-bit, and it means the 7.9× ratio in the table
below is a bit-budget difference far more than a quantizer difference. The row
is kept, because it is what a reader gets from `llama-quantize` on that model —
but it is not evidence about quantizer quality, and it should not be read as
any.

And it shows up in the perplexity, on the same corpus at the same chunk count:

| Model | fp16 | llama.cpp Q4_K_M | cost | `rai convert` 4-bit | cost | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12.54 | **12.91** | +3.0% | **15.49** | **+23.5%** | 7.9× |
| Qwen2.5-1.5B | 8.68 | **9.08** | +4.6% | **9.83** | **+13.2%** | 2.9× |
| Qwen2.5-3B | 7.42 | **7.87** | +6.1% | **8.98** | **+21.0%** | 3.5× |
| SmolLM2-1.7B | 7.91 | **8.52** | +7.7% | **10.30** | **+30.3%** | 3.9× |

Every model, both engines, one corpus. RAI's round-to-nearest 4-bit costs
between three and four times what Q4_K_M costs on the three checkpoints where
both are really 4-bit files, and 7.9× on Qwen2.5-0.5B, where — as the tensor
table above shows — llama.cpp is not producing a 4-bit file at all.

On the one model where RAI's calibrated exporter can also run, the gap is
mostly a quantizer gap and not a format one:

| SmolLM2-1.7B | file | bits/weight | perplexity | cost over fp16 |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp Q4_K_M | 1,006.7 MiB | 4.93 | **8.52** | +7.7% |
| RAI, `export_raimodel.py` (GPTQ) | **918.4 MiB** | **4.50** | **8.80** | **+11.2%** |
| RAI, `rai convert` (RTN) | **918.4 MiB** | **4.50** | 10.30 | +30.3% |

RAI's calibrated 4-bit spends 0.43 fewer bits per weight than Q4_K_M and pays
1.45× its quantization cost, against 3.9× for round-to-nearest.

Q4_K_M is still ahead by 3.5 percentage points, and the tensor table above
says what that buys: 24 promoted tensors out of SmolLM2's 168, offset by RAI
spending 8 bits on the embedding where Q4_K_M spends 6. That is the
mixed-precision decision described below, stated in tensors rather than in
adjectives — and on this model it is now the *whole* of the remaining gap,
because the quantizer half of it is already closed.

**The two engines were checked against each other before this table was
believed.** `llama-perplexity` scores through its own GGUF tokenizer and
prepends a BOS to the corpus, where `rai perplexity` does neither — so llama.cpp
was also run on the *unquantized* F16 GGUF, to see whether the two pipelines
agree on a model neither of them is quantizing:

| Model, fp16 | `reference_ppl.py` | `llama-perplexity` | apart |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12.5421 | 12.5422 | 0.0008% |
| Qwen2.5-1.5B | 8.6817 | 8.6819 | 0.0023% |
| Qwen2.5-3B | 7.4201 | 7.4198 | 0.0040% |

`reference_ppl.py` runs transformers on a GPU with the HuggingFace tokenizer
and no BOS; `llama-perplexity` runs llama.cpp on a CPU with the GGUF tokenizer
and a prepended BOS. Two implementations that share no code, on three models,
agreeing to four decimal places. The tokenizer difference and the BOS offset
are therefore not material at this corpus size, which is what makes the Q4_K_M
column above a controlled comparison rather than a rough one — and it is also
the strongest available evidence that neither pipeline is simply wrong.

So RAI spends fewer bits per weight and pays for it in perplexity. That is a
design difference with two clear directions of improvement, neither of them a
mystery.

**The cheaper one is already written.** Three fifths of the round-to-nearest
loss closes with the GPTQ exporter that ships in this repository, at the same
file size, in the same container, on the same decode path — it simply is not
reachable from `rai convert`, and the Python path that has it writes container
v1 and so refuses most of the model list. Nothing new has to be invented; the
two halves have to meet.

**The other one is a format change.** The container already stores a
per-projection `bias_mask`, and a per-projection *bit-width* would let the same
format spend its bits where they are worth most, which is what Q4_K_M does with
`ffn_down` and `attn_v`. Nothing in this repository does that today. It would
be container v3.

### The two writers agree, byte for byte, on a real checkpoint

`rai convert` (Rust, no Python) and `scripts/export_rtn.py` (Python, torch) are
two independent implementations of the same conversion. The repository has
always tested that they agree on a synthetic fixture. On **SmolLM2-1.7B**, a
real 1.7-billion-parameter checkpoint:

| | |
| --- | ---: |
| `rai convert` | 962,996,000 bytes, SHA-256 `8d32b963…e69eadb9` |
| `export_rtn.py` | 962,996,000 bytes, SHA-256 `8d32b963…e69eadb9` |

Identical. Reproduce with `scripts/export_rtn.py --model <dir> --output <file>
--max-context 2048` against the same `rai convert` options and compare hashes.

**This check earned its keep the first time it was run.** Before the fix in
this release the two differed — by exactly three bytes, all inside the header's
`rope_theta` field: `rai convert` wrote the checkpoint's 130000 and the Python
exporter wrote 10000, because transformers 5 moved that value and the exporter
read it through a defaulting `getattr`. A 963 MB file agreeing everywhere
except one float is what made the cause findable at all; a whole-file "differs"
would have said nothing. See the CHANGELOG entry for the blast radius.

### Conversion cost, on real checkpoints

Peak resident set during `rai convert`, measured with `/usr/bin/time -v`:

| Model | checkpoint | `.raimodel` | ratio | wall | **peak RSS** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 988.1 MB | 335.0 MB | 2.95× | 7.9 s | **87.5 MB** |
| Qwen2.5-1.5B | 3,087.5 MB | 944.6 MB | 3.27× | 20.9 s | **73.9 MB** |
| Qwen2.5-3B | 6,171.9 MB | 1,805.6 MB | 3.42× | 40.7 s | **83.0 MB** |

Peak memory is flat across a 6× range of checkpoint size — a 6.2 GB checkpoint
converted in 83 MB. That is the streaming claim at the top of this file,
measured on real checkpoints rather than asserted.

### Decode and prefill, same machine

Greedy (`--temperature 0`) with top-k, top-p and repetition penalty all
neutralised, so this measures the engine and not the sampler. Median of five
runs of 128 tokens; the range is the spread across those five.

| Model | decode | prefill | load | peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | **26.8 tok/s** (25.4–27.5) | 134–155 tok/s | 0.94 s | 434 MB |
| Qwen2.5-1.5B | **12.9 tok/s** (12.8–13.7) | 67–81 tok/s | 1.37 s | 1,044 MB |
| Qwen2.5-3B | **8.5 tok/s** (8.1–9.0) | 46–49 tok/s | 1.93 s | 1,923 MB |

### Reproducing it

```bash
rai convert /path/to/Qwen2.5-1.5B -o out/qwen.raimodel --max-context 2048
rai perplexity out/qwen.raimodel --text wiki.test.raw --context 512 --max-chunks 40

python rai-infer/scripts/reference_ppl.py \
  --model /path/to/Qwen2.5-1.5B --text wiki.test.raw \
  --context 512 --max-chunks 40 --dtype float16
```

Add `--dump-topk 64 --dump-path <file>` to either and
`scripts/compare_topk.py` will report top-1 agreement, top-K overlap and the
exact paired change in the correct token's log-probability — which is a
stronger statement than perplexity, because it is measured on the same tokens
at the same positions under both models.

The calibrated path, against the same corpus and the same chunk count — an hour
on an L4 for a 1.7B checkpoint, and it needs a checkpoint container v1 can hold
(SmolLM2 can, Qwen2.5 cannot):

```bash
python rai-infer/scripts/export_raimodel.py \
  --model /path/to/SmolLM2-1.7B --output out/smollm2-gptq.raimodel \
  --max-context 2048 --cal-chunks 128
rai perplexity out/smollm2-gptq.raimodel --text wiki.test.raw \
  --context 512 --max-chunks 40 --dump-topk 64 --dump-path out/gptq-top64.bin
```

Export into a directory of its own: both paths write `tokenizer.json` beside the
model, and `rai convert` refuses — correctly — to overwrite one belonging to a
different model.

The 95% intervals on the paired differences come from a block bootstrap over the
40 scoring blocks, resampling blocks rather than tokens because tokens inside a
block share a context. What it consumes is the two `--dump-topk` records, so it
needs no model and no GPU: read the per-token target log-probabilities with
`compare_topk.read_record`, reshape to `(40, 255)`, and resample block indices.

And the llama.cpp tensor table, which is `llama-quantize`'s own log:

```bash
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M 2>&1 \
  | grep -oE 'converting to [a-z0-9_]+' | sort | uniq -c
```

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
- These GPTQ figures describe a path most of the advertised model list cannot
  take. The Python exporters write container v1, so they refuse any checkpoint
  with projection biases, per-head QK norms, RoPE scaling, MoE routing or logit
  softcapping — Qwen2/2.5, Qwen3, Gemma, OLMo2 and Phi-3 among them.
  `rai convert` writes v2 and accepts all of those, and it does
  round-to-nearest, not GPTQ. **Quantization cost, measured** below is the
  round-to-nearest number, because that is the path the quickstart takes — and
  **The calibrated quantizer, measured end to end** is the same comparison run
  as a perplexity on SmolLM2, the one advertised checkpoint both paths accept.
  The Hessian-weighted output errors in the table above are per-layer proxies;
  that section is what they are worth in the only units a user experiences.

  **The proxy held up.** It predicted 2.8× lower output error on average. The
  end-to-end measurement puts GPTQ's loss against fp16 at 0.1064 nats where
  round-to-nearest costs 0.2644 — **2.49× lower**, on a different model, a
  different corpus and a metric with no code in common. A per-layer proxy
  landing within 12% of the whole-model result is the reason to keep using it
  for the layer-level work, where a perplexity sweep per experiment is not
  affordable.
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
