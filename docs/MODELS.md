# Which models work

RAI runs one architecture: the plain Llama/Mistral decoder. RMSNorm, rotary
position embeddings built from a single theta, grouped-query attention, SwiGLU
feed-forward, no bias vectors anywhere. If your checkpoint is that shape, it
converts and runs. If it is not, the exporter refuses it by name before it
writes a byte.

Read this page before you download a checkpoint.

**Works:** TinyLlama-1.1B-Chat, SmolLM / SmolLM2 (135M, 360M, 1.7B), Llama-2
7B and 13B, Mistral-7B v0.1 / v0.2 / v0.3, and the fine-tunes built on those —
Zephyr-7B-beta, OpenHermes-2.5-Mistral-7B, Vicuna, Nous-Hermes-Llama2.

**Refused:** Qwen2, Qwen2.5, Qwen3, Llama-3.1, Llama-3.2, every Gemma, Mixtral
and other mixture-of-experts models, Phi, OLMo2, Falcon, GPT-NeoX, MPT.

TinyLlama-1.1B-Chat is the one checkpoint measured end to end — 47.0 s to
convert, 619.5 MB on disk, 21.8 tok/s decoding
([BENCHMARKS.md](../BENCHMARKS.md)). The rest of the "works" list is supported
by architecture, not by a retained run.

## Why the line falls where it does

The `.raimodel` container stores exactly this and nothing else:

- a 64-byte header holding the architecture dimensions, `rope_theta`, and
  `norm_eps`
- one 8-bit quantized embedding table
- per layer: seven 4-bit projections (`q`, `k`, `v`, `o`, `gate`, `up`, `down`)
  and two f32 RMSNorm weight vectors
- one final f32 RMSNorm
- an optional 4-bit `lm_head` (omitted when the head is tied to the embedding)

The kernels implement exactly this and nothing else: RMSNorm as `x / rms * w`,
rotary embeddings from one theta, full causal attention with grouped-query head
mapping, and SwiGLU with a SiLU gate.

Any weight or behaviour outside those two lists has no place to live. It would
be dropped at export, and the resulting file would load cleanly and generate
nonsense. So the exporter rejects it instead.

## Compatibility table

| Architecture family | Status | Reason |
| --- | --- | --- |
| Llama (1 / 2) | **Supported** | Reference architecture for the format. |
| Mistral (v0.1 / v0.2 / v0.3) | **Supported** | Identical module tree to Llama plus a sliding window (see below). |
| TinyLlama | **Supported** | Llama architecture at 1.1B; the measured reference model. |
| SmolLM / SmolLM2 | **Supported** | `LlamaForCausalLM` with no bias and no RoPE scaling. |
| Llama-2 / Mistral fine-tunes (Zephyr, OpenHermes, Vicuna, Nous-Hermes) | **Supported** | Fine-tuning changes weights, not architecture. |
| Llama-3 / Llama-3 Instruct (8B) | **Accepted by preflight** | The 3.0 configs declare no `rope_scaling`, no bias, and no QK norm. Not measured here. Llama-3-70B needs a raised `--group-size` (see shape constraints). |
| Qwen2, Qwen2.5 | **Supported** | Projection bias vectors are stored as f32 in the layer section (container v2). Qwen2.5-0.5B-Instruct verified generating. |
| Qwen3 | Not supported | Per-head Q and K RMSNorms; the layer section has two norm vectors, both already used. |
| Qwen3-MoE, Mixtral, other MoE | Not supported | A router plus per-expert MLPs; the layer section holds one gate/up/down triple. |
| Llama-3.1, Llama-3.2 | **Supported** | The `llama3` frequency rescaling is stored in the v2 header and applied when the RoPE table is built, pinned by unit test against the transformers reference to under 2e-7. Llama-3.2-1B-Instruct verified generating. |
| Gemma | **Supported** | GeGLU is a second activation kernel selected by the v2 header; the `(1 + w)` RMSNorm folds into the stored norm weights at conversion; the `sqrt(hidden_size)` embedding scale is a header field applied at lookup (it cannot be folded — Gemma ties `lm_head` to the embedding, so a pre-scaled table would scale every logit). gemma-2b-it verified generating. |
| Gemma2 | Not supported | Attention and final logit softcapping — extra operations, not extra parameters. |
| Gemma3, Gemma3-text | Not supported | Per-head QK norm and interleaved sliding-window attention. |
| OLMo2 | Not supported | Per-head QK norms. |
| Phi, Falcon, GPT-NeoX, MPT, GPT-2 | Not supported | No Llama-style module tree. Either `model.model.layers` is absent, or the projections are fused or named differently (Phi-3 fuses QKV into one `qkv_proj` and gate/up into one `gate_up_proj`), so the exporter cannot find the seven tensors it stores. |

A "not supported" row is a hard error at export, not a warning.

### Sliding-window models

Mistral declares `sliding_window: 4096`. RAI always runs full causal attention.
Inside the window that is identical arithmetic; past it, the two diverge. The
exporter therefore accepts a sliding-window checkpoint up to the window length
and refuses a longer `--max-context`, naming the window in the error. Export
Mistral-7B with `--max-context 4096` or lower.

### Shape constraints that apply to every model

These are enforced identically by the Python exporter and by the Rust loader,
so a file that exports always loads:

| Constraint | Consequence |
| --- | --- |
| `hidden_size == num_heads * head_dim` | A checkpoint declaring a decoupled `head_dim` is refused; the format has no field for it. |
| `head_dim` is a multiple of 8 | Required by the SIMD attention kernels. |
| `hidden_size` and `intermediate_size` are even | Required by nibble-packed 4-bit weights. |
| `num_heads` divisible by `num_kv_heads` | Required by the grouped-query head mapping. |
| At most 128 quantization groups per row | At the default `--group-size 128`, `hidden_size` and `intermediate_size` must each be at most 16,384. Llama-2-70B's intermediate size of 28,672 exceeds this; it needs `--group-size 224` or larger. |
| Group sizes are even and at most 254 | The group size is one header byte. |
| RoPE table at most 512 MB | The table costs `max_context * head_dim / 2 * 8` bytes. At `head_dim 128` that allows 1,048,576 positions, so the separate hard cap of 1,000,000 on `max_context` binds first. |

## Check your model before downloading 14 GB

Every blocker is visible in the checkpoint's `config.json`, which is a few
kilobytes. Fetch that file alone first.

```bash
curl -sL https://huggingface.co/<org>/<model>/raw/main/config.json -o config.json
python - <<'PY'
import json
c = json.load(open("config.json"))
print("model_type       ", c.get("model_type"))
print("architectures    ", c.get("architectures"))
print("rope_scaling     ", c.get("rope_scaling"))
print("experts          ", c.get("num_experts"), c.get("num_local_experts"))
print("sliding_window   ", c.get("sliding_window"))
print("softcapping      ", c.get("attn_logit_softcapping"), c.get("final_logit_softcapping"))
print("head_dim         ", c.get("head_dim"), "hidden", c.get("hidden_size"),
      "heads", c.get("num_attention_heads"))
PY
```

Read the output against this:

| Field | Convert it | Do not download |
| --- | --- | --- |
| `model_type` | `llama`, `mistral` | `gemma`, `gemma2`, `gemma3`, `gemma3_text`, `mixtral`, `qwen3_moe`, `phi3`, `falcon`, anything else |
| `rope_scaling` | `null`, absent, or `{"rope_type": "default"}` | any other `rope_type`, including `llama3`, `linear`, `dynamic`, `yarn` |
| `num_experts` / `num_local_experts` | absent or zero | any positive value |
| `attn_logit_softcapping` / `final_logit_softcapping` | absent or null | any value |
| `sliding_window` | absent, or present and at least your `--max-context` | present and below your `--max-context` |
| `head_dim` | absent, or equal to `hidden_size / num_attention_heads` | anything else |

`config.json` does not record attention bias, so judge that by family: **Qwen2
and Qwen2.5 carry it, Llama and Mistral do not.**

The `config.json` read is a filter, not the verdict. The exporter loads the
real weights and re-checks them — it walks every layer looking for a `bias`
tensor on any of the seven projections and for `q_norm` / `k_norm` modules on
the attention block, and it collects every problem it finds before failing.
The error names the count, the first offending tensor, and the reason — one
bullet per problem found:

```
this checkpoint cannot be represented by the .raimodel format:
  - <count> projection(s) carry bias vectors (e.g. layer 0.q_proj); the format
    stores weights only, so the biases would be silently dropped.
    Qwen2/Qwen2.5 are the common case here.
```

It fails before calibration, so a refusal costs seconds, not hours.

## What it would take to support the rest

Each of these is tractable. None is free. They are listed roughly in ascending
cost.

### Attention and MLP bias vectors — unlocks Qwen2, Qwen2.5

Add optional bias sections to the layer payload: up to seven f32 vectors,
sized by each projection's output rows. The header gains a flag byte saying
which are present, and the format version goes from 1 to 2 so old readers
reject new files instead of misreading them. The kernels gain an add after each
matvec, which costs one pass over the output vector. The exporter stops
treating a bias as an error and writes it. Cost: a format version bump, one new
section layout, and an epsilon of runtime.

### Per-head QK norm — unlocks Qwen3, OLMo2, and half of Gemma3

Two more f32 norm vectors per layer, each `head_dim` long, applied to Q and K
per head after projection and before RoPE. The layer section grows by
`2 * head_dim * 4` bytes and the reader grows two more `RMSNormWeights` views.
The existing `rms_norm` kernel does the arithmetic unchanged; it is called per
head instead of per token. Cost: a format version bump and two extra section
fields. This is the cheapest of the maths changes.

### GeGLU activation — a prerequisite for Gemma

A second activation kernel: `gelu_mul_inplace` alongside `silu_mul_inplace`,
with the same AVX2 treatment, plus one header byte selecting which the model
uses. Gemma also needs its `(1 + w)` RMSNorm variant and its
`sqrt(hidden_size)` embedding scale — both are one-line changes to the kernels
but both need a header flag, because the same weight values mean different
things under each convention. Cost: one new kernel, two behaviour flags, and
careful conformance testing against the reference implementation, since a wrong
flag produces plausible-looking garbage rather than a crash.

### RoPE scaling — unlocks Llama-3.1, Llama-3.2, and every YaRN model

The header carries `rope_theta` and nothing else, so it gains a scaling type
byte and the three-to-four f32 parameters those schemes need (`factor`,
`low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`).
`RoPETable::new` gains a matching frequency-transform stage per scheme;
`llama3`, `linear`, `dynamic`, and `yarn` each compute their inverse
frequencies differently. The 64-byte header has room for this. Cost: a format
version bump, a scaling-parameter block in the header, one table builder per
scheme, and a positional-accuracy test per scheme — this is where a subtle
error shows up as quality loss at long context rather than as a failure.

### Mixture of experts — unlocks Mixtral, Qwen3-MoE

The largest change. The layer section becomes a router matrix plus N expert
gate/up/down triples instead of one, so the section layout and the size
accounting change shape. The forward pass gains a top-k router, per-token
expert selection, and a gather over the selected experts' weights, which turns
the MLP from a fixed sequence of three matvecs into a data-dependent dispatch.
The memory profile changes with it: a 4-bit Mixtral-8x7B is roughly eight times
the MLP weight of a 7B for the same per-token compute, and the loader's
one-read-into-heap strategy would need revisiting at that size. Cost: a new
section layout, a new forward path, new capacity limits, and a benchmark
rerun — a release of its own, not an increment.

## See also

- [INSTALL.md](./INSTALL.md#converting-a-model) — the conversion commands.
- [BENCHMARKS.md](../BENCHMARKS.md) — the measured TinyLlama-1.1B conversion
  and decode run.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — where the format and the kernels live
  in the workspace.
