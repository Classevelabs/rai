# Which models work

RAI runs the Llama-family decoder: RMSNorm, rotary position embeddings,
grouped-query attention, and a gated feed-forward block. Container v2 added the
pieces that the closest neighbours of that architecture need — projection bias
vectors, `llama3` RoPE frequency rescaling, and a GeGLU activation — which is
what brings Qwen2/2.5, Llama-3.1/3.2, and Gemma inside the line.

Anything the container cannot represent is refused by name at conversion time,
before a byte is written. Read this page before you download a checkpoint.

**Runs:** Qwen2 and Qwen2.5, Llama-3.1 and Llama-3.2, Gemma, Llama-2,
Mistral-7B v0.1/v0.2/v0.3, TinyLlama-1.1B-Chat, SmolLM and SmolLM2 (135M, 360M,
1.7B), and fine-tunes of those — Zephyr-7B-beta, OpenHermes-2.5-Mistral-7B,
Vicuna, Nous-Hermes-Llama2.

**Refused:** Qwen3, OLMo2, Gemma2, Gemma3, Mixtral and other mixture-of-experts
models, Phi, Falcon, GPT-NeoX, MPT.

Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, and gemma-2b-it were each
converted and generated coherent text. TinyLlama-1.1B-Chat is the one
checkpoint measured end to end — 619.5 MB on disk, 21.8 tok/s decoding
([BENCHMARKS.md](../BENCHMARKS.md)). The rest of the list is supported by
architecture, not by a retained run.

## The two conversion paths do not cover the same models

`rai convert` writes container v2. The Python exporters
(`export_raimodel.py`, `export_fast.py`, `export_rtn.py`) write container v1
only, so they still refuse everything v2 added:

| Checkpoint | `rai convert` | Python exporters |
| --- | --- | --- |
| Llama, Mistral, TinyLlama, SmolLM | Converts | Converts |
| Qwen2, Qwen2.5 | Converts | Refused — projection bias vectors |
| Llama-3.1, Llama-3.2 | Converts | Refused — `rope_type: "llama3"` |
| Gemma | Converts | Refused — embedding scale and `(1 + w)` RMSNorm |

Use `rai convert` unless you specifically want calibrated GPTQ quantization,
which only the Python path implements.

## Why the line falls where it does

The `.raimodel` container stores exactly this and nothing else:

- a header holding the architecture dimensions, `rope_theta`, `norm_eps`, the
  activation code, the RoPE scaling parameters, the bias mask, and the
  embedding scale
- one 8-bit quantized embedding table
- per layer: seven 4-bit projections (`q`, `k`, `v`, `o`, `gate`, `up`,
  `down`), an optional f32 bias vector per projection, and two f32 RMSNorm
  weight vectors
- one final f32 RMSNorm
- an optional 4-bit `lm_head` (omitted when the head is tied to the embedding)

The kernels implement exactly this and nothing else: RMSNorm as `x / rms * w`,
rotary embeddings from one theta with an optional `llama3` frequency rescale,
full causal attention with grouped-query head mapping, and a gated MLP with
either a SiLU or a GeLU-tanh gate.

A model needing a weight or an operation outside those two lists has nowhere to
put it. It would be dropped at export and the resulting file would load cleanly
and generate nonsense, so the exporter rejects it instead.

## Compatibility table

| Architecture family | Status | Reason |
| --- | --- | --- |
| Llama (1 / 2) | **Supported** | Reference architecture for the format. |
| Mistral (v0.1 / v0.2 / v0.3) | **Supported** | Identical module tree to Llama plus a sliding window (see below). |
| TinyLlama | **Supported** | Llama architecture at 1.1B; the measured reference model. |
| SmolLM / SmolLM2 | **Supported** | `LlamaForCausalLM` with no bias and no RoPE scaling. |
| Llama-2 / Mistral fine-tunes (Zephyr, OpenHermes, Vicuna, Nous-Hermes) | **Supported** | Fine-tuning changes weights, not architecture. |
| Llama-3 / Llama-3 Instruct (8B) | **Supported** | The 3.0 configs declare no `rope_scaling`, no bias, and no QK norm, so they convert at container v1. Llama-3-70B needs a raised `--group-size` (see shape constraints). |
| Llama-3.1, Llama-3.2 | **Supported** | The `llama3` frequency rescaling is stored in the v2 header and applied when the RoPE table is built, pinned by unit test against the transformers reference to under 2e-7. Llama-3.2-1B-Instruct verified generating. |
| Qwen2, Qwen2.5 | **Supported** | Projection bias vectors are stored as f32 in the layer section (container v2). Qwen2.5-0.5B-Instruct verified generating. |
| Gemma | **Supported** | GeGLU is a second activation kernel selected by the v2 header; the `(1 + w)` RMSNorm folds into the stored norm weights at conversion; the `sqrt(hidden_size)` embedding scale is a header field applied at lookup, because Gemma ties `lm_head` to the embedding and a pre-scaled table would scale every logit. gemma-2b-it verified generating. |
| Qwen3, OLMo2 | Not supported | Per-head Q and K RMSNorms; the layer section has two norm vectors, both already used. |
| Gemma2 | Not supported | Attention and final logit softcapping — extra operations, not extra parameters. |
| Gemma3, Gemma3-text | Not supported | Per-head QK norm and interleaved sliding-window attention. |
| Qwen3-MoE, Mixtral, other MoE | Not supported | A router plus per-expert MLPs; the layer section holds one gate/up/down triple. |
| Phi, Falcon, GPT-NeoX, MPT, GPT-2 | Not supported | No Llama-style module tree. Either `model.model.layers` is absent, or the projections are fused or named differently (Phi-3 fuses QKV into one `qkv_proj` and gate/up into one `gate_up_proj`), so the exporter cannot find the seven tensors it stores. |

A "not supported" row is a hard error at export, not a warning.

### Sliding-window models

Mistral declares `sliding_window: 4096`. RAI always runs full causal attention.
Inside the window that is identical arithmetic; past it, the two diverge. The
exporter therefore accepts a sliding-window checkpoint up to the window length
and refuses a longer `--max-context`, naming the window in the error. Convert
Mistral-7B with `--max-context 4096` or lower.

### Shape constraints that apply to every model

These are enforced identically by the converter and by the Rust loader, so a
file that converts always loads:

| Constraint | Consequence |
| --- | --- |
| `head_dim` is a multiple of 8 | Required by the SIMD attention kernels. A decoupled `head_dim` (`num_heads * head_dim != hidden_size`, as Gemma uses) is fine: only the interior of the attention block changes width. |
| `num_heads * head_dim` is even | Required by nibble-packed 4-bit weights. |
| `hidden_size` and `intermediate_size` are even | Required by nibble-packed 4-bit weights. |
| `num_heads` divisible by `num_kv_heads` | Required by the grouped-query head mapping. |
| At most 128 quantization groups per row | At the default `--group-size 128`, `hidden_size` and `intermediate_size` must each be at most 16,384. Llama-2-70B's intermediate size of 28,672 exceeds this; it needs `--group-size 224` or larger. |
| Group sizes are even and at most 254 | The group size is one header byte. |
| RoPE table at most 512 MB | The table costs `max_context * head_dim / 2 * 8` bytes. At `head_dim 128` that allows 1,048,576 positions, so the separate hard cap of 1,000,000 on `--max-context` binds first. |

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
print("hidden_act       ", c.get("hidden_act"), c.get("hidden_activation"))
PY
```

Read the output against this:

| Field | Convert it | Do not download |
| --- | --- | --- |
| `model_type` | `llama`, `mistral`, `qwen2`, `gemma` | `gemma2`, `gemma3`, `gemma3_text`, `qwen3`, `mixtral`, `qwen3_moe`, `olmo2`, `phi3`, `falcon`, anything else |
| `rope_scaling` | `null`, absent, `{"rope_type": "default"}`, or `{"rope_type": "llama3", ...}` | any other `rope_type`, including `linear`, `dynamic`, `yarn` |
| `num_experts` / `num_local_experts` | absent or zero | any positive value |
| `attn_logit_softcapping` / `final_logit_softcapping` | absent or null | any value |
| `sliding_window` | absent, or present and at least your `--max-context` | present and below your `--max-context` |
| `hidden_act` / `hidden_activation` | `silu`, `swish`, `gelu_pytorch_tanh`, or absent | any other named activation |

The `config.json` read is a filter, not the verdict. The converter loads the
real weights and re-checks them — it walks every layer looking for `q_norm` /
`k_norm` modules on the attention block and for the seven projections it
stores, and it collects every problem it finds before failing. The error names
the count, the first offending tensor, and the reason, one bullet per problem:

```
this checkpoint cannot be represented by the .raimodel format:
  - 56 per-head QK norm(s) present (e.g. model.layers.0.self_attn.q_norm);
    the format has no place to store them.
```

It fails before calibration, so a refusal costs seconds, not hours.

## What it would take to support the rest

Each of these is tractable. None is free. They are listed roughly in ascending
cost.

### Per-head QK norm — unlocks Qwen3, OLMo2, and half of Gemma3

Two more f32 norm vectors per layer, each `head_dim` long, applied to Q and K
per head after projection and before RoPE. The layer section grows by
`2 * head_dim * 4` bytes and the reader grows two more `RMSNormWeights` views.
The existing `rms_norm` kernel does the arithmetic unchanged; it is called per
head instead of per token. Cost: a header flag and two extra section fields.
This is the cheapest of the remaining maths changes.

### Logit softcapping — unlocks Gemma2

`softcap * tanh(x / softcap)` applied to the attention scores and again to the
final logits, with the two cap values carried in the header. No new parameters,
but two new points in the forward pass and a `tanh` over the whole vocabulary
on every step. Gemma2 also interleaves sliding-window and full attention layers,
which the KV cache and the attention kernel currently have no way to express.

### YaRN and other RoPE schemes — unlocks long-context fine-tunes

The header already carries a scaling-type byte and the `llama3` parameters.
`linear`, `dynamic`, and `yarn` each compute their inverse frequencies
differently, so each needs a `RoPETable::new` branch and its own
positional-accuracy test — this is where a subtle error shows up as quality
loss at long context rather than as a failure.

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
