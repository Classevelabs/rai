# Which models work

## I have a folder — what now?

Point the converter at it. Nothing is written until the checkpoint has passed
the preflight, so a wrong answer costs seconds:

```bash
rai convert /path/to/Qwen2.5-0.5B-Instruct -o qwen2.5-0.5b-q4.raimodel
```

Or press **Check** in Studio (`rai serve`, then the Convert screen). It calls
`POST /api/inspect`, which runs the same preflight the converter runs and
reports the same verdict — with the difference that Studio reads folders only.
It has no HTTP client, so a repository id is answered with the download command
rather than a fetch:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
```

Either way the answer is one of three things: it converts, it converts with a
different flag, or it is refused by name. A refusal always names the blocker;
the table below says what to run instead.

On Windows, download into a plain directory: `huggingface_hub` fills its cache
with symlinks, which Windows refuses without Developer Mode
([INSTALL.md](./INSTALL.md#downloading-on-windows)).

## What has to be in the folder

Any directory holding these converts. Where it came from and what it is called
do not matter.

| File | Needed | Notes |
| --- | --- | --- |
| `config.json` | Yes | Read first, and alone when you inspect a config-only folder. |
| `model.safetensors` | Yes, or the shards below | Single-file checkpoints. |
| `model-00001-of-0000N.safetensors` + `model.safetensors.index.json` | Yes, or the single file above | Sharded checkpoints; the index is what the converter walks. |
| `tokenizer.json` | Yes | Conversion refuses without it, because the runtime needs it next to the model. It is copied there for you. |
| `pytorch_model.bin`, `*.pth`, `*.gguf`, `*.msgpack` | Not read | A repository that ships only these will not convert today. |

Two layouts fail for reasons that have nothing to do with the architecture, and
both are fixable where you stand:

- **Weights only as `pytorch_model.bin`.** Most such models have a
  `.safetensors` sibling revision on the Hub; take that one, or write the
  tensors out yourself with `safetensors.torch.save_file` before pointing RAI
  at the folder.
- **A `tokenizer.model` and no `tokenizer.json`.** Older SentencePiece
  repositories ship only the former. Load the folder once with
  `transformers.AutoTokenizer.from_pretrained(...)` and
  `.save_pretrained(...)`, which writes the fast `tokenizer.json` beside it.

## Will my model convert?

Find your family. The **Use instead** column is a checkpoint of comparable size
that this format does store.

| Your checkpoint | Verdict | Use instead | Why |
| --- | --- | --- | --- |
| Llama-2, Llama-3, Llama-3.1, Llama-3.2 | **Converts** | — | Reference architecture. `llama3` RoPE rescaling is stored in the v2 header. |
| Mistral-7B v0.1 / v0.2 / v0.3 | **Converts** | — | Identical module tree to Llama. Convert with `--max-context 4096` or lower (see sliding windows). |
| Qwen2, Qwen2.5 | **Converts** | — | Projection bias vectors are stored as f32 in the layer section. |
| Gemma (first generation) | **Converts** | — | GeGLU is a stored activation code; the `(1 + w)` norm and embedding scale are folded at conversion. |
| TinyLlama-1.1B | **Converts** | — | The measured reference model. |
| SmolLM, SmolLM2 (135M / 360M / 1.7B) | **Converts** | — | `LlamaForCausalLM` with no bias and no RoPE scaling. |
| Llama-2 / Mistral fine-tunes (Zephyr, OpenHermes, Vicuna, Nous-Hermes) | **Converts** | — | Fine-tuning changes weights, not architecture. |
| Qwen3 (dense) | *Support landing — see below* | `Qwen/Qwen2.5-*-Instruct` at your size | Per-head QK norm; support in flight. |
| OLMo2 | *Support landing — see below* | `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` | Per-head QK norm; support in flight. |
| Gemma2 | *Support landing — see below* | `google/gemma-2b-it`, `google/gemma-7b-it` | Logit softcapping; support in flight. |
| Gemma3, Gemma3-text | *Support landing — see below* | `google/gemma-2b-it` | Verdict pending; a refusal names the exact blocker. |
| Mixtral-8x7B, Mixtral-8x22B | **Refused** | `mistralai/Mistral-7B-Instruct-v0.3` | A router plus per-expert MLPs; a layer holds one gate/up/down triple. |
| Qwen3-MoE (30B-A3B, 235B-A22B) | **Refused** | `Qwen/Qwen2.5-14B-Instruct` | Mixture-of-experts routing, independently of the dense-Qwen3 line above. |
| Any config with `num_experts` or `num_local_experts` | **Refused** | The dense model of the same family | Same reason. |
| Phi-2, Phi-3, Phi-3.5 | **Refused** | `meta-llama/Llama-3.2-3B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` | Phi-3 fuses QKV into one `qkv_proj` and gate/up into one `gate_up_proj`, so the seven tensors a layer stores are not there. |
| Falcon | **Refused** | `mistralai/Mistral-7B-Instruct-v0.3` | No Llama-style module tree. |
| GPT-NeoX, Pythia | **Refused** | `meta-llama/Llama-3.2-1B-Instruct` (or SmolLM2 below 1B) | No Llama-style module tree. |
| MPT | **Refused** | `mistralai/Mistral-7B-Instruct-v0.3` | No Llama-style module tree. |
| GPT-2 | **Refused** | `HuggingFaceTB/SmolLM2-360M-Instruct` | Learned position embeddings, fused attention, no `model.layers`. |
| Anything with `rope_scaling` other than `default` or `llama3` (`linear`, `dynamic`, `yarn`) | **Refused** | The unscaled base checkpoint, or a Llama-3.1 model | Positions would be computed wrong; the container stores two schemes. |
| Anything carrying `lm_head.bias` | **Refused** | The base checkpoint it was tuned from | Biases are stored for the seven layer projections only, so it would be dropped in silence. |

A refusal is a hard error at export, never a warning, and never a file that
loads cleanly and generates nonsense.

### Four families in flight

<!-- FINALISE: per-head QK norm (Qwen3, OLMo2) and logit softcapping (Gemma2),
     possibly Gemma3-text, are being implemented. Replace each line below with
     the verdict once that work reports, and set the matching table rows. -->

**Support is landing for these four. The verdict below is not set yet — set it
when the architecture work reports.**

- **Qwen3 (dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B)** — per-head QK norm support is
  being implemented. *Verdict: pending.* If it is refused, use
  `Qwen/Qwen2.5-0.5B-Instruct` for Qwen3-0.6B, `Qwen/Qwen2.5-1.5B-Instruct` for
  1.7B, `Qwen/Qwen2.5-3B-Instruct` for 4B, `Qwen/Qwen2.5-7B-Instruct` for 8B,
  `Qwen/Qwen2.5-14B-Instruct` for 14B, `Qwen/Qwen2.5-32B-Instruct` for 32B.
- **OLMo2 (7B, 13B)** — same per-head QK norm work. *Verdict: pending.* If it is
  refused, use `meta-llama/Llama-3.1-8B-Instruct` or
  `mistralai/Mistral-7B-Instruct-v0.3`.
- **Gemma2 (2B, 9B, 27B)** — logit softcapping support is being implemented.
  *Verdict: pending.* If it is refused, use `google/gemma-2b-it` (the Gemma
  checkpoint verified generating) or `google/gemma-7b-it`.
- **Gemma3, Gemma3-text (1B, 4B, 12B, 27B)** — needs more than the work in
  flight covers; its sliding and global attention layers do not share one RoPE
  base. *Verdict: pending.* If it is refused, use `google/gemma-2b-it`.

Studio does not read this page: it reports whatever the server's preflight
says, so the moment a family is accepted it stops being offered an alternative.

### What has actually been run

Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, and gemma-2b-it were each
converted and generated coherent text. TinyLlama-1.1B-Chat is the one
checkpoint measured end to end: 2,200 MB of fp16 weights become a 619.5 MB
`.raimodel` in 7.6 s at 22.9 MB peak RSS, and decode 21.8 tok/s
([BENCHMARKS.md](../BENCHMARKS.md)). Zephyr-7B-beta and SmolLM2-1.7B-Instruct
were converted and measured on disk. Every other row above is supported by
architecture, not by a retained run.

## Check before downloading 14 GB

Every config-level blocker is visible in `config.json`, which is a few
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

The three fields that decide it most often are `model_type`, `num_experts`, and
`rope_scaling`. Read the output against this:

| Field | Convert it | Do not download |
| --- | --- | --- |
| `model_type` | `llama`, `mistral`, `qwen2`, `gemma` | `mixtral`, `qwen3_moe`, `phi3`, `falcon`, `mpt`, `gpt_neox`, `gpt2` — and see the four families in flight for `qwen3`, `olmo2`, `gemma2`, `gemma3` |
| `num_experts` / `num_local_experts` | Absent or zero | Any positive value — no exceptions, whatever the family |
| `rope_scaling` | `null`, absent, `{"rope_type": "default"}`, or `{"rope_type": "llama3", ...}` | Any other `rope_type`, including `linear`, `dynamic`, `yarn` |
| `sliding_window` | Absent, or at least your `--max-context` | Present and below your `--max-context` — lower the context rather than dropping the model |
| `hidden_act` / `hidden_activation` | `silu`, `swish`, `gelu_pytorch_tanh`, or absent | Any other named activation |

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

<!-- FINALISE: if per-head QK norm lands, the per-layer list above gains two
     more f32 norm vectors and this section needs the same edit. -->

The kernels implement exactly this and nothing else: RMSNorm as `x / rms * w`,
rotary embeddings from one theta with an optional `llama3` frequency rescale,
full causal attention with grouped-query head mapping, and a gated MLP with
either a SiLU or a GeLU-tanh gate.

A model needing a weight or an operation outside those two lists has nowhere to
put it. It would be dropped at export and the resulting file would load cleanly
and generate nonsense, so the exporter rejects it instead.

### Sliding-window models

Mistral declares `sliding_window: 4096`. RAI always runs full causal attention.
Inside the window that is identical arithmetic; past it, the two diverge. The
exporter therefore accepts a sliding-window checkpoint up to the window length
and refuses a longer `--max-context`, naming the window in the error. Convert
Mistral-7B with `--max-context 4096` or lower. This is the one refusal that
needs no different model — the same folder converts at a shorter context.

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

A model refused for one of these is refused over a flag, not over its
architecture: raise `--group-size`, or lower `--max-context`, and convert it
again.

## What it would take to support the rest

Each of these is tractable. None is free. They are listed roughly in ascending
cost.

### Per-head QK norm — Qwen3, OLMo2, and half of Gemma3

<!-- FINALISE: in flight. Rewrite this subsection as shipped capability if it
     lands, or leave it here if it does not. -->

Two more f32 norm vectors per layer, each `head_dim` long, applied to Q and K
per head after projection and before RoPE. The layer section grows by
`2 * head_dim * 4` bytes and the reader grows two more `RMSNormWeights` views.
The existing `rms_norm` kernel does the arithmetic unchanged; it is called per
head instead of per token. Cost: a header flag and two extra section fields.
This is the cheapest of the remaining maths changes, and it is the work
currently in flight.

### Logit softcapping — Gemma2

<!-- FINALISE: in flight, same as above. -->

`softcap * tanh(x / softcap)` applied to the attention scores and again to the
final logits, with the two cap values carried in the header. No new parameters,
but two new points in the forward pass and a `tanh` over the whole vocabulary
on every step. Gemma2 also interleaves sliding-window and full attention layers,
which the KV cache and the attention kernel currently have no way to express.

### YaRN and other RoPE schemes — long-context fine-tunes

The header already carries a scaling-type byte and the `llama3` parameters.
`linear`, `dynamic`, and `yarn` each compute their inverse frequencies
differently, so each needs a `RoPETable::new` branch and its own
positional-accuracy test — this is where a subtle error shows up as quality
loss at long context rather than as a failure.

### Mixture of experts — Mixtral, Qwen3-MoE

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
