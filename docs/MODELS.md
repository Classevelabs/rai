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
| Qwen3 (dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B) | **Converts** | — | Per-head QK norm is stored in the v2 layer section. Verified: Qwen3-0.6B converts to 399 MB and generates. |
| Gemma2 (2B, 9B, 27B) | **Converts** | — | Logit softcapping and the two block-output norms are stored. Convert with `--max-context 4096` or lower (see sliding windows). Verified: gemma-2-2b-it converts to 1.70 GB and generates. |
| OLMo2 | **Refused** | `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` | Its QK norm spans all heads at once rather than one head, and it has no `input_layernorm`. Two separate gaps; see below. |
| Gemma3, Gemma3-text | **Refused** | `google/gemma-2b-it` | Its sliding and global layers use different RoPE bases; the header stores one. See below. |
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

### Two families that look close but are not

Qwen3 and Gemma2 look like near neighbours of models RAI already ran, and they
are — they convert now. OLMo2 and Gemma3 look equally close and are not. The
difference is worth stating, because the family name does not tell you.

- **OLMo2 (7B, 13B).** It has `q_norm` and `k_norm`, the same tensor names
  Qwen3 uses, so it reads as the same feature. It is not. Qwen3's norm is one
  `head_dim`-long vector applied inside each head; OLMo2's is
  `num_heads * head_dim` long and applied to the whole projection output before
  the tensor is ever split into heads. Different arithmetic, not a different
  size. OLMo2 also has no `input_layernorm` at all — it normalizes after each
  block rather than before, and a layer here stores two norms in the pre-block
  positions. Two independent gaps. Use `meta-llama/Llama-3.1-8B-Instruct` or
  `mistralai/Mistral-7B-Instruct-v0.3`.
- **Gemma3, Gemma3-text (1B, 4B, 12B, 27B).** Per-head QK norm and softcapping
  are both stored now, so neither is what stops it. Gemma3 interleaves local
  and global attention layers that use *different RoPE bases* —
  `rope_local_base_freq` of 10,000 against `rope_theta` of 1,000,000. The
  header stores one theta and the runtime builds one rotary table, so five
  layers in six would rotate at the wrong frequency: a file that loads cleanly
  and degrades quietly. That needs per-layer RoPE tables, not a flag. Use
  `google/gemma-2b-it`.

Studio does not read this page: it reports whatever the server's preflight
says, so a family that becomes supported stops being offered an alternative.

### What has actually been run

Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, gemma-2b-it, **Qwen3-0.6B** and
**gemma-2-2b-it** were each converted and generated coherent text. The last two
are the checkpoints that exercise the capabilities added for them: Qwen3-0.6B
writes 399 MB with the QK-norm flag set, gemma-2-2b-it writes 1.70 GB with the
sandwich-norm flag and both softcaps. Their stored norm vectors were also read
back from the container at documented byte offsets and compared against the
source `.safetensors` — bit-exact, so the capability is stored correctly and
not merely accepted. TinyLlama-1.1B-Chat is the one
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
| `model_type` | `llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `gemma2` | `mixtral`, `qwen3_moe`, `olmo2`, `gemma3`, `gemma3_text`, `phi3`, `falcon`, `mpt`, `gpt_neox`, `gpt2` |
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
tensor 'model.layers.0.self_attn.q_norm.weight' holds 4096 values but
head_dim is 128; this container implements the per-head QK norm shared
across heads (Qwen3, Gemma3), not a norm over the whole projection (OLMo2).
```

That is the OLMo2 case, and it is deliberately specific: the tensor has the
name RAI supports and the wrong width, which a message saying only
"unsupported architecture" would have hidden.

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

- a header holding the architecture dimensions, **one** `rope_theta`,
  `norm_eps`, the activation code, the RoPE scaling parameters, the bias mask,
  the embedding scale, the two logit softcaps, and the attention scale
- one 8-bit quantized embedding table
- per layer: seven 4-bit projections (`q`, `k`, `v`, `o`, `gate`, `up`,
  `down`), an optional f32 bias vector per projection, an optional pair of
  `head_dim`-long f32 QK norm vectors, an optional pair of hidden-sized f32
  block-output norms, and two f32 RMSNorm weight vectors
- one final f32 RMSNorm
- an optional 4-bit `lm_head` (omitted when the head is tied to the embedding)

Each optional block is *absent* rather than zero-length when unused, which is
why a model needing none of them still writes a file byte-identical to what
earlier versions produced.

The kernels implement exactly this and nothing else: RMSNorm as `x / rms * w`,
applied per token or per head; rotary embeddings from **one** theta with an
optional `llama3` frequency rescale; full causal attention with grouped-query
head mapping, an optional non-default scale and an optional `tanh` softcap on
the scores; and a gated MLP with either a SiLU or a GeLU-tanh gate.

The emphasis on *one* theta is where Gemma3 falls out: a model that rotates
different layers at different frequencies has nowhere to say so.

A model needing a weight or an operation outside those two lists has nowhere to
put it. It would be dropped at export and the resulting file would load cleanly
and generate nonsense, so the exporter rejects it instead.

### Sliding-window models

Mistral and Gemma2 declare `sliding_window: 4096`. RAI always runs full causal
attention.
Inside the window that is identical arithmetic; past it, the two diverge. The
exporter therefore accepts a sliding-window checkpoint up to the window length
and refuses a longer `--max-context`, naming the window in the error. Convert
Mistral-7B and Gemma2 with `--max-context 4096` or lower. This is the one
refusal that needs no different model — the same folder converts at a shorter
context, and Studio offers that as a button rather than an alternative model.

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

### Full-width QK norm — OLMo2

RAI stores a `head_dim`-long norm applied inside each head. OLMo2 needs a
`num_heads * head_dim`-long norm applied to the projection output before it is
split into heads. That is a different kernel, not a wider vector, and it would
need its own flag so the two cannot be confused at load time. OLMo2 would also
need pre-norm made optional, since it has no `input_layernorm` — a structural
change to the layer, not a stored weight.

### Per-layer RoPE tables — Gemma3

Gemma3 rotates its local and global attention layers at different bases. The
header carries one `rope_theta` and the runtime builds one table shared by
every layer, so this needs a second stored base, a per-layer selector, and two
live tables in memory. The arithmetic is already implemented; what is missing
is the ability to say "this layer, not that one", which touches the header, the
layer section, and the attention entry points at once.

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
