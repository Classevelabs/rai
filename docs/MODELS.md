# Models

RAI runs models stored in its own file format, `.raimodel`. You get one by
converting a model from [Hugging Face](https://huggingface.co), and
`rai-get-model` does the downloading and converting for you.

- [Starter models](#starter-models)
- [Any other model](#any-other-model)
- [Will my model work?](#will-my-model-work)
- [A model you already downloaded](#a-model-you-already-downloaded)
- [Conversion options](#conversion-options)
- [The Python exporters](#the-python-exporters)
- [Details](#details): why some models are refused, and the limits that apply
  to all of them

## Starter models

```bash
rai-get-model qwen2.5-1.5b
```

| Name | Download | On disk | |
| --- | --- | --- | --- |
| `qwen2.5-0.5b` | 1.0 GB | 0.3 GB | Small and fast. Good for trying RAI out on any computer. |
| `qwen2.5-1.5b` | 3.1 GB | 0.9 GB | Recommended. Clear answers at a good speed on a laptop. |
| `qwen2.5-7b` | 15.2 GB | 4.0 GB | Best answers, but slower. For computers with 8 GB of memory or more. |

These are Qwen2.5 Instruct models, released by Alibaba's Qwen team under the
Apache-2.0 license and free to download without a Hugging Face account. Each is
pinned to one upload, and every file is checked against its SHA-256
fingerprint before it is converted, so everyone gets exactly the same model.
The model's license is copied next to it.

The models go into `~/RAI/models/<name>` (on Windows,
`%USERPROFILE%\RAI\models\<name>`), or under `RAI_HOME` if you set it. The
download is deleted once the model is converted.

## Any other model

Give `rai-get-model` the model's name as Hugging Face shows it:

```bash
rai-get-model HuggingFaceTB/SmolLM2-1.7B-Instruct
```

It downloads the model's small settings files first and asks RAI whether it
can run it. Only then does it download the weights, convert them, and keep the
result in `~/RAI/models`. Add `--keep` to keep the downloaded originals too.

Some models, Llama and Gemma among them, ask you to accept their license on
Hugging Face before you can download them. Accept it on the model's page,
create a read token at <https://huggingface.co/settings/tokens>, and run:

```bash
HF_TOKEN=hf_your_token rai-get-model meta-llama/Llama-3.2-1B-Instruct
```

For chatting, take a model whose name ends in `-Instruct`, `-Chat` or `-it`.
The others are base models: they continue a text rather than answer it.

## Will my model work?

| Family | | Notes |
| --- | --- | --- |
| Llama 2, 3, 3.1, 3.2 | Works | |
| Mistral 7B (v0.1 to v0.3) | Works | |
| Mixtral 8x7B | Works | About 24 GB on disk, and it needs that much memory |
| Qwen 2 and 2.5 | Works | |
| Qwen 3 (0.6B to 32B) and Qwen3 30B-A3B | Works | |
| Gemma (first generation) | Works | |
| Gemma 2 | Works | Conversations up to 4,096 tokens ([why](#sliding-window-models)) |
| Gemma 3 270M and 1B | Works | Conversations up to 512 tokens |
| Gemma 3 4B, 12B and 27B | No | These also see images; take Gemma 2 or Qwen 3 of a similar size |
| Phi-3 mini 4k | Works | Conversations up to 2,047 tokens |
| Phi-3 128k, Phi-3.5 | No | Take `microsoft/Phi-3-mini-4k-instruct` |
| OLMo 2, OLMoE | Works | |
| TinyLlama, SmolLM, SmolLM2 | Works | |
| Fine-tunes of the above (Zephyr, OpenHermes, Vicuna, Nous-Hermes, ...) | Works | Fine-tuning changes the weights, not the design |
| Mixtral 8x22B, Qwen3 235B-A22B | No | Too big for one RAI file; take Mixtral 8x7B or Qwen3 30B-A3B |
| SmolLM3, StableLM, Granite, MiniCPM | No | Take SmolLM2 1.7B, or a Llama, Mistral or Qwen 3 of the same size |
| Phi-2, Falcon, GPT-NeoX, Pythia, MPT, GPT-2 | No | A different design; take Llama 3.2, Mistral 7B or SmolLM2 |
| Long-context versions that change position handling (`yarn`, `linear`, `dynamic` or `longrope` scaling) | No | Take the original model they were made from, or Llama 3.1 |
| Models with a shared expert (some Qwen MoE and DeepSeek-style models) | No | Take the same family without one |

You don't have to work this out yourself: `rai-get-model` and `rai check`
answer from the model's settings before anything large is downloaded, and the
answer names the reason.

Each of these has been converted and run, and gave sensible answers:
Qwen2.5 0.5B, Qwen3 0.6B, Llama 3.2 1B, Gemma 2B, Gemma 2 2B, Gemma 3 1B,
Phi-3 mini 4k, OLMo 2 1B, OLMoE 1B-7B and TinyLlama 1.1B. Every release is
installed on clean computers and has to answer a question correctly with the
Qwen2.5 0.5B starter before it is published.

## A model you already downloaded

Point RAI at the folder. `rai check` answers in a second, from the settings
alone:

```bash
rai check ./Qwen2.5-3B-Instruct
rai convert ./Qwen2.5-3B-Instruct -o qwen2.5-3b-q4.raimodel
```

RAI Studio's **Convert** screen does the same with buttons: **Check**, then
**Convert**.

The folder needs:

| File | |
| --- | --- |
| `config.json` | The model's settings |
| `model.safetensors`, or `model-00001-of-0000N.safetensors` files with `model.safetensors.index.json` | The weights |
| `tokenizer.json` | How text becomes tokens. Copied next to the converted model |

RAI does not read `pytorch_model.bin`, `.pt` or `.pth` files: loading those
can run code hidden inside them. Most models offer `.safetensors` too. For one
that doesn't, the error message gives you the command that converts it. A
folder with `tokenizer.model` and no `tokenizer.json` needs the fast tokenizer
saved once with Python's `transformers`
(`AutoTokenizer.from_pretrained(folder).save_pretrained(folder)`).

RAI also doesn't read `.gguf` files. Those are already compressed by another
program; RAI starts from the original weights so it controls the compression
itself.

Converting writes three files side by side: the model (`.raimodel`), its
tokenizer (`tokenizer.json`) and its chat format (`<name>.chat.json`). Keep
them together. Conversion reads the weights a piece at a time, so it needs
little memory: a 7B model converts on a 16 GB laptop.

## Conversion options

| Option | Default | What it does |
| --- | --- | --- |
| `-o`, `--output FILE` | `<folder>-q4.raimodel` | Where the model is written |
| `--max-context N` | the model's own | The longest conversation, in tokens, the file allows. Lower it to save memory |
| `--group-size N` | 128 | Smaller is more accurate and a little bigger: 64 adds about 5% to the file, 32 about 15% |
| `--calibration-text FILE` | none | Tunes the conversion on a sample of real text. Same file size, better accuracy, takes minutes instead of seconds |
| `--calibration-sequences N`, `--calibration-seq-len N` | 16, 512 | How much of the sample text is used |
| `--embed-group-size N` | 64 | The same as `--group-size`, for the word table |
| `--quiet` | off | No progress output |

**Use `--calibration-text` if you can.** Any plain text file of a few thousand
words that looks like what you will use the model for will do:

```bash
rai convert ./SmolLM2-1.7B-Instruct --calibration-text my-notes.txt
```

On SmolLM2 1.7B it lowered perplexity (lower is better) from 10.30 to 9.22.
It works for Llama, Mistral, TinyLlama, SmolLM, Qwen 2 and 2.5, Llama 3.1
and 3.2, and the first Gemma. For Qwen 3, Gemma 2 and 3, OLMo 2, Mixtral and
Qwen3-MoE it is refused with the reason; convert those without it.

A model converted with a `--group-size` other than 128 opens in RAI 0.2.5 and
later only.

## The Python exporters

`rai-infer/scripts` holds Python exporters, including GPTQ, the most accurate
of the conversions. They need Python 3.11, PyTorch and a calibration dataset,
and they support fewer families than `rai convert`: Llama, Mistral, TinyLlama
and SmolLM, but not Qwen, Llama 3.1 and 3.2, Gemma, Qwen 3, OLMo 2 or Phi-3.

```bash
python -m venv .venv
.venv/bin/pip install -r rai-infer/scripts/requirements-lock.txt     # Windows: .venv\Scripts\pip.exe
.venv/bin/python rai-infer/scripts/export_raimodel.py --model ./SmolLM2-1.7B-Instruct --output smollm2-gptq.raimodel
```

On SmolLM2 1.7B, GPTQ cost 11.2% in perplexity against the original model,
`rai convert --calibration-text` 16.6% and plain `rai convert` 30.3%
([BENCHMARKS.md](../BENCHMARKS.md#the-calibrated-quantizer-measured-end-to-end)).
`export_rtn.py` writes exactly the same file as plain `rai convert`, much more
slowly; it exists to check one against the other.

## Details

### Why some models are refused

A `.raimodel` file stores a fixed set of things: the model's dimensions, its
position settings (`rope_theta`, a second base for models whose layers
alternate, and `llama3` rescaling), the activation (SiLU or GeLU-tanh),
optional biases on the seven layer projections, optional per-head or
projection-wide QK norms, optional norms around each block, Gemma's embedding
scale and logit softcaps, the attention scale, and routed experts. RAI's
arithmetic does exactly these things and nothing else.

A model that needs anything outside that list would load and quietly compute
the wrong thing, so it is refused at conversion instead:

- **A shared expert** that runs for every token next to the routed ones.
- **Position scaling other than the default or `llama3`.** `yarn`, `linear`,
  `dynamic` and `longrope` each compute positions differently, and a mistake
  there only shows as worse answers in long conversations.
- **A bias on the output layer** (`lm_head.bias`).
- **Scales outside the weights**: Granite's and MiniCPM's multipliers.
- **Position encoding on part of a head or some layers only**
  (`partial_rotary_factor` below 1, `no_rope_layers`).
- **A LayerNorm with a bias** where RAI stores RMSNorm.
- **A different layer layout**: Falcon, GPT-NeoX, MPT, GPT-2, Phi-2.
- **Any `model_type` that has not been checked.** Matching names prove
  nothing: ERNIE-4.5 and EXAONE-4 use Llama's tensor names with different
  arithmetic. The families that convert are those whose `model_type` is
  `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_moe`, `gemma`, `gemma2`,
  `gemma3_text`, `phi3`, `olmo2`, `olmoe` or `mixtral`.

The converter checks the real weights too, not just the settings, and lists
every problem it finds, naming the first tensor and the reason.

### Sliding-window models

Some models look back over only their most recent tokens: Mistral 7B v0.1 and
Gemma 2 over 4,096, Phi-3 mini 4k over 2,047, Gemma 3 over 512 on most layers.
RAI always looks back over the whole conversation, which is the same
arithmetic up to that length and different past it. So RAI converts these
with their window as the longest conversation, without being asked, and
refuses a longer `--max-context` with the window named.

### Limits that apply to every model

| Limit | |
| --- | --- |
| Head size a multiple of 8 | What the fast attention code is written for |
| Even layer widths | Two 4-bit weights share each byte |
| Query heads a multiple of key-value heads | Grouped-query attention |
| At most 1,024 groups per row | Width divided by `--group-size`. Gemma 2 27B needs `--group-size 64` or more |
| Group sizes even, 2 to 254 | Stored in one byte |
| `--max-context` at most 1,000,000 | |
| A file at most 64 GiB | |

A model refused for one of these converts with a different `--group-size` or
`--max-context`.

### The file format

`rai-infer/src/format.rs` documents the byte layout, and
[ARCHITECTURE.md](ARCHITECTURE.md) shows where everything lives in the code.
