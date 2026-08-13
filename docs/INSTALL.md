# Installation

**Most people want [INSTALL.md](../INSTALL.md) at the repository root**, which
covers downloading a release archive, verifying its checksum, putting the
binaries on `PATH`, and a first generation. This page is the reference: build
requirements, source installs, the container, and model conversion.

RAI is distributed as prebuilt release archives, its public source repository,
and five crates.io packages. The currently published v0.1.0 artifacts predate
the unreleased hardening work in this checkout; do not describe an untagged
checkout as the published crate release.

## Requirements

- Rust 1.87 or newer. This repository pins Rust 1.95.0 for repeatable checks.
- x86-64 with AVX2, FMA, and F16C for the optimized inference path. Scalar
  fallbacks exist, but ARM acceleration is not implemented.
- Python 3.9+ for calibrated (GPTQ) model export and draft-model training.
  Round-to-nearest conversion runs through the `rai-convert` binary and needs
  no Python. See [Converting a model](#converting-a-model).

## Source checkout

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --workspace --release --locked
```

The repository's `.cargo/config.toml` uses `target-cpu=native`; binaries built
that way are for the build machine and die with SIGILL on any CPU that lacks an
instruction the compiler chose to use. Override the flag before moving a binary
to a different CPU:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --workspace --release --locked
```

`x86-64-v2` is the floor the published release archives use. It costs nothing:
the AVX2/FMA/F16C kernels are dispatched at runtime, not selected by the
compile-time baseline.

Install the two end-user binary crates from a checkout:

```bash
cargo install --locked --path rai-infer     # rai-convert, rai-generate, rai-chat
cargo install --locked --path rai-server    # rai-server
```

The public v0.1.0 crates can be installed explicitly with:

```bash
cargo install classeve-rai-infer --version 0.1.0 --locked
cargo install classeve-rai-server --version 0.1.0 --locked
```

## Container

The container is an amd64/x86-64-v3 MCP stdio image and runs as a non-root
user. It does not expose a REST port because the built-in REST listener is
intentionally loopback-only.

```bash
docker build -t rai-server .
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run --rm -i rai-server mcp
```

Persisted memory must be mounted at a path writable by UID 10001 and explicitly
configured with `RAI_DATA_PATH`. Set `RAI_MCP_MUTATIONS_ENABLED=true` only when
the launching MCP client should be allowed to modify that state.

The same `Dockerfile` also builds an inference-CLI image, which carries
`rai-convert` and `rai-generate` and no server:

```bash
docker build --target cli -t rai-cli .
docker run --rm -v "$PWD:/work" -w /work rai-cli \
  rai-convert --model ./TinyLlama-1.1B-Chat-v1.0 --output ./tinyllama.raimodel
```

Models are mounted, never baked in. `rai-chat` is not in that image on purpose:
it binds 127.0.0.1 and rejects non-localhost `Host`/`Origin` headers, so a
published container port would only ever refuse the connection. Run `rai-chat`
on the host.

Both images default to `x86-64-v3`. Pass
`--build-arg RUST_TARGET_CPU=x86-64-v2` for the portable floor the release
archives use.

## Verification

See [RELEASE_READINESS.md](./RELEASE_READINESS.md) for the complete candidate
gate. A successful local build alone is not release approval.

## Converting a model

**Check compatibility first.** RAI runs plain Llama- and Mistral-architecture
checkpoints. Qwen, Gemma, Llama-3.1/3.2, and mixture-of-experts models are
refused at export. [docs/MODELS.md](./MODELS.md) has the full table, named
checkpoints in both columns, and a recipe for checking a model's `config.json`
before you download its weights.

Two conversion paths produce the same `.raimodel` format:

| Path | Tool | Needs | Use it when |
| --- | --- | --- | --- |
| Round-to-nearest | `rai-convert` | Nothing beyond the RAI build | Default. No Python, no torch. |
| Round-to-nearest | `export_rtn.py` | Python, torch, transformers | You already have the Python environment set up. |
| GPTQ (calibrated) | `export_raimodel.py`, `export_fast.py` | Python, torch, transformers, datasets, and a calibration corpus | You want the calibrated quantization and can spend the time. |

### Default path: `rai-convert` (no Python)

`rai-convert` is built with the workspace and reads a HuggingFace checkpoint
directory directly, so round-to-nearest conversion needs no Python
installation, no torch, and no virtual environment.

```bash
cargo build --workspace --release --locked

./target/release/rai-convert --model /path/to/TinyLlama-1.1B-Chat-v1.0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <hf-dir>` | required | Directory holding `config.json`, the safetensors weights, and the tokenizer |
| `--output <file>` | derived from the model name | Destination `.raimodel` path |
| `--group-size 128` | 128 | Columns per quantization group for the 4-bit linears |
| `--embed-group-size 64` | 64 | Columns per quantization group for the 8-bit embedding table |
| `--max-context 2048` | 2048 | Context length baked into the RoPE table |

Raise `--group-size` for models whose `hidden_size` or `intermediate_size`
exceeds 16,384; see the shape constraints in
[docs/MODELS.md](./MODELS.md#shape-constraints-that-apply-to-every-model).
Lower `--max-context` for sliding-window models such as Mistral-7B.

### Calibrated path: the Python exporters

GPTQ needs a calibration corpus and a torch runtime. The Python exporters also
remain the reference implementation of the format. This sequence was verified
end to end on 2026-08-09 (Windows 11, Python 3.11) converting
TinyLlama-1.1B-Chat; the measured results are in
[BENCHMARKS.md](../BENCHMARKS.md).

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip.exe
.venv/bin/pip install -r rai-infer/scripts/requirements-lock.txt
```

`requirements-lock.txt` pins the exact versions the benchmark ran on.
`requirements.txt` carries lower bounds and will resolve to whatever is newest,
which is not the combination those numbers were measured on. Neither file
installs `accelerate`: the exporters read weights on CPU and do not use
`device_map`. No GPU is required.

```bash
# Round-to-nearest, the Python equivalent of rai-convert
python rai-infer/scripts/export_rtn.py \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-q4.raimodel

# GPTQ, calibrated against wikitext by default
python rai-infer/scripts/export_raimodel.py \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-gptq.raimodel
```

`export_rtn.py` takes well under a minute for a 1B model. `export_raimodel.py`
and `export_fast.py` run GPTQ instead: they need calibration data and far more
time. On the measured TinyLlama run, round-to-nearest took 47 s and a
deliberately short GPTQ run took 1,780 s.

Every path writes `tokenizer.json` next to the model file and refuses to
overwrite a different tokenizer already sitting there.

### Refusals

Both paths enforce the same architecture preflight and fail with the reason
named rather than writing a file that loads cleanly and generates nonsense:
bias vectors on the attention or MLP projections (Qwen2/Qwen2.5), per-head QK
norms (Qwen3, Gemma3, OLMo2), RoPE scaling of any type other than `default`
(Llama-3.1/3.2), mixture-of-experts routing, logit softcapping, Gemma's
modified RMSNorm, a decoupled `head_dim`, and any module tree that is not
Llama-style. See [docs/MODELS.md](./MODELS.md) for what each refusal means and
what it would take to lift it.

### Downloading on Windows

`huggingface_hub` populates its cache with symlinks, which Windows refuses
without Developer Mode or an elevated shell:

```
OSError: [WinError 1314] A required privilege is not held by the client
```

Either enable Developer Mode, or set `HF_HUB_DISABLE_SYMLINKS=1` and download
into a plain directory with `snapshot_download(..., local_dir=...)`.

### Running a chat model

Instruction-tuned checkpoints need the prompt format they were trained on.
Passing a bare instruction makes the model emit end-of-sequence immediately and
produce no output at all:

```bash
rai-generate --model tinyllama-1.1b-q4.raimodel --tokenizer tokenizer.json \
  --chat-template zephyr --prompt "Explain photosynthesis in simple terms."
```

Available templates: `none`, `few-shot`, `mistral`, `llama3`, `chatml`,
`zephyr`, and `auto`. `auto` identifies a family by probing the tokenizer for
sentinel tokens, so it detects Mistral, Llama-3, and ChatML — but **not**
Zephyr-style models such as TinyLlama-Chat, whose `<|user|>` markers are
ordinary text rather than vocabulary entries. Pass `--chat-template zephyr`
explicitly for those.
