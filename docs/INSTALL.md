# Installation

RAI is distributed through its public source repository and five crates.io
packages. The currently published v0.1.0 artifacts predate the unreleased
hardening work in this checkout; do not describe an untagged checkout as the
published crate release.

## Requirements

- Rust 1.87 or newer. This repository pins Rust 1.95.0 for repeatable checks.
- x86-64 with AVX2, FMA, and F16C for the optimized inference path. Scalar
  fallbacks exist, but ARM acceleration is not implemented.
- Python 3.9+ only for model export or draft-model training. See
  [Converting a model](#converting-a-model) for the exact, verified setup.

## Source checkout

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --workspace --release --locked
```

The repository's `.cargo/config.toml` uses `target-cpu=native`; binaries built
that way are for the build machine. Remove or override that flag before moving
a binary to a different CPU.

Install the two end-user binary crates from a checkout:

```bash
cargo install --path rai-infer --locked
cargo install --path rai-server --locked
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

## Verification

See [RELEASE_READINESS.md](./RELEASE_READINESS.md) for the complete candidate
gate. A successful local build alone is not release approval.

## Converting a model

The exporters are plain Python and need no GPU. This sequence was verified end
to end on 2026-08-09 (Windows 11, Python 3.11) converting TinyLlama-1.1B-Chat;
the measured results are in [BENCHMARKS.md](../BENCHMARKS.md).

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip.exe
.venv/bin/pip install -r rai-infer/scripts/requirements-lock.txt
```

`requirements-lock.txt` pins the exact versions the benchmark ran on.
`requirements.txt` carries lower bounds only and will resolve to whatever is
newest, which is not the combination those numbers were measured on. Neither
file installs `accelerate`: the exporters read weights on CPU and do not use
`device_map`.

Download a checkpoint, then convert it:

```bash
python rai-infer/scripts/export_rtn.py \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-q4.raimodel
```

`export_rtn.py` is round-to-nearest and takes well under a minute for a 1B
model. `export_raimodel.py` and `export_fast.py` run GPTQ instead: better
quality, but they need calibration data and far more time.

The exporter refuses checkpoints this format cannot represent — bias vectors on
the attention or MLP projections (Qwen2/Qwen2.5), per-head QK norms (Qwen3,
Gemma3), RoPE scaling (Llama-3.1/3.2), mixture-of-experts routing, logit
softcapping, and Gemma's modified RMSNorm. It fails before calibration with the
reason named, rather than writing a file that loads cleanly and generates
nonsense. Plain Llama- and Mistral-architecture checkpoints convert directly.

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
