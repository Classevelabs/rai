# Installation

**Most people want [INSTALL.md](../INSTALL.md) at the repository root**, which
covers downloading a release archive, verifying its checksum, putting the
binaries on `PATH`, and a first generation. This page is the reference: build
requirements, source installs, the container, and model conversion.

RAI is distributed as prebuilt release archives, its public source repository,
and two crates.io packages, `classeve-rai-infer` and `classeve-rai-compress`.
The memory-service crates (`rai-server`, `rai-core`, `rem-nra`) build from
source only — they are `publish = false` on purpose; the note at the top of
their manifests says why.

## Requirements

- Rust 1.87 or newer. This repository pins 1.95.0 for repeatable checks, which
  `rustup` installs automatically from `rust-toolchain.toml`.
- x86-64 with AVX2, FMA, and F16C for the optimized inference path. Scalar
  fallbacks exist; ARM acceleration is not implemented.
- Python 3.9+ for calibrated (GPTQ) export and draft-model training. The `rai`
  binary converts without it. See [Converting a model](#converting-a-model).

## Source checkout

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --workspace --release --locked
```

That produces `rai` (the whole command-line surface) and `rai-server` (the
memory service), plus the deprecated `rai-convert` / `rai-generate` /
`rai-chat` wrappers.

The repository's `.cargo/config.toml` pins x86-64 builds to `target-cpu=x86-64-v2`,
the same floor the published release archives use, so the binary you just built
is safe to copy to another machine. It costs nothing: the AVX2/FMA/F16C kernels
are dispatched at runtime, not selected by the compile-time baseline. aarch64
builds are left at the toolchain default.

To tune a build to the machine in front of you — worth doing only for a local
benchmark, never for a binary anyone else will run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --workspace --release --locked
```

That binary is for the build machine alone. Run it on any CPU that lacks an
instruction the compiler chose to use and it dies with SIGILL, on startup, with
no diagnostic naming the cause.

Install the two end-user binary crates from a checkout:

```bash
cargo install --locked --path rai-infer     # rai, and the deprecated wrappers
cargo install --locked --path rai-server    # rai-server
```

The published crate can be installed by name — this covers `rai` and the
wrappers; `rai-server` installs from a checkout only:

```bash
cargo install classeve-rai-infer --locked
```

## Container

The container is an MCP stdio image and runs as a non-root user. It does not
expose a REST port because the built-in REST listener is loopback-only. Built
on amd64 it targets `x86-64-v3`; built on arm64 — which is what `docker build`
does by default on an Apple Silicon Mac — it takes the toolchain default and
runs the scalar kernels, because there is no NEON implementation.

```bash
docker build -t rai-server .
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run --rm -i rai-server mcp
```

Persisted memory must be mounted at a path writable by UID 10001 and explicitly
configured with `RAI_DATA_PATH`. Set `RAI_MCP_MUTATIONS_ENABLED=true` only when
the launching MCP client should be allowed to modify that state.

The same `Dockerfile` also builds an inference-CLI image carrying `rai` and no
server:

```bash
docker build --target cli -t rai-cli .
docker run --rm -v "$PWD:/work" -w /work rai-cli \
  rai convert ./TinyLlama-1.1B-Chat-v1.0 -o ./tinyllama.raimodel
```

Models are mounted, never baked in. `rai serve` is not exposed from that image
on purpose: it binds 127.0.0.1 and rejects non-localhost `Host`/`Origin`
headers, so a published container port would only ever refuse the connection.
Run `rai serve` on the host.

Both images default to `x86-64-v3` **when the image being built is x86-64**.
Pass `--build-arg RUST_TARGET_CPU=x86-64-v2` for the portable floor the release
archives use. The flag is not applied to arm64 builds, where an x86 processor
name means nothing.

## Converting a model

**Check compatibility first.** [MODELS.md](./MODELS.md) has the full table,
named checkpoints in both columns, and a recipe for checking a model's
`config.json` before you download its weights. It also records the one place
the two conversion paths differ: `rai convert` writes container v2 and handles
Qwen2/2.5, Llama-3.1/3.2, and Gemma; the Python exporters write container v1
and refuse all three.

| Path | Tool | Needs | Use it when |
| --- | --- | --- | --- |
| Round-to-nearest | `rai convert` | Nothing beyond the RAI build | Default. No Python, no torch. |
| Round-to-nearest | `export_rtn.py` | Python, torch, transformers | You want the reference implementation to compare against. |
| GPTQ (calibrated) | `export_raimodel.py`, `export_fast.py` | Python, torch, transformers, datasets, and a calibration corpus | You want calibrated quantization and can spend the time. |

### Default path: `rai convert` (no Python)

`rai convert` reads a HuggingFace checkpoint directory directly and streams the
`.safetensors` tensor by tensor, so peak memory does not grow with the model —
a 7B checkpoint converts on a 16 GB machine.

```bash
cargo build --workspace --release --locked

./target/release/rai convert /path/to/TinyLlama-1.1B-Chat-v1.0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `<MODEL_DIR>` | required | Positional: directory holding `config.json`, the safetensors weights, and `tokenizer.json` |
| `-o`, `--output <FILE>` | `<dirname>-q4.raimodel` | Destination `.raimodel` path |
| `--group-size <N>` | 128 | Columns per quantization group for the 4-bit linears |
| `--embed-group-size <N>` | 64 | Columns per quantization group for the 8-bit embedding table |
| `--max-context <TOKENS>` | the model's declared context | Context length baked into the RoPE table (hard cap 1,000,000) |
| `--tokenizer-out <FILE>` | next to the output | Where `tokenizer.json` is copied |
| `--quiet` | off | Suppress progress output |

Raise `--group-size` for models whose `hidden_size` or `intermediate_size`
exceeds 16,384; see the shape constraints in
[MODELS.md](./MODELS.md#shape-constraints-that-apply-to-every-model).
Lower `--max-context` for sliding-window models such as Mistral-7B.

### Calibrated path: the Python exporters

GPTQ needs a calibration corpus and a torch runtime.

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
# Round-to-nearest, the Python equivalent of `rai convert`
python rai-infer/scripts/export_rtn.py \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-q4.raimodel

# GPTQ, calibrated against wikitext by default
python rai-infer/scripts/export_raimodel.py \
  --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output tinyllama-1.1b-gptq.raimodel
```

`rai convert` and `export_rtn.py` produce byte-identical output for the same
checkpoint and options. On TinyLlama-1.1B the Rust path took 7.6 s against
188.8 s for the Python one, in 22.9 MB of RAM against 4,981 MB
([BENCHMARKS.md](../BENCHMARKS.md)). GPTQ is far slower than either: it needs
calibration data and a Cholesky inverse per layer.

Every path writes `tokenizer.json` next to the model file and refuses to
overwrite a different tokenizer already sitting there.

### Refusals

Both paths fail with the reason named rather than writing a file that loads
cleanly and generates nonsense. `rai convert` refuses a shared expert running
alongside the routed ones, `rope_type` values other than `default` and
`llama3`, an `lm_head` bias, activations other than SiLU and GeLU-tanh, and
any module tree that is not Llama-style. Routed mixture-of-experts (Mixtral,
Qwen3-MoE), per-head and full-width QK norms (Qwen3, OLMo2), Gemma3's
per-layer RoPE bases, and Gemma2's softcapping are stored and converted — they
are no longer refusals. The Python exporters refuse all of the above plus
projection bias vectors (Qwen2/2.5), every `rope_scaling` type
(Llama-3.1/3.2), and every Gemma variant. See [MODELS.md](./MODELS.md) for what
each refusal means, what to use instead, and what it would take to lift it.

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
rai run tinyllama-1.1b-q4.raimodel \
  --chat-template zephyr --prompt "Explain photosynthesis in simple terms."
```

Available templates: `none`, `few-shot`, `mistral`, `llama3`, `chatml`,
`zephyr`, `phi3`, `gemma`, and `auto`. `auto` identifies a family by probing
the tokenizer for sentinel tokens, so it detects Gemma, Mistral, Llama-3,
ChatML, Phi-3, and Zephyr-style vocabularies — but **not** TinyLlama-Chat,
whose `<|user|>` markers are ordinary text rather than vocabulary entries.
Pass `--chat-template zephyr` explicitly for it.
