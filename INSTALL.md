# Installing RAI

RAI is four command-line programs and no installer:

| Binary | What it is for |
| --- | --- |
| `rai-convert` | Turn a HuggingFace checkpoint into a `.raimodel` file |
| `rai-generate` | Generate text from a `.raimodel` on the CPU |
| `rai-chat` | Serve a local chat UI over HTTP |
| `rai-server` | Local REST + MCP memory service |

Nothing runs as a service, nothing writes outside the paths you name, and no
GPU is required. `rai-chat` and `rai-server` listen on loopback only.

If you want to build from source instead, skip to
[Installing from source](#installing-from-source). For model export details,
Python requirements, and chat-template behaviour, see
[docs/INSTALL.md](./docs/INSTALL.md).

## 1. Download a release archive

Every tagged release publishes one archive per platform, plus a `SHA256SUMS`
file, at <https://github.com/Classevelabs/rai/releases>.

| Platform | Archive |
| --- | --- |
| Linux, x86-64 | `rai-<version>-x86_64-unknown-linux-gnu.tar.gz` |
| macOS, Intel | `rai-<version>-x86_64-apple-darwin.tar.gz` |
| Windows, x86-64 | `rai-<version>-x86_64-pc-windows-msvc.zip` |

There are no ARM builds. Apple Silicon and ARM64 Linux have to build from
source, and will get the scalar path — the optimized kernels are x86-64 AVX2
and there is no NEON implementation.

The released binaries are built for the **x86-64-v2** baseline, so they start on
any x86-64 CPU from roughly 2009 onward. The fast kernels need AVX2, FMA, and
F16C; RAI probes for those at runtime and uses them when present, so the same
download is both portable and fast on a modern machine.

## 2. Verify the checksum

Do this before you run anything. Put `SHA256SUMS` in the same directory as the
archive you downloaded.

Linux and macOS:

```bash
sha256sum --ignore-missing -c SHA256SUMS
# rai-0.2.0-x86_64-unknown-linux-gnu.tar.gz: OK
```

macOS without GNU coreutils:

```bash
shasum -a 256 --ignore-missing -c SHA256SUMS
```

Windows PowerShell:

```powershell
$file = "rai-0.2.0-x86_64-pc-windows-msvc.zip"
$want = (Select-String -Path SHA256SUMS -Pattern ([regex]::Escape($file))).Line.Split()[0]
$got  = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
if ($want -eq $got) { "OK" } else { throw "checksum mismatch" }
```

`--ignore-missing` is what lets you check one archive against a `SHA256SUMS`
that lists all three. If the check fails, delete the download; do not run it.

## 3. Put the binaries on your PATH

Unpack, then move the four binaries somewhere already on `PATH`. They are
self-contained and can live anywhere.

Linux and macOS:

```bash
tar xzf rai-0.2.0-x86_64-unknown-linux-gnu.tar.gz
cd rai-0.2.0-x86_64-unknown-linux-gnu
install -m 0755 rai-convert rai-generate rai-chat rai-server ~/.local/bin/
rai-generate --help
```

If `~/.local/bin` is not on your `PATH`, add it (`export
PATH="$HOME/.local/bin:$PATH"` in your shell profile), or use `/usr/local/bin`
with `sudo`.

macOS marks downloaded files with a quarantine attribute, and these binaries
are not notarized. Gatekeeper will refuse them until you clear it:

```bash
xattr -d com.apple.quarantine rai-convert rai-generate rai-chat rai-server
```

Windows PowerShell:

```powershell
Expand-Archive rai-0.2.0-x86_64-pc-windows-msvc.zip -DestinationPath $HOME\bin
$dir = "$HOME\bin\rai-0.2.0-x86_64-pc-windows-msvc"
[Environment]::SetEnvironmentVariable(
  "Path", "$([Environment]::GetEnvironmentVariable('Path','User'));$dir", "User")
# open a new terminal, then:
rai-generate --help
```

The Windows binaries are unsigned, so SmartScreen may warn on first run. The
checksum you verified in step 2 is the assurance here.

## 4. Sixty seconds to your first generation

You need a checkpoint on disk. Any plain Llama- or Mistral-architecture model
works; a ~1B model converts in well under a minute. TinyLlama-1.1B-Chat is the
one these instructions were verified against.

```bash
# Convert it once. No Python, no torch — rai-convert reads the checkpoint
# directory directly and writes tokenizer.json next to the model file.
rai-convert --model /path/to/TinyLlama-1.1B-Chat-v1.0 \
            --output tinyllama-q4.raimodel

# Generate.
rai-generate --model tinyllama-q4.raimodel \
             --tokenizer tokenizer.json \
             --chat-template zephyr \
             --prompt "Explain photosynthesis in simple terms."
```

Then, if you want a browser instead of a terminal:

```bash
rai-chat --model tinyllama-q4.raimodel --tokenizer tokenizer.json
# open http://127.0.0.1:8090
```

Not every checkpoint converts. RAI runs plain Llama- and Mistral-architecture
models; Qwen, Gemma, Llama-3.1/3.2, and mixture-of-experts checkpoints are
refused at conversion time with the reason named. Check
[docs/MODELS.md](./docs/MODELS.md) before downloading weights.

**The one thing that trips everybody up:** instruction-tuned checkpoints need
the prompt format they were trained on. Pass a bare instruction without
`--chat-template` and the model emits end-of-sequence immediately and prints
nothing at all. That is not a crash and not a broken conversion. TinyLlama-Chat
needs `--chat-template zephyr` specifically — `auto` cannot detect it, because
its `<|user|>` markers are ordinary text rather than vocabulary entries. See
[Running a chat model](./docs/INSTALL.md#running-a-chat-model).

The memory service is separate from inference and needs no model:

```bash
rai-server rest      # REST API on 127.0.0.1:3000
rai-server mcp       # MCP over stdio, for an MCP client
```

Read [docs/OPERATIONS.md](./docs/OPERATIONS.md) before pointing anything real at
`rai-server`; its configuration is entirely environment variables, and the
defaults are deliberately conservative.

## Installing from source

You need Rust 1.87 or newer. The repository pins 1.95.0 for repeatable checks,
which `rustup` will install automatically from `rust-toolchain.toml`.

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo install --locked --path rai-infer     # rai-convert, rai-generate, rai-chat
cargo install --locked --path rai-server    # rai-server
```

`cargo install` puts them in `~/.cargo/bin`, which `rustup` already added to
your `PATH`.

**Read this before you copy a source-built binary to another machine.** The
repository's `.cargo/config.toml` sets `target-cpu=native`, which is right for
a machine building for itself and wrong for anything else: the compiler is free
to emit instructions your build machine has and the target machine does not, and
the result dies with SIGILL — an illegal-instruction crash, on startup, with no
useful message. Override it the way the release workflow does:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo install --locked --path rai-infer
```

That is the same portable floor the published archives use, and it costs
nothing measurable: the hot kernels are runtime-dispatched, not compiled by
baseline.

## Container

The published `Dockerfile` builds the MCP stdio image. See
[docs/INSTALL.md](./docs/INSTALL.md#container) for how to run it, and
[Dockerfile](./Dockerfile) for the `--target` stages, including one that carries
the inference CLI as well as the server.

## Uninstalling

Delete the binaries. RAI creates no registry entries, no services, and no
config files. The only state it can leave behind is whatever you pointed
`RAI_DATA_PATH` at, and whatever `.raimodel` files you converted.
