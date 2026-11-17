# RAI

[![Latest release](https://img.shields.io/github/v/release/Classevelabs/rai)](https://github.com/Classevelabs/rai/releases/latest) [![License](https://img.shields.io/github/license/Classevelabs/rai)](LICENSE)

RAI runs language models on your own computer, on the processor alone. There
is no graphics card to buy, no Python to set up and no account to make, and
what you type stays on your machine.

Built by [ClassEve](https://classeve.com). Free to use under the Apache-2.0
license.

> **Official repository.** This is the only official repository for RAI.
> ClassEve's complete list of official accounts is at [classeve.com/official](https://classeve.com/official).
> The GitHub account `github.com/ClassEve` is an unrelated third party, not affiliated with ClassEve.

## Install

**Windows 10 and 11.** Download `rai-<version>-windows-x64-setup.exe` from the
[latest release](https://github.com/Classevelabs/rai/releases/latest) and run
it. If Windows says "Windows protected your PC", choose **More info**, then
**Run anyway**. Or paste this into PowerShell:

```powershell
irm https://github.com/Classevelabs/rai/releases/latest/download/install.ps1 | iex
```

**macOS and Linux.** Paste this into a terminal:

```bash
curl -fsSL https://github.com/Classevelabs/rai/releases/latest/download/install.sh | sh
```

Ubuntu, Debian, Fedora, RHEL and openSUSE can also use the `.deb` or `.rpm`
package from the release page. [INSTALL.md](INSTALL.md) has every way to
install, how to update and remove RAI, and what to do when something goes
wrong.

## Start chatting

Open **RAI Studio** from the Start menu or your app menu, or type `rai-studio`
in a terminal. The first time, it offers to download a model to talk to. Say
yes, wait for the download, and the chat opens in your browser at
<http://localhost:8090>.

Prefer the terminal?

```bash
rai-get-model qwen2.5-1.5b
rai run ~/RAI/models/qwen2.5-1.5b/qwen2.5-1.5b-q4.raimodel --prompt "Why is the sky blue?"
```

On Windows the model lands in `%USERPROFILE%\RAI\models` instead.
[docs/USING.md](docs/USING.md) covers Studio, every command and the chat API.

## Models

Three models are ready to download by name:

| Name | Download | On disk | |
| --- | --- | --- | --- |
| `qwen2.5-0.5b` | 1.0 GB | 0.3 GB | Small and fast. Good for trying RAI out on any computer. |
| `qwen2.5-1.5b` | 3.1 GB | 0.9 GB | Recommended. Clear answers at a good speed on a laptop. |
| `qwen2.5-7b` | 15.2 GB | 4.0 GB | Best answers, but slower. For computers with 8 GB of memory or more. |

`rai-get-model` fetches each one from Hugging Face at a fixed version, checks
every file against its known fingerprint, converts it and deletes the
download, so only the "on disk" size stays.

Most other models on Hugging Face from these families work too: Llama 2, 3,
3.1 and 3.2, Mistral, Mixtral 8x7B, Qwen 2, 2.5 and 3, Gemma, Gemma 2, the
text-only Gemma 3 sizes, Phi-3 mini 4k, OLMo 2, OLMoE, TinyLlama, SmolLM and
their fine-tunes. Give `rai-get-model` the model's name, such as
`rai-get-model Qwen/Qwen2.5-3B-Instruct`; it reads the model's settings first
and stops before the big download if RAI cannot run it.
[docs/MODELS.md](docs/MODELS.md) lists what works, what doesn't and why.

## What it needs

- **Windows** 10 or 11, **macOS** 10.12 or newer (Intel) or 11 or newer
  (Apple Silicon), or **Linux** with glibc 2.28 or newer: Ubuntu 20.04,
  Debian 10, RHEL 8, or anything newer.
- A 64-bit Intel or AMD processor from 2011 or later, or an Apple Silicon Mac.
  Processors with AVX2, which most Intel Core and AMD Ryzen chips since 2015
  have, take RAI's fast path.
- Free memory of about the model's size on disk, plus some room for the
  conversation.
- No graphics card.

## Private by default

The `rai` program never connects to the internet. RAI Studio listens only to
your own computer and refuses requests from websites. `rai-get-model` talks to
Hugging Face, and only when you run it. There are no accounts, no telemetry
and no update checks.

## How fast

Measured with RAI 0.2.0 on a laptop processor with 4 cores, using the
TinyLlama 1.1B model:

- RAI writes 21.8 tokens a second in 629 MB of memory. The same model in
  Hugging Face `transformers` writes 4.3 tokens a second and needs 7 times the
  memory. A token is about three quarters of a word.
- `rai convert` turns the model into a RAI file in 7.6 seconds, using 23 MB of
  memory. The same conversion in Python with PyTorch takes 188.8 seconds and
  4,981 MB.
- A 7B model writes about 3 tokens a second on the same laptop, and converts
  in 82.6 seconds in 26 MB of memory, so even a 16 GB laptop converts models
  that size.

[BENCHMARKS.md](BENCHMARKS.md) has the method and the full record, including
what 4-bit storage costs in accuracy.

## Commands

| Command | What it does |
| --- | --- |
| `rai-studio` | Opens the chat in your browser |
| `rai-get-model <name>` | Downloads a model and converts it |
| `rai run <model> --prompt "..."` | Answers one prompt in the terminal |
| `rai serve <model>` | Starts the chat page and the chat API |
| `rai check <folder>` | Says whether a downloaded model will convert |
| `rai convert <folder>` | Converts a Hugging Face model to a `.raimodel` file |
| `rai models` | Lists the models on this computer |
| `rai perplexity <model> --text <file>` | Measures how well a model predicts a text |

`rai <command> --help` shows each command's options, with examples.

## For developers

RAI is written in Rust. To build it from source:

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --release --locked -p classeve-rai-infer
```

The engine is also a library on crates.io,
[`classeve-rai-infer`](https://crates.io/crates/classeve-rai-infer).
[docs/BUILDING.md](docs/BUILDING.md) covers building, the container image and
using the library. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how
the engine is put together, and [CONTRIBUTING.md](CONTRIBUTING.md) how to send
a change.

The workspace holds two crates that make up RAI, `rai-infer` (the engine and
the `rai` program) and `rai-compress` (quantization research tools), and three
that belong to a separate memory service and are not part of RAI: `rai-core`,
`rai-server` and `rem-nra`. Those three are in no release download.

## Help

- Something not working: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md),
  then [SUPPORT.md](SUPPORT.md).
- Security issues: [SECURITY.md](SECURITY.md). Please don't file them as
  public issues.
- What changed in each release: [CHANGELOG.md](CHANGELOG.md).
- Why RAI behaves the way it does: [BEHAVIOUR-DECISIONS.md](BEHAVIOUR-DECISIONS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
Copyright 2025-2026 ClassEve.
