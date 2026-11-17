# When something goes wrong

Find the message you see, or the closest description, below. If nothing here
helps, [SUPPORT.md](../SUPPORT.md) says where to ask.

## Installing

### "Windows protected your PC"

Windows shows this for programs it hasn't seen often. Choose **More info**,
then **Run anyway**. To check the file first, compare its fingerprint with the
release's `SHA256SUMS` ([INSTALL.md](../INSTALL.md#check-your-download)).

### "rai" cannot be opened (macOS)

macOS holds back programs downloaded through a browser. The terminal installer
avoids this. For an archive you unpacked yourself, allow the folder once:

```bash
xattr -dr com.apple.quarantine /path/to/the/unpacked/folder
```

RAI Studio's launcher prints this command, with your folder filled in, when it
sees the problem.

### `rai: command not found`, or "'rai' is not recognized"

The installer added RAI to your PATH, but a terminal that was already open
doesn't know yet. Open a new terminal window. If it still isn't found:

- **macOS and Linux:** run `export PATH="$HOME/.local/bin:$PATH"`, or reinstall
  without `--no-path`.
- **Windows:** sign out and back in once, or reinstall without
  `RAI_NO_PATH`.

### `version 'GLIBC_2.xx' not found` (Linux)

Your Linux is older than RAI supports: it needs glibc 2.28 or newer (Ubuntu
20.04, Debian 10, RHEL 8 or later). On an older system, build RAI from source
([docs/BUILDING.md](BUILDING.md)).

### "RAI has Linux builds for x86-64 only"

There is no ready-made build for ARM Linux (a Raspberry Pi, or an ARM server).
Build from source there ([docs/BUILDING.md](BUILDING.md)).

### "Permission denied" when starting a launcher (macOS and Linux)

The file lost its "can run" mark on the way, which happens when an archive is
unpacked by some tools. Restore it once:

```bash
chmod +x launchers/rai-studio.sh launchers/rai-studio.command launchers/rai-get-model
```

## Getting a model

### "not enough free space"

`rai-get-model` needs about one and a half times the download free while it
works. Free some space, or keep your models on another drive by setting
`RAI_HOME` to a folder there before running it:

```bash
RAI_HOME=/mnt/big-disk/RAI rai-get-model qwen2.5-1.5b      # macOS and Linux
```

```powershell
$env:RAI_HOME = 'D:\RAI'; rai-get-model qwen2.5-1.5b       # Windows
```

Set `RAI_HOME` for good in your system settings or shell profile, so RAI
Studio finds the models there too.

### The download stopped halfway

Run the same command again. It picks up where it stopped.

### "Hugging Face refused ... This model needs a login"

Some models, such as Llama and Gemma, ask you to accept their license on
Hugging Face first.

1. Open the model's page on huggingface.co while signed in, and accept the
   license.
2. Create a token with read access at
   <https://huggingface.co/settings/tokens>.
3. Run the command again with the token set:

```bash
HF_TOKEN=hf_your_token rai-get-model meta-llama/Llama-3.2-1B-Instruct
```

On Windows: `$env:HF_TOKEN = 'hf_your_token'`, then the same command.

### "RAI cannot convert this model"

The model is from a family RAI doesn't run, or uses a feature RAI doesn't
store. The message names the reason, and nothing large was downloaded.
[MODELS.md](MODELS.md) lists what works, with a similar model to take instead.

### The folder has `pytorch_model.bin` and no `.safetensors`

RAI doesn't read that format on purpose: loading it can run code hidden in the
file. Most models offer `.safetensors` files too; the error message tells you
how to convert the ones that don't.

## Chatting

### The browser didn't open

Open <http://localhost:8090> yourself. If you set `RAI_PORT`, use that number
instead of 8090.

### "port 8090 is already in use"

Another program uses port 8090, often a RAI Studio that is already running:
look for its window, or open <http://localhost:8090> to use it. To start
another one on a different port:

```bash
RAI_PORT=9000 rai-studio          # macOS and Linux
```

```powershell
$env:RAI_PORT = '9000'; rai-studio        # Windows
```

### RAI Studio says "No model yet"

There is no model in `~/RAI/models` (on Windows, `%USERPROFILE%\RAI\models`)
or next to the program. Run the command the page shows. When it finishes, the
model appears in RAI Studio's model list.

### The answers are nonsense, repeat themselves, or stop at once

- Use a model whose name ends in `-Instruct` or `-it`. Base models continue
  text; they aren't trained to answer questions.
- A model converted by RAI 0.2.5 or earlier has no record of its chat format.
  Convert it again, or get it again with `rai-get-model`.
- Larger models answer better. If `qwen2.5-0.5b` disappoints, try
  `qwen2.5-1.5b`.

### It's slow

- Close other busy programs: RAI uses every core it can.
- Try a smaller model. Speed falls as models grow.
- Open the system monitor in RAI Studio (the button at the top). If **AVX2**
  isn't lit, your processor runs RAI's portable code, which is slower. Apple
  Silicon Macs also run the portable code. A smaller model helps most there.

### The computer freezes, or RAI stops with an out-of-memory error

The model needs more memory than is free. Close other programs or take a
smaller model: it needs free memory of about its size on disk, plus room for
the conversation.

## Still stuck?

Open an issue at <https://github.com/Classevelabs/rai/issues> with what you
ran, what you expected and the full message you got. `rai --version` and your
operating system help too.
