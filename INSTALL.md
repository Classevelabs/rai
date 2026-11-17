# Installing RAI

Pick the way that suits you. Every one of them installs RAI for you alone,
needs no administrator password (except the Linux packages), and leaves your
models in a folder of their own, so you can reinstall or remove RAI without
losing them.

- [Windows](#windows)
- [macOS and Linux](#macos-and-linux)
- [Ubuntu, Debian, Fedora, RHEL and openSUSE packages](#linux-packages)
- [Without installing](#without-installing)
- [Your first model](#your-first-model)
- [Check your download](#check-your-download)
- [Updating](#updating)
- [Removing RAI](#removing-rai)

If something goes wrong, [docs/TROUBLESHOOTING.md](https://github.com/Classevelabs/rai/blob/main/docs/TROUBLESHOOTING.md)
lists the usual problems and their fixes.

## Windows

You need Windows 10 or 11 on a 64-bit Intel or AMD processor.

**With Setup.** Download `rai-<version>-windows-x64-setup.exe` from the
[latest release](https://github.com/Classevelabs/rai/releases/latest) and run
it. If Windows says "Windows protected your PC", choose **More info**, then
**Run anyway**.

Setup installs RAI into `%LOCALAPPDATA%\Programs\RAI`, puts **RAI Studio** in
the Start menu and on the desktop, and adds RAI to your PATH, so `rai`,
`rai-studio` and `rai-get-model` work in any new terminal. When it finishes,
open RAI Studio.

**With PowerShell.** The same install from one line, without the desktop
shortcut:

```powershell
irm https://github.com/Classevelabs/rai/releases/latest/download/install.ps1 | iex
```

It downloads the release, checks it against the release's `SHA256SUMS`,
installs it and offers to download a model. To change what it does, set one
of these first, in the same PowerShell window:

| Setting | What it does |
| --- | --- |
| `$env:RAI_INSTALL_MODEL = 'qwen2.5-0.5b'` | Downloads this model without asking |
| `$env:RAI_NO_MODEL = '1'` | Doesn't offer a model |
| `$env:RAI_VERSION = '0.2.6'` | Installs that release instead of the latest |
| `$env:RAI_INSTALL_DIR = 'D:\Tools\RAI'` | Installs somewhere else |
| `$env:RAI_NO_PATH = '1'` | Leaves your PATH alone |

## macOS and Linux

You need macOS 10.12 or newer on an Intel Mac, macOS 11 or newer on an Apple
Silicon Mac, or a 64-bit Intel or AMD Linux with glibc 2.28 or newer (Ubuntu
20.04, Debian 10, RHEL 8, or anything newer). In a terminal:

```bash
curl -fsSL https://github.com/Classevelabs/rai/releases/latest/download/install.sh | sh
```

It downloads the release for your computer, checks it against the release's
`SHA256SUMS`, and installs it into `~/.local/share/rai`, with `rai`,
`rai-studio` and `rai-get-model` linked into `~/.local/bin`. If that folder is
not on your PATH yet, it adds a line to your shell's profile (`~/.zshrc`,
`~/.bashrc`, `~/.bash_profile` or `~/.profile`), so open a new terminal
afterwards. On a Linux desktop it also adds RAI Studio to your app menu.
Nothing is written outside your home folder, and `sudo` is never needed.

Options go after `sh -s --`:

```bash
curl -fsSL https://github.com/Classevelabs/rai/releases/latest/download/install.sh | sh -s -- --model qwen2.5-0.5b
```

| Option | What it does |
| --- | --- |
| `--model NAME` | Downloads this model without asking |
| `--no-model` | Doesn't offer a model |
| `--version 0.2.6` | Installs that release instead of the latest |
| `--dir DIR` | Installs into DIR instead of `~/.local/share/rai` |
| `--no-path` | Leaves your PATH and shell profile alone |

## Linux packages

Download the package from the
[latest release](https://github.com/Classevelabs/rai/releases/latest), then:

```bash
# Ubuntu and Debian
sudo apt install ./rai_<version>_amd64.deb

# Fedora, RHEL, Rocky and AlmaLinux
sudo dnf install ./rai-<version>-1.x86_64.rpm

# openSUSE
sudo zypper install --allow-unsigned-rpm ./rai-<version>-1.x86_64.rpm
```

The package puts `rai`, `rai-studio` and `rai-get-model` in `/usr/bin` and adds
RAI Studio to the app menu.

## Without installing

Each release also has a plain archive per platform. Unpack it anywhere and
start RAI Studio from the `launchers` folder inside. On a Mac, do the step
below the table first.

| Computer | Archive | Start |
| --- | --- | --- |
| Windows | `rai-<version>-x86_64-pc-windows-msvc.zip` | double-click `launchers\rai-studio.cmd` |
| Mac with Apple Silicon | `rai-<version>-aarch64-apple-darwin.tar.gz` | double-click `launchers/rai-studio.command` |
| Mac with Intel | `rai-<version>-x86_64-apple-darwin.tar.gz` | double-click `launchers/rai-studio.command` |
| Linux | `rai-<version>-x86_64-unknown-linux-gnu.tar.gz` | run `launchers/rai-studio.sh` |

On an Apple Silicon Mac, take the Apple Silicon archive: the Intel one would
run under translation and be slower.

macOS holds back programs downloaded through a browser, and an archive
unpacked in Finder passes that on: double-clicking `rai-studio.command` would
only say it cannot be opened. Allow the unpacked folder once, in Terminal,
before the first double-click:

```bash
xattr -dr com.apple.quarantine /path/to/the/unpacked/folder
```

## Your first model

RAI Studio offers a model the first time it starts, and the installers offer
one too. To get one yourself:

```bash
rai-get-model qwen2.5-1.5b
```

| Name | Download | On disk | |
| --- | --- | --- | --- |
| `qwen2.5-0.5b` | 1.0 GB | 0.3 GB | Small and fast. Good for trying RAI out on any computer. |
| `qwen2.5-1.5b` | 3.1 GB | 0.9 GB | Recommended. Clear answers at a good speed on a laptop. |
| `qwen2.5-7b` | 15.2 GB | 4.0 GB | Best answers, but slower. For computers with 8 GB of memory or more. |

Models go into `~/RAI/models` (on Windows, `%USERPROFILE%\RAI\models`), one
folder each. Set `RAI_HOME` to keep them somewhere else; RAI Studio, `rai
models` and `rai-get-model` all follow it. While it works, `rai-get-model`
needs free space of about one and a half times the download. When it is done,
it deletes the download and keeps only the converted model.

Any other model from a supported family works the same way, by its Hugging
Face name. [docs/MODELS.md](https://github.com/Classevelabs/rai/blob/main/docs/MODELS.md) has the list.

```bash
rai-get-model HuggingFaceTB/SmolLM2-1.7B-Instruct
```

## Check your download

The installers do this for you. For a file you downloaded yourself, every
release has a `SHA256SUMS` file listing each download's fingerprint. Put it
next to your download, then:

```bash
sha256sum --ignore-missing -c SHA256SUMS         # Linux
shasum -a 256 --ignore-missing -c SHA256SUMS     # macOS
```

On Windows, in PowerShell, compare the output of
`Get-FileHash <file> -Algorithm SHA256` with the line for that file in
`SHA256SUMS`; capital letters don't matter. If they differ in anything else,
delete the download and don't run it.

Each file also carries a signed build record that ties it to this repository
and the exact commit it was built from. With the
[GitHub CLI](https://cli.github.com):

```bash
gh attestation verify <file> --repo Classevelabs/rai
```

## Updating

Run the same installer again: the new Setup, the same PowerShell or terminal
line, or the new package. It replaces the program and keeps your models.

The folder RAI is installed in belongs to RAI, and an update replaces it.
Keep your own files, models included, in `~/RAI` or anywhere else.

## Removing RAI

Close RAI Studio first: a running copy keeps its files in use.

**Windows, installed with Setup.** Open **Settings > Apps**, find **RAI**, and
choose **Uninstall**. It asks whether to delete your downloaded models too;
the answer is no unless you say yes.

**Windows, installed with PowerShell.** In PowerShell:

```powershell
$dir = Join-Path $env:LOCALAPPDATA 'Programs\RAI'
Remove-Item -Recurse -Force $dir
Remove-Item -Force (Join-Path ([Environment]::GetFolderPath('Programs')) 'RAI Studio.lnk')
$key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
$path = $key.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
$key.SetValue('Path', (($path.Split(';') | Where-Object { $_ -and $_.TrimEnd('\') -ne $dir }) -join ';'), 'ExpandString')
```

**macOS and Linux, installed with the terminal line:**

```bash
rm -rf ~/.local/share/rai ~/.local/bin/rai ~/.local/bin/rai-studio ~/.local/bin/rai-get-model
rm -f ~/.local/share/applications/rai-studio.desktop
```

If the installer added RAI to your PATH, your shell profile also has a line
`# Added by the RAI installer` with an `export PATH=` line under it. Delete
both.

**Linux packages:** `sudo apt remove rai`, `sudo dnf remove rai` or
`sudo zypper remove rai`.

**Your models** are in `~/RAI` (on Windows, `%USERPROFILE%\RAI`). Delete that
folder if you don't want them any more.
