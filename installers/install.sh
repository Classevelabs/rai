#!/bin/sh
# Install RAI on Linux or macOS:
#
#   curl -fsSL https://github.com/Classevelabs/rai/releases/latest/download/install.sh | sh
#
# It downloads the RAI release for this computer, checks it against the
# release's SHA256SUMS, installs it for you alone (no sudo, nothing outside
# your home folder), and offers to download a model to chat with.
#
# Options go after `sh -s --`, for example `... | sh -s -- --model qwen2.5-0.5b`:
#   --model NAME   also download this starter model (qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-7b)
#   --no-model     do not offer a model
#   --version X    install release X instead of the latest (for example 0.2.6)
#   --dir DIR      install into DIR (default: ~/.local/share/rai)
#   --no-path      do not add RAI to your PATH
# The same as environment variables: RAI_INSTALL_MODEL, RAI_NO_MODEL=1,
# RAI_VERSION, RAI_INSTALL_DIR, RAI_NO_PATH=1.

set -eu

REPO_URL="https://github.com/Classevelabs/rai"
VERSION="${RAI_VERSION:-}"
DIR="${RAI_INSTALL_DIR:-$HOME/.local/share/rai}"
BIN="${RAI_BIN_DIR:-$HOME/.local/bin}"
# A folder that already holds the release files, used instead of the internet.
# This is how the release is tested on clean machines before it is published.
FROM="${RAI_INSTALL_FROM:-}"
MODEL="${RAI_INSTALL_MODEL:-}"
NO_MODEL="${RAI_NO_MODEL:-0}"
NO_PATH="${RAI_NO_PATH:-0}"

say() { printf '%s\n' "$*"; }
die() { printf '\nRAI installer: %s\n\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --model) shift; [ $# -gt 0 ] || die "--model needs a name"; MODEL="$1" ;;
        --model=*) MODEL="${1#--model=}" ;;
        --no-model) NO_MODEL=1 ;;
        --version) shift; [ $# -gt 0 ] || die "--version needs a number"; VERSION="$1" ;;
        --version=*) VERSION="${1#--version=}" ;;
        --dir) shift; [ $# -gt 0 ] || die "--dir needs a folder"; DIR="$1" ;;
        --dir=*) DIR="${1#--dir=}" ;;
        --no-path) NO_PATH=1 ;;
        -h|--help) sed -n '2,19p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done
VERSION="${VERSION#v}"

# --- which build this computer needs -----------------------------------------
os=$(uname -s)
arch=$(uname -m)
case "$os" in
    Linux)
        case "$arch" in
            x86_64|amd64) target=x86_64-unknown-linux-gnu ;;
            *) die "RAI has Linux builds for x86-64 only, and this computer is $arch." ;;
        esac
        # The build needs glibc 2.28 or newer: Ubuntu 20.04, Debian 10,
        # RHEL 8, Fedora 29 and everything after them.
        if ldd --version 2>&1 | grep -qi musl; then
            die "RAI needs glibc, which this system (Alpine or another musl system) does not use."
        fi
        glibc=$(ldd --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+$' || true)
        if [ -n "$glibc" ]; then
            major=${glibc%%.*}; minor=${glibc#*.}
            if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 28 ]; }; then
                die "RAI needs glibc 2.28 or newer, and this system has $glibc. Ubuntu 20.04, Debian 10, RHEL 8 or newer will do."
            fi
        fi
        ;;
    Darwin)
        # An Intel terminal on Apple Silicon (Rosetta) still gets the native build.
        if [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
            arch=arm64
        fi
        case "$arch" in
            arm64) target=aarch64-apple-darwin ;;
            x86_64) target=x86_64-apple-darwin ;;
            *) die "RAI has no macOS build for $arch." ;;
        esac
        ;;
    *)
        die "this installer is for Linux and macOS. On Windows, run in PowerShell:
    irm $REPO_URL/releases/latest/download/install.ps1 | iex"
        ;;
esac

# --- tools ------------------------------------------------------------------------
if [ -z "$FROM" ]; then
    if command -v curl >/dev/null 2>&1; then
        download() { curl -fL --retry 5 --retry-delay 5 --progress-bar -o "$2" "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        download() { wget -q --show-progress -O "$2" "$1"; }
    else
        die "curl or wget is needed. Install one, for example: sudo apt install curl"
    fi
fi
command -v tar >/dev/null 2>&1 || die "tar is needed to unpack RAI."
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "sha256sum or shasum is needed to check the download."
fi

# --- which release ----------------------------------------------------------------
if [ -z "$VERSION" ]; then
    if [ -n "$FROM" ]; then
        found=$(ls "$FROM" 2>/dev/null | sed -n "s/^rai-\(.*\)-$target\.tar\.gz$/\1/p" | head -n 1)
        VERSION="$found"
    elif command -v curl >/dev/null 2>&1; then
        # GitHub answers /releases/latest with a redirect to the release's tag.
        latest=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$REPO_URL/releases/latest") || latest=""
        VERSION=${latest##*/v}
    else
        latest=$(wget -S --spider "$REPO_URL/releases/latest" 2>&1 | sed -n 's/^ *[Ll]ocation: .*\/tag\/v\(.*\)$/\1/p' | tail -n 1)
        VERSION="$latest"
    fi
fi
case "$VERSION" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) die "could not find out the latest RAI version. Name one with --version, for example --version 0.2.6" ;;
esac

archive="rai-$VERSION-$target.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

say ""
say "Installing RAI $VERSION for $target"
if [ -n "$FROM" ]; then
    cp "$FROM/$archive" "$FROM/SHA256SUMS" "$tmp/" || die "$FROM does not hold $archive and SHA256SUMS."
else
    base="$REPO_URL/releases/download/v$VERSION"
    download "$base/$archive" "$tmp/$archive" || die "could not download $base/$archive"
    download "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "could not download the release's SHA256SUMS"
fi

expected=$(awk -v f="$archive" '$2 == f || $2 == "*"f { print $1 }' "$tmp/SHA256SUMS")
actual=$(sha256 "$tmp/$archive")
[ -n "$expected" ] || die "SHA256SUMS has no entry for $archive."
[ "$actual" = "$expected" ] || die "the download does not match SHA256SUMS (got $actual, expected $expected). Nothing was installed."
say "Checked: SHA-256 matches the release."

tar -xzf "$tmp/$archive" -C "$tmp"
unpacked="$tmp/rai-$VERSION-$target"
[ -f "$unpacked/rai" ] || die "the archive has no rai program in it."

# Replace any earlier install in one move, so a failure leaves the old one.
# Only a folder that is empty or holds a previous RAI is ever replaced: --dir
# pointed at a folder of other things must not delete them.
if [ -d "$DIR" ] && [ ! -f "$DIR/rai" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
    die "$DIR already holds other files and is not an earlier RAI install. Choose an empty folder with --dir."
fi
mkdir -p "$(dirname -- "$DIR")"
rm -rf "$DIR.new" "$DIR.old"
cp -R "$unpacked" "$DIR.new"
if [ -d "$DIR" ]; then mv "$DIR" "$DIR.old"; fi
mv "$DIR.new" "$DIR"
rm -rf "$DIR.old"
chmod +x "$DIR/rai" "$DIR/launchers/rai-studio.sh" "$DIR/launchers/rai-get-model" 2>/dev/null || true

mkdir -p "$BIN"
ln -sf "$DIR/rai" "$BIN/rai"
ln -sf "$DIR/launchers/rai-studio.sh" "$BIN/rai-studio"
ln -sf "$DIR/launchers/rai-get-model" "$BIN/rai-get-model"
"$BIN/rai" --version >/dev/null 2>&1 || die "the installed rai does not start on this computer. Please report it at $REPO_URL/issues"
say "Installed: $DIR"

# --- PATH ---------------------------------------------------------------------------
on_path=0
case ":$PATH:" in *":$BIN:"*) on_path=1 ;; esac
if [ "$on_path" = 0 ] && [ "$NO_PATH" != 1 ]; then
    case "${SHELL:-}" in
        */zsh) profile="$HOME/.zshrc" ;;
        */bash) if [ "$os" = Darwin ]; then profile="$HOME/.bash_profile"; else profile="$HOME/.bashrc"; fi ;;
        *) profile="$HOME/.profile" ;;
    esac
    line="export PATH=\"$BIN:\$PATH\""
    if ! grep -qsF "$line" "$profile"; then
        printf '\n# Added by the RAI installer\n%s\n' "$line" >> "$profile"
        say "Added $BIN to your PATH in $profile."
    fi
fi

# --- a menu entry on a Linux desktop -----------------------------------------------
if [ "$os" = Linux ] && [ -f "$DIR/launchers/rai-studio.desktop" ]; then
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    launchers=$(printf '%s\n' "$DIR/launchers" | sed -e 's/\\/\\\\/g' -e 's/[\\&|]/\\&/g')
    sed "s|@LAUNCHER_DIR@|$launchers|" "$DIR/launchers/rai-studio.desktop" > "$apps/rai-studio.desktop"
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$apps" 2>/dev/null || true
    fi
fi

# --- a model to chat with --------------------------------------------------------------
home="${RAI_HOME:-$HOME/RAI}"
have_model=0
for path in "$home"/models/*/*.raimodel; do
    if [ -f "$path" ]; then have_model=1; break; fi
done
if [ "$NO_MODEL" != 1 ] && [ "$have_model" = 0 ]; then
    # Piped into sh, the script's own input is the pipe, so the question goes
    # to the terminal. A container build or a provisioning script has a
    # /dev/tty that passes -r and -w and still cannot be opened, and a failed
    # redirection would end this script after the install had succeeded, so
    # opening it is the test.
    if [ -z "$MODEL" ] && (: </dev/tty >/dev/tty) 2>/dev/null; then
        {
            printf '\nRAI needs a model to chat with. The recommended one is Qwen2.5 1.5B\n'
            printf 'Instruct, a 3.1 GB download. Download it now? [Y/n] '
        } > /dev/tty
        answer=""
        read -r answer < /dev/tty || answer=n
        case "$answer" in [nN]*) ;; *) MODEL=qwen2.5-1.5b ;; esac
    fi
    if [ -n "$MODEL" ]; then
        say ""
        "$BIN/rai-get-model" "$MODEL" --yes || say "The model download did not finish. Run it again any time with: rai-get-model $MODEL"
    fi
fi

say ""
say "RAI $VERSION is ready."
say ""
say "  Chat in your browser:   rai-studio"
say "  Get a model:            rai-get-model"
say "  Everything else:        rai --help"
if [ "$on_path" = 0 ]; then
    say ""
    say "Open a new terminal first, or run:  export PATH=\"$BIN:\$PATH\""
fi
say ""
