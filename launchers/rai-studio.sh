#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rai-studio.sh - chat with a model in your browser (Linux and macOS).
#
# It finds the rai program and a model, starts `rai serve`, and opens your
# browser at http://localhost:8090. With no model on this computer yet, it
# offers to download the recommended one, and starts the page either way.
# Ctrl+C, or closing the terminal, stops the server.
#
# Settings, as environment variables:
#   RAI_PORT=9000                       another port
#   RAI_MODEL=/path/to/model.raimodel   the model to start with
#   RAI_HOME=/data/rai                  where models live (default: ~/RAI)
#   RAI_NO_BROWSER=1                    do not open a browser
# ---------------------------------------------------------------------------
set -u

# The folder this script really lives in: installs link to it as rai-studio
# from a bin folder, and rai-get-model and the archive root are found from here.
SELF="${BASH_SOURCE[0]}"
while [ -L "${SELF}" ]; do
    LINK="$(readlink "${SELF}")"
    case "${LINK}" in
        /*) SELF="${LINK}" ;;
        *) SELF="$(dirname -- "${SELF}")/${LINK}" ;;
    esac
done
HERE="$(cd -- "$(dirname -- "${SELF}")" && pwd)"

# The release archive ships the binary at its root and the launchers one level
# down, so the parent directory counts as home too. Looking only beside this
# script is what made a downloaded archive fail on the first double-click.
# A program is a regular file that can run: a folder also passes -x, and in
# the packages' layout /usr/lib/rai/../rai is one.
is_program() { [ -f "$1" ] && [ -x "$1" ]; }

ROOT="${HERE}"
if ! is_program "${HERE}/rai" && is_program "${HERE}/../rai"; then
    ROOT="$(cd -- "${HERE}/.." && pwd)"
fi
PORT="${RAI_PORT:-8090}"
URL="http://localhost:${PORT}/"
RAI_HOME="${RAI_HOME:-${HOME}/RAI}"
export RAI_HOME

die() {
    printf '\n%s\n\n' "$*" >&2
    # Keep the message on screen when this was double-clicked from a file
    # manager or Finder and the window would otherwise close at once.
    if [ -t 0 ]; then
        read -r -p "Press Enter to close. " _ || true
    fi
    exit 1
}

# --- locate the rai binary: beside this script, the archive root, then PATH --
if is_program "${HERE}/rai"; then
    RAI="${HERE}/rai"
elif is_program "${ROOT}/rai"; then
    RAI="${ROOT}/rai"
elif RAI="$(command -v rai 2>/dev/null)"; then
    :
else
    die "  Could not find the rai program.

  Put it in ${ROOT} or on your PATH, then run this again."
fi

# macOS holds back a program it has marked as downloaded through a browser
# until the mark is cleared. The installer script never sets it; an archive
# opened in Finder does, so say exactly how to clear it.
if [ "$(uname -s)" = Darwin ] && xattr -p com.apple.quarantine "${RAI}" >/dev/null 2>&1; then
    die "  macOS is holding back rai because it was downloaded through a browser.
  Allow it once by running this in Terminal, then open this again:

    xattr -dr com.apple.quarantine \"${ROOT}\""
fi

# --- pick a model: RAI_MODEL, then beside rai, then $RAI_HOME/models ---------
MODEL=""
count_models() {
    # Counts the files among its arguments into COUNT (a glob that matched
    # nothing is not one) and keeps the last of them in LAST.
    local n=0 path
    for path in "$@"; do
        [ -f "${path}" ] || continue
        n=$((n + 1))
        LAST="${path}"
    done
    COUNT="${n}"
}

if [ -n "${RAI_MODEL:-}" ]; then
    [ -f "${RAI_MODEL}" ] || die "  RAI_MODEL is set to \"${RAI_MODEL}\" but that file does not exist."
    MODEL="${RAI_MODEL}"
else
    # One model next to rai, as an unpacked archive keeps it, starts at once;
    # several start the page with none loaded, so you choose there.
    LAST=""
    count_models "${ROOT}"/*.raimodel
    if [ "${COUNT}" -eq 0 ]; then
        # The models rai-get-model converted, one folder each.
        count_models "${RAI_HOME}"/models/*/*.raimodel
    fi
    if [ "${COUNT}" -eq 1 ]; then
        MODEL="${LAST}"
    elif [ "${COUNT}" -eq 0 ]; then
        GET=""
        if is_program "${HERE}/rai-get-model"; then
            GET="${HERE}/rai-get-model"
        elif command -v rai-get-model >/dev/null 2>&1; then
            GET="$(command -v rai-get-model)"
        fi
        if [ -n "${GET}" ] && [ -t 0 ]; then
            printf '\n  RAI needs a model to chat with, and there is none on this computer yet.\n'
            printf '  The recommended one is Qwen2.5 1.5B Instruct, a 3.1 GB download. It is\n'
            printf '  saved in %s and ready to chat as soon as it is converted.\n\n' "${RAI_HOME}"
            read -r -p "  Download it now? [Y/n] " answer || answer=n
            case "${answer}" in
                [nN]*) ;;
                *)
                    if "${GET}" qwen2.5-1.5b --yes; then
                        candidate="${RAI_HOME}/models/qwen2.5-1.5b/qwen2.5-1.5b-q4.raimodel"
                        [ -f "${candidate}" ] && MODEL="${candidate}"
                    fi
                    ;;
            esac
        fi
    fi
fi

if [ -n "${MODEL}" ]; then
    printf '  Model:  %s\n' "${MODEL}"
else
    printf '  Model:  none yet - choose or get one in the page that opens\n'
fi
printf '  Server: %s\n\n' "${URL}"
printf '  Waiting for the server, then opening your browser.\n'
printf '  Press Ctrl+C to stop it.\n\n'

# --- start the server and make sure it dies with this script ----------------
# From the folder rai is in, so the page's library lists models kept beside it
# as well as those under $RAI_HOME.
cd -- "${ROOT}" || die "  Cannot enter ${ROOT}."
if [ -n "${MODEL}" ]; then
    "${RAI}" serve "${MODEL}" --port "${PORT}" &
else
    "${RAI}" serve --port "${PORT}" &
fi
SERVER_PID=$!
# Ctrl+C or a close stops the server; STOPPING tells the end of this script
# that the exit code it then sees was asked for.
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
trap 'STOPPING=1; kill "${SERVER_PID}" 2>/dev/null || true' INT TERM

# --- wait for the port, then open the browser -------------------------------
open_url() {
    # `open` is macOS's; on Debian and Ubuntu the same name is openvt.
    local openers="xdg-open gio sensible-browser x-www-browser firefox"
    [ "$(uname -s)" = Darwin ] && openers="open"
    for opener in ${openers}; do
        if command -v "${opener}" >/dev/null 2>&1; then
            case "${opener}" in
                gio) "${opener}" open "${URL}" >/dev/null 2>&1 && return 0 ;;
                *) "${opener}" "${URL}" >/dev/null 2>&1 && return 0 ;;
            esac
        fi
    done
    return 1
}

port_is_open() {
    # Bash's /dev/tcp needs no external tool and is present everywhere bash is.
    # Both loopback families: rai serve binds IPv6 loopback on a machine
    # without an IPv4 one, and a probe of 127.0.0.1 alone never saw it.
    local host
    for host in 127.0.0.1 ::1; do
        if (exec 3<>"/dev/tcp/${host}/${PORT}") 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

deadline=$((SECONDS + 120))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}"
        code=$?
        trap - EXIT
        die "  rai serve exited with code ${code} before it was ready; see the messages above."
    fi
    if port_is_open; then
        if [ -z "${RAI_NO_BROWSER:-}" ]; then
            open_url || printf '  Could not open a browser. Visit %s yourself.\n' "${URL}"
        fi
        break
    fi
    sleep 0.5
done
if [ "${SECONDS}" -ge "${deadline}" ]; then
    # A large model can take longer to load than this waits.
    printf '  Still not answering after two minutes. If the model is still loading,\n'
    printf '  open %s yourself once it is ready.\n' "${URL}"
fi

# Hand the terminal back to the server so Ctrl+C reaches it. A server that
# stops on its own with an error says so even after the page opened: another
# program holding the port answers the probe above, and rai serve then exits.
code=0
wait "${SERVER_PID}" || code=$?
trap - EXIT
if [ "${code}" -ne 0 ] && [ -z "${STOPPING:-}" ]; then
    printf '\n  rai serve exited with code %s; see the messages above.\n' "${code}" >&2
    exit "${code}"
fi
