#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rai-studio.command - launcher for the local rai chat UI (macOS).
#
# A .command file is double-clickable in Finder. Put it in the same folder as
# the `rai` binary and your .raimodel file, make it executable once
# (`chmod +x rai-studio.command`), then double-click it. It starts
# `rai serve`, waits for the port to come up, and opens your default browser.
# Ctrl+C, or closing the Terminal window, stops the server.
#
# Override the defaults with environment variables:
#   RAI_PORT=9000 ./rai-studio.command
#   RAI_MODEL=/models/tinyllama-1.1b-q4.raimodel ./rai-studio.command
# ---------------------------------------------------------------------------
set -u

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${RAI_PORT:-8090}"
URL="http://localhost:${PORT}/"

die() {
    printf '\n%s\n\n' "$*" >&2
    # Finder-launched .command windows close on exit; hold the message.
    if [ -t 0 ]; then
        read -r -p "Press Enter to close. " _ || true
    fi
    exit 1
}

# --- locate the rai binary: beside this script first, then PATH -------------
if [ -x "${HERE}/rai" ]; then
    RAI="${HERE}/rai"
elif RAI="$(command -v rai 2>/dev/null)"; then
    :
else
    die "  Could not find the rai binary.

  Put it in ${HERE} or on your PATH, then run this launcher again."
fi

# --- pick a model -----------------------------------------------------------
if [ -n "${RAI_MODEL:-}" ]; then
    [ -f "${RAI_MODEL}" ] || die "  RAI_MODEL is set to \"${RAI_MODEL}\" but that file does not exist."
    MODEL="${RAI_MODEL}"
else
    models=()
    for path in "${HERE}"/*.raimodel; do
        [ -f "${path}" ] && models+=("${path}")
    done

    case "${#models[@]}" in
        0)
            die "  No .raimodel file found in this folder:
    ${HERE}

  Convert a HuggingFace checkpoint first, for example:
    \"${RAI}\" convert /path/to/TinyLlama-1.1B-Chat -o \"${HERE}/tinyllama.raimodel\"

  Then run this launcher again."
            ;;
        1)
            MODEL="${models[0]}"
            ;;
        *)
            listing=""
            for path in "${models[@]}"; do
                listing="${listing}    $(basename -- "${path}")"$'\n'
            done
            die "  More than one .raimodel file is in this folder, so it is not
  obvious which one to start:

${listing}
  Pick one by setting RAI_MODEL and running this launcher again:
    RAI_MODEL=${HERE}/<name>.raimodel $0"
            ;;
    esac
fi

printf '  Model:  %s\n' "${MODEL}"
printf '  Server: %s\n\n' "${URL}"
printf '  Waiting for the model to load, then opening your browser.\n'
printf '  Press Ctrl+C to stop the server.\n\n'

# --- start the server and make sure it dies with this script ----------------
"${RAI}" serve "${MODEL}" --port "${PORT}" &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

port_is_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && exec 3<&- 3>&-
}

deadline=$((SECONDS + 120))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        die "  The server stopped before it was ready; see the messages above."
    fi
    if port_is_open; then
        open "${URL}" >/dev/null 2>&1 ||
            printf '  Could not open a browser. Visit %s yourself.\n' "${URL}"
        break
    fi
    sleep 0.5
done

# Hand the terminal back to the server so Ctrl+C reaches it.
wait "${SERVER_PID}"
