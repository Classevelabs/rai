#!/bin/bash
# rai-studio.command - double-click in Finder to chat with a model in your
# browser. It runs rai-studio.sh beside it, the one launcher for macOS and
# Linux, so the two can never drift apart.
exec "$(cd -- "$(dirname -- "$0")" && pwd)/rai-studio.sh" "$@"
