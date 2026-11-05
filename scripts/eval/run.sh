#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
exec "$ARENA_PY" -m embodied_ai_lab.eval.runner "$@"
