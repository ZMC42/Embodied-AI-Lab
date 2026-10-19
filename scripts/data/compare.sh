#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
cd "$LAB_ROOT"
exec "$ARENA_PY" -m embodied_ai_lab.data.compare_datasets "$@"
