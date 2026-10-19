#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
exec "$GROOT_PY" -m embodied_ai_lab.sft.launch "$@"
