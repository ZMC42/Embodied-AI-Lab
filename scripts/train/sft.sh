#!/usr/bin/env bash
# Run sustained training under systemd --user; do not enable boot startup.
set -eo pipefail
source "$(dirname "$0")/../env.sh"
exec "$GROOT_PY" -m embodied_ai_lab.sft.launch "$@"
