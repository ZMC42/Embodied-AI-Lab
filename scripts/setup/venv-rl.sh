#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export UV_PROJECT_ENVIRONMENT="$(dirname "$(dirname "$RL_PY")")"
export GIT_LFS_SKIP_SMUDGE=1
cd "$ARENA_ROOT"
uv sync --frozen --inexact
# The validated freeze deliberately includes overrides of upstream requirements.
uv pip install --python "$RL_PY" --no-deps -r "$LAB_ROOT/scripts/setup/rl-requirements.txt"
"$RL_PY" "$LAB_ROOT/scripts/setup/fix-newton-mujoco-export.py"
uv pip install --python "$RL_PY" --no-deps -e "$RLINF_ROOT" -e "$GROOT_REF_ROOT"
