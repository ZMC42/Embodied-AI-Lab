#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export UV_PROJECT_ENVIRONMENT="$(dirname "$(dirname "$ARENA_PY")")"
export GIT_LFS_SKIP_SMUDGE=1
cd "$ARENA_ROOT"
uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$ARENA_PY" 'cmake<4'
PATH="$(dirname "$ARENA_PY"):$PATH" uv pip install --python "$ARENA_PY" egl-probe==1.0.2
uv sync --frozen --extra dev --group gr00t-client
# Preserve the viewer and its validated additions from the existing environment.
uv pip install --python "$ARENA_PY" debugpy 'viser==1.1.1' 'rich==14.3.4' \
  'msgspec==0.22.0' 'zstandard==0.25.0'
