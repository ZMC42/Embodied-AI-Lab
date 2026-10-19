#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export UV_PROJECT_ENVIRONMENT="$(dirname "$(dirname "$GROOT_PY")")"
export GIT_LFS_SKIP_SMUDGE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
cd "$GROOT_ROOT"
uv sync --python 3.10
uv pip install --python "$GROOT_PY" debugpy
