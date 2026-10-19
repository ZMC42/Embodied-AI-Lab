#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
# Limit glibc arena retention while staging CPU checkpoints.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
recipe=ppo-offload
if [[ $# -gt 0 && "$1" != *=* && "$1" != -* ]]; then
  recipe="$1"
  shift
fi
export LAB_ROOT
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export RLINF_EXT_MODULE="${RLINF_EXT_MODULE:-embodied_ai_lab.rlinf.extension}"
export LD_LIBRARY_PATH="$("$RL_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$GROOT_REF_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/erl-training
cd "$ARENA_ROOT"
exec "$RL_PY" -m embodied_ai_lab.rlinf.train \
  --config-path "$LAB_ROOT/configs/rl/locomanip" --config-name "$recipe" \
  "hydra.searchpath=[file://$EMBODIED_PATH/config]" "$@"
