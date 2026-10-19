#!/usr/bin/env bash
# Shared paths; virtual environments stay on local storage.
export LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARENA_ROOT="$LAB_ROOT/third_party/IsaacLab-Arena"
export GROOT_ROOT="$ARENA_ROOT/submodules/Isaac-GR00T"
export GROOT_REF_ROOT="$LAB_ROOT/third_party/Isaac-GR00T-reference"
export GROOT_RL_ROOT="$LAB_ROOT/third_party/Isaac-GR00T-rl"
export RLINF_ROOT="$LAB_ROOT/third_party/RLinf"
export ARENA_PY="$LAB_ROOT/.venvs/arena/bin/python"
export GROOT_PY="$LAB_ROOT/.venvs/groot/bin/python"
export RL_PY="$LAB_ROOT/.venvs/rl/bin/python"
export HF_HOME="$LAB_ROOT/cache/huggingface"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$LAB_ROOT/tmp/runtime"
mkdir -p "$TMPDIR" "$LAB_ROOT/logs" "$LAB_ROOT/data/assets/"{http,https,git}
ln -sfn "$LAB_ROOT/data/assets/http" "$TMPDIR/http"
ln -sfn "$LAB_ROOT/data/assets/https" "$TMPDIR/https"
ln -sfn "$LAB_ROOT/data/assets/git" "$TMPDIR/asset_cache"
# Upstream RecorderManager writes to this fixed path.
mkdir -p /tmp/isaaclab "$LAB_ROOT/data/isaaclab-recorder-logs"
ln -sfn "$LAB_ROOT/data/isaaclab-recorder-logs" /tmp/isaaclab/logs
