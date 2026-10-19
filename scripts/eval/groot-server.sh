#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
cd "$GROOT_ROOT"
exec "$GROOT_PY" -m gr00t.eval.run_gr00t_server \
  --modality-config-path "$ARENA_ROOT/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py" \
  --model-path "$LAB_ROOT/models/GR00T-N1.6-G1/checkpoint-20000" \
  --embodiment-tag NEW_EMBODIMENT --device cuda --host 127.0.0.1 --port 5555
} 2>&1 | tee -a "$LAB_ROOT/logs/groot-server.log"
