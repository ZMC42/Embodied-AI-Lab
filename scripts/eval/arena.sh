#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
PYTHONPATH="$ARENA_ROOT:$PYTHONPATH" "$GROOT_PY" -m isaaclab_arena_gr00t.utils.wait_for_gr00t_server \
  --host 127.0.0.1 --port 5555 --timeout-sec 600 --poll-interval-sec 5 --request-timeout-ms 5000
cd "$ARENA_ROOT"
# WebRTC 扩展自带 CUDA 12.0；优先使用 torch 配套的 runtime，避免缺少 CUDA 符号。
export LD_LIBRARY_PATH="$("$ARENA_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ARENA_PY" -m isaaclab_arena.evaluation.policy_runner \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml \
  --remote_host 127.0.0.1 --remote_port 5555 \
  --num_episodes 1 --num_envs 1 --seed 42 \
  --enable_cameras --device cpu --policy_device cuda \
  --kit_args="--/log/file=$LAB_ROOT/logs/kit.log" \
  --output_base_dir "$LAB_ROOT/logs/evaluation" "$@" \
  galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint
} 2>&1 | tee -a "$LAB_ROOT/logs/arena-eval.log"
