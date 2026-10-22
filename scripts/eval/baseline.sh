#!/usr/bin/env bash
# Evaluate one physics engine and clean up its child processes.
set -eo pipefail
source "$(dirname "$0")/../env.sh"
preset=physx
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --presets)
      [[ $# -ge 2 ]] || { echo '--presets requires a value' >&2; exit 2; }
      preset="$2"
      shift 2 ;;
    --presets=*) preset="${1#*=}"; shift ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]}"
[[ $# -ge 1 && $# -le 6 ]] || {
  echo 'Usage: baseline.sh OUTPUT_DIR [cpu|gpu|both] [episodes] [num_envs] [model_path] [seed] [--presets NAME]' >&2
  exit 2
}
run_dir="$1"
backend="${2:-both}"
episodes="${3:-100}"
num_envs="${4:-5}"
seed="${6:-42}"
case "$preset" in
  physx) ;;
  newton)
    [[ "$backend" == gpu ]] || { echo 'newton requires gpu (cpu/both are unsupported)' >&2; exit 2; } ;;
  *) echo 'physics preset must be physx or newton' >&2; exit 2 ;;
esac
case "$backend" in
  both) backends=(cpu gpu) ;;
  cpu|gpu) backends=("$backend") ;;
  *) echo 'physics device must be cpu, gpu or both' >&2; exit 2 ;;
esac
model_path="$(realpath "${5:-$LAB_ROOT/models/GR00T-N1.6-G1/checkpoint-20000}")"
server_python="$GROOT_PY"
server_source="$GROOT_ROOT"
# RL 导出必须使用训练端的 reference/SDPA，避免不同 attention 栈改变 BF16 动作。
# 对官方起点设 EVAL_RL_SERVER=1，可在相同推理环境下建立 RL 对照。
if [[ -f "$model_path/rl_export.json" || "${EVAL_RL_SERVER:-0}" == 1 ]]; then
  export EVAL_RL_SERVER=1
  server_python="$RL_PY"
  server_source="$GROOT_REF_ROOT"
else
  export EVAL_RL_SERVER=0
fi
mkdir -p "$run_dir"
run_dir="$(realpath "$run_dir")"
server_pid=
monitor_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for physics in "${backends[@]}"; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 \
    > "$run_dir/gpu-$physics.csv" &
  monitor_pid=$!
  cd "$server_source"
  PYTHONPATH="$server_source:$PYTHONPATH" EVAL_SEED="$seed" "$server_python" -m embodied_ai_lab.eval.server \
    --modality-config-path "$ARENA_ROOT/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py" \
    --model-path "$model_path" \
    --embodiment-tag NEW_EMBODIMENT --device cuda --host 127.0.0.1 --port 5555 \
    > "$run_dir/server-$physics.log" 2>&1 &
  server_pid=$!
  PYTHONPATH="$ARENA_ROOT:$PYTHONPATH" "$GROOT_PY" -m isaaclab_arena_gr00t.utils.wait_for_gr00t_server \
    --host 127.0.0.1 --port 5555 --timeout-sec 600 --poll-interval-sec 5 --request-timeout-ms 5000
  cd "$ARENA_ROOT"
  device=cpu
  [[ "$physics" == gpu ]] && device=cuda
  LD_LIBRARY_PATH="$("$ARENA_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ARENA_PY" -m embodied_ai_lab.eval.baseline \
    --presets "$preset" \
    --device "$device" --num_envs "$num_envs" --seed "$seed" --enable_cameras \
    --episodes "$episodes" --output "$run_dir/$physics" --model-path "$model_path" \
    --kit_args="--/log/file=$run_dir/kit-$physics.log" \
    galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint \
    > "$run_dir/eval-$physics.log" 2>&1
  cleanup
done
if [[ "$backend" == both ]]; then
  "$ARENA_PY" -m embodied_ai_lab.eval.summary "$run_dir" \
    > "$run_dir/summary-stdout.log"
fi
