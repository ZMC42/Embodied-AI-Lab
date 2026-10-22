#!/usr/bin/env bash
# Export an RL policy, evaluate it against its SFT parent, and summarize both.
set -eo pipefail

usage() {
  cat <<'EOF'
Usage: compare-checkpoints.sh SFT_CHECKPOINT RL_WEIGHTS [gpu|cpu|both] [episodes] [num_envs] [seed] [output_dir]

SFT_CHECKPOINT must be the checkpoint used to initialize RL training.
RL_WEIGHTS is global_step_N/actor/model_state_dict/full_weights.pt.
Defaults: gpu, 100 episodes per model/backend, 5 environments, seed 2000.
Results: logs/checkpoint-eval/<timestamp>/{sft,rl}/, comparison.json, report.md.
The exported policy is saved under models/rl-export-<timestamp>/.
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if (( $# < 2 || $# > 7 )); then
  usage >&2
  exit 2
fi

source "$(dirname "$0")/../env.sh"
sft_model="$(realpath "$1")"
rl_weights="$(realpath "$2")"
backend="${3:-gpu}"
episodes="${4:-100}"
num_envs="${5:-5}"
seed="${6:-2000}"
case "$backend" in
  gpu|cpu|both) ;;
  *) echo 'backend must be gpu, cpu or both' >&2; exit 2 ;;
esac
run_id="$(date +%Y%m%d-%H%M%S-%N)"
output="$(realpath -m "${7:-$LAB_ROOT/logs/checkpoint-eval/$run_id}")"
rl_model="$LAB_ROOT/models/rl-export-$run_id"
cd "$LAB_ROOT"
mkdir -p "$(dirname "$output")"
mkdir "$output"
printf 'Evaluation output: %s\nExported RL model: %s\n' "$output" "$rl_model"

"$LAB_ROOT/scripts/eval/export-rl.sh" \
  --weights "$rl_weights" --base-model "$sft_model" --output "$rl_model"

# Use the same reference/SDPA, BF16, single-sample inference for both models.
export EVAL_RL_SERVER=1
"$LAB_ROOT/scripts/eval/baseline.sh" \
  "$output/sft" "$backend" "$episodes" "$num_envs" "$sft_model" "$seed"
"$LAB_ROOT/scripts/eval/baseline.sh" \
  "$output/rl" "$backend" "$episodes" "$num_envs" "$rl_model" "$seed"

exec "$RL_PY" -m embodied_ai_lab.eval.compare_checkpoints "$output" "$backend"
