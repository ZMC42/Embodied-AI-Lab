#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
export LD_LIBRARY_PATH="$("$ARENA_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ARENA_ROOT"
exec "$ARENA_PY" -m submodules.IsaacLab.scripts.imitation_learning.isaaclab_mimic.generate_dataset \
  --input_file "$LAB_ROOT/data/locomanipulation/official/arena_g1_loco_manipulation_dataset_annotated.hdf5" \
  --output_file "$LAB_ROOT/data/locomanipulation/generated.hdf5" \
  --generation_num_trials 100 --num_envs 1 --device cpu --enable_cameras \
  --kit_args="--/log/file=$LAB_ROOT/logs/generation-kit.log" \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task galileo_g1_locomanip_pick_and_place \
  --mimic --object brown_box --embodiment g1_wbc_pink "$@"
} 2>&1 | tee -a "$LAB_ROOT/logs/generate-demos.log"
