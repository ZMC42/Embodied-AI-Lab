#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
cd "$LAB_ROOT"
exec "$ARENA_PY" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "nvidia/Arena-G1-Loco-Manipulation-Task",
    repo_type="dataset",
    revision="07de42fdade76102fc09d921c7b6200d0b31ec01",
    allow_patterns=["README.md", "arena_g1_loco_manipulation_dataset_annotated.hdf5", "lerobot/meta/*"],
    local_dir="data/locomanipulation/official",
)
PY
} 2>&1 | tee -a "$LAB_ROOT/logs/download-demos.log"
