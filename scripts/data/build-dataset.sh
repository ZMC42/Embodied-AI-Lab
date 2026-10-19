#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
cd "$LAB_ROOT"

# 上游生成器会覆盖 HDF5；已有产物必须先由调用者保全。
test ! -e data/locomanipulation/generated.hdf5
test ! -e data/locomanipulation/generated/lerobot

echo "$(date -Is) generation started"
bash scripts/data/generate-demos.sh
"$ARENA_PY" - <<'PY'
import h5py

with h5py.File("data/locomanipulation/generated.hdf5", "r") as source:
    assert len(source["data"]) == 100, "Expected 100 generated demos"
    assert all(bool(demo.attrs["success"]) for demo in source["data"].values())
PY
echo "$(date -Is) conversion started"
bash scripts/data/convert-demos.sh
echo "$(date -Is) validation started"
"$ARENA_PY" -m embodied_ai_lab.data.compare_datasets
echo "$(date -Is) dataset build completed"
} 2>&1 | tee -a "$LAB_ROOT/logs/build-dataset.log"
