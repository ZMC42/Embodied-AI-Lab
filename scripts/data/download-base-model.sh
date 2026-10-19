#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
cd "$LAB_ROOT"
exec "$GROOT_PY" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "nvidia/GR00T-N1.6-3B",
    revision="d0814e7ecb19202e7c8468b46098b0b7ef3a6d61",
    allow_patterns=["*.json", "*.safetensors"],
    local_dir="models/GR00T-N1.6-3B",
)
PY
} 2>&1 | tee -a "$LAB_ROOT/logs/download-base-model.log"
