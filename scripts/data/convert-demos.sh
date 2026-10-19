#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../env.sh"
{
input_hdf5="$(realpath "${1:-$LAB_ROOT/data/locomanipulation/generated.hdf5}")"
conversion_yaml="$LAB_ROOT/logs/conversion/$(basename "${input_hdf5%.hdf5}").yaml"
cd "$ARENA_ROOT"
"$ARENA_PY" - "$input_hdf5" "$conversion_yaml" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
config_path = Path(sys.argv[2])
template = Path("isaaclab_arena_gr00t/lerobot/config/g1_locomanip_config.yaml")
config = yaml.safe_load(template.read_text())
config.update(data_root=str(source.parent), hdf5_name=source.name)
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
"$ARENA_PY" -m isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot --yaml_file "$conversion_yaml"
cd "$GROOT_ROOT"
exec "$GROOT_PY" - "${input_hdf5%.hdf5}/lerobot" <<'PY'
import json
from pathlib import Path
import sys
from gr00t.data.stats import generate_stats

dataset = Path(sys.argv[1])
# 上游转换器对不足一个 chunk 的数据写入 0，并沿用模板的固定 train split。
info_path = dataset / "meta/info.json"
info = json.loads(info_path.read_text())
info["total_chunks"] = (info["total_episodes"] + info["chunks_size"] - 1) // info["chunks_size"]
info["splits"] = {"train": f"0:{info['total_episodes']}"}
info_path.write_text(json.dumps(info, indent=4) + "\n")
generate_stats(dataset)
PY
} 2>&1 | tee -a "$LAB_ROOT/logs/convert-demos.log"
