"""Export RLinf full_weights.pt to the official GR00T sharded BF16 layout."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]


def export_checkpoint(weights, base_model, output):
    index = json.loads((base_model / "model.safetensors.index.json").read_text())
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {key: value for key, value in state.items()
             if not key.startswith("action_head.value_head.")}
    if set(state) != set(index["weight_map"]):
        raise ValueError("Policy keys do not match the official checkpoint index")
    output.mkdir(parents=True, exist_ok=False)
    for path in base_model.iterdir():
        if path.name == ".cache" or path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    total_size = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {key: state[key].to(torch.bfloat16).contiguous()
                   for key, name in index["weight_map"].items() if name == shard}
        save_file(tensors, output / shard, metadata={"format": "pt"})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"Saved {shard}: {len(tensors)} tensors", flush=True)
    index["metadata"]["total_size"] = total_size
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (output / "rl_export.json").write_text(json.dumps({
        "weights": str(weights.resolve()), "base_model": str(base_model.resolve()),
        "inference_runtime": "rl-reference-sdpa",
    }, indent=2) + "\n")
    print(f"Exported {len(state)} policy tensors ({total_size} bytes) to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/GR00T-N1.6-G1/checkpoint-20000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.weights, args.base_model, args.output)


if __name__ == "__main__":
    main()
