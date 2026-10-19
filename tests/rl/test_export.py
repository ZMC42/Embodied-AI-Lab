"""Exercise the checkpoint export CLI with the official sharded layout."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

class ExportTest(unittest.TestCase):
    def test_export_preserves_policy_weights_and_auxiliary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "official"
            base.mkdir()
            (base / "experiment_cfg").mkdir()
            (base / "experiment_cfg/config.yaml").write_text("embodiment: g1\n")
            for name in ("config", "processor_config", "statistics", "embodiment_id"):
                (base / f"{name}.json").write_text('{"test": true}\n')
            weight_map = {"backbone.weight": "model-00001-of-00002.safetensors",
                          "action_head.action_decoder.weight": "model-00002-of-00002.safetensors"}
            for key, shard in weight_map.items():
                save_file({key: torch.zeros(3, dtype=torch.bfloat16)}, base / shard)
            (base / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 12}, "weight_map": weight_map,
            }))
            weights = {key: torch.arange(3, dtype=torch.float32) + i
                       for i, key in enumerate(weight_map)}
            weights["action_head.value_head.weight"] = torch.ones(3)
            torch.save(weights, root / "full_weights.pt")
            subprocess.run([
                sys.executable, "-m", "embodied_ai_lab.rlinf.export",
                "--weights", str(root / "full_weights.pt"),
                "--base-model", str(base), "--output", str(root / "export"),
            ], check=True, capture_output=True, text=True)
            output = root / "export"
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(index["weight_map"], weight_map)
            self.assertEqual(index["metadata"]["total_size"], 12)
            manifest = json.loads((output / "rl_export.json").read_text())
            self.assertEqual(manifest["inference_runtime"], "rl-reference-sdpa")
            self.assertEqual(manifest["weights"], str(root / "full_weights.pt"))
            for key, shard in weight_map.items():
                tensors = load_file(output / shard)
                self.assertEqual(set(tensors), {key})
                self.assertEqual(tensors[key].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(tensors[key], weights[key].bfloat16()))
            for path in base.rglob("*"):
                if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
                    self.assertEqual(path.read_bytes(), (output / path.relative_to(base)).read_bytes())


if __name__ == "__main__":
    unittest.main()
