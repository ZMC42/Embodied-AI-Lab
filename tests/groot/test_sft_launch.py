"""Recipe translation needs no CUDA device or training artifacts."""

from pathlib import Path
import unittest

import yaml

from embodied_ai_lab.sft.launch import build_command


ROOT = Path(__file__).resolve().parents[2]


def recipe(name):
    with (ROOT / "configs/sft" / f"{name}.yaml").open() as source:
        return yaml.safe_load(source)


class SftLaunchTest(unittest.TestCase):
    def test_smoke_argv_paths_debug_and_passthrough(self):
        command = build_command(
            recipe("smoke"), lab_root=ROOT, python="python", num_gpus=1,
            debug=True, extra_args=["--max-steps", "20", "--save-steps", "20"],
        )
        self.assertEqual(command[:7], [
            "python", "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=1", "--no-python", "python",
        ])
        self.assertEqual(command[7:15], [
            "-m", "debugpy", "--listen", "127.0.0.1:5678", "--wait-for-client",
            "--configure-subProcess", "false", "gr00t/experiment/launch_finetune.py",
        ])
        expected = {
            "--base-model-path": str(ROOT / "models/GR00T-N1.6-3B"),
            "--dataset-path": str(ROOT / "data/locomanipulation/generated/lerobot"),
            "--modality-config-path": str(ROOT / "third_party/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py"),
            "--embodiment-tag": "NEW_EMBODIMENT",
            "--output-dir": str(ROOT / "models/GR00T-N1.6-smoke"),
            "--num-gpus": "1", "--global-batch-size": "1",
            "--gradient-accumulation-steps": "1", "--max-steps": "200",
            "--save-steps": "200", "--save-total-limit": "1",
            "--tune-top-llm-layers": "0", "--model-dtype": "bfloat16",
            "--dataloader-num-workers": "0", "--shard-size": "128",
            "--num-shards-per-epoch": "32",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        for flag in ["--no-tune-llm", "--no-tune-visual", "--tune-projector",
                     "--tune-diffusion-model", "--no-use-wandb"]:
            self.assertIn(flag, command)
        jitter = command.index("--color-jitter-params")
        self.assertEqual(command[jitter + 1:jitter + 9], [
            "brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08",
        ])
        self.assertEqual(command[-4:], ["--max-steps", "20", "--save-steps", "20"])

    def test_full_effective_batch_and_per_device_override(self):
        for devices, batch, accum in [(1, None, 4), (2, None, 2),
                                      (4, None, 1), (2, 12, 4)]:
            with self.subTest(devices=devices, batch=batch):
                command = build_command(recipe("g1-full"), lab_root=ROOT, python="python",
                                        num_gpus=devices, per_device_batch=batch)
                self.assertEqual(command[command.index("--num-gpus") + 1], str(devices))
                global_batch = int(command[command.index("--global-batch-size") + 1])
                self.assertEqual(global_batch, devices * (24 if batch is None else batch))
                self.assertEqual(command[command.index("--gradient-accumulation-steps") + 1], str(accum))
                self.assertEqual(global_batch * accum, 96)

    def test_non_divisible_effective_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            build_command(recipe("g1-full"), lab_root=ROOT, python="python",
                          num_gpus=3, per_device_batch=24)


if __name__ == "__main__":
    unittest.main()
