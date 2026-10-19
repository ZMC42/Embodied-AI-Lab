"""Reload a smoke checkpoint and predict G1 action chunks from real episodes.

Set LAB_GPU_SMOKE to the saved SFT smoke checkpoint. Also requires
models/GR00T-N1.6-3B and data/locomanipulation/generated/lerobot
(episodes 0, 49, 99, frame 100). LAB_GPU_SMOKE_STEPS sets the exact expected saved global step
(default 20). Requires CUDA and the groot environment; missing artifacts fail the test.
"""

import argparse
import json
import unittest
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch



ROOT = Path(__file__).resolve().parents[2]


def tensor_from_checkpoint(path, name):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    with safe_open(path / index["weight_map"][name], framework="pt") as weights:
        return weights.get_tensor(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/GR00T-N1.6-smoke/checkpoint-200")
    parser.add_argument("--expected-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/training-checkpoint-check.json")
    args = parser.parse_args()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    checkpoint = args.checkpoint
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == args.expected_steps, state["global_step"]
    losses = [entry for entry in state["log_history"] if "loss" in entry]
    assert losses and all(np.isfinite(entry["loss"]) and np.isfinite(entry["grad_norm"]) for entry in losses)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "processor_config.json", "statistics.json", "embodiment_id.json"):
        assert (checkpoint / name).is_file(), name
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["tune_top_llm_layers"] == 0
    assert not config["tune_llm"] and not config["tune_visual"]
    assert config["tune_projector"] and config["tune_diffusion_model"]
    assert config["torch_dtype"] == "bfloat16"

    base = ROOT / "models/GR00T-N1.6-3B"
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    comparisons = {}
    for prefix in ("backbone.model.language_model.", "backbone.model.vision_model.", "action_head.model.transformer_blocks.", "action_head.action_decoder."):
        name = next(name for name in index if name.startswith(prefix) and name.endswith(("weight", ".W")))
        before = tensor_from_checkpoint(base, name)
        after = tensor_from_checkpoint(checkpoint, name)
        assert torch.isfinite(after).all(), name
        changed = not torch.equal(before, after)
        assert changed == prefix.startswith("action_head."), name
        comparisons[name] = {"changed": changed, "max_abs_delta": float((after.float() - before.float()).abs().max())}

    policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, str(checkpoint), device="cuda", strict=True)
    loader = LeRobotEpisodeLoader(
        dataset_path=ROOT / "data/locomanipulation/generated/lerobot",
        modality_configs=policy.modality_configs,
    )
    predictions = []
    for episode_index in (0, 49, 99):
        episode = loader[episode_index]
        step = 100
        observation = {"video": {}, "state": {}, "language": {}}
        for key in policy.modality_configs["video"].modality_keys:
            observation["video"][key] = np.stack([episode[f"video.{key}"].iloc[step]])[None]
        for key in policy.modality_configs["state"].modality_keys:
            observation["state"][key] = np.asarray(episode[f"state.{key}"].iloc[step], dtype=np.float32)[None, None]
        for key in policy.modality_configs["language"].modality_keys:
            observation["language"][key] = [[episode[f"language.{key}"].iloc[step]]]
        action, _ = policy.get_action(observation)
        assert set(action) == set(policy.modality_configs["action"].modality_keys)
        assert all(value.shape[:2] == (1, 50) and np.isfinite(value).all() for value in action.values())
        assert sum(value.shape[-1] for value in action.values()) == 35
        predictions.append({"episode": episode_index, "step": step, "action_shapes": {k: list(v.shape) for k, v in action.items()}, "finite": True})
    report = {
        "checkpoint": str(checkpoint), "global_step": state["global_step"],
        "losses": losses, "sampled_weight_changes": comparisons, "predictions": predictions,
        "inference_peak_cuda_allocated": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


class SmokeCheckpointGpuTest(unittest.TestCase):
    def test_saved_smoke_checkpoint_and_real_episode_inference(self):
        import os
        from tests.gpu import DATASET, REPORTS, ROOT, artifact, require, run_worker

        checkpoint = artifact("SMOKE")
        require(self, checkpoint, ROOT / "models/GR00T-N1.6-3B", DATASET)
        report = REPORTS / "sft-smoke.json"
        run_worker(self, __name__, "--checkpoint", checkpoint, "--expected-steps",
                   os.environ.get("LAB_GPU_SMOKE_STEPS", "20"), "--output", report,
                   runtime="sft", environment="groot", report=report)


if __name__ == "__main__":
    main()
