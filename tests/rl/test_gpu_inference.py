"""Compare fixed-seed N1.6 action chunks using real dataset observations.

Requires models/GR00T-N1.6-G1/checkpoint-20000 and data/locomanipulation/generated/lerobot
(episodes 0, 49, 99, frame 100). Runs the reference and RL fork in fresh RL Python
subprocesses; CUDA is required. The fork comparison retains the original 1e-5 tolerance and
reports source selection: the known letterbox preprocessing difference does not fail this
regression. Missing prerequisites fail the test.
"""

import argparse
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observation", type=Path, default=ROOT / "logs/rl-checks/observation.pt")
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if args.capture_only:
        import gr00t.model  # Register the official processor.
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(ROOT / "models/GR00T-N1.6-G1/checkpoint-20000")
        modalities = processor.get_modality_configs()["new_embodiment"]
    else:
        policy = Gr00tPolicy(
            EmbodimentTag.NEW_EMBODIMENT,
            str(ROOT / "models/GR00T-N1.6-G1/checkpoint-20000"),
            device=args.device,
        )
        modalities = policy.modality_configs
    loader = LeRobotEpisodeLoader(
        ROOT / "data/locomanipulation/generated/lerobot",
        modality_configs=modalities,
        video_backend="opencv",
    )
    arrays, samples = {}, []
    for episode_index in (0, 49, 99):
        episode = loader._load_parquet_data(episode_index)
        frames = loader._load_video_data(episode_index, np.array([100]))
        obs = {"video": {}, "state": {}, "language": {}}
        for key in modalities["video"].modality_keys:
            obs["video"][key] = frames[key][None]
        for key in modalities["state"].modality_keys:
            obs["state"][key] = np.asarray(
                episode[f"state.{key}"].iloc[100], dtype=np.float32
            )[None, None]
        for key in modalities["language"].modality_keys:
            obs["language"][key] = [[episode[f"language.{key}"].iloc[100]]]
        if episode_index == 0:
            args.observation.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obs, args.observation)
            if args.capture_only:
                return
        random.seed(42 + episode_index)
        np.random.seed(42 + episode_index)
        torch.manual_seed(42 + episode_index)
        action, _ = policy.get_action(obs)
        assert sum(a.shape[-1] for a in action.values()) == 35
        for key, value in action.items():
            assert value.shape[:2] == (1, 50) and np.isfinite(value).all()
            arrays[f"{episode_index}/{key}"] = value
        samples.append(
            {
                "episode": episode_index,
                "step": 100,
                "shapes": {k: list(v.shape) for k, v in action.items()},
            }
        )
    report = {
        "samples": samples,
        "device": args.device,
        "attention": "sdpa",
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
    }
    if args.reference:
        reference = np.load(args.reference)
        assert set(reference.files) == set(arrays)
        report["comparison"] = {}
        equivalent = True
        for key, value in arrays.items():
            delta = np.abs(value - reference[key])
            report["comparison"][key] = {
                "max_abs_delta": float(delta.max()),
                "mean_abs_delta": float(delta.mean()),
            }
            equivalent = equivalent and np.allclose(
                value, reference[key], atol=1e-5, rtol=1e-5
            )
        report["equivalent"] = bool(equivalent)
        report["selected_source"] = "rlinf_fork" if equivalent else "e29d8fc"
    np.savez(args.output.with_suffix(".npz"), **arrays)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


class InferenceGpuTest(unittest.TestCase):
    def test_reference_and_fork_fixed_seed_actions(self):
        from tests.gpu import BASE_MODEL, DATASET, REPORTS, require, run_worker

        require(self, BASE_MODEL, DATASET)
        reference = REPORTS / "inference-reference.json"
        fork = REPORTS / "inference-fork.json"
        run_worker(self, __name__, "--output", reference, report=reference)
        run_worker(self, __name__, "--output", fork, "--reference",
                   reference.with_suffix(".npz"), runtime="fork", report=fork)


if __name__ == "__main__":
    main()
