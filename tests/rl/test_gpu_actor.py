"""Run RLinf's N1.6 flow-SDE actor and value head on a real G1 observation.

Requires models/GR00T-N1.6-G1/checkpoint-20000 and data/locomanipulation/generated/lerobot
(episodes 0, 49, 99, frame 100). Captures its own episode-0 observation before checking the
actor; CUDA and the RL environment are required.
"""

import argparse
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/rl-checks/actor.json")
    parser.add_argument("--observation", type=Path, default=ROOT / "logs/rl-checks/observation.pt")
    args = parser.parse_args()
    from rlinf.models.embodiment.gr00t.gr00t_n1d6 import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(ROOT / "models/GR00T-N1.6-G1/checkpoint-20000"),
            "embodiment_tag": "new_embodiment",
            "action_dim": 35,
            "denoising_steps": 4,
            "num_action_chunks": 50,
            # Supply processor tensors directly without env converters.
            "obs_converter_type": "libero",
            "rl_head_config": {
                "add_value_head": True,
                "disable_dropout": True,
                "noise_method": "flow_sde",
                "noise_level": 0.3,
                "noise_anneal": False,
                "joint_logprob": False,
                "detach_critic_input": True,
                "use_vlm_value": False,
                "value_vlm_mode": "mean_token",
                "action_noise_scale": 0.0,
            },
        }
    )
    model = get_model(cfg).to(args.device)
    model.requires_grad_(False)
    model.action_head.action_decoder.requires_grad_(True)
    model.action_head.model.transformer_blocks[-args.blocks :].requires_grad_(True)
    model.action_head.value_head.requires_grad_(True)
    model.eval()
    observation = torch.load(args.observation, weights_only=False)
    flat = {}
    for modality in ("video", "state"):
        flat.update(
            {f"{modality}.{key}": value for key, value in observation[modality].items()}
        )
    flat["annotation.human.action.task_description"] = next(
        iter(observation["language"].values())
    )
    inputs = model.apply_transforms(flat)
    inputs = {
        key: torch.stack(value) if isinstance(value, list) else value
        for key, value in inputs.items()
    }
    inputs = {
        key: value.to(
            device=args.device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    with torch.no_grad():
        actions, rollout = model._get_rl_action(inputs, mode="train")
    output = model(
        forward_inputs=rollout["forward_inputs"],
        prev_logprobs=rollout["prev_logprobs"],
        compute_values=True,
    )
    assert output["logprobs"].shape == (1, 50, 35)
    for key in ("logprobs", "prev_logprobs", "values"):
        assert torch.isfinite(output[key]).all(), key
    ratio = torch.exp((output["logprobs"] - output["prev_logprobs"]).mean(dim=(1, 2)))
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=0.02, rtol=0.02)
    # Exercise policy and critic gradients with a nonzero advantage and target.
    policy_loss = -ratio.mean()
    value_loss = (output["values"].float() - 1).square().mean()
    (policy_loss + value_loss).backward()
    gradients = {}
    for name, module in (
        ("action_decoder", model.action_head.action_decoder),
        ("last_dit_blocks", model.action_head.model.transformer_blocks[-args.blocks :]),
        ("value_head", model.action_head.value_head),
    ):
        grads = [
            p.grad
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        norm = sum(g.float().square().sum().item() for g in grads) ** 0.5
        assert norm > 0, name
        gradients[name] = norm
    report = {
        "model": "gr00t_n1d6",
        "blocks": args.blocks,
        "device": args.device,
        "raw_action_shape": list(actions.shape),
        "logprobs_shape": list(output["logprobs"].shape),
        "values_shape": list(output["values"].shape),
        "ratio": ratio.tolist(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "gradient_norms": gradients,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


class ActorGpuTest(unittest.TestCase):
    def test_flow_sde_actor_value_and_gradients(self):
        from tests.gpu import BASE_MODEL, DATASET, REPORTS, require, run_worker

        require(self, BASE_MODEL, DATASET)
        observation = REPORTS / "actor-observation.pt"
        observation.unlink(missing_ok=True)
        run_worker(self, "tests.rl.test_gpu_inference", "--capture-only",
                   "--output", REPORTS / "actor-capture.json", "--observation", observation)
        require(self, observation)
        report = REPORTS / "actor.json"
        run_worker(self, __name__, "--observation", observation, "--report", report,
                   report=report)


if __name__ == "__main__":
    main()
