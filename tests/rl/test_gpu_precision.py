"""严格重载混合存储精度 checkpoint，逐张量验证权重与 dtype。

Set LAB_GPU_RL_CONFIG and LAB_GPU_RL_WEIGHTS to the resolved RL training config
and full_weights.pt. The config must point to an available
base model. This GPU-suite regression loads full model tensors in host memory; missing
artifacts fail the test.
"""

import argparse
import json
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from embodied_ai_lab.rlinf.extension import build_model, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    register()
    model = build_model(cfg.actor.model, torch.float32)
    weights = torch.load(args.weights, weights_only=True, map_location="cpu", mmap=True)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.dtype == weights[name].dtype and torch.equal(p, weights[name]), name
    model.load_state_dict(weights, strict=True)
    for name, tensor in model.state_dict().items():
        assert tensor.dtype == weights[name].dtype, (name, tensor.dtype, weights[name].dtype)
        assert torch.equal(tensor, weights[name]), name
    trainable = frozen = 0
    for name, p in model.named_parameters():
        assert p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16), name
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    report = {"strict_load": True, "all_tensors_equal": True, "frozen_matches_start": True,
              "keys": len(weights),
              "trainable_fp32_elements": trainable, "frozen_bf16_elements": frozen,
              "weights": str(args.weights), "config": str(args.config)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


class PrecisionGpuTest(unittest.TestCase):
    def test_mixed_precision_checkpoint_tensor_equality(self):
        from tests.gpu import REPORTS, artifact, require, run_worker

        config = artifact("RL_CONFIG")
        weights = artifact("RL_WEIGHTS")
        require(self, config, weights)
        report = REPORTS / "precision.json"
        run_worker(self, __name__, "--config", config, "--weights", weights,
                   "--output", report, report=report)


if __name__ == "__main__":
    main()
