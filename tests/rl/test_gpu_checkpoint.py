"""重载 RLinf 保存的完整 N1.6 权重，在真实 Arena 环境闭环执行两个动作块。

Set LAB_GPU_RL_CONFIG and LAB_GPU_RL_WEIGHTS to the resolved RL training config
and full_weights.pt. The config must point to an available
base model. CUDA, Isaac Sim and Arena assets are required; missing artifacts fail the test.
The exact weight assertion uses actor precision; closed-loop inference uses rollout precision.
"""

import argparse
import json
import unittest
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.utils.precision import torch_dtype_from_precision

    from embodied_ai_lab.rlinf.env import make_sim
    from embodied_ai_lab.rlinf.extension import build_model, register

    cfg = OmegaConf.load(args.config)
    register()
    # Preserve the training precision for the exact checkpoint assertion.
    model = build_model(cfg.actor.model, torch_dtype_from_precision(cfg.actor.model.precision))
    model.eval()
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    count = sum(value.numel() for value in weights.values())
    # 验证训练后的value权重确实进入模型，而非只重载SFT基座。
    key = next(
        key
        for key in weights
        if "value_head" in key and weights[key].is_floating_point()
    )
    torch.testing.assert_close(
        model.state_dict()[key].cpu(), weights[key], rtol=0, atol=0
    )
    del weights
    # Match build_model: frozen weights stay BF16; only trainable weights follow precision.
    rollout_dtype = torch_dtype_from_precision(cfg.rollout.model.precision)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(rollout_dtype)
    model.to(device="cuda")
    torch.manual_seed(2000)
    cfg.env.eval.device = args.device
    cfg.env.eval.viser_envs = 0
    env, app = make_sim(cfg.env.eval, 2, 2000)
    try:
        obs, _ = env.reset(seed=2000)
        start = time.monotonic()
        shapes = []
        values = []
        with torch.inference_mode():
            for _ in range(2):
                actions, result = model.predict_action_batch(obs, mode="eval")
                assert actions.shape == (2, 50, 50)
                assert np.isfinite(actions).all()
                value = result["prev_values"].float()
                assert torch.isfinite(value).all()
                values.extend(value.cpu().flatten().tolist())
                obs_list, reward, terminated, truncated, _ = env.step(
                    torch.as_tensor(actions)
                )
                obs = obs_list[-1]
                assert torch.isfinite(reward).all()
                assert torch.isfinite(obs["states"]).all()
                shapes.append(list(actions.shape))
        report = dict(
            weights=str(args.weights),
            loaded_parameters=count,
            strict_load=True,
            checked_value_key=key,
            device=args.device,
            seed=2000,
            chunks=2,
            control_steps_per_env=100,
            total_control_steps=200,
            action_shapes=shapes,
            values=values,
            elapsed_s=time.monotonic() - start,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()
        app.close()


class CheckpointGpuTest(unittest.TestCase):
    def test_strict_checkpoint_reload_and_closed_loop(self):
        from tests.gpu import REPORTS, artifact, require, run_worker

        config = artifact("RL_CONFIG")
        weights = artifact("RL_WEIGHTS")
        require(self, config, weights)
        report = REPORTS / "checkpoint.json"
        run_worker(self, __name__, "--config", config, "--weights", weights,
                   "--output", report, report=report)


if __name__ == "__main__":
    main()
