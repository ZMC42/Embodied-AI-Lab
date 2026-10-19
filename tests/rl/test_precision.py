"""FP32 主权重须能保留 BF16 前向下的小幅 Adam 更新。"""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from embodied_ai_lab.rlinf.extension import build_model


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 32)
        self.action_head = nn.Module()
        self.action_head.action_decoder = nn.Linear(32, 64)
        self.action_head.model = nn.Module()
        self.action_head.model.transformer_blocks = nn.ModuleList([nn.Linear(32, 32)])
        self.action_head.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.action_head.model.transformer_blocks[0](x)
        # 未使用的输出槽模拟其他 embodiment 的零梯度参数。
        return self.action_head.action_decoder(x)[:, :32], self.action_head.value_head(x)


class PrecisionTest(unittest.TestCase):
    def test_small_updates_survive_bfloat16_forward_and_strict_reload(self):
        torch.manual_seed(7)
        cfg = OmegaConf.create({"train_last_blocks": 1, "no_split_modules": [],
                                "obs_converter_type": "libero"})
        with patch("rlinf.models.embodiment.gr00t.gr00t_n1d6.get_model",
                   side_effect=lambda cfg, dtype: Policy().to(dtype)):
            actor = build_model(cfg, torch.float32).cuda()
            rollout = build_model(cfg, torch.bfloat16).cuda()
        self.assertTrue(all(p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16)
                            for p in actor.parameters()))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in rollout.parameters()))
        rollout.load_state_dict(actor.state_dict(), strict=True)
        reloaded = deepcopy(actor)
        frozen = {n: p.clone() for n, p in actor.named_parameters() if not p.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group("gloo", rank=0, world_size=1,
                init_method=f"file://{Path(directory) / 'init'}")
            try:
                model = FSDP(actor, device_id=0, use_orig_params=False,
                    auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32))
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=5e-6, betas=(0.9, 0.95),
                                             eps=1e-8, weight_decay=0)
                x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
                expected = rollout(x)
                actions, values = model(x)
                self.assertEqual(actions.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(actions, expected[0]))
                (actions.float().square().mean() + values.float().square().mean()).backward()
                updates = [(p, p.detach().clone(), p.grad != 0) for p in parameters]
                self.assertGreater(sum((~active).sum().item() for _, _, active in updates), 0)
                optimizer.step()
                for p, before, active in updates:
                    self.assertGreater(active.sum().item(), 0)
                    self.assertGreater((p[active] != before[active]).float().mean().item(), 0.98)
                    self.assertTrue(torch.equal(p[~active], before[~active]))
                for state in optimizer.state.values():
                    self.assertEqual(state["exp_avg"].dtype, torch.float32)
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
                with FSDP.summon_full_params(model):
                    for name, p in actor.named_parameters():
                        name = name.replace("_fsdp_wrapped_module.", "")
                        if not p.requires_grad:
                            self.assertTrue(torch.equal(p, frozen[name]))
                saved = model.state_dict()
                reloaded.load_state_dict(saved, strict=True)
                self.assertTrue(all(torch.equal(t, saved[n]) for n, t in reloaded.state_dict().items()))
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
