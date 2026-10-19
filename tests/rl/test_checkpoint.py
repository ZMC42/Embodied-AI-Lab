"""单卡混合精度 checkpoint 不应克隆一整份 CUDA 权重，并须恢复 Adam。"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from embodied_ai_lab.rlinf.extension import cpu_checkpoint_state_dict
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


class CheckpointTest(unittest.TestCase):
    def test_cpu_clone_memory_and_fresh_model_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group("gloo", rank=0, world_size=1,
                init_method=f"file://{Path(directory) / 'init'}")
            try:
                def build():
                    backbone = nn.Linear(1024, 1024, dtype=torch.bfloat16)
                    backbone.requires_grad_(False)
                    model = FSDP(nn.Sequential(backbone, nn.Linear(1024, 1024)).cuda(),
                        device_id=0, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                        use_orig_params=False, mixed_precision=MixedPrecision(
                            param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad], lr=5e-6)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                    model(torch.ones(1, 1024, device="cuda", dtype=torch.bfloat16)
                          ).float().square().mean().backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    return model, optimizer, scheduler

                model, optimizer, scheduler = build()
                strategy = FSDPStrategy(cfg=None, world_size=1)
                expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                expected_optim = {n: {k: v.cpu().clone() for k, v in optimizer.state[p].items()}
                                  for n, p in model.named_parameters() if p.requires_grad}

                def export_peak():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    start = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    state = strategy.get_model_state_dict(model, True, True)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - start
                    self.assertTrue(all(v.device.type == "cpu" for v in state.values()))
                    return state, peak

                original, original_peak = export_peak()
                with cpu_checkpoint_state_dict(model):
                    moved, moved_peak = export_peak()
                    strategy.save_checkpoint(model, optimizer, scheduler, directory)
                self.assertLess(moved_peak, original_peak / 2)
                self.assertEqual(original.keys(), moved.keys())
                for key in original:
                    self.assertEqual(original[key].dtype, moved[key].dtype)
                    self.assertTrue(torch.equal(original[key], moved[key]))
                # 保存 hook 已撤销；训练权重同步仍然可以导出 CUDA 张量。
                synced = strategy.get_model_state_dict(model, False, False)
                self.assertTrue(all(v.device.type == "cuda" for v in synced.values()))
                del synced

                fresh, fresh_optim, fresh_scheduler = build()
                self.assertTrue(any(not torch.equal(p.cpu(), expected[n])
                                    for n, p in fresh.named_parameters()))
                strategy.load_checkpoint(fresh, fresh_optim, fresh_scheduler, directory)
                for name, parameter in fresh.named_parameters():
                    self.assertEqual(parameter.dtype, expected[name].dtype)
                    self.assertTrue(torch.equal(parameter.cpu(), expected[name]))
                    if parameter.requires_grad:
                        for key, value in fresh_optim.state[parameter].items():
                            self.assertEqual(value.dtype, expected_optim[name][key].dtype)
                            self.assertTrue(torch.equal(value.cpu(), expected_optim[name][key]))
                self.assertEqual(scheduler.state_dict(), fresh_scheduler.state_dict())
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
