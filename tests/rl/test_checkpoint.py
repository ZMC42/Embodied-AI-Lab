"""混合精度保存/同步避免整份 CUDA 克隆，保留分桶权重及 Adam 状态。"""

import asyncio
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from embodied_ai_lab.rlinf.extension import cpu_checkpoint_state_dict, install_cpu_state_dict_exports
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.hybrid_engines.weight_syncer import BucketWeightSyncer, WeightSyncer
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class CheckpointTest(unittest.TestCase):
    def test_cpu_clone_memory_and_fresh_model_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group("gloo", rank=0, world_size=1,
                init_method=f"file://{Path(directory) / 'init'}")
            try:
                def build():
                    backbone = nn.Linear(1024, 1024, dtype=torch.bfloat16)
                    backbone.requires_grad_(False)
                    decoder = nn.Linear(1024, 1024)
                    decoder.register_buffer("counter", torch.tensor([7]))
                    decoder.register_buffer("offset", torch.tensor([1.234567], dtype=torch.float32))
                    model = FSDP(nn.Sequential(backbone, decoder).cuda(),
                        device_id=0, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                        use_orig_params=False, mixed_precision=MixedPrecision(
                            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                            buffer_dtype=torch.bfloat16))
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
                self.assertEqual(next(value.dtype for name, value in model.named_buffers()
                                      if name.endswith(".offset")), torch.bfloat16)
                strategy = FSDPStrategy(cfg=None, world_size=1)
                expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                expected_optim = {n: {k: v.cpu().clone() for k, v in optimizer.state[p].items()}
                                  for n, p in model.named_parameters() if p.requires_grad}

                def export_peak(export=None):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    start = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    state = export() if export else strategy.get_model_state_dict(model, True, True)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - start
                    self.assertTrue(all(v.device.type == "cpu" for v in state.values()))
                    return state, peak

                original, original_peak = export_peak()
                with cpu_checkpoint_state_dict(model):
                    moved, moved_peak = export_peak()
                    strategy.save_checkpoint(model, optimizer, scheduler, directory)
                original_save = EmbodiedFSDPActor.save_checkpoint
                original_rollout = EmbodiedFSDPActor.get_rollout_state_dict
                install_cpu_state_dict_exports()
                try:
                    syncer = BucketWeightSyncer(256 * 1024, None, "cuda")
                    actor = SimpleNamespace(model=model, enable_offload=True,
                        weight_syncer=syncer,
                        param_names_need_sync=["1.weight", "1.bias", "1.counter", "1.offset"],
                        get_model_state_dict=partial(
                            strategy.get_model_state_dict, model))
                    before = [(name, value.dtype, value.shape) for name, value in model.named_parameters()]
                    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
                    rollout, rollout_peak = export_peak(partial(
                        EmbodiedFSDPActor.get_rollout_state_dict, actor))
                    self.assertEqual(before, [(name, value.dtype, value.shape)
                                              for name, value in model.named_parameters()])
                    for name, value in model.named_buffers():
                        self.assertEqual(value.dtype, buffers_before[name].dtype)
                        self.assertTrue(torch.equal(value, buffers_before[name]))
                    patch_cfg = Path(__file__).resolve().parents[2] / (
                        "third_party/RLinf/examples/embodiment/config/weight_syncer/patch_syncer.yaml")
                    patch_syncer = WeightSyncer.create(OmegaConf.load(patch_cfg))
                    for offload, weight_syncer in ((False, syncer), (True, patch_syncer)):
                        actor.enable_offload = offload
                        actor.weight_syncer = weight_syncer
                        fallback = EmbodiedFSDPActor.get_rollout_state_dict(actor)
                        self.assertTrue(all(value.is_cuda for value in fallback.values()))
                        del fallback
                finally:
                    EmbodiedFSDPActor.save_checkpoint = original_save
                    EmbodiedFSDPActor.get_rollout_state_dict = original_rollout
                self.assertLess(moved_peak, original_peak / 2)
                self.assertLess(rollout_peak, original_peak / 2)
                self.assertEqual(original.keys(), moved.keys())
                self.assertEqual(set(rollout), {"1.weight", "1.bias", "1.counter", "1.offset"})
                self.assertEqual(rollout["1.offset"].dtype, torch.float32)
                for key in original:
                    self.assertEqual(original[key].dtype, moved[key].dtype)
                    self.assertTrue(torch.equal(original[key], moved[key]))
                for key in rollout:
                    self.assertEqual(original[key].dtype, rollout[key].dtype)
                    self.assertTrue(torch.equal(original[key], rollout[key]))
                target = nn.Sequential(nn.Linear(1024, 1024, dtype=torch.bfloat16),
                                       nn.Linear(1024, 1024)).cuda()
                target[1].register_buffer("counter", torch.tensor([0], device="cuda"))
                target[1].register_buffer("offset", torch.tensor([0.0], device="cuda"))
                frozen = target[0].weight.detach().clone()
                async def transfer(target):
                    await syncer.init_sender(rollout, list(rollout), send=None)
                    buckets = syncer.iter_buckets(rollout, version=2)

                    async def receive():
                        return next(buckets)

                    self.assertEqual(await syncer.apply(target, receive), 2)

                asyncio.run(transfer(target))
                for key, value in target.state_dict().items():
                    if key not in rollout:
                        continue
                    self.assertEqual(value.dtype, rollout[key].dtype)
                    self.assertTrue(torch.equal(value.cpu(), rollout[key]))
                self.assertTrue(torch.equal(target[0].weight, frozen))
                del target, frozen
                # 临时 hook 已撤销，直接导出仍保持上游 CUDA 语义。
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
