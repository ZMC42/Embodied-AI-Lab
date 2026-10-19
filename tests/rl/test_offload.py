"""Check that single-rank FSDP offload releases module references to CUDA storage."""

import tempfile
import unittest
from pathlib import Path

import torch
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class OffloadTest(unittest.TestCase):
    def test_no_shard_offload_moves_module_views(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{Path(directory) / 'init'}",
                rank=0,
                world_size=1,
            )
            try:
                model = FSDP(torch.nn.Linear(1024, 1024).cuda(), device_id=0)
                strategy = FSDPStrategy(cfg=None, world_size=1)
                state_dict = strategy.get_model_state_dict(
                    model, cpu_offload=True, full_state_dict=True
                )
                self.assertTrue(
                    all(t.device.type == "cpu" for t in state_dict.values())
                )
                strategy.offload_param_and_grad(model, offload_grad=False)
                self.assertEqual(next(model.parameters()).device.type, "cpu")
                self.assertEqual(model.module.weight.device.type, "cpu")
                strategy.onload_param_and_grad(
                    model, torch.device("cuda:0"), onload_grad=False
                )
                result = model(torch.ones(1, 1024, device="cuda"))
                self.assertTrue(torch.isfinite(result).all())
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
