"""First PPO update audit must reject drift without changing outputs or gradients."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from embodied_ai_lab.rlinf.parity import audit_micro_batch, logprob_parity


class Policy(torch.nn.Module):
    def __init__(self, gap=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.gap = gap

    def forward(self, value):
        new = value * self.weight
        return {"logprobs": new, "prev_logprobs": new.detach() + self.gap}


def train(actor, micro_batch, metrics, *, is_last):
    output = actor.model(micro_batch["value"])
    output["logprobs"].square().sum().backward()
    return output


class FirstUpdateParityTest(unittest.TestCase):
    def actor(self, directory, gap=0.0):
        return SimpleNamespace(
            cfg=SimpleNamespace(algorithm={"audit_first_update": True},
                runner=SimpleNamespace(logger=SimpleNamespace(log_path=directory))),
            optimizer_steps=0, version=0, _rank=2, model=Policy(gap),
        )

    def test_complete_batch_preserves_outputs_gradients_and_rng_then_skips_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = self.actor(directory)
            control = self.actor(directory)
            batch = {"value": torch.tensor([[1., 2.], [3., 4.]])}
            wrapped = audit_micro_batch(train)
            rng = torch.random.get_rng_state()
            for is_last in (False, True):
                expected = train(control, batch, {}, is_last=is_last)
                actual = wrapped(actor, batch, {}, is_last=is_last)
                self.assertTrue(torch.equal(actual["logprobs"], expected["logprobs"]))
                self.assertTrue(torch.equal(actor.model.weight.grad, control.model.weight.grad))
                self.assertEqual(len(actor.model._forward_hooks), 0)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            path = Path(directory) / "first-update-parity-rank2.json"
            evidence = json.loads(path.read_text())
            self.assertTrue(evidence["complete"] and evidence["passed"])
            self.assertEqual(evidence["micro_batches"], 2)
            self.assertEqual(evidence["max_abs_chunk_logratio"], 0.)
            actor.optimizer_steps = 1
            actor.model.gap = 100.
            wrapped(actor, batch, {}, is_last=True)
            self.assertEqual(json.loads(path.read_text()), evidence)

    def test_drift_and_nonfinite_fail_before_backward_and_remove_hook(self):
        for gap in (0.1, float("nan"), 1e30):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as directory:
                actor = self.actor(directory, gap)
                with self.assertRaisesRegex(RuntimeError, "parity failed"):
                    audit_micro_batch(train)(actor, {"value": torch.ones(1, 3)}, {}, is_last=False)
                evidence = json.loads((Path(directory) / "first-update-parity-rank2.json").read_text())
                self.assertFalse(evidence["passed"])
                self.assertIsNone(actor.model.weight.grad)
                self.assertEqual(len(actor.model._forward_hooks), 0)

    def test_tolerance_and_loss_mask(self):
        new = torch.zeros(2, 2)
        old = torch.tensor([[0., 0.], [1., 1.]])
        result = logprob_parity({"logprobs": new, "prev_logprobs": old}, torch.tensor([True, False]))
        self.assertEqual(result["valid_chunks"], 1)
        self.assertEqual(result["max_abs_chunk_logratio"], 0.)
        with tempfile.TemporaryDirectory() as directory:
            actor = self.actor(directory, 1e-5)
            audit_micro_batch(train)(actor, {"value": torch.zeros(1, 2)}, {}, is_last=True)
            self.assertTrue(json.loads((Path(directory) / "first-update-parity-rank2.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
