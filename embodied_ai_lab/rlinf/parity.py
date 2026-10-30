"""Observe the first real PPO batch without adding a model forward or changing it."""

import json
import math
from pathlib import Path

import torch


# Historical same-architecture runs were exactly zero. Fix this small FP32
# comparison tolerance before execution; it is not a tunable training threshold.
CHUNK_LOGRATIO_TOLERANCE = 1e-4


def logprob_parity(output, loss_mask=None):
    new = output["logprobs"].detach().to(device="cpu", dtype=torch.float32)
    old = output["prev_logprobs"].detach().to(device="cpu", dtype=torch.float32)
    finite = bool(torch.isfinite(new).all() and torch.isfinite(old).all())
    # Match chunk_level preprocessing: sum each input separately in FP32.
    delta = new.flatten(1).sum(1) - old.flatten(1).sum(1)
    if loss_mask is not None:
        delta = delta[loss_mask.detach().cpu().bool().flatten()]
    ratio_error = torch.expm1(delta.double()).abs()
    finite = finite and bool(torch.isfinite(delta).all() and torch.isfinite(ratio_error).all())
    return {
        "finite": finite,
        "valid_chunks": delta.numel(),
        "max_abs_element_logprob_error": (new - old).abs().max().item() if finite else None,
        "max_abs_chunk_logratio": delta.abs().max().item() if finite and delta.numel() else None,
        "max_abs_ratio_error": ratio_error.max().item()
        if finite and delta.numel() else None,
    }


def audit_micro_batch(original):
    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        if not self.cfg.algorithm.get("audit_first_update", False) or self.optimizer_steps != 0:
            return original(self, micro_batch, metrics, is_last=is_last)
        state = getattr(self, "_first_update_parity", None)
        if state is None:
            state = self._first_update_parity = {
                "rank": self._rank,
                "global_step": self.version,
                "optimizer_steps_before_update": self.optimizer_steps,
                "micro_batches": 0, "valid_chunks": 0,
                "complete": False, "passed": True, "finite": True,
                "max_abs_element_logprob_error": 0.0,
                "max_abs_chunk_logratio": 0.0,
                "max_abs_ratio_error": 0.0,
                "chunk_logratio_tolerance": CHUNK_LOGRATIO_TOLERANCE,
                "comparison": "Detached CPU FP32 chunk sums; historical same-architecture result was exact zero.",
            }

        def observe(module, inputs, output):
            sample = logprob_parity(output, micro_batch.get("loss_mask"))
            state["micro_batches"] += 1
            state["valid_chunks"] += sample["valid_chunks"]
            state["finite"] = state["finite"] and sample["finite"]
            for key in ("max_abs_element_logprob_error", "max_abs_chunk_logratio", "max_abs_ratio_error"):
                if sample[key] is not None:
                    state[key] = max(state[key], sample[key])
            state["complete"] = is_last
            state["passed"] = (
                state["finite"]
                and math.isfinite(state["max_abs_ratio_error"])
                and state["max_abs_chunk_logratio"] <= CHUNK_LOGRATIO_TOLERANCE
                and (not is_last or state["valid_chunks"] > 0)
            )
            if is_last or not state["passed"]:
                path = Path(self.cfg.runner.logger.log_path) / f"first-update-parity-rank{self._rank}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
            if not state["passed"]:
                raise RuntimeError(f"First-update logprob parity failed: {state}")

        handle = self.model.register_forward_hook(observe)
        try:
            return original(self, micro_batch, metrics, is_last=is_last)
        finally:
            handle.remove()

    return train_micro_batch


def install_first_update_audit():
    """Patch before Ray copies the actor class, like the checkpoint-only hook."""
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    EmbodiedFSDPActor.train_micro_batch = audit_micro_batch(EmbodiedFSDPActor.train_micro_batch)
