"""Run the existing GR00T server with an explicit inference RNG seed."""

import os
import time
from pathlib import Path
from threading import Thread

import tyro
import numpy as np
from gr00t.eval.run_gr00t_server import ServerConfig, main
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer
from transformers import set_seed


def warm_weights(model_path):
    # Read alongside mmap: CIFS may invalidate cached pages after the last close.
    for path in sorted(Path(model_path).glob("*.safetensors")):
        start = time.monotonic()
        with path.open("rb") as stream:
            while stream.read(4 * 1024 * 1024):
                pass
        print(f"Pre-read {path.name}: {time.monotonic() - start:.1f}s", flush=True)


class SingleSamplePolicy(Gr00tPolicy):
    """Match the local RL rollout's BF16 forward shape for any evaluation env count."""

    def _get_action(self, observation, options=None):
        n = len(next(iter(observation["video"].values())))
        chunks = []
        for i in range(n):
            sample = {modality: {key: value[i:i + 1] for key, value in group.items()}
                      for modality, group in observation.items()}
            actions, _ = super()._get_action(sample, options)
            chunks.append(actions)
        return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}, {}


def load_rl_policy(config):
    return SingleSamplePolicy(config.embodiment_tag, config.model_path,
                              device=config.device, strict=config.strict)


def run():
    config = tyro.cli(ServerConfig)
    set_seed(int(os.environ.get("EVAL_SEED", "42")))
    Thread(target=warm_weights, args=(config.model_path,), daemon=True).start()
    if os.environ["EVAL_RL_SERVER"] == "1":
        print("RL inference: reference/SDPA, BF16, one sample per forward", flush=True)
        policy = load_rl_policy(config)
        PolicyServer(policy, host=config.host, port=config.port).run()
    else:
        main(config)


if __name__ == "__main__":
    run()
