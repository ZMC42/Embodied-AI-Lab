"""Frozen maple-table screening and the Arena closed-loop reference.

Run one GPU worker at a time. A worker failure stops the sweep. Action parity is
a separate prerequisite; run this module only after that hard gate passes.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "third_party/IsaacLab-Arena"
CANDIDATES = ARENA / "isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml"


def variants():
    """Read candidates in official YAML order without duplicating their assets."""
    document = yaml.safe_load(CANDIDATES.read_text())
    return [
        {
            "name": name,
            "environment": {**document["shared"]["environment"], **run.get("environment", {})},
            "environment_builder": {
                **document["shared"]["environment_builder"],
                **run.get("environment_builder", {}),
            },
        }
        for name, run in document["runs"].items()
    ]


def select_variant(results):
    eligible = [result for result in results if .2 <= result["success_rate"] <= .7]
    return min(eligible, key=lambda result: abs(result["success_rate"] - .5)) if eligible else None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def import_paths():
    names = (
        "embodied_ai_lab", "embodied_ai_lab.environments.droid_io",
        "embodied_ai_lab.rlinf.extension", "rlinf", "isaaclab", "isaaclab_arena",
        "isaaclab_arena_gr00t.policy.gr00t_core", "gr00t",
        "gr00t.policy.gr00t_policy",
    )
    return {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in names}


def compose_recipe():
    import rlinf  # noqa: F401 -- upstream OmegaConf resolvers
    from hydra import compose, initialize_config_dir

    os.environ["LAB_ROOT"] = str(ROOT)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs/rl/maple-table")):
        return compose(config_name="ppo", overrides=[
            f"hydra.searchpath=[file://{ROOT / 'third_party/RLinf/examples/embodiment/config'}]"
        ])


def prepare_eval_model(model, weights, device):
    """Load complete actor weights on CPU before casting to rollout precision."""
    import torch

    loaded = None
    if weights is not None:
        weights = weights.resolve()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        loaded = {"path": str(weights), "strict": True, "tensors": len(state)}
        del state
        with weights.open("rb") as stream:
            loaded["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    model.to(dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return loaded


def run_rlinf(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import set_seed
    from embodied_ai_lab.rlinf.env import make_sim
    from embodied_ai_lab.rlinf.extension import build_model, register

    candidate = variants()[args.variant_index]
    cfg = compose_recipe()
    env_cfg = cfg.env.eval
    for key, value in candidate["environment"].items():
        if key != "type":
            env_cfg.arena[key] = value
    env_cfg.arena.episode_length_s = 70.4
    env_cfg.language_instruction = candidate["environment_builder"]["language_instruction"]
    env_cfg.device = args.device
    env_cfg.seed = args.seed
    env_cfg.is_eval = True
    env_cfg.viser_envs = 0
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    set_seed(args.seed)
    env, app = make_sim(env_cfg, num_envs=4, seed=args.seed)
    start = time.monotonic()
    try:
        assert env.env.max_episode_length == 1056
        assert abs(env.env.step_dt - 1 / 15) < 1e-9
        assert env.env.get_language_instruction() == env_cfg.language_instruction
        assert set(env.env.termination_manager.active_terms) == {"object_dropped", "success", "time_out"}
        original_step = env.env.step
        active = torch.ones(4, dtype=torch.bool, device=env.device)
        dropped = torch.zeros_like(active)

        def record_drops(actions):
            result = original_step(actions)
            dropped.logical_or_(env.env.termination_manager.get_term("object_dropped") & active)
            active.logical_and_(~(result[2] | result[3]))
            return result

        env.env.step = record_drops
        register()
        model = build_model(cfg.actor.model, torch.float32 if args.weights else torch.bfloat16)
        loaded_weights = prepare_eval_model(model, args.weights, args.device)
        obs, _ = env.reset(seed=args.seed)
        inference_seed = args.seed if args.inference_seed is None else args.inference_seed
        set_seed(inference_seed)
        records = []
        write_json(args.output / "protocol.json", {
            "candidate": candidate, "seed": args.seed, "episodes": args.episodes,
            "num_envs": 4, "chunk_size": 32, "max_episode_steps": 1056,
            "step_dt": env.env.step_dt,
            "language_instruction": env.env.get_language_instruction(),
            "model_mode": "eval", "inference_micro_batch_size": cfg.actor.model.inference_micro_batch_size,
            "base_model": str(Path(cfg.actor.model.model_path).resolve()),
            "loaded_weights": loaded_weights, "rollout_precision": "bf16",
            "inference_seed": inference_seed,
            "import_paths": import_paths(),
        })
        with (args.output / "episodes.jsonl").open("w") as stream, torch.inference_mode():
            while len(records) < args.episodes:
                assert set(obs["task_descriptions"]) == {env_cfg.language_instruction}
                actions, _ = model.predict_action_batch(env_obs=obs, mode="eval")
                assert actions.shape == (4, 32, 8) and np.isfinite(actions).all()
                active.fill_(True)
                dropped.zero_()
                observations, _, terminated, truncated, infos = env.step(torch.from_numpy(actions))
                obs = observations[0]
                done = terminated[:, -1] | truncated[:, -1]
                if not done.any():
                    continue
                episode = infos[0]["final_info"]["episode"]
                for env_id in done.nonzero().flatten().tolist():
                    if len(records) == args.episodes:
                        break
                    success = bool(episode["success_once"][env_id])
                    record = {
                        "episode": len(records), "env_id": env_id,
                        "success": success,
                        "object_dropped": bool(dropped[env_id]),
                        "timeout": bool(truncated[env_id, -1]),
                        "fall_once": bool(episode["fall_once"][env_id]),
                        "return": float(episode["return"][env_id]),
                        "episode_len": int(episode["episode_len"][env_id]),
                    }
                    assert not record["fall_once"], "Fixed-base DROID unexpectedly fell"
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
        result = {
            "name": candidate["name"], "variant_index": args.variant_index,
            "episodes": len(records), "successes": sum(r["success"] for r in records),
            "success_rate": sum(r["success"] for r in records) / len(records),
            "object_dropped": sum(r["object_dropped"] for r in records),
            "timeouts": sum(r["timeout"] for r in records),
            "falls": sum(r["fall_once"] for r in records),
            "elapsed_s": time.monotonic() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        write_json(args.output / "complete.json", result)
    finally:
        env.close()
        app.close()


def run_reference(args):
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app
    try:
        from transformers import set_seed
        from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
        from isaaclab_arena.evaluation.run_execution import execute_experiment
        from embodied_ai_lab.eval import maple_reference_policy  # noqa: F401 -- policy registration

        candidate = variants()[args.variant_index]
        reference_config = {
            "runs": {candidate["name"]: {
                "environment": {**candidate["environment"], "episode_length_s": 70.4},
                "environment_builder": {
                    **candidate["environment_builder"], "num_envs": 4, "seed": args.seed,
                },
                "policy": {
                    "type": "embodied_ai_lab.eval.maple_reference_policy.MapleReferencePolicy",
                    "policy_config_yaml_path": str(ARENA / "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"),
                    "policy_device": args.device, "num_envs": 4,
                    "model_path": str((ROOT / "models/GR00T-N1.6-DROID").resolve()),
                },
                "rollout_limit": {"num_episodes": args.episodes},
            }}
        }
        yaml_path = args.output / "experiment.yaml"
        yaml_path.write_text(yaml.safe_dump(reference_config, sort_keys=False))
        set_seed(args.seed)
        cfg = load_arena_experiment_from_config_file(yaml_path, device=args.device)
        write_json(args.output / "protocol.json", {
            "runner": "isaaclab_arena.evaluation.run_execution.execute_experiment",
            "candidate": candidate, "seed": args.seed, "episodes": args.episodes,
            "num_envs": 4, "import_paths": import_paths(),
        })
        results = execute_experiment(cfg, output_dir=args.output)
        records_path = args.output / candidate["name"] / "episode_results_rebuild0.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        assert len(records) == args.episodes
        assert all(result.status.value == "completed" for result in results)
        write_json(args.output / "complete.json", {
            "name": candidate["name"], "episodes": len(records),
            "successes": sum(record["success"] for record in records),
            "success_rate": sum(record["success"] for record in records) / len(records),
            "runner_status": [result.status.value for result in results],
        })
    finally:
        app.close()


def screen(args):
    parity = json.loads(args.parity_report.read_text())
    assert parity["passed"], "Action parity must pass before variant screening"
    results = []
    for index, candidate in enumerate(variants()):
        output = args.output / candidate["name"]
        command = [
            sys.executable, "-m", __spec__.name, "rlinf", "--output", str(output),
            "--variant-index", str(index), "--episodes", "32",
            "--seed", str(args.seed), "--device", args.device,
        ]
        with (args.output / f"{candidate['name']}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ARENA)
        results.append(json.loads((output / "complete.json").read_text()))
    selected = select_variant(results)
    write_json(args.output / "complete.json", {
        "results": results,
        "selection_rule": "SR in [0.2, 0.7], closest to 0.5, tie by YAML order",
        "selected": selected,
        "outcome": "selected" if selected else "maple_table_eliminated",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rlinf", "reference", "screen"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant-index", default=0, type=int)
    parser.add_argument("--episodes", default=4, type=int)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--weights", type=Path, help="Complete RLinf actor full_weights.pt for frozen evaluation")
    parser.add_argument("--inference-seed", type=int, help="Inference RNG seed after reset; defaults to --seed")
    args = parser.parse_args()
    if args.weights is not None:
        assert args.stage == "rlinf", "--weights is supported only for the RLinf worker"
        args.weights = args.weights.resolve()
    if args.stage == "screen":
        assert args.parity_report is not None, "screen requires --parity-report"
        args.parity_report = args.parity_report.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # The official policy YAML contains paths relative to Arena.
    os.chdir(ARENA)
    {"rlinf": run_rlinf, "reference": run_reference, "screen": screen}[args.stage](args)


if __name__ == "__main__":
    main()
