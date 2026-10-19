"""Check RLinf eval versus an exported GR00T policy over the real server protocol.

Run all stages with .venvs/rl + Isaac-GR00T-reference, sequentially. Capture the same real
dataset frames as test_gpu_inference.py, including nonzero state. Each request uses batch size
one, as in the local RL rollout.

Requires the resolved RL config and full_weights.pt (LAB_GPU_RL_CONFIG / LAB_GPU_RL_WEIGHTS),
an export with rl_export.json (LAB_GPU_EXPORT),
models/GR00T-N1.6-G1/checkpoint-20000, and data/locomanipulation/generated/lerobot. Set
all three artifact environment variables. Requires CUDA and a free local port 5556;
missing artifacts fail the test.
"""

import argparse
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture(args):
    import gr00t.model  # Register the official processor.
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from transformers import AutoProcessor
    from isaaclab_arena_gr00t.utils.joints_conversion import remap_policy_joints_to_sim_joints_np
    from embodied_ai_lab.environments.g1_io import joint_configs

    processor = AutoProcessor.from_pretrained(args.model_path)
    modalities = processor.get_modality_configs()["new_embodiment"]
    loader = LeRobotEpisodeLoader(
        ROOT / "data/locomanipulation/generated/lerobot",
        modality_configs=modalities, video_backend="opencv",
    )
    policy_joints, sim_joints = joint_configs()
    states, images, descriptions, samples = [], [], [], []
    for episode_index in (0, 49, 99):
        episode = loader._load_parquet_data(episode_index)
        frames = loader._load_video_data(episode_index, np.array([100]))
        grouped = {key: np.asarray(episode[f"state.{key}"].iloc[100], dtype=np.float32)[None, None]
                   for key in modalities["state"].modality_keys}
        # The policy only observes these 31 upper-body joints; other sim slots are unused.
        state = remap_policy_joints_to_sim_joints_np(grouped, policy_joints, sim_joints, "NEW_EMBODIMENT")
        states.append(torch.from_numpy(state[0, 0]).float())
        images.append(torch.from_numpy(frames["ego_view"][0]))
        descriptions.append(episode[f"language.{modalities['language'].modality_keys[0]}"].iloc[100])
        samples.append({"episode": episode_index, "step": 100, "seed": 42 + episode_index})
    obs = {"states": torch.stack(states), "main_images": torch.stack(images),
           "task_descriptions": descriptions}
    torch.save({"actor": obs, "samples": samples}, args.output / "observations.pt")
    print("Captured real dataset episodes 0/49/99, frame 100", flush=True)


def actor(args):
    from omegaconf import OmegaConf
    from embodied_ai_lab.rlinf.extension import build_model, register

    register()
    cfg = OmegaConf.load(args.config)
    model = build_model(cfg.actor.model)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval()
    model.to("cuda")
    captured = torch.load(args.output / "observations.pt", weights_only=False)
    obs = captured["actor"]
    assert torch.isfinite(obs["states"]).all()
    assert all(image.float().std() > 0 for image in obs["main_images"])
    # Use the actual Arena observation builder, including its image size and raw state.
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations, load_gr00t_joint_configs
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file

    policy_cfg = create_config_from_yaml(
        "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml",
        Gr00tClosedloopPolicyCfg,
    )
    policy_joints, _, state_joints = load_gr00t_joint_configs(policy_cfg)
    server_obs = build_gr00t_policy_observations(
        [obs["main_images"].numpy()], obs["states"].numpy(),
        obs["task_descriptions"][0], policy_cfg, state_joints, policy_joints,
        load_gr00t_modality_config_from_file(policy_cfg.modality_config_path, policy_cfg.embodiment_tag),
    )
    torch.save(server_obs, args.output / "server-observation.pt")
    arrays = {}
    noise = []
    hook = model.action_head.action_encoder.register_forward_pre_hook(
        lambda module, inputs: noise.append(inputs[0].detach().float().cpu().numpy())
    )
    try:
        for i, sample in enumerate(captured["samples"]):
            one = {key: value[i:i + 1] for key, value in obs.items()}
            seed_rng(sample["seed"])
            noise.clear()
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(one, mode="eval")
            arrays[f"{i}/actions"] = actions
            arrays[f"{i}/initial_noise"] = noise[0]
            assert actions.shape == (1, 50, 50) and np.isfinite(actions).all()
        indices = [0, 1, 2, 0, 1]
        batch = {key: [value[i] for i in indices] if isinstance(value, list)
                 else value[indices] for key, value in obs.items()}
        seed_rng(4242)
        noise.clear()
        with torch.inference_mode():
            arrays["batch/actions"], _ = model.predict_action_batch(batch, mode="eval")
        arrays["batch/initial_noise"] = np.concatenate(noise[::model.action_head.num_inference_timesteps])
    finally:
        hook.remove()
    np.savez(args.output / "actor.npz", **arrays)
    print("Saved strict-loaded RLinf eval action blocks", flush=True)


def server(args):
    from threading import Thread
    from gr00t.eval.run_gr00t_server import ServerConfig
    from gr00t.policy.server_client import PolicyClient, PolicyServer

    from embodied_ai_lab.eval import server as implementation
    policy = implementation.load_rl_policy(ServerConfig(model_path=str(args.model_path)))
    obs = torch.load(args.output / "server-observation.pt", weights_only=False)
    samples = torch.load(args.output / "observations.pt", weights_only=False)["samples"]
    service = PolicyServer(policy, host="127.0.0.1", port=args.port)
    thread = Thread(target=service.run, daemon=True)
    thread.start()
    client = PolicyClient(host="127.0.0.1", port=args.port)
    arrays, noise = {}, []
    hook = policy.model.action_head.action_encoder.register_forward_pre_hook(
        lambda module, inputs: noise.append(inputs[0].detach().float().cpu().numpy())
    )
    try:
        assert client.ping()
        for i, sample in enumerate(samples):
            one = {modality: {key: value[i:i + 1] for key, value in group.items()}
                   for modality, group in obs.items()}
            seed_rng(sample["seed"])
            noise.clear()
            actions, _ = client.get_action(one)
            for key, value in actions.items():
                assert value.shape[:2] == (1, 50) and np.isfinite(value).all()
                arrays[f"{i}/{key}"] = value
            arrays[f"{i}/initial_noise"] = noise[0]
        indices = [0, 1, 2, 0, 1]
        batch = {modality: {key: [value[i] for i in indices] if isinstance(value, list)
                           else value[indices] for key, value in group.items()}
                 for modality, group in obs.items()}
        seed_rng(4242)
        noise.clear()
        actions, _ = client.get_action(batch)
        arrays.update({f"batch/{key}": value for key, value in actions.items()})
        arrays["batch/initial_noise"] = np.concatenate(noise[::policy.model.action_head.num_inference_timesteps])
    finally:
        hook.remove()
        client.kill_server()
        thread.join()
        service.socket.close()
        service.context.term()
    np.savez(args.output / "server.npz", **arrays)
    print("Saved exported policy actions via GR00T PolicyServer/PolicyClient", flush=True)


def compare(args):
    from embodied_ai_lab.environments.g1_io import policy_actions

    actor_data = np.load(args.output / "actor.npz")
    server_data = np.load(args.output / "server.npz")
    captured = torch.load(args.output / "observations.pt", weights_only=False)
    samples = []
    for i, sample in enumerate(captured["samples"]):
        actions = {key.split("/", 1)[1]: server_data[key]
                   for key in server_data.files if key.startswith(f"{i}/") and not key.endswith("initial_noise")}
        evaluated = policy_actions(actions)
        trained = actor_data[f"{i}/actions"]
        delta = np.abs(evaluated - trained)
        noise_equal = np.array_equal(actor_data[f"{i}/initial_noise"], server_data[f"{i}/initial_noise"])
        samples.append({**sample, "shape": list(evaluated.shape),
                        "initial_noise_equal": noise_equal,
                        "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
                        "equivalent": bool(np.allclose(evaluated, trained, atol=1e-5, rtol=1e-5))})
    report = {"weights": str(args.weights), "model_path": str(args.model_path),
              "mode": "eval", "atol": 1e-5, "rtol": 1e-5, "samples": samples,
              "equivalent": all(s["equivalent"] and s["initial_noise_equal"] for s in samples)}
    actions = {key.removeprefix("batch/"): server_data[key] for key in server_data.files
               if key.startswith("batch/") and key != "batch/initial_noise"}
    evaluated = policy_actions(actions)
    delta = np.abs(evaluated - actor_data["batch/actions"])
    report["batched_request"] = {
        "seed": 4242, "shape": list(evaluated.shape),
        "initial_noise_equal": bool(np.array_equal(actor_data["batch/initial_noise"], server_data["batch/initial_noise"])),
        "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
        "equivalent": bool(np.allclose(evaluated, actor_data["batch/actions"], atol=1e-5, rtol=1e-5)),
    }
    report["equivalent"] &= report["batched_request"]["equivalent"] and report["batched_request"]["initial_noise_equal"]
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["equivalent"], "RLinf and exported GR00T policies differ"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "actor", "server", "compare"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args)


class ExportGpuTest(unittest.TestCase):
    def test_exported_policy_server_matches_rlinf_eval(self):
        from tests.gpu import BASE_MODEL, DATASET, REPORTS, artifact, require, run_worker

        config = artifact("RL_CONFIG")
        weights = artifact("RL_WEIGHTS")
        exported = artifact("EXPORT")
        require(self, config, weights, exported / "rl_export.json", BASE_MODEL, DATASET)
        # Reject a stale export from a different run before checking its actions.
        manifest = json.loads((exported / "rl_export.json").read_text())
        self.assertEqual(Path(manifest["weights"]).resolve(), weights.resolve())
        output = REPORTS / "export"
        report = output / "comparison.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.unlink(missing_ok=True)
        options = ["--config", config, "--weights", weights,
                   "--model-path", exported, "--output", output]
        for stage in ("capture", "actor", "server", "compare"):
            run_worker(self, __name__, stage, *options)
        self.assertTrue(report.is_file(), f"Missing export comparison report: {report}")


if __name__ == "__main__":
    main()
