"""Evaluate a G1 checkpoint with Arena's rollout and episode recorder."""

import json
import os
import time
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import (
    get_isaaclab_arena_cli_parser,
    requested_physics_backend_from_argparse,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.physics_backend import validate_physics_device


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parents[2] / "models/GR00T-N1.6-G1/checkpoint-20000")
    args, _ = parser.parse_known_args()
    validate_physics_device(args.presets, args.device)
    args.output.mkdir(parents=True, exist_ok=False)

    with SimulationAppContext(args):
        import torch
        from isaaclab.managers.recorder_manager import RecorderTermCfg
        from isaaclab_arena.evaluation.policy_runner import rollout_policy
        from isaaclab_arena.recording.episode_recorder_manager import (
            EpisodeRecorderTermCfg,
        )
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )
        from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
            Gr00tRemoteClosedloopPolicy,
            Gr00tRemoteClosedloopPolicyCfg,
        )
        from embodied_ai_lab.environments.baseline_recorder import (
            BaselineDiagnosticsRecorder,
            record_diagnostics,
        )
        from embodied_ai_lab.physics import physics_protocol

        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args = parser.parse_args()
        builder = get_arena_builder_from_cli(args)
        builder.arena_env.episode_recorder_terms["baseline"] = EpisodeRecorderTermCfg(func=record_diagnostics)
        cfg, kwargs = builder.compose_manager_cfg()
        cfg.recorders.baseline_diagnostics = RecorderTermCfg(class_type=BaselineDiagnosticsRecorder)
        cfg.recorders.dataset_export_dir_path = str(args.output)
        env = builder.make_registered(cfg, kwargs)
        base = env.unwrapped
        assert base.max_episode_length == 1500
        assert abs(base.step_dt - 0.02) < 1e-8
        base.episode_recorder.set_job_name(args.model_path.name)
        base.episode_recorder.set_output_path(args.output / "episodes.jsonl")
        policy = Gr00tRemoteClosedloopPolicy(
            Gr00tRemoteClosedloopPolicyCfg(
                policy_config_yaml_path="isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml",
                policy_device="cuda",
                num_envs=args.num_envs,
                remote_host="127.0.0.1",
                remote_port=args.port,
            )
        )
        assert policy.action_chunk_length == 50
        rl_inference = os.environ["EVAL_RL_SERVER"] == "1"
        protocol = {
            **physics_protocol(base),
            "requested_physics_backend": requested_physics_backend_from_argparse(args),
            "seed": args.seed,
            "episodes": args.episodes,
            "num_envs": args.num_envs,
            "device": str(base.device),
            "camera": type(base.scene["robot_head_cam"]).__name__,
            "step_dt": base.step_dt,
            "max_episode_steps": base.max_episode_length,
            "action_chunk_length": policy.action_chunk_length,
            "checkpoint": str(args.model_path.resolve()),
            "inference_runtime": "rl-reference-sdpa" if rl_inference else "official-flash-attention",
            "inference_micro_batch_size": 1 if rl_inference else args.num_envs,
            "navigation_threshold_m": 0.5,
            "fall_diagnostic_drop_m": 0.4,
        }
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        start = time.monotonic()
        try:
            rollout_policy(env, policy, num_steps=None, num_episodes=args.episodes)
            records = [json.loads(line) for line in (args.output / "episodes.jsonl").read_text().splitlines()]
            assert len(records) == args.episodes
            result = {
                "episodes": len(records),
                "successes": sum(record["success"] for record in records),
                "elapsed_s": time.monotonic() - start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            (args.output / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            policy.close()
            env.close()


if __name__ == "__main__":
    main()
