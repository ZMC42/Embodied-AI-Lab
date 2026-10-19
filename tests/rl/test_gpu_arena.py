"""真实仿真：外部工厂注册、成功奖励、超时和掉箱终止。

Requires the RL environment, CUDA/Isaac Sim and Arena assets linked under data/assets. Runs
real simulated environments on the CPU physics device by default, with GPU cameras; no training
checkpoint is needed.
"""

import argparse
import json
import unittest
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--viser-envs", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-factory", action="store_true")
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf

    from embodied_ai_lab.rlinf.env import make_sim

    cfg = OmegaConf.create(
        dict(
            device=args.device,
            viser_envs=args.viser_envs,
            task="galileo_g1_locomanip_pick_and_place_rl",
            io_module="embodied_ai_lab.environments.g1_io",
            arena_args=[
                "--object",
                "brown_box",
                "--embodiment",
                "g1_wbc_joint",
                "--external_environment_class_path",
                "embodied_ai_lab.environments.locomanip:LocomanipRLEnvironment",
            ],
        )
    )
    if args.legacy_factory:
        cfg.task = "galileo_g1_locomanip_pick_and_place_rl_legacy"
        cfg.arena_args[-1] = "tests.rl.legacy_factory:LegacyLocomanipEnvironment"
    env, app = make_sim(cfg, args.num_envs, 1000)
    base = env.env
    env_zero_only = [True] + [False] * (args.num_envs - 1)
    try:
        assert str(base.device) == args.device
        obs, _ = env.reset(seed=1000)
        robot = base.scene["robot"]
        standing_pose = robot.data.root_pose_w.clone()
        standing_height = (standing_pose[:, 2] - base.scene.env_origins[:, 2]).tolist()
        assert not base.episode_signals()["fall_once"].any(), standing_height
        fallen_pose = standing_pose.clone()
        fallen_pose[0, 2] -= 0.5
        robot.write_root_pose_to_sim(fallen_pose)
        assert base.episode_signals()["fall_once"].tolist() == env_zero_only
        robot.write_root_pose_to_sim(standing_pose)
        assert not base.episode_signals()["fall_once"].any()
        assert base.max_episode_length == 1500
        assert base.step_dt == 0.02
        assert base._active_episode_mask.all()
        actions = torch.zeros(args.num_envs, 2, 50, device=args.device)
        actions[:, :, :43] = obs["states"].to(args.device)[:, None]
        actions[:, :, 46] = 0.75
        # 注入终止管理器的成功信号，验证真实 Lab 的 termination→reward 顺序。
        # 不把它计作自然任务成功；训练不会使用此注入。
        success_cfg = base.termination_manager.get_term_cfg("success")
        original_success = success_cfg.func
        success_cfg.func = lambda e, **kw: torch.tensor(env_zero_only, device=e.device)
        base.termination_manager.set_term_cfg("success", success_cfg)
        base.episode_length_buf[0] = 1499
        out, reward, term, trunc, info = env.step(actions)
        viser_http_status = None
        if args.viser_envs > 0:
            with urlopen("http://127.0.0.1:8080", timeout=10) as response:
                viser_http_status = response.status
            assert viser_http_status == 200, viser_http_status
        assert reward.sum(1).tolist() == [1.0] + [0.0] * (args.num_envs - 1), reward
        assert term[:, -1].tolist() == env_zero_only, term
        assert not trunc.any()
        assert info[-1]["_final_observation"].tolist() == env_zero_only
        assert base.episode_length_buf.tolist() == [0] + [2] * (args.num_envs - 1)
        assert base._active_episode_mask.all()
        success_cfg.func = original_success
        base.termination_manager.set_term_cfg("success", success_cfg)
        env.reset(seed=1000)
        base.episode_length_buf[0] = 1499
        _, reward2, term2, trunc2, _ = env.step(actions)
        assert trunc2[:, -1].tolist() == env_zero_only, trunc2
        assert not term2.any() and not reward2.any()
        env.reset(seed=1000)
        box = base.scene["brown_box"]
        pose = box.data.root_pose_w.clone()
        pose[0, 2] = -10.0
        box.write_root_pose_to_sim(pose)
        _, reward3, term3, trunc3, _ = env.step(actions)
        assert term3[:, -1].tolist() == env_zero_only, term3
        assert not trunc3.any() and not reward3.any()
        report = dict(
            device=args.device,
            num_envs=args.num_envs,
            viser_envs=args.viser_envs,
            viser_http_status=viser_http_status,
            episode_steps=1500,
            camera=type(base.scene["robot_head_cam"]).__name__,
            factory="ExampleEnvironmentBase" if args.legacy_factory else "ArenaEnvironmentFactory",
            success_signal_injected=True,
            termination_overrides_coincident_timeout=True,
            success_rewards=reward.sum(1).tolist(),
            timeout_truncated=trunc2[:, -1].tolist(),
            dropped_terminated=term3[:, -1].tolist(),
            checkpoint_loop=False,
            standing_pelvis_z_relative_env_m=standing_height,
            fall_diagnostic_drop_m=0.4,
            fall_injection_verified=True,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()
    # Kit close 可直接退出进程；只在检查成功后调用，保留断言失败的退出码。
    app.close()


class ArenaGpuTest(unittest.TestCase):
    def test_real_arena_factory_rewards_and_termination(self):
        from tests.gpu import REPORTS, run_worker

        # Exercise both the current factory and the supported legacy adapter.
        for legacy in (False, True):
            with self.subTest(legacy_factory=legacy):
                report = REPORTS / f"arena-{'legacy' if legacy else 'factory'}.json"
                options = ["--legacy-factory"] if legacy else []
                run_worker(self, __name__, "--device", "cpu", "--output", report,
                           *options, report=report)


if __name__ == "__main__":
    main()
