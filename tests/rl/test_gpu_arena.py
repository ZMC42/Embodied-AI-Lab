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
    parser.add_argument("--physics-backend", choices=("isaacsim_physx", "newton_mjwarp"),
                        default="isaacsim_physx")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--viser-envs", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-factory", action="store_true")
    parser.add_argument("--stability-steps", type=int, default=0)
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf

    from embodied_ai_lab.rlinf.env import make_sim

    cfg = OmegaConf.create(
        dict(
            device=args.device,
            physics_backend=args.physics_backend,
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
    from isaaclab_arena.utils.physics_backend import physics_protocol

    actual_physics = physics_protocol(base)
    assert actual_physics["physics_backend"] == args.physics_backend, actual_physics
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
        stability = {}
        if args.stability_steps:
            # Exercise WBC with real physics, without policy checkpoints or injected outcomes.
            hold = actions[:, 0].clone()
            minimum_height = robot.data.root_pos_w[:, 2].clone()
            base.defer_resets = True

            def advance(command, steps):
                nonlocal minimum_height
                for step in range(steps):
                    _, _, terminated, truncated, _ = base.step(command)
                    assert torch.isfinite(robot.data.root_state_w).all()
                    assert torch.isfinite(robot.data.joint_pos).all()
                    assert not terminated.any() and not truncated.any(), (terminated, truncated)
                    assert not base.episode_signals()["fall_once"].any(), robot.data.root_pos_w
                    minimum_height = torch.minimum(minimum_height, robot.data.root_pos_w[:, 2])
                    if (step + 1) % 100 == 0:
                        print(json.dumps({"phase_steps": step + 1,
                                          "pelvis_z": robot.data.root_pos_w[:, 2].tolist()}), flush=True)

            advance(hold, args.stability_steps)
            camera = base.scene["robot_head_cam"].data.output["rgb"]
            assert torch.isfinite(camera).all() and camera.max() > 0, camera.shape
            start = robot.data.root_pos_w.clone()
            navigate = hold.clone()
            navigate[:, 43] = -0.15
            advance(navigate, 100)
            navigation_distance = torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - start[:, :2], dim=1)
            assert (navigation_distance > 0.05).all(), navigation_distance
            arm_ids, _ = robot.find_joints("left_shoulder_pitch_joint")
            arm_start = robot.data.joint_pos[:, arm_ids].clone()
            arm_command = hold.clone()
            arm_command[:, arm_ids] += 0.2
            advance(arm_command, 50)
            arm_motion = (robot.data.joint_pos[:, arm_ids] - arm_start).abs()
            assert (arm_motion > 0.05).all(), arm_motion
            unaffected = robot.data.root_state_w[1:].clone()
            untouched_joint = robot.data.joint_pos[1:].clone()
            episode_length = base.episode_length_buf[1:].clone()
            base.defer_resets = False
            env.reset(env_ids=torch.tensor([0], device=base.device))
            assert torch.equal(robot.data.root_state_w[1:], unaffected)
            assert torch.equal(robot.data.joint_pos[1:], untouched_joint)
            assert base.episode_length_buf[0] == 0
            assert torch.equal(base.episode_length_buf[1:], episode_length)
            assert torch.allclose(robot.data.root_pos_w[0], standing_pose[0, :3], atol=0.01)
            contact_sensor = base.scene["contact_sensor_brown_box"]
            contact_force = contact_sensor.data.net_forces_w.torch
            assert torch.isfinite(contact_force).all()
            stability = dict(
                standing_steps=args.stability_steps,
                minimum_pelvis_height=minimum_height.tolist(),
                navigation_distance_m=navigation_distance.tolist(),
                arm_joint_motion_rad=arm_motion.tolist(),
                camera_shape=list(camera.shape),
                independent_reset_verified=True,
                contact_force_shape=list(contact_force.shape),
                finite_states=True,
            )
            print(json.dumps({"stability": stability}), flush=True)
            obs, _ = env.reset(seed=1000)
            actions[:, :, :43] = obs["states"].to(args.device)[:, None]
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
            physics_backend=args.physics_backend,
            physics_config=type(base.cfg.sim.physics).__name__,
            actual_physics=actual_physics,
            stability=stability,
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

    def test_newton_g1_stability_and_independent_reset(self):
        from tests.gpu import REPORTS, run_worker

        report = REPORTS / "arena-newton-g1.json"
        run_worker(self, __name__, "--device", "cuda:0", "--physics-backend", "newton_mjwarp",
                   "--stability-steps", 300, "--output", report, report=report)


if __name__ == "__main__":
    main()
