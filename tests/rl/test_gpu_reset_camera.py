"""Reset must invalidate cached pixels and renderer history without stepping physics."""

import argparse
import json
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from embodied_ai_lab.rlinf.env import make_sim

    cfg = OmegaConf.create(dict(
        device=args.device, viser_envs=0,
        task="galileo_g1_locomanip_pick_and_place_rl",
        external_environment_class_path="embodied_ai_lab.environments.locomanip:LocomanipRLEnvironment",
        io_module="embodied_ai_lab.environments.g1_io",
        arena=dict(object="brown_box", destination="blue_sorting_bin",
                   embodiment="g1_wbc_joint", enable_cameras=True),
    ))
    chunk, app = make_sim(cfg, 2, 1000)
    env = chunk.env
    try:
        import isaaclab
        import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx_utils

        assert env.cfg.num_rerenders_on_reset == 0
        camera = env.scene["robot_head_cam"]
        robot = env.scene["robot"]
        box = env.scene.rigid_objects["brown_box"]

        def physical_state():
            return [value.cpu().clone() for value in (
                robot.data.root_state_w, box.data.root_state_w,
                robot.data.joint_pos, robot.data.joint_vel,
            )]

        def counters():
            return [env.sim.get_physics_step_count(), env._sim_step_counter, env.common_step_counter]

        images, states, checks = [], [], []
        # ABBA separates differing histories from ordinary repeat variation.
        for offset in (0.4, -0.4, -0.4, 0.4):
            chunk.reset(seed=1000)
            moved = box.data.root_state_w.clone()
            moved[:, 0] += offset
            box.write_root_state_to_sim(moved)
            env.sim.forward()
            for _ in range(24):
                env.sim.render()
                camera.update(0.0, force_recompute=True)
            old = camera.data.output["rgb"].cpu().clone()
            before_count = counters()
            surviving_state = [value[1].clone() for value in physical_state()]

            # No extra rendering or cache refresh outside the production reset path.
            observations, _ = chunk.reset(seed=1000, env_ids=torch.tensor([0], device=env.device))
            current = observations["main_images"]
            assert not torch.equal(current[0], old[0, :, :, :3]), "Reset returned the previous episode's pixels"
            assert counters() == before_count, "Reset advanced physics or environment steps"
            after = physical_state()
            for expected, actual in zip(surviving_state, after):
                torch.testing.assert_close(expected, actual[1], rtol=0, atol=0)
            images.append(current[0].numpy().astype(float))
            states.append([value[0] for value in after])
            checks.append(dict(history_box_dx=offset, no_physics_advance=True, fresh_pixels=True))

        for state in states[1:]:
            for expected, actual in zip(states[0], state):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        difference = lambda left, right: float(np.abs(images[left] - images[right]).mean())
        across = [difference(0, 1), difference(3, 2)]
        repeats = [difference(0, 3), difference(1, 2)]
        # Permit ordinary stochastic RTX variation; reject excess caused by old geometry.
        assert np.mean(across) <= np.mean(repeats) + 0.5, (across, repeats)
        report = dict(
            num_envs=2, seed=1000, device=args.device,
            num_rerenders_on_reset=env.cfg.num_rerenders_on_reset,
            isaaclab_source=isaaclab.__file__, rtx_source=rtx_utils.__file__,
            cross_history_rgb_mad=across, same_history_rgb_mad=repeats,
            same_reset_state=True, checks=checks,
            temporal_reset_scope="all RTX cameras, including surviving environments",
            policy_success_rate_tested=False,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        chunk.close()
    app.close()


class ResetCameraGpuTest(unittest.TestCase):
    def test_partial_reset_discards_stale_pixels_and_history(self):
        from tests.gpu import REPORTS, run_worker

        report = REPORTS / "reset-camera.json"
        run_worker(self, __name__, "--output", report, report=report)


if __name__ == "__main__":
    main()
