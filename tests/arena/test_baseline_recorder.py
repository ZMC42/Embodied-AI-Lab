"""Exercise episode diagnostics without starting Isaac Sim."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


class RecorderTerm:
    def __init__(self, cfg, env):
        self._env = env


recorder_module = ModuleType("isaaclab.managers.recorder_manager")
recorder_module.RecorderTerm = RecorderTerm
with patch.dict(sys.modules, {"isaaclab.managers.recorder_manager": recorder_module}):
    from embodied_ai_lab.environments import baseline_recorder as diagnostics


class Scene(dict):
    pass


def make_env():
    origins = torch.tensor([[0., 0., 0.], [2., 0., 10.], [4., 0., -5.], [6., 0., 20.]],
                           dtype=torch.float32)
    standing = torch.tensor([0., -0.3, 0.2, -0.8])
    defaults = torch.zeros(4, 13)
    defaults[:, 2] = standing
    roots = origins.clone()
    roots[:, 2] += standing
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=roots, default_root_state=defaults))
    scene = Scene(robot=robot)
    scene.env_origins = origins
    positions = {"robot": roots, "brown_box": origins + torch.tensor([1., 0., 0.])}
    terminations = {"success": torch.tensor([False, False, False, True]),
                    "object_dropped": torch.zeros(4, dtype=torch.bool)}
    env = SimpleNamespace(
        num_envs=4, device="cpu", scene=scene,
        active_episode_mask=torch.tensor([True, True, False, True]),
        arena_world=SimpleNamespace(get_position_w=positions.__getitem__),
        termination_manager=SimpleNamespace(active_terms=list(terminations), get_term=terminations.__getitem__),
    )
    return env


class BaselineRecorderTest(unittest.TestCase):
    def test_fall_threshold_uses_relative_default_standing_height(self):
        env = make_env()
        recorder = diagnostics.BaselineDiagnosticsRecorder(None, env)
        recorder.record_post_reset(torch.tensor([0, 1, 3]))
        recorder.record_post_step()
        self.assertEqual(recorder.fall_once.tolist(), [False] * 4)
        roots = env.scene["robot"].data.root_pos_w
        roots[0, 2] -= 0.4
        recorder.record_post_step()
        self.assertFalse(bool(recorder.fall_once[0]), "The threshold is strictly greater than 0.4 m")
        roots[0, 2] -= 0.001
        roots[1, 2] -= 0.5
        recorder.record_post_step()
        self.assertEqual(recorder.fall_once.tolist(), [True, True, False, False])

    def test_terminal_step_latches_global_id_and_partial_reset_preserves_neighbors(self):
        env = make_env()
        recorder = diagnostics.BaselineDiagnosticsRecorder(None, env)
        recorder.record_post_reset(torch.tensor([0, 1, 3]))
        roots = env.scene["robot"].data.root_pos_w
        roots[2:, 2] -= 0.5  # Env 2 is inactive; env 3 finishes on this step.
        recorder.record_post_step()
        terminal = diagnostics.record_diagnostics(env, 3)
        self.assertTrue(terminal["fall_once"])
        self.assertTrue(terminal["termination_terms"]["success"])
        self.assertEqual(terminal["initial_box_xy_m"], [7., 0.])
        self.assertEqual(terminal["min_base_box_xy_distance_m"], 1.)
        self.assertFalse(diagnostics.record_diagnostics(env, 0)["fall_once"])
        self.assertFalse(bool(recorder.fall_once[2]))

        roots[3, 2] += 0.5  # Recovery must not erase a fall earlier in the episode.
        recorder.record_post_step()
        recorder.record_post_reset(torch.tensor([1]))
        self.assertTrue(diagnostics.record_diagnostics(env, 3)["fall_once"])
        recorder.record_post_reset(torch.tensor([3]))
        self.assertFalse(diagnostics.record_diagnostics(env, 3)["fall_once"])
        self.assertTrue(terminal["fall_once"], "The written terminal record survives the reset")


if __name__ == "__main__":
    unittest.main()
