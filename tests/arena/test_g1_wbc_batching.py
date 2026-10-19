"""Exercise G1 WBC reset and stand/walk selection across independent envs."""

import unittest
from collections import deque
from unittest.mock import Mock, patch

import numpy as np
import torch
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy import (
    g1_homie_policy as homie,
)


def make_policy(num_envs=2):
    sessions = []
    for value in (1.0, 2.0):
        session = Mock(input_names=["obs"])
        session.run.side_effect = lambda inputs, value=value: {
            "action": np.full((inputs["obs"].shape[0], 15), value, dtype=np.float32)
        }
        sessions.append(session)
    with (
        patch.object(homie, "retrieve_file_path", side_effect=lambda path, **_: path),
        patch.object(homie, "OnnxInferenceSession", side_effect=sessions),
    ):
        policy = homie.G1HomiePolicyV2(None, "config/g1_homie_v2.yaml", "stand.onnx,walk.onnx", num_envs)
    policy.set_goal({
        "navigate_cmd": np.zeros((num_envs, 3), dtype=np.float32),
        "base_height_command": np.full((num_envs, 1), 0.75, dtype=np.float32),
        "torso_orientation_rpy_cmd": np.zeros((num_envs, 3), dtype=np.float32),
    })
    policy.obs_tensor = torch.from_numpy(policy.obs_buffer)
    return policy


class WBCBatchingTest(unittest.TestCase):
    def test_partial_reset_preserves_other_env_history_and_action(self):
        policy = make_policy()
        policy.action[:] = [[1] * 15, [2] * 15]
        policy.gait_indices[:] = torch.tensor([[0.25], [0.75]])
        policy.obs_buffer[:] = 3
        policy.cmd[:] = [[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]]
        policy.obs_history = deque(
            [np.full((2, 86), 4.0, dtype=np.float32) for _ in range(policy.config["obs_history_len"])],
            maxlen=policy.config["obs_history_len"],
        )
        policy.reset(torch.tensor([1]))
        np.testing.assert_array_equal(policy.action[0], 1)
        np.testing.assert_array_equal(policy.action[1], 0)
        np.testing.assert_array_equal(policy.obs_buffer[0], 3)
        np.testing.assert_array_equal(policy.obs_buffer[1], 0)
        np.testing.assert_array_equal(policy.obs_tensor.numpy(), policy.obs_buffer)
        np.testing.assert_allclose(policy.cmd[0], [0.3, 0.2, 0.1])
        np.testing.assert_array_equal(policy.cmd[1], policy.config["cmd_init"])
        self.assertEqual(float(policy.gait_indices[0]), 0.25)
        self.assertEqual(float(policy.gait_indices[1]), 0)
        self.assertEqual(len(policy.obs_history), policy.config["obs_history_len"])
        for frame in policy.obs_history:
            np.testing.assert_array_equal(frame[0], 4)
            np.testing.assert_array_equal(frame[1], 0)

    def test_neighbor_motion_does_not_change_standing_policy(self):
        policy = make_policy()
        reference = policy.get_action()["body_action"][0].copy()
        policy.cmd[1, 0] = 0.4
        mixed = policy.get_action()["body_action"]
        np.testing.assert_array_equal(mixed[0], reference)
        expected_walk = 2 * policy.config["action_scale"] + policy.config["default_angles"]
        np.testing.assert_array_equal(mixed[1], expected_walk)


if __name__ == "__main__":
    unittest.main()
