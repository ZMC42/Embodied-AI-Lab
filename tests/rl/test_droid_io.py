"""DROID 的两路图像、状态与绝对关节动作和 Arena 官方路径一致。"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_ai_lab.environments import droid_io


class DroidIOTest(unittest.TestCase):
    def test_observations_equal_official_builder_with_distinct_cameras(self):
        from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations

        states = torch.linspace(-0.8, 0.9, 26).reshape(2, 13)
        images = torch.arange(2 * 720 * 1280 * 3).remainder(256).byte().reshape(2, 720, 1280, 3)
        cameras = [images, 255 - images]
        source = {
            "policy": {"robot_joint_pos": states},
            "camera_obs": {"external_camera_rgb": cameras[0], "wrist_camera_rgb": cameras[1]},
        }
        wrapped = droid_io.wrap_obs(source, "Pick up the cube and put it in the bowl.")
        policy, _, state = droid_io.joint_configs()
        modalities = {
            name: SimpleNamespace(modality_keys=keys)
            for name, keys in {
                "video": ["exterior_image_1_left", "wrist_image_left"],
                "state": ["joint_position", "gripper_position"],
                "language": ["annotation.language.language_instruction"],
            }.items()
        }
        official = build_gr00t_policy_observations(
            [image.numpy() for image in cameras], states.numpy(), wrapped["task_descriptions"][0],
            SimpleNamespace(target_image_size=(180, 320, 3)), state, policy, modalities,
        )
        converted = droid_io.policy_obs(wrapped)
        for group in ("video", "state"):
            for key, value in official[group].items():
                np.testing.assert_array_equal(converted[f"{group}.{key}"], value)
        self.assertEqual(converted["video.exterior_image_1_left"].shape, (2, 1, 180, 320, 3))
        np.testing.assert_array_equal(converted["state.joint_position"][:, 0], states[:, :7])
        np.testing.assert_array_equal(converted["state.gripper_position"][:, 0], states[:, 7:8])
        self.assertEqual(converted["annotation.language.language_instruction"],
                         [item[0] for item in official["language"]["annotation.language.language_instruction"]])
        states.zero_()
        cameras[0].zero_()
        self.assertTrue(wrapped["states"].any())
        self.assertTrue(wrapped["main_images"].any())

    def test_actions_equal_official_builder_in_absolute_joint_order(self):
        from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import TaskMode
        from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_action_np

        joints = np.linspace(-0.7, 0.8, 2 * 32 * 7, dtype=np.float32).reshape(2, 32, 7)
        gripper = np.linspace(0, 1, 64, dtype=np.float32).reshape(2, 32, 1)
        actions = {"joint_position": joints, "gripper_position": gripper}
        policy, action, _ = droid_io.joint_configs()
        official = build_gr00t_action_np(
            actions, TaskMode.DROID_MANIPULATION, policy, action, embodiment_tag="OXE_DROID",
        ).astype(np.float32)
        converted = droid_io.policy_actions(actions)
        np.testing.assert_array_equal(converted, official)
        np.testing.assert_array_equal(converted, np.concatenate([joints, gripper], axis=-1))
        np.testing.assert_array_equal(droid_io.policy_actions(actions, chunk_size=5), official[:, :5])
        self.assertEqual(converted.shape, (2, 32, 8))


class MapleTableFactoryTest(unittest.TestCase):
    def test_sparse_reward_final_observation_and_episode_length(self):
        from embodied_ai_lab.environments.maple_table import (
            MapleTableRLEnvironment,
            MapleTableRLEnvironmentCfg,
            success_reward,
        )
        from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
            PickAndPlaceMapleTableEnvironment,
        )

        cfg = MapleTableRLEnvironmentCfg()
        self.assertEqual(cfg.episode_length_s, 70.4)
        self.assertEqual(cfg.embodiment, "droid_abs_joint_pos")
        description = SimpleNamespace(env_cfg_callback=lambda env_cfg: env_cfg)
        with patch.dict(sys.modules, {"isaaclab.managers": SimpleNamespace(RewardTermCfg=SimpleNamespace)}):
            with patch.object(PickAndPlaceMapleTableEnvironment, "build", return_value=description) as build:
                factory = object.__new__(MapleTableRLEnvironment)
                result = factory.build(cfg)
                build.assert_called_once_with(cfg)
                env_cfg = result.env_cfg_callback(SimpleNamespace())
        self.assertTrue(env_cfg.compute_final_obs)
        self.assertEqual(set(env_cfg.rewards), {"success"})
        self.assertIs(env_cfg.rewards["success"].func, success_reward)
        self.assertEqual(env_cfg.rewards["success"].weight / 15, 1)
        env = SimpleNamespace(termination_manager=SimpleNamespace(
            get_term=lambda name: torch.tensor([True, False]),
        ))
        torch.testing.assert_close(success_reward(env), torch.tensor([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
