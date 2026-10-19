"""RLinf 动作块的终局奖励和 bootstrap 观测契约。"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from embodied_ai_lab.environments.g1_io import joint_configs, policy_actions, policy_obs
from embodied_ai_lab.rlinf.chunk_env import ChunkEnv
from embodied_ai_lab.rlinf.extension import predict_micro_batches


class Simulator:
    """第2步成功且达时间上限，第3步另一环境超时；reset 会替换观测。"""

    device = "cpu"
    num_envs = 2
    defer_resets = False

    def __init__(self):
        self.steps = 0
        self.resets = []

    def reset(self, seed=None, env_ids=None):
        if env_ids is not None:
            self.resets.append(env_ids.tolist())
        return {"states": torch.full((2, 1), -1.0)}, {}

    def step(self, actions):
        assert self.defer_resets
        self.steps += 1
        return (
            {"states": torch.full((2, 1), float(self.steps))},
            torch.tensor([float(self.steps >= 2), 0.0]),
            torch.tensor([self.steps >= 2, False]),
            torch.tensor([self.steps >= 2, self.steps >= 3]),
            {},
        )

    def close(self):
        pass

    def episode_signals(self):
        return {"success_once": torch.tensor([self.steps >= 2, False]),
                "fall_once": torch.tensor([False, self.steps == 1])}


class ChunkEnvTest(unittest.TestCase):
    def test_g1_observations_match_arena_evaluation_builder(self):
        from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations

        states = torch.linspace(-0.91, 0.83, 86).reshape(2, 43)
        images = torch.arange(2 * 480 * 640 * 3).remainder(256).byte().reshape(2, 480, 640, 3)
        obs = {"states": states, "main_images": images, "task_descriptions": ["pick", "pick"]}
        groups = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
        modality = {name: SimpleNamespace(modality_keys=keys) for name, keys in {
            "state": groups, "video": ["ego_view"], "language": ["task_description"],
        }.items()}
        policy, sim = joint_configs()
        official = build_gr00t_policy_observations(
            [images.numpy()], states.numpy(), "pick",
            SimpleNamespace(target_image_size=(480, 640, 3)), sim, policy, modality,
        )
        converted = policy_obs(obs)
        for name in ("video", "state"):
            for key, value in official[name].items():
                np.testing.assert_array_equal(converted[f"{name}.{key}"], value)

    def test_single_sample_inference_preserves_actions_and_training_input_order(self):
        def predict(env_obs, mode):
            self.assertEqual(mode, "train")
            self.assertEqual(env_obs["states"].shape[0], 1)
            state = env_obs["states"]
            return state.numpy(), {
                "prev_logprobs": state + 10,
                "forward_inputs": {"state": state, "label": env_obs["labels"]},
                "unused": None,
            }

        states = torch.tensor([[3.0], [1.0], [2.0]])
        actions, result = predict_micro_batches(
            predict, {"states": states, "labels": ["c", "a", "b"]}, 1, mode="train"
        )
        np.testing.assert_array_equal(actions, states.numpy())
        torch.testing.assert_close(result["prev_logprobs"], states + 10)
        torch.testing.assert_close(result["forward_inputs"]["state"], states)
        self.assertEqual(result["forward_inputs"]["label"], ["c", "a", "b"])
        self.assertIsNone(result["unused"])

    def test_g1_roundtrip_preserves_upper_body_navigation_and_image_format(self):
        states = torch.arange(86).reshape(2, 43).float()
        observations = policy_obs(
            {
                "states": states,
                "main_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
                "task_descriptions": ["pick", "pick"],
            }
        )
        self.assertEqual(observations["video.ego_view"].shape, (2, 1, 480, 640, 3))
        actions = {
            key.removeprefix("state."): np.repeat(value, 50, axis=1)
            for key, value in observations.items()
            if key.startswith("state.")
        }
        actions["navigate_command"] = np.full((2, 50, 3), 0.2)
        actions["base_height_command"] = np.full((2, 50, 1), 0.75)
        result = policy_actions(actions)
        self.assertEqual(result.shape, (2, 50, 50))
        controlled = [2, 5, 8, 11, 12, 15, 16, *range(19, 43)]
        np.testing.assert_array_equal(
            result[:, 0, controlled], states[:, controlled].numpy()
        )
        np.testing.assert_allclose(result[:, :, 43:46], 0.2)
        np.testing.assert_allclose(result[:, :, 46], 0.75)
        np.testing.assert_array_equal(result[:, :, 47:], 0)

    def test_terminal_reward_once_final_obs_before_reset_and_done_at_chunk_end(self):
        sim = Simulator()
        env = ChunkEnv(sim, lambda x: x)
        env.reset()
        obs, rewards, terminated, truncated, infos = env.step(torch.zeros(2, 5, 1))
        self.assertEqual(rewards.tolist(), [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.assertEqual(terminated.tolist(), [[False] * 4 + [True], [False] * 5])
        self.assertEqual(truncated.tolist(), [[False] * 5, [False] * 4 + [True]])
        self.assertEqual(
            infos[-1]["final_observation"]["states"].flatten().tolist(), [2, 3]
        )
        self.assertEqual(obs[-1]["states"].flatten().tolist(), [-1, -1])
        self.assertEqual(sim.resets, [[0, 1]])
        episode = infos[-1]["final_info"]["episode"]
        self.assertEqual(episode["success_once"].tolist(), [True, False])
        self.assertEqual(episode["fall_once"].tolist(), [False, True])
        self.assertFalse(any(signal.any() for signal in env.signals.values()))
        self.assertFalse(sim.defer_resets)

    def test_reward_does_not_imply_success_and_post_terminal_fall_is_ignored(self):
        sim = Simulator()
        sim.episode_signals = lambda: {
            "success_once": torch.tensor([False, False]),
            "fall_once": torch.tensor([sim.steps >= 3, False]),
        }
        env = ChunkEnv(sim, lambda x: x)
        env.reset()
        _, _, _, _, infos = env.step(torch.zeros(2, 5, 1))
        episode = infos[-1]["final_info"]["episode"]
        self.assertGreater(episode["return"][0], 0)
        self.assertEqual(episode["success_once"].tolist(), [False, False])
        self.assertEqual(episode["fall_once"].tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
