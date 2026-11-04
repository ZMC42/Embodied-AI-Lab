"""DROID 转换复用 Arena 官方的观测构建、关节映射与动作构建。"""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


@lru_cache
def joint_configs():
    directory = (
        Path(__file__).resolve().parents[2]
        / "third_party/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/droid"
    )
    return tuple(
        yaml.safe_load((directory / name).read_text())["joints"]
        for name in (
            "gr00t_8dof_joint_space.yaml", "8dof_joint_space.yaml", "13dof_joint_space.yaml",
        )
    )


def wrap_obs(obs, description):
    return {
        "states": obs["policy"]["robot_joint_pos"].detach().cpu().clone(),
        "main_images": obs["camera_obs"]["external_camera_rgb"].detach().cpu().clone(),
        "wrist_images": obs["camera_obs"]["wrist_camera_rgb"].detach().cpu().clone(),
        "task_descriptions": [description] * obs["policy"]["robot_joint_pos"].shape[0],
    }


def policy_obs(env_obs):
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations

    policy, _, state = joint_configs()
    modalities = {
        name: SimpleNamespace(modality_keys=keys)
        for name, keys in {
            "video": ["exterior_image_1_left", "wrist_image_left"],
            "state": ["joint_position", "gripper_position"],
            "language": ["annotation.language.language_instruction"],
        }.items()
    }
    observations = build_gr00t_policy_observations(
        [env_obs[key].cpu().numpy() for key in ("main_images", "wrist_images")],
        env_obs["states"].cpu().float().numpy(),
        env_obs["task_descriptions"][0],
        SimpleNamespace(target_image_size=(180, 320, 3)),
        state, policy, modalities,
    )
    result = {
        f"{group}.{key}": value
        for group in ("video", "state")
        for key, value in observations[group].items()
    }
    result["annotation.language.language_instruction"] = env_obs["task_descriptions"]
    return result


def policy_actions(actions, chunk_size=32):
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import TaskMode
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_action_np

    policy, action, _ = joint_configs()
    result = build_gr00t_action_np(
        actions, TaskMode.DROID_MANIPULATION, policy, action, embodiment_tag="OXE_DROID",
    )
    return result[:, :chunk_size].astype(np.float32)
