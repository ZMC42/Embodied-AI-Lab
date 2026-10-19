"""G1 任务转换复用 Arena 的关节顺序和图像预处理。"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


@lru_cache
def joint_configs():
    directory = (
        Path(__file__).resolve().parents[2]
        / "third_party/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1"
    )
    return tuple(
        yaml.safe_load((directory / name).read_text())["joints"]
        for name in ("gr00t_43dof_joint_space.yaml", "43dof_joint_space.yaml")
    )


def wrap_obs(obs, description):
    return {
        "states": obs["policy"]["robot_joint_pos"].detach().cpu().clone(),
        "main_images": obs["camera_obs"]["robot_head_cam_rgb"].detach().cpu().clone(),
        "task_descriptions": [description] * obs["policy"]["robot_joint_pos"].shape[0],
    }


def policy_obs(env_obs):
    from isaaclab_arena_gr00t.policy.gr00t_core import resize_rgb_for_policy
    from isaaclab_arena_gr00t.utils.joints_conversion import (
        remap_sim_joints_to_policy_joints_from_np,
    )

    policy, sim = joint_configs()
    states = remap_sim_joints_to_policy_joints_from_np(
        env_obs["states"].cpu().float().numpy(), sim, policy
    )
    rgb = resize_rgb_for_policy([env_obs["main_images"].cpu().numpy()], (480, 640, 3))[
        0
    ]
    result = {
        "video.ego_view": rgb[:, None],
        "annotation.human.action.task_description": env_obs["task_descriptions"],
    }
    for group in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
        result[f"state.{group}"] = states[group][:, None]
    return result


def policy_actions(actions, chunk_size=50):
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
        TaskMode,
    )
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_action_np

    policy, sim = joint_configs()
    result = build_gr00t_action_np(actions, TaskMode.G1_LOCOMANIPULATION, policy, sim)
    return result[:, :chunk_size].astype(np.float32)
