"""Record navigation and falls before Arena resets a finishing episode."""

import torch
from isaaclab.managers.recorder_manager import RecorderTerm


class BaselineDiagnosticsRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.initial_box_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.min_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        self.fall_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.baseline_diagnostics = self

    def record_post_reset(self, env_ids):
        self.initial_box_xy[env_ids] = self._env.arena_world.get_position_w("brown_box")[env_ids, :2]
        self.min_distance[env_ids] = float("inf")
        self.fall_once[env_ids] = False
        return None, None

    def record_post_step(self):
        env = self._env
        active = env.active_episode_mask
        base_xy = env.arena_world.get_position_w("robot")[:, :2]
        distance = torch.linalg.vector_norm(base_xy - self.initial_box_xy, dim=-1)
        self.min_distance[active] = torch.minimum(self.min_distance[active], distance[active])
        # Match the training diagnostic: pelvis drop relative to the default standing pose.
        robot = env.scene["robot"]
        height_delta = (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                        - robot.data.default_root_state[:, 2])
        self.fall_once[active] |= height_delta[active] < -0.4
        return None, None


def record_diagnostics(env, env_id):
    """Arena supplies the global environment id, including during partial resets."""
    diagnostics = env.baseline_diagnostics
    return {
        "fall_once": bool(diagnostics.fall_once[env_id]),
        "min_base_box_xy_distance_m": float(diagnostics.min_distance[env_id]),
        "initial_box_xy_m": diagnostics.initial_box_xy[env_id].tolist(),
        "final_box_xyz_m": env.arena_world.get_position_w("brown_box")[env_id].tolist(),
        "termination_terms": {
            name: bool(env.termination_manager.get_term(name)[env_id])
            for name in env.termination_manager.active_terms
        },
    }
