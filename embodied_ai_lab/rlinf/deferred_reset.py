"""仅推迟自动 reset，不改变官方任务的终止判定。须在启动 Sim 后导入。"""

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnv,
)


class DeferredResetArenaEnv(IsaacLabArenaManagerBasedRLEnv):
    defer_resets = False

    def step(self, action):
        if not self.defer_resets:
            return super().step(action)
        # 父类即使未实际 reset 也会重渲染；延期期间只保留正常物理步渲染。
        rerenders = self.cfg.num_rerenders_on_reset
        self.cfg.num_rerenders_on_reset = 0
        try:
            return super().step(action)
        finally:
            self.cfg.num_rerenders_on_reset = rerenders

    def episode_signals(self):
        # 仅作诊断；场景地板低于原点，使用相对初始站立位置的下降量。
        robot = self.scene["robot"]
        height_delta = (robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
                        - robot.data.default_root_state[:, 2])
        return {
            "success_once": self.termination_manager.get_term("success"),
            "fall_once": height_delta < -0.4,
        }

    def _reset_idx(self, env_ids):
        if not self.defer_resets:
            super()._reset_idx(env_ids)
