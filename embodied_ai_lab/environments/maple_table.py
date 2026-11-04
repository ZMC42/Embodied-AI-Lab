"""官方 Maple table DROID 任务加一次性成功奖励。"""

from dataclasses import dataclass

from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)


@dataclass
class MapleTableRLEnvironmentCfg(PickAndPlaceMapleTableEnvironmentCfg):
    episode_length_s: float = 70.4


def success_reward(env):
    return env.termination_manager.get_term("success").float()


class MapleTableRLEnvironment(PickAndPlaceMapleTableEnvironment):
    name = "pick_and_place_maple_table_rl"
    _legacy_argparse_cfg_type = MapleTableRLEnvironmentCfg

    def build(self, cfg):
        from isaaclab.managers import RewardTermCfg

        description = super().build(cfg)
        original_callback = description.env_cfg_callback

        def callback(env_cfg):
            env_cfg = original_callback(env_cfg)
            # RewardManager 乘以 step_dt；15Hz 下权重15使成功当步奖励为1。
            env_cfg.rewards = {
                "success": RewardTermCfg(func=success_reward, weight=15.0)
            }
            env_cfg.compute_final_obs = True
            return env_cfg

        description.env_cfg_callback = callback
        return description
