"""官方 G1 搬运任务加一次性成功奖励。"""

from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import (
    GalileoG1LocomanipPickAndPlaceEnvironment,
)


def success_reward(env):
    return env.termination_manager.get_term("success").float()


class LocomanipRLEnvironment(GalileoG1LocomanipPickAndPlaceEnvironment):
    name = "galileo_g1_locomanip_pick_and_place_rl"

    def build(self, cfg):
        from isaaclab.managers import RewardTermCfg

        description = super().build(cfg)
        original_callback = description.env_cfg_callback

        def callback(env_cfg):
            env_cfg = original_callback(env_cfg)
            # RewardManager 乘以 step_dt；50Hz 下权重50使成功当步奖励为1。
            env_cfg.rewards = {
                "success": RewardTermCfg(func=success_reward, weight=50.0)
            }
            env_cfg.compute_final_obs = True
            return env_cfg

        description.env_cfg_callback = callback
        return description
