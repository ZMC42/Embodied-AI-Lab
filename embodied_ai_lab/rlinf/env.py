"""使用 RLinf 的仿真子进程承载注册后的 Arena Gym 环境。"""

import importlib
from functools import partial

import gymnasium as gym
from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv


def make_sim(cfg, num_envs, seed, debug=False):
    if debug:
        from embodied_ai_lab.rlinf.debugging import wait_for_debugger

        wait_for_debugger("env")

    from isaaclab.app import AppLauncher

    launcher_kwargs = {}
    if cfg.viser_envs > 0:
        launcher_kwargs.update(visualizer="viser", max_visible_envs=cfg.viser_envs)
    app = AppLauncher(
        headless=True, enable_cameras=True, device=cfg.device, **launcher_kwargs
    ).app
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import (
        GalileoG1LocomanipPickAndPlaceEnvironmentCfg,
    )

    from embodied_ai_lab.environments.locomanip import LocomanipRLEnvironment
    from embodied_ai_lab.rlinf.chunk_env import ChunkEnv
    from embodied_ai_lab.rlinf.deferred_reset import DeferredResetArenaEnv

    class Builder(ArenaEnvBuilder):
        def get_entry_point(self):
            return DeferredResetArenaEnv

    description = LocomanipRLEnvironment().build(
        GalileoG1LocomanipPickAndPlaceEnvironmentCfg(**cfg.arena)
    )
    env = Builder(
        description, ArenaEnvBuilderCfg(num_envs=num_envs, seed=seed, device=cfg.device)
    ).make_registered().unwrapped
    converter = importlib.import_module(cfg.io_module)
    return ChunkEnv(
        env, partial(converter.wrap_obs, description=env.get_language_instruction())
    ), app


class ArenaEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.env = SubProcIsaacLabEnv(partial(
            make_sim, cfg, num_envs, self.seed,
            debug=seed_offset == 0 and not cfg.is_eval,
        ))
        self._is_start = True

    def reset(self, seed=None, env_ids=None):
        return self.env.reset(seed=self.seed if seed is None else seed, env_ids=env_ids)

    def chunk_step(self, actions):
        return self.env.step(actions)

    def update_reset_state_ids(self):
        pass

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def close(self):
        self.env.close()
