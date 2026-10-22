"""使用 RLinf 的仿真子进程承载注册后的 Arena Gym 环境。"""

import importlib
import json
import os
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv

from embodied_ai_lab.rlinf.configuration import arena_arguments


def make_sim(cfg, num_envs, seed, debug=False, worker_index=0):
    if debug:
        from embodied_ai_lab.rlinf.debugging import wait_for_debugger

        wait_for_debugger("env")

    arguments = arena_arguments(cfg)
    backend = getattr(cfg, "physics_backend", "isaacsim_physx")
    from isaaclab.app import AppLauncher

    launcher_kwargs = {}
    if cfg.viser_envs > 0:
        launcher_kwargs.update(visualizer="viser", max_visible_envs=cfg.viser_envs)
    app = AppLauncher(
        headless=True, enable_cameras=True, device=cfg.device, **launcher_kwargs
    ).app
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback,
    )

    from embodied_ai_lab.rlinf.chunk_env import ChunkEnv
    from embodied_ai_lab.rlinf.deferred_reset import DeferredResetArenaEnv

    argv = sys.argv
    sys.argv = [
        argv[0],
        "--task",
        cfg.task,
        "--headless",
        "--enable_cameras",
        "--device",
        cfg.device,
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        *arguments,
    ]
    try:
        environment_registration_callback()
    finally:
        sys.argv = argv
    spec = gym.spec(cfg.task)
    spec.entry_point = DeferredResetArenaEnv
    env_cfg = spec.kwargs["env_cfg_entry_point"]
    env_cfg.seed = seed
    env_cfg.sim.device = cfg.device
    env_cfg.scene.num_envs = num_envs
    from isaaclab_physx.physics import PhysxCfg

    if isinstance(env_cfg.sim.physics, PhysxCfg):
        env_cfg.sim.physics.gpu_found_lost_pairs_capacity = 2**16
        env_cfg.sim.physics.gpu_found_lost_aggregate_pairs_capacity = 2**16
        env_cfg.sim.physics.gpu_total_aggregate_pairs_capacity = 2**16
    env = gym.make(cfg.task, cfg=env_cfg).unwrapped
    if getattr(cfg, "physics_log_dir", None):
        from embodied_ai_lab.physics import physics_protocol

        role = "eval" if cfg.is_eval else "train"
        directory = Path(cfg.physics_log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(
            role=role, worker_index=worker_index, pid=os.getpid(), seed=seed,
            requested_physics_backend=backend,
            **physics_protocol(env),
        )
        path = directory / f"{role}-worker_{worker_index}-pid_{os.getpid()}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
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
            worker_index=seed_offset,
        ))
        # 等待当前仿真初始化完成，避免 train/eval 同时导入模型的主存峰值叠加。
        self.env.device()
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
