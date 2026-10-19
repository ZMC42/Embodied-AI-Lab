"""将 Gym 向量环境转换为 RLinf 的动作块与终局观测接口。"""

import torch


class ChunkEnv:
    """块内终止只记录一次，块末统一 reset，保留真实终局观测供 bootstrap。"""

    def __init__(self, env, convert_obs):
        self.env = env
        self.convert_obs = convert_obs
        self.device = env.device
        self.num_envs = env.num_envs
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.lengths = torch.zeros(self.num_envs, device=self.device)
        self.signals = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        for name in ("success_once", "fall_once")}

    def reset(self, seed=None, env_ids=None):
        obs, _ = self.env.reset(seed=seed, env_ids=env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.returns[ids] = 0
        self.lengths[ids] = 0
        for signal in self.signals.values():
            signal[ids] = False
        return self.convert_obs(obs), {}

    def step(self, actions):
        actions = actions.to(self.device)
        n, horizon, _ = actions.shape
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        terminated = torch.zeros_like(done)
        truncated = torch.zeros_like(done)
        rewards = torch.zeros(n, horizon, device=self.device)
        self.env.defer_resets = True
        final_obs = None
        for i in range(horizon):
            obs, reward, term, trunc, _ = self.env.step(actions[:, i])
            active = ~done
            for name, signal in self.env.episode_signals().items():
                self.signals[name] |= signal & active
            rewards[:, i] = reward * active
            self.returns += rewards[:, i]
            self.lengths += active
            term = term.clone() & active
            # RLinf按truncated补bootstrap；终止与时间上限同一步发生时，以终止为准。
            trunc = trunc.clone() & active & ~term
            ended = term | trunc
            converted = self.convert_obs(obs)
            if final_obs is None:
                final_obs = {
                    key: value.clone()
                    if isinstance(value, torch.Tensor)
                    else list(value)
                    for key, value in converted.items()
                }
            for key, value in converted.items():
                if isinstance(value, torch.Tensor):
                    final_obs[key][active.cpu()] = value[active.cpu()]
            terminated |= term
            truncated |= trunc
            done |= ended
        self.env.defer_resets = False
        terminations = torch.zeros(n, horizon, dtype=torch.bool, device=self.device)
        truncations = torch.zeros_like(terminations)
        terminations[:, -1] = terminated
        truncations[:, -1] = truncated
        infos = {}
        if done.any():
            episode = {
                "return": self.returns.cpu().clone(),
                "episode_len": self.lengths.cpu().clone(),
                **{name: signal.cpu().clone() for name, signal in self.signals.items()},
            }
            converted, _ = self.reset(env_ids=done.nonzero().flatten())
            infos = {
                "final_observation": final_obs,
                "_final_observation": done.cpu(),
                "final_info": {"episode": episode},
                "_final_info": done.cpu(),
            }
        # 释放仿真子进程的空闲缓存，为同卡 PPO 更新留出显存。
        torch.cuda.empty_cache()
        return (
            [converted],
            rewards.cpu(),
            terminations.cpu(),
            truncations.cpu(),
            [infos],
        )

    def close(self):
        self.env.close()
