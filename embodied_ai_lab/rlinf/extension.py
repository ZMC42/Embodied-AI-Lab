"""每个 RLinf worker 通过 RLINF_EXT_MODULE 加载任务与模型转换。"""

from contextlib import contextmanager
from functools import partial

import torch


@contextmanager
def cpu_checkpoint_state_dict(model):
    """在 FSDP 克隆 state_dict 之前搬到 CPU，避免单卡复制整个模型到 GPU。"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    def to_cpu(module, state, prefix, metadata):
        for key, value in state.items():
            if key.startswith(prefix):
                state[key] = value.cpu()

    handles = [module.module.register_state_dict_post_hook(to_cpu)
               for module in model.modules() if isinstance(module, FSDP)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def install_checkpoint_cpu_clone():
    """在 driver 复制 Ray worker 类之前安装，仅影响 checkpoint 保存。"""
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    original_save = EmbodiedFSDPActor.save_checkpoint

    def save_checkpoint(self, save_path, step=0):
        with cpu_checkpoint_state_dict(self.model):
            return original_save(self, save_path, step)

    EmbodiedFSDPActor.save_checkpoint = save_checkpoint


def predict_g1(model, env_obs, mode="train", **kwargs):
    """和 Arena 评测使用相同的原始状态、图像及 processor token 长度。"""
    observations = model.obs_convert_fn(env_obs)
    decode_state = {key.removeprefix("state."): value
                    for key, value in observations.items() if key.startswith("state.")}
    inputs = model.apply_transforms(observations)
    inputs = {key: value.bfloat16() if isinstance(value, torch.Tensor)
              and value.dtype == torch.float32 else value for key, value in inputs.items()}
    normalized, result = model._get_rl_action(inputs, mode=mode)
    decoded = model._get_unnormalized_action(normalized, state=decode_state)
    actions = model.action_convert_fn(decoded, chunk_size=model.output_action_chunks)
    return actions, result


def predict_micro_batches(predict, env_obs, batch_size, **kwargs):
    """固定推理批次大小，使本地 BF16 的采样和训练重算使用相同前向形状。"""
    from rlinf.data.schema.embodied_types import merge_batch_values, split_batch_value

    n = env_obs["states"].shape[0]
    sizes = [min(batch_size, n - start) for start in range(0, n, batch_size)]
    outputs = [
        predict(env_obs=obs, **kwargs) for obs in split_batch_value(env_obs, sizes)
    ]
    return (
        merge_batch_values([actions for actions, _ in outputs]),
        merge_batch_values([result for _, result in outputs]),
    )


def build_model(cfg, torch_dtype=torch.bfloat16):
    from embodied_ai_lab.rlinf.debugging import wait_for_model_debugger
    from rlinf.models.embodiment.gr00t.gr00t_n1d6 import get_model

    wait_for_model_debugger()
    # 冻结基座始终 BF16；actor 的 FP32 仅用于可训练主权重。
    model = get_model(cfg, torch.bfloat16)
    model.requires_grad_(False)
    blocks = cfg.get("train_last_blocks", None)
    if blocks is None:
        model.action_head.requires_grad_(True)
    else:
        model.action_head.action_decoder.requires_grad_(True)
        model.action_head.model.transformer_blocks[-blocks:].requires_grad_(True)
        model.action_head.value_head.requires_grad_(True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch_dtype)
    model._no_split_modules = list(cfg.no_split_modules)
    if cfg.obs_converter_type == "arena_g1":
        model.predict_action_batch = partial(predict_g1, model)
    if cfg.get("inference_micro_batch_size"):
        model.predict_action_batch = partial(
            predict_micro_batches,
            model.predict_action_batch,
            batch_size=cfg.inference_micro_batch_size,
        )
    return model


def register():
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from rlinf.models import register_model
    from rlinf.models.embodiment.gr00t.simulation_io import (
        ACTION_CONVERSION_N1D6,
        OBS_CONVERSION,
    )

    from embodied_ai_lab.environments import g1_io
    from embodied_ai_lab.rlinf.env import ArenaEnv

    REGISTER_ISAACLAB_ENVS["Arena-v0"] = ArenaEnv
    OBS_CONVERSION["arena_g1"] = g1_io.policy_obs
    ACTION_CONVERSION_N1D6["arena_g1"] = g1_io.policy_actions
    register_model("gr00t_n1d6", build_model, force=True)
