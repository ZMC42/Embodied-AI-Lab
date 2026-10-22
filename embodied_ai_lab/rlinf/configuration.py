"""在启动模型和仿真进程前确定各角色的物理后端。"""

import json
from pathlib import Path

from embodied_ai_lab.physics import arena_preset


def arena_arguments(cfg):
    """项目字段唯一决定物理后端，转换为 Arena 的原生 preset 参数。"""
    preset = arena_preset(getattr(cfg, "physics_backend", "isaacsim_physx"), cfg.device)
    arguments = list(cfg.arena_args)
    if any(argument.partition("=")[0] in ("--presets", "--physics-backend") for argument in arguments):
        raise ValueError("Set physics_backend in the environment config, not a selector in arena_args")
    return [*arguments, "--presets", preset.value]


def _source_backends(checkpoint):
    """读取源运行产物，旧 checkpoint 缺少后端记录时保持未知。"""
    run_dir = Path(checkpoint).resolve().parent.parent
    record_path = run_dir / "physics/run.json"
    if record_path.is_file():
        return json.loads(record_path.read_text())["current_backends"]
    return {"train": "unknown", "eval": "unknown"}


def prepare_training_config(cfg):
    """在模型或仿真启动前验证两套配置，并记录当前与来源后端。"""
    for role in ("train", "eval"):
        arena_arguments(cfg.env[role])

    current = {
        role: cfg.env[role].physics_backend
        for role in ("train", "eval")
    }
    checkpoint = cfg.runner.resume_dir
    record = {
        "source_checkpoint": checkpoint,
        "source_backends": _source_backends(checkpoint) if checkpoint else None,
        "current_backends": current,
    }
    directory = Path(cfg.env.train.physics_log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return cfg
