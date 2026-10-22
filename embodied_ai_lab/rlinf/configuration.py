"""在启动模型和仿真进程前确定各角色的物理后端。"""

import json
from pathlib import Path

from omegaconf import OmegaConf

from isaaclab_arena.utils.physics_backend import (
    PhysicsBackend,
    canonical_physics_backend,
    physics_backend_from_name,
    validate_physics_device,
)


def arena_arguments(cfg):
    """结构化字段决定后端；兼容且移除一致的旧 Arena 选择器。"""
    name = getattr(cfg, "physics_backend", "isaacsim_physx")
    backend = physics_backend_from_name(name)
    validate_physics_device(backend, cfg.device)
    arguments = list(cfg.arena_args)
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option, separator, value = argument.partition("=")
        if option in ("--presets", "--physics-backend"):
            if not separator:
                index += 1
                if index == len(arguments):
                    raise ValueError(f"{option} requires a value")
                value = arguments[index]
            selected = (
                PhysicsBackend(value) if option == "--presets"
                else physics_backend_from_name(value)
            )
            if selected != backend:
                raise ValueError(
                    f"arena_args {option}={value} conflicts with "
                    f"physics_backend={name}"
                )
        else:
            result.append(argument)
        index += 1
    return result


def _source_backends(checkpoint):
    """读取源运行产物，旧 checkpoint 缺少后端记录时保持未知。"""
    run_dir = Path(checkpoint).resolve().parent.parent
    record_path = run_dir / "physics/run.json"
    if record_path.is_file():
        return json.loads(record_path.read_text())["current_backends"]
    candidates = (
        run_dir / "physics/config.yaml",
        run_dir.parent / "tensorboard/all/config.yaml",
        run_dir.parent / "tensorboard/config.yaml",
    )
    for path in candidates:
        if path.is_file():
            cfg = OmegaConf.load(path)
            if path != run_dir / "physics/config.yaml":
                log_path = OmegaConf.select(cfg, "runner.logger.log_path")
                experiment = OmegaConf.select(cfg, "runner.logger.experiment_name")
                # TensorBoard 配置由同一日志目录下的实验共享，必须确认属于源运行。
                if not log_path or not experiment or not Path(log_path).is_absolute():
                    continue
                if (Path(log_path) / experiment).resolve() != run_dir:
                    continue
            return {
                role: OmegaConf.select(cfg, f"env.{role}.physics_backend", default="unknown")
                for role in ("train", "eval")
            }
    return {"train": "unknown", "eval": "unknown"}


def prepare_training_config(cfg):
    """先处理 train，使插值继承 arena_args 的 eval 可独立覆盖后端。"""
    for role in ("train", "eval"):
        env_cfg = cfg.env[role]
        env_cfg.arena_args = arena_arguments(env_cfg)

    current = {
        role: canonical_physics_backend(physics_backend_from_name(cfg.env[role].physics_backend))
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
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    return cfg
