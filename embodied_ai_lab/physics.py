"""项目配置与 Arena preset 的边界，以及本项目的物理补丁记录。"""

import hashlib
from importlib.metadata import distribution

from isaaclab_arena.utils.physics_backend import (
    PhysicsBackend,
    physics_protocol as arena_physics_protocol,
    validate_physics_device,
)


PHYSICS_PRESETS = {
    "isaacsim_physx": PhysicsBackend.PHYSX,
    "newton_mjwarp": PhysicsBackend.NEWTON,
}


def arena_preset(name, device):
    """验证项目配置并一次转换为 Arena preset。"""
    if name not in PHYSICS_PRESETS:
        raise ValueError(f"Unknown physics backend {name!r}; choose from {', '.join(PHYSICS_PRESETS)}")
    preset = PHYSICS_PRESETS[name]
    validate_physics_device(preset, device)
    return preset


def physics_protocol(env):
    """在 Arena 实际协议上补充本项目安装的 Newton 源码摘要。"""
    return {
        **arena_physics_protocol(env),
        "newton_solver_sha256": hashlib.sha256(
            distribution("newton").locate_file("newton/_src/solvers/mujoco/solver_mujoco.py").read_bytes()
        ).hexdigest(),
    }
