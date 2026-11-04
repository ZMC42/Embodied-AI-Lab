"""Local reference model behind Arena's official GR00T IO and chunk scheduler.

Import after SimulationApp starts, like the Arena policy it adapts.
"""

from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
    Gr00tClosedloopPolicyCfg,
    TaskMode,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyCfg,
    compute_action_dim,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml


@dataclass
class MapleReferencePolicyCfg(Gr00tBasePolicyCfg):
    model_path: str = ""
    num_envs: int = 4


@register_policy
class MapleReferencePolicy(PolicyBase[MapleReferencePolicyCfg]):
    name = "maple_reference"

    def __init__(self, config):
        super().__init__(config)
        self.policy_config = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyCfg
        )
        self.device = config.policy_device
        self.num_envs = config.num_envs
        self.task_mode = TaskMode(self.policy_config.task_mode_name)
        (self.policy_joints_config, self.robot_action_joints_config,
         self.robot_state_joints_config) = load_gr00t_joint_configs(self.policy_config)
        self._client = Gr00tPolicy(
            EmbodimentTag.OXE_DROID, config.model_path, device=self.device
        )
        self.modality_configs = self._client.modality_configs
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length
        self._chunking_state = ActionChunkScheduler(
            num_envs=self.num_envs, action_chunk_length=self.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=self.action_dim, device=self.device,
        )
        self.task_description = None

    # Retain the official preparation, action conversion and reset scheduling.
    get_action = Gr00tRemoteClosedloopPolicy.get_action
    _get_action_chunk = Gr00tRemoteClosedloopPolicy._get_action_chunk
    _extract_hold_action = Gr00tRemoteClosedloopPolicy._extract_hold_action
    set_task_description = Gr00tRemoteClosedloopPolicy.set_task_description
    reset = Gr00tRemoteClosedloopPolicy.reset

    def close(self):
        self._client = None
        self._chunking_state = None
        self.modality_configs = None
