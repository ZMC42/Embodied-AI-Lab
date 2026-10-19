"""只用于验证旧 ExampleEnvironmentBase 注册入口。"""

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class LegacyLocomanipEnvironment(ExampleEnvironmentBase):
    name = "galileo_g1_locomanip_pick_and_place_rl_legacy"

    @staticmethod
    def add_cli_args(parser):
        from isaaclab_arena_environments.cli import add_environment_cli_args

        from embodied_ai_lab.environments.locomanip import LocomanipRLEnvironment

        add_environment_cli_args(parser, LocomanipRLEnvironment)

    def get_env(self, args_cli):
        from isaaclab_arena_environments.cli import build_environment_from_cli

        from embodied_ai_lab.environments.locomanip import LocomanipRLEnvironment

        description = build_environment_from_cli(LocomanipRLEnvironment, args_cli)
        description.name = self.name
        return description
