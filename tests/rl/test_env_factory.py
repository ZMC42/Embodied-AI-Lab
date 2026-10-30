"""配置选择任务和外部工厂；隔离 Isaac Sim 启动与构建。"""

import importlib
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

from isaaclab_arena.assets import registries

from embodied_ai_lab.rlinf.env import make_sim


@dataclass
class PickCfg:
    object: str = "box"


@dataclass
class ReachCfg:
    target: str = "shelf"


class PickFactory:
    name = "pick"
    _legacy_argparse_cfg_type = PickCfg

    def build(self, cfg):
        assert isinstance(cfg, PickCfg)
        return f"pick {cfg.object}"


class ReachFactory:
    name = "reach"
    _legacy_argparse_cfg_type = ReachCfg

    def build(self, cfg):
        assert isinstance(cfg, ReachCfg)
        return f"reach {cfg.target}"


class EnvironmentFactoryTest(unittest.TestCase):
    def setUp(self):
        registry = registries.EnvironmentRegistry()
        self.enterContext(patch.object(registry, "_components", {
            "pick": PickFactory, "reach": ReachFactory,
        }))
        self.enterContext(patch.object(registries, "ensure_assets_registered"))
        self.app = object()
        self.launcher = Mock(return_value=SimpleNamespace(app=self.app))
        self.entry_point = type("DeferredReset", (), {})
        built = self.built = []

        class Builder:
            def __init__(builder, description, cfg):
                builder.description = description
                builder.cfg = cfg

            def make_registered(builder):
                built.append(builder)
                env = SimpleNamespace(
                    device=builder.cfg.device,
                    num_envs=builder.cfg.num_envs,
                    get_language_instruction=lambda: builder.description,
                )
                return SimpleNamespace(unwrapped=env)

        def build_environment(factory, namespace):
            return factory().build(factory._legacy_argparse_cfg_type(**vars(namespace)))

        def parse_external(path):
            module, name = path.split(":")
            factory = getattr(importlib.import_module(module), name)
            return factory.name, factory

        self.build_environment = Mock(side_effect=build_environment)
        self.enterContext(patch.dict(sys.modules, {
            "isaaclab.app": SimpleNamespace(AppLauncher=self.launcher),
            "isaaclab_arena.environments.arena_env_builder": SimpleNamespace(
                ArenaEnvBuilder=Builder,
            ),
            "isaaclab_arena.environments.arena_env_builder_cfg": SimpleNamespace(
                ArenaEnvBuilderCfg=SimpleNamespace,
            ),
            "isaaclab_arena_environments.cli": SimpleNamespace(
                ensure_environments_registered=Mock(),
                parse_and_return_external_environment_from_string=parse_external,
                build_environment_from_cli=self.build_environment,
            ),
            "isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment":
                SimpleNamespace(GalileoG1LocomanipPickAndPlaceEnvironmentCfg=PickCfg),
            "embodied_ai_lab.environments.locomanip": SimpleNamespace(
                LocomanipRLEnvironment=PickFactory,
            ),
            "embodied_ai_lab.rlinf.deferred_reset": SimpleNamespace(
                DeferredResetArenaEnv=self.entry_point,
            ),
            "factory_fixture": SimpleNamespace(ReachFactory=ReachFactory),
            "factory_io_fixture": SimpleNamespace(
                wrap_obs=lambda obs, description: (obs, description),
            ),
        }))

    def config(self, task, arena, external=None):
        return SimpleNamespace(
            task=task, arena=arena, external_environment_class_path=external,
            device="cpu", viser_envs=0, io_module="factory_io_fixture",
        )

    def test_registered_task_selects_its_factory_and_config(self):
        for task, arena, expected in (
            ("pick", {"object": "crate"}, "pick crate"),
            ("reach", {"target": "table"}, "reach table"),
        ):
            with self.subTest(task=task):
                env, app = make_sim(self.config(task, arena), 3, 73)
                self.assertIs(app, self.app)
                self.assertEqual(env.convert_obs("observation"), ("observation", expected))
                self.assertEqual(vars(self.built[-1].cfg), {
                    "num_envs": 3, "seed": 73, "device": "cpu",
                })
                self.assertIs(self.built[-1].get_entry_point(), self.entry_point)
        self.assertEqual([call.args[0] for call in self.build_environment.call_args_list],
                         [PickFactory, ReachFactory])

    def test_external_factory_is_used_without_registry_registration(self):
        registry = registries.EnvironmentRegistry()
        self.enterContext(patch.object(registry, "_components", {}))
        env, _ = make_sim(self.config(
            "reach", {"target": "cabinet"}, "factory_fixture:ReachFactory",
        ), 2, 42)
        self.assertEqual(env.convert_obs("observation"), ("observation", "reach cabinet"))
        self.assertIs(self.build_environment.call_args.args[0], ReachFactory)

    def test_external_factory_task_mismatch_fails_before_building(self):
        with self.assertRaisesRegex(AssertionError, "pick|reach"):
            make_sim(self.config(
                "pick", {"target": "table"}, "factory_fixture:ReachFactory",
            ), 2, 42)
        self.assertEqual(self.built, [])
        self.build_environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
