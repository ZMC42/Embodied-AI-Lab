"""Compose the training recipe with RLinf's registered OmegaConf resolvers."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

import rlinf  # noqa: F401 -- registers the upstream multiply/int_div resolvers


ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigTest(unittest.TestCase):
    def test_offload_batch_and_typed_arena_train_eval_contract(self):
        searchpath = ROOT / "third_party/RLinf/examples/embodiment/config"
        with patch.dict(os.environ, LAB_ROOT=str(ROOT)), initialize_config_dir(
            version_base=None, config_dir=str(ROOT / "configs/rl/locomanip")
        ):
            for overrides, expected_batch in (([], 60), (["env.train.total_num_envs=4"], 120)):
                with self.subTest(overrides=overrides):
                    cfg = compose(
                        config_name="ppo-offload",
                        overrides=[f"hydra.searchpath=[file://{searchpath}]", *overrides],
                    )
                    self.assertEqual(cfg.actor.global_batch_size, expected_batch)
                    for env in (cfg.env.train, cfg.env.eval):
                        self.assertEqual(env.viser_envs, 0)
                        self.assertTrue(env.auto_reset)
                        self.assertFalse(env.ignore_terminations)
                        self.assertEqual(env.max_episode_steps, 1500)
                        self.assertTrue(env.arena.enable_cameras)
                        self.assertEqual(env.arena.embodiment, "g1_wbc_joint")
                        self.assertEqual(env.task, "galileo_g1_locomanip_pick_and_place_rl")
                        self.assertEqual(
                            env.external_environment_class_path,
                            "embodied_ai_lab.environments.locomanip:LocomanipRLEnvironment",
                        )

    def test_task_and_external_factory_overrides_reach_train_and_eval(self):
        searchpath = ROOT / "third_party/RLinf/examples/embodiment/config"
        with patch.dict(os.environ, LAB_ROOT=str(ROOT)), initialize_config_dir(
            version_base=None, config_dir=str(ROOT / "configs/rl/locomanip")
        ):
            for task, factory in (("lift_object", "null"), ("custom_task", "custom.env:Factory")):
                with self.subTest(task=task):
                    cfg = compose(
                        config_name="ppo-offload",
                        overrides=[
                            f"hydra.searchpath=[file://{searchpath}]",
                            f"env.train.task={task}",
                            f"env.train.external_environment_class_path={factory}",
                        ],
                    )
                    for env in (cfg.env.train, cfg.env.eval):
                        self.assertEqual(env.task, task)
                        self.assertEqual(env.external_environment_class_path,
                                         None if factory == "null" else factory)


if __name__ == "__main__":
    unittest.main()
