"""Compose the training recipe with RLinf's registered OmegaConf resolvers."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

import rlinf  # noqa: F401 -- registers the upstream multiply/int_div resolvers


ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigTest(unittest.TestCase):
    def test_maple_table_demo_keeps_six_optimizer_updates(self):
        searchpath = ROOT / "third_party/RLinf/examples/embodiment/config"
        with patch.dict(os.environ, LAB_ROOT=str(ROOT)), initialize_config_dir(
            version_base=None, config_dir=str(ROOT / "configs/rl/maple-table")
        ):
            cfg = compose(
                config_name="ppo",
                overrides=[f"hydra.searchpath=[file://{searchpath}]"],
            )
            self.assertEqual(cfg.actor.global_batch_size, 11)
            self.assertEqual(cfg.env.train.total_num_envs, 2)
            self.assertEqual(cfg.cluster.component_placement["actor,env,rollout"], "0-0")
            self.assertEqual(cfg.runner.max_epochs, 3)
            self.assertEqual(cfg.runner.max_steps, 3)
            self.assertEqual(cfg.runner.save_interval, 3)
            self.assertEqual(cfg.runner.logger.experiment_name, "maple-table-blue-block-ppo-demo")
            chunks = cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
            self.assertEqual(chunks, 33)
            self.assertEqual(cfg.env.train.total_num_envs * chunks // cfg.actor.global_batch_size, 6)
            self.assertEqual(cfg.algorithm.update_epoch, 1)
            self.assertEqual(cfg.actor.optim.lr, 2.5e-6)
            self.assertEqual(cfg.actor.optim.value_lr, 1e-4)
            self.assertTrue(cfg.actor.enable_offload)
            self.assertTrue(cfg.rollout.enable_offload)
            self.assertEqual(cfg.actor.micro_batch_size, 1)
            self.assertEqual(cfg.actor.model.train_last_blocks, 1)
            self.assertEqual(cfg.weight_syncer.type, "bucket")
            self.assertEqual(cfg.weight_syncer.bucket.bucket_size, 536870912)
            self.assertIsNone(cfg.weight_syncer.bucket.bucket_dtype)
            self.assertFalse(cfg.weight_syncer.bucket.is_agent)
            self.assertTrue(cfg.weight_syncer.bucket.load_instant)
            self.assertNotIn("patch", cfg.weight_syncer)
            self.assertEqual(cfg.actor.model.action_dim, 8)
            self.assertEqual(cfg.actor.model.denoising_steps, 8)
            self.assertEqual(cfg.actor.model.embodiment_tag, "oxe_droid")
            self.assertEqual(cfg.actor.model.obs_converter_type, "arena_droid")
            for env in (cfg.env.train, cfg.env.eval):
                self.assertEqual(env.max_episode_steps, 1056)
                self.assertEqual(env.max_steps_per_rollout_epoch, 1056)
                self.assertEqual(env.arena.episode_length_s, 70.4)
                self.assertTrue(env.auto_reset)
                self.assertFalse(env.ignore_terminations)
                self.assertEqual(env.arena.embodiment, "droid_abs_joint_pos")
                self.assertEqual(env.task, "pick_and_place_maple_table_rl")
                self.assertEqual(env.arena.pick_up_object, "blue_block_basic_robolab")
                self.assertEqual(env.arena.destination_location, "wooden_bowl_hot3d_robolab")
                self.assertEqual(env.arena.hdr, "home_office_robolab")
                self.assertEqual(env.language_instruction, "Pick up the blue block and place it in the bowl.")
            self.assertEqual(cfg.env.eval.language_instruction, cfg.env.train.language_instruction)

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
