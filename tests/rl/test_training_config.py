"""Compose the training recipe with RLinf's registered OmegaConf resolvers."""

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from hydra import compose, initialize_config_dir

import rlinf  # noqa: F401 -- registers the upstream multiply/int_div resolvers

from embodied_ai_lab.rlinf.configuration import _source_backends, arena_arguments, prepare_training_config
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigTest(unittest.TestCase):
    def test_serial_initialization_keeps_resume_after_all_workers_ready(self):
        from embodied_ai_lab.rlinf.extension import install_serial_worker_initialization
        from rlinf.runners.embodied_runner import EmbodiedRunner

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_2"
            (checkpoint / "actor").mkdir(parents=True)
            for resume in (None, str(checkpoint)):
                with self.subTest(resume=resume), patch.object(EmbodiedRunner, "init_workers"):
                    events = Mock()
                    runner = SimpleNamespace(
                        env=events.env, rollout=events.rollout, reward=events.reward,
                        actor=events.actor, cfg=OmegaConf.create({"runner": {"resume_dir": resume}}),
                        logger=Mock(), global_step=0,
                    )
                    install_serial_worker_initialization()
                    EmbodiedRunner.init_workers(runner)
                    expected = []
                    for role in ("env", "rollout", "reward", "actor"):
                        expected.extend([getattr(call, role).init_worker(),
                                         getattr(call, role).init_worker().wait()])
                    if resume:
                        expected.extend([call.actor.load_checkpoint(str(checkpoint / "actor")),
                                         call.actor.load_checkpoint().wait()])
                    self.assertEqual(events.mock_calls, expected)
                    self.assertEqual(runner.global_step, 2 if resume else 0)

    def compose(self, overrides=()):
        searchpath = ROOT / "third_party/RLinf/examples/embodiment/config"
        with patch.dict(os.environ, LAB_ROOT=str(ROOT)), initialize_config_dir(
            version_base=None, config_dir=str(ROOT / "configs/rl/locomanip")
        ):
            cfg = compose(
                config_name="ppo-offload",
                overrides=[f"hydra.searchpath=[file://{searchpath}]", *overrides],
            )
            # 环境变量在使用配置时仍需可用；只解析环境路径，保留 train/eval 的动态插值。
            cfg.actor.model.model_path = cfg.actor.model.model_path
            cfg.runner.logger.log_path = cfg.runner.logger.log_path
            return cfg

    def test_physics_backend_default_follow_and_override(self):
        for overrides, expected_train, expected_eval in (
            ([], "isaacsim_physx", "isaacsim_physx"),
            (["env.train.physics_backend=newton_mjwarp"], "newton_mjwarp", "newton_mjwarp"),
            (["env.train.physics_backend=newton_mjwarp",
              "env.eval.physics_backend=isaacsim_physx"], "newton_mjwarp", "isaacsim_physx"),
            (["env.eval.physics_backend=newton_mjwarp"], "isaacsim_physx", "newton_mjwarp"),
        ):
            with self.subTest(overrides=overrides):
                cfg = self.compose(overrides)
                self.assertEqual(cfg.env.train.physics_backend, expected_train)
                self.assertEqual(cfg.env.eval.physics_backend, expected_eval)
                self.assertEqual(cfg.env.train.device, "cuda:0")
                self.assertEqual(cfg.env.eval.device, "cuda:0")

    def test_legacy_backend_selectors_removed_or_rejected(self):
        cfg = self.compose().env.train
        cfg.arena_args = ["--object", "brown_box", "--presets=physx",
                          "--physics-backend", "isaacsim_physx", "--presets", "physx"]
        self.assertEqual(arena_arguments(cfg), ["--object", "brown_box"])
        for arguments in (["--presets", "newton"],
                          ["--physics-backend=newton_mjwarp"], ["--presets"]):
            cfg.arena_args = arguments
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                arena_arguments(cfg)

    def test_eval_override_does_not_inherit_train_backend_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.compose(["env.eval.physics_backend=newton_mjwarp"])
            cfg.env.train.physics_log_dir = directory
            cfg.env.train.arena_args.extend(["--presets", "physx"])
            prepare_training_config(cfg)
            self.assertNotIn("--presets", cfg.env.train.arena_args)
            self.assertNotIn("--presets", cfg.env.eval.arena_args)
            self.assertEqual(cfg.env.eval.physics_backend, "newton_mjwarp")
            cfg.env.eval.arena_args = ["--presets", "physx"]
            with self.assertRaisesRegex(ValueError, "conflicts"):
                prepare_training_config(cfg)

    def test_newton_cpu_rejected_before_creating_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.compose(["env.train.physics_backend=newton_mjwarp", "env.train.device=cpu"])
            cfg.env.train.physics_log_dir = f"{directory}/physics"
            with self.assertRaisesRegex(ValueError, "CUDA"):
                prepare_training_config(cfg)
            self.assertFalse(Path(cfg.env.train.physics_log_dir).exists())

    def test_cross_backend_resume_records_source_and_current_without_rewriting_resume(self):
        for source, target in (("isaacsim_physx", "newton_mjwarp"),
                               ("newton_mjwarp", "isaacsim_physx")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                original = Path(directory) / "old"
                (original / "physics").mkdir(parents=True)
                (original / "physics/run.json").write_text(json.dumps({
                    "current_backends": {"train": source, "eval": source},
                }))
                checkpoint = str(original / "checkpoints/global_step_2")
                cfg = self.compose([f"env.train.physics_backend={target}"])
                cfg.runner.resume_dir = checkpoint
                cfg.env.train.physics_log_dir = str(Path(directory) / "new/physics")
                prepare_training_config(cfg)
                record = json.loads((Path(cfg.env.train.physics_log_dir) / "run.json").read_text())
                self.assertEqual(record["source_backends"], {"train": source, "eval": source})
                self.assertEqual(record["current_backends"], {"train": target, "eval": target})
                self.assertEqual(cfg.runner.resume_dir, checkpoint)
                self.assertEqual(record["source_checkpoint"], checkpoint)

    def test_resume_uses_original_config_and_allows_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "old"
            (original / "physics").mkdir(parents=True)
            checkpoint = str(original / "checkpoints/global_step_2")
            cfg = self.compose(["env.train.physics_backend=newton_mjwarp"])
            cfg.runner.resume_dir = checkpoint
            cfg.env.train.physics_log_dir = str(Path(directory) / "new/physics")
            for known in (False, True):
                if known:
                    OmegaConf.save(self.compose(), original / "physics/config.yaml", resolve=True)
                prepare_training_config(cfg)
                record = json.loads((Path(cfg.env.train.physics_log_dir) / "run.json").read_text())
                expected = "isaacsim_physx" if known else "unknown"
                self.assertEqual(record["source_backends"], {"train": expected, "eval": expected})

    def test_shared_tensorboard_config_requires_matching_source_run(self):
        for tensorboard in ("tensorboard", "tensorboard/all"):
            with self.subTest(tensorboard=tensorboard), tempfile.TemporaryDirectory() as directory:
                log_path = Path(directory)
                checkpoint = log_path / "source/checkpoints/global_step_2"
                config_path = log_path / tensorboard / "config.yaml"
                config_path.parent.mkdir(parents=True)
                for logger, expected in (
                    ({"log_path": directory, "experiment_name": "source"}, "newton_mjwarp"),
                    ({"log_path": directory, "experiment_name": "later-run"}, "unknown"),
                    ({"log_path": str(log_path / "other"), "experiment_name": "source"}, "unknown"),
                    ({}, "unknown"),
                ):
                    with self.subTest(logger=logger):
                        saved = OmegaConf.create({
                            "runner": {"logger": logger},
                            "env": {
                                "train": {"physics_backend": "newton_mjwarp"},
                                "eval": {"physics_backend": "newton_mjwarp"},
                            },
                        })
                        OmegaConf.save(saved, config_path)
                        self.assertEqual(_source_backends(checkpoint), {"train": expected, "eval": expected})

    def test_offload_batch_tracks_parallel_environment_count(self):
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
                    self.assertEqual(cfg.env.train.viser_envs, 0)


if __name__ == "__main__":
    unittest.main()
