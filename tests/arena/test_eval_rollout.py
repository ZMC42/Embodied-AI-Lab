"""Exercise the evaluation construction/protocol seam without Isaac Sim."""

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class EvalRolloutTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "evaluation"
        # The parent runner creates the attempt directory before starting rollout.
        self.output.mkdir()
        self.args = SimpleNamespace(output=self.output, episodes=2, num_envs=24,
                                    port=5555, seed=1000, model_path=Path(directory.name))
        parser = Mock()
        parser.parse_known_args.return_value = (self.args, [])
        parser.parse_args.return_value = self.args
        cfg = SimpleNamespace(num_rerenders_on_reset=0, recorders=SimpleNamespace())
        kwargs = {"render_mode": None}
        self.base = SimpleNamespace(
            max_episode_length=1500, step_dt=0.02, device="cpu",
            cfg=SimpleNamespace(sim=SimpleNamespace(physics=object())),
            scene={"robot_head_cam": object()}, episode_recorder=Mock(),
        )
        self.env = Mock(unwrapped=self.base)

        def make_registered(manager_cfg, env_kwargs):
            self.assertIs(manager_cfg, cfg)
            self.assertIs(env_kwargs, kwargs)
            self.assertEqual(manager_cfg.num_rerenders_on_reset, 0)
            # Read the built environment's actual configuration, not the input cfg.
            self.base.cfg.num_rerenders_on_reset = 2
            return self.env

        builder = Mock()
        builder.compose_manager_cfg.return_value = (cfg, kwargs)
        builder.make_registered.side_effect = make_registered
        self.policy = Mock(action_chunk_length=50)
        self.make_policy = Mock(return_value=self.policy)

        def rollout(actual_env, actual_policy, **options):
            self.assertIs(actual_env, self.env)
            self.assertIs(actual_policy, self.policy)
            self.assertEqual(options, {"num_steps": None, "num_episodes": 2})
            (self.output / "episodes.jsonl").write_text(
                '{"success": true}\n{"success": false}\n')

        self.rollout = Mock(side_effect=rollout)
        modules = {
            "isaaclab_arena.cli.isaaclab_arena_cli": SimpleNamespace(
                get_isaaclab_arena_cli_parser=lambda: parser),
            "isaaclab_arena.utils.isaaclab_utils.simulation_app": SimpleNamespace(
                SimulationAppContext=lambda _: nullcontext()),
            "isaaclab_arena.evaluation.policy_runner": SimpleNamespace(rollout_policy=self.rollout),
            "isaaclab_arena_environments.cli": SimpleNamespace(
                get_arena_builder_from_cli=lambda _: builder,
                get_isaaclab_arena_environments_cli_parser=lambda _: parser),
            "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy": SimpleNamespace(
                Gr00tRemoteClosedloopPolicy=self.make_policy,
                Gr00tRemoteClosedloopPolicyCfg=SimpleNamespace),
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        source = Path(__file__).resolve().parents[2] / "embodied_ai_lab/eval/rollout.py"
        spec = importlib.util.spec_from_file_location("rollout_under_test", source)
        self.implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.implementation)
        version_patch = patch.object(self.implementation.importlib.metadata, "version", return_value="fixture")
        version_patch.start()
        self.addCleanup(version_patch.stop)
        env_patch = patch.dict("os.environ", {"EVAL_RL_SERVER": "0"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_records_native_successes_and_actual_reset_render(self):
        self.implementation.main()
        protocol = json.loads((self.output / "protocol.json").read_text())
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual(protocol["num_rerenders_on_reset"], 2)
        self.assertEqual(protocol["seed"], 1000)
        self.assertEqual(protocol["checkpoint"], str(self.args.model_path.resolve()))
        self.assertEqual(protocol["inference_runtime"], "official-flash-attention")
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["successes"], 1)
        self.assertGreaterEqual(result["elapsed_s"], 0)
        self.assertFalse((self.output / "complete.json").exists())
        self.env.close.assert_called_once()
        self.policy.close.assert_called_once()

    def test_records_rl_inference_shape(self):
        with patch.dict("os.environ", {"EVAL_RL_SERVER": "1"}):
            self.implementation.main()
        protocol = json.loads((self.output / "protocol.json").read_text())
        self.assertEqual(protocol["inference_runtime"], "rl-reference-sdpa")
        self.assertEqual(protocol["inference_micro_batch_size"], 1)

    def test_rollout_failure_closes_resources_without_result(self):
        self.rollout.side_effect = RuntimeError("rollout failed")
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            self.implementation.main()
        self.assertFalse((self.output / "result.json").exists())
        self.env.close.assert_called_once()
        self.policy.close.assert_called_once()

    def test_policy_construction_failure_closes_environment(self):
        self.make_policy.side_effect = RuntimeError("policy failed")
        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            self.implementation.main()
        self.assertFalse((self.output / "result.json").exists())
        self.env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
