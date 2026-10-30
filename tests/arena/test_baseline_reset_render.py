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


class BaselineResetRenderTest(unittest.TestCase):
    def test_eval_builds_with_reset_render_and_records_runtime_value(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            args = SimpleNamespace(output=output, episodes=1, num_envs=24,
                                   port=5555, seed=1000, model_path=Path(directory))
            parser = Mock()
            parser.parse_known_args.return_value = (args, [])
            parser.parse_args.return_value = args
            cfg = SimpleNamespace(num_rerenders_on_reset=0, recorders=SimpleNamespace())
            kwargs = {"render_mode": None}
            base = SimpleNamespace(
                max_episode_length=1500, step_dt=0.02, device="cpu",
                cfg=SimpleNamespace(sim=SimpleNamespace(physics=object())),
                scene={"robot_head_cam": object()}, episode_recorder=Mock(),
            )
            env = Mock(unwrapped=base)

            def make_registered(manager_cfg, env_kwargs):
                self.assertIs(manager_cfg, cfg)
                self.assertIs(env_kwargs, kwargs)
                self.assertEqual(manager_cfg.num_rerenders_on_reset, 4)
                # Protocol must read the built environment's actual configuration.
                base.cfg.num_rerenders_on_reset = 2
                return env

            builder = Mock()
            builder.arena_env = SimpleNamespace(episode_recorder_terms={})
            builder.compose_manager_cfg.return_value = (cfg, kwargs)
            builder.make_registered.side_effect = make_registered
            policy = Mock(action_chunk_length=50)

            def rollout(actual_env, actual_policy, **options):
                self.assertIs(actual_env, env)
                self.assertIs(actual_policy, policy)
                (output / "episodes.jsonl").write_text('{"success": true}\n')

            modules = {
                "isaaclab_arena.cli.isaaclab_arena_cli": SimpleNamespace(
                    get_isaaclab_arena_cli_parser=lambda: parser),
                "isaaclab_arena.utils.isaaclab_utils.simulation_app": SimpleNamespace(
                    SimulationAppContext=lambda _: nullcontext()),
                "torch": SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
                "isaaclab.managers.recorder_manager": SimpleNamespace(RecorderTermCfg=SimpleNamespace),
                "isaaclab_arena.evaluation.policy_runner": SimpleNamespace(rollout_policy=rollout),
                "isaaclab_arena.recording.episode_recorder_manager": SimpleNamespace(
                    EpisodeRecorderTermCfg=SimpleNamespace),
                "isaaclab_arena_environments.cli": SimpleNamespace(
                    get_arena_builder_from_cli=lambda _: builder,
                    get_isaaclab_arena_environments_cli_parser=lambda _: parser),
                "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy": SimpleNamespace(
                    Gr00tRemoteClosedloopPolicy=lambda _: policy,
                    Gr00tRemoteClosedloopPolicyCfg=SimpleNamespace),
                "embodied_ai_lab.environments.baseline_recorder": SimpleNamespace(
                    BaselineDiagnosticsRecorder=object, record_diagnostics=object()),
            }
            source = Path(__file__).resolve().parents[2] / "embodied_ai_lab/eval/baseline.py"
            spec = importlib.util.spec_from_file_location("baseline_under_test", source)
            baseline = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, modules), patch.dict("os.environ", {"EVAL_RL_SERVER": "0"}):
                spec.loader.exec_module(baseline)
                with patch.object(baseline.importlib.metadata, "version", return_value="fixture"):
                    baseline.main()
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["num_rerenders_on_reset"], 2)
            self.assertEqual(json.loads((output / "complete.json").read_text())["successes"], 1)
            env.close.assert_called_once()
            policy.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
