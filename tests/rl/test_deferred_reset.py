"""Run the upstream step/reset control flow without launching Isaac Sim."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from embodied_ai_lab.rlinf.chunk_env import ChunkEnv


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "third_party/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab/isaaclab/envs"


def upstream_methods(path, name, methods, base):
    """Keep executable upstream methods; replace only simulation construction."""
    source = ast.parse(path.read_text())
    node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
    node.bases = [ast.Name(id="base", ctx=ast.Load())]
    node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    scope = {"base": base, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


class SimulatorCore:
    def _reset_idx(self, env_ids):
        self.actual_resets.append(env_ids.tolist())


Base = upstream_methods(LAB / "manager_based_env.py", "ManagerBasedEnv", {"reset"}, SimulatorCore)
RL = upstream_methods(LAB / "manager_based_rl_env.py", "ManagerBasedRLEnv", {"step"}, Base)
Arena = upstream_methods(
    ROOT / "third_party/IsaacLab-Arena/isaaclab_arena/environments/isaaclab_arena_manager_based_env.py",
    "IsaacLabArenaManagerBasedRLEnv", {"step"}, RL,
)


class DeferredResetRenderTest(unittest.TestCase):
    def setUp(self):
        module_path = ROOT / "embodied_ai_lab/rlinf/deferred_reset.py"
        spec = importlib.util.spec_from_file_location("deferred_reset_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "isaaclab_arena.environments.isaaclab_arena_manager_based_env":
                SimpleNamespace(IsaacLabArenaManagerBasedRLEnv=Arena),
        }):
            spec.loader.exec_module(module)
        self.env = env = module.DeferredResetArenaEnv()
        env.device, env.num_envs = "cpu", 2
        env.cfg = SimpleNamespace(
            decimation=1, sim=SimpleNamespace(render_interval=1),
            num_rerenders_on_reset=2, compute_final_obs=False, wait_for_textures=False,
        )
        env.sim = Mock(is_rendering=True)
        env.sim.consume_reset_request.return_value = False
        env.scene = Mock()
        env.action_manager = Mock()
        env.recorder_manager = Mock(active_terms=[])
        env.observation_manager = Mock()
        env.observation_manager.compute.side_effect = lambda **_: {
            "states": torch.full((2, 1), float(env.common_step_counter))}
        env.reward_manager = Mock()
        env.reward_manager.compute.return_value = torch.zeros(2)
        env.termination_manager = Mock()

        def termination():
            env.termination_manager.terminated = torch.tensor([env.common_step_counter >= 2, False])
            env.termination_manager.time_outs = torch.zeros(2, dtype=torch.bool)
            return env.termination_manager.terminated.clone()

        env.termination_manager.compute.side_effect = termination
        env.command_manager = Mock()
        env.event_manager = Mock(active_terms={})
        env.video_recorders = []
        env.render_enabled = env.has_rtx_sensors = True
        env._physics_handles_decimation = False
        env._sim_step_counter = env.common_step_counter = 0
        env.physics_dt = env.step_dt = 0.02
        env.episode_length_buf = torch.zeros(2, dtype=torch.long)
        env._active_episode_mask = torch.ones(2, dtype=torch.bool)
        env._reset_env_ids = torch.empty(0, dtype=torch.long)
        env.extras = {}
        env.actual_resets = []
        env.episode_signals = lambda: {
            "success_once": env.termination_manager.terminated,
            "fall_once": torch.zeros(2, dtype=torch.bool),
        }

    def test_chunk_renders_only_on_physics_steps_and_real_reset(self):
        env = self.env
        chunk = ChunkEnv(env, lambda obs: obs)
        with patch("torch.cuda.empty_cache"):
            _, _, terminated, _, infos = chunk.step(torch.zeros(2, 5, 1))
        self.assertEqual(env.actual_resets, [[0]])
        # Five normal step renders, then two configured renders for the real reset.
        calls = env.sim.render.call_args_list
        self.assertEqual(sum(bool(call.kwargs) for call in calls), 5)
        self.assertEqual(sum(not call.kwargs for call in calls), 2)
        self.assertEqual(env.cfg.num_rerenders_on_reset, 2)
        self.assertEqual(terminated[0].tolist(), [False] * 4 + [True])
        self.assertEqual(infos[0]["final_observation"]["states"][0].item(), 2)

    def test_non_deferred_step_keeps_upstream_reset_render(self):
        self.env.common_step_counter = 1
        self.env.step(torch.zeros(2, 1))
        self.assertEqual(self.env.actual_resets, [[0]])
        self.assertEqual(self.env.sim.render.call_count, 3)
        self.assertEqual(self.env.cfg.num_rerenders_on_reset, 2)

    def test_failed_deferred_step_restores_reset_render_setting(self):
        self.env.defer_resets = True
        self.env.action_manager.process_action.side_effect = RuntimeError("step failed")
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            self.env.step(torch.zeros(2, 1))
        self.assertEqual(self.env.cfg.num_rerenders_on_reset, 2)


if __name__ == "__main__":
    unittest.main()
