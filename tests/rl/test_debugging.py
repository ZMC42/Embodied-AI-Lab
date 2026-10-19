"""RL 调试端口与进程选择契约；不启动 Ray worker 或 Isaac Sim。"""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from embodied_ai_lab.rlinf import debugging
from embodied_ai_lab.rlinf import env as arena_env


class DebuggingTest(unittest.TestCase):
    def test_disabled_debugging_does_not_import_debugpy_or_scheduler(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(
            sys.modules, {"debugpy": None, "rlinf.scheduler": None}
        ):
            for role in debugging.DEBUG_PORTS:
                debugging.wait_for_debugger(role)
            debugging.wait_for_model_debugger()

    def test_selected_role_waits_on_loopback_with_child_attachment_disabled(self):
        for role, port in debugging.DEBUG_PORTS.items():
            with self.subTest(role=role):
                debugpy = Mock()
                with patch.dict(os.environ, RL_DEBUG=role), patch.dict(
                    sys.modules, {"debugpy": debugpy}
                ), patch("builtins.print"):
                    debugging.wait_for_debugger(role)
                self.assertEqual(debugpy.mock_calls, [
                    call.configure(subProcess=False),
                    call.listen(("127.0.0.1", port)),
                    call.wait_for_client(),
                ])

    def test_other_roles_and_nonzero_ranks_do_not_import_debugpy(self):
        with patch.dict(os.environ, RL_DEBUG="actor"), patch.dict(
            sys.modules, {"debugpy": None}
        ):
            debugging.wait_for_debugger("rollout")
            debugging.wait_for_debugger("driver")
            debugging.wait_for_debugger("env")
            debugging.wait_for_debugger("actor", rank=1)

    def test_invalid_debug_role_fails_before_listening(self):
        with patch.dict(os.environ, RL_DEBUG="all"), patch.dict(
            sys.modules, {"debugpy": None}
        ):
            with self.assertRaisesRegex(ValueError, "RL_DEBUG"):
                debugging.wait_for_debugger("driver")

    def test_model_debugging_selects_actual_worker_group_and_rank(self):
        cfg = {
            "actor": SimpleNamespace(group_name="custom-trainer"),
            "rollout": SimpleNamespace(group_name="custom-sampler"),
        }
        for target in ("actor", "rollout"):
            for actual in ("actor", "rollout"):
                for rank in (0, 1):
                    with self.subTest(target=target, actual=actual, rank=rank):
                        worker = SimpleNamespace(
                            cfg=cfg, _rank=rank,
                            worker_address=SimpleNamespace(
                                root_group_name=cfg[actual].group_name
                            ),
                        )
                        scheduler = SimpleNamespace(
                            Worker=SimpleNamespace(current_worker=worker)
                        )
                        debugpy = Mock()
                        with patch.dict(os.environ, RL_DEBUG=target), patch.dict(
                            sys.modules,
                            {"rlinf.scheduler": scheduler, "debugpy": debugpy},
                        ), patch("builtins.print"):
                            debugging.wait_for_model_debugger()
                        if target == actual and rank == 0:
                            debugpy.listen.assert_called_once_with(
                                ("127.0.0.1", debugging.DEBUG_PORTS[target])
                            )
                            debugpy.wait_for_client.assert_called_once_with()
                        else:
                            self.assertEqual(debugpy.mock_calls, [])

    def test_simulator_debug_hook_runs_before_app_launcher(self):
        class DebugReached(Exception):
            pass

        launcher = Mock()
        with patch.dict(sys.modules, {
            "isaaclab.app": SimpleNamespace(AppLauncher=launcher),
        }), patch.object(
            debugging, "wait_for_debugger", side_effect=DebugReached
        ) as wait:
            with self.assertRaises(DebugReached):
                arena_env.make_sim(
                    SimpleNamespace(viser_envs=0, device="cpu"), 1, 42, debug=True
                )
        wait.assert_called_once_with("env")
        launcher.assert_not_called()

    def test_only_first_training_simulator_is_eligible_to_wait(self):
        for is_eval, offset, expected in (
            (False, 0, True), (False, 1, False),
            (True, 0, False), (True, 1, False),
        ):
            with self.subTest(is_eval=is_eval, seed_offset=offset):
                cfg = SimpleNamespace(seed=42, is_eval=is_eval)
                with patch.object(arena_env, "SubProcIsaacLabEnv") as subprocess:
                    arena_env.ArenaEnv(cfg, 2, offset, 2, None)
                factory = subprocess.call_args.args[0]
                self.assertIs(factory.func, arena_env.make_sim)
                self.assertEqual(factory.keywords, {"debug": expected})

    def test_vscode_attach_ports_match_debug_listeners(self):
        root = Path(__file__).resolve().parents[2]
        launch = json.loads((root / ".vscode/launch.json").read_text())
        configurations = {cfg["name"]: cfg for cfg in launch["configurations"]}
        ports = []
        for role, port in debugging.DEBUG_PORTS.items():
            config = configurations[f"RLinf PPO {role} (attach)"]
            self.assertEqual(config["request"], "attach")
            self.assertEqual(config["connect"], {"host": "127.0.0.1", "port": port})
            self.assertFalse(config["subProcess"])
            ports.append(port)
        sft_port = configurations["GR00T N1.6 SFT training (attach)"]["connect"]["port"]
        self.assertEqual(len(set([sft_port, *ports])), 1 + len(ports))


if __name__ == "__main__":
    unittest.main()
