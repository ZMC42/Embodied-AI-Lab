"""仿真创建边界上的后端透传、PhysX 专用参数和运行记录。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from embodied_ai_lab.rlinf.env import make_sim


class PhysicsBackendTest(unittest.TestCase):
    def test_make_sim_selects_backend_and_only_physx_gets_gpu_capacities(self):
        class PhysxCfg:
            pass

        class NewtonCfg:
            __slots__ = ()

        for backend, physics, legacy in (("isaacsim_physx", PhysxCfg(), False),
                                         ("newton_mjwarp", NewtonCfg(), False),
                                         ("isaacsim_physx", PhysxCfg(), True)):
            with self.subTest(backend=backend, legacy=legacy), tempfile.TemporaryDirectory() as directory:
                cfg = OmegaConf.create(dict(
                    task="fake-task", device="cuda:0", physics_backend=backend,
                    physics_log_dir=directory, is_eval=True, viser_envs=0,
                    io_module="test_physics_converter", arena_args=["--object", "brown_box"],
                ))
                if legacy:
                    del cfg.physics_backend
                    del cfg.physics_log_dir
                    cfg = SimpleNamespace(**OmegaConf.to_container(cfg))
                env_cfg = SimpleNamespace(
                    sim=SimpleNamespace(physics=physics), scene=SimpleNamespace(),
                )
                spec = SimpleNamespace(kwargs={"env_cfg_entry_point": env_cfg})
                environment = Mock()
                registration_arguments = []
                register = lambda: registration_arguments.extend(sys.argv)
                launcher = Mock(return_value=SimpleNamespace(app="sim-app"))
                modules = {
                    "isaaclab.app": SimpleNamespace(AppLauncher=launcher),
                    "isaaclab_physx.physics": SimpleNamespace(PhysxCfg=PhysxCfg),
                    "isaaclab_arena.environments.isaaclab_interop": SimpleNamespace(
                        environment_registration_callback=register,
                    ),
                    "embodied_ai_lab.rlinf.deferred_reset": SimpleNamespace(
                        DeferredResetArenaEnv=object,
                    ),
                    "test_physics_converter": SimpleNamespace(wrap_obs=Mock()),
                }
                original_arguments = sys.argv
                with patch.dict(sys.modules, modules), \
                     patch("embodied_ai_lab.rlinf.env.gym.spec", return_value=spec), \
                     patch("embodied_ai_lab.rlinf.env.gym.make", return_value=SimpleNamespace(unwrapped=environment)), \
                     patch("embodied_ai_lab.rlinf.chunk_env.ChunkEnv", return_value="chunk-env"), \
                     patch("isaaclab_arena.utils.physics_backend.physics_protocol", return_value={
                         "physics": type(physics).__name__, "physics_backend": backend,
                     }) as protocol:
                    self.assertEqual(make_sim(cfg, 2, 42, worker_index=3), ("chunk-env", "sim-app"))
                self.assertIs(sys.argv, original_arguments)
                self.assertEqual(registration_arguments.count("--physics-backend"), 1)
                self.assertEqual(registration_arguments[-1], backend)
                self.assertEqual(env_cfg.sim.device, "cuda:0")
                self.assertEqual(env_cfg.scene.num_envs, 2)
                if backend == "isaacsim_physx":
                    self.assertEqual(physics.gpu_found_lost_pairs_capacity, 2**16)
                    self.assertEqual(physics.gpu_found_lost_aggregate_pairs_capacity, 2**16)
                    self.assertEqual(physics.gpu_total_aggregate_pairs_capacity, 2**16)
                paths = list(Path(directory).glob("eval-worker_3-pid_*.json"))
                if legacy:
                    protocol.assert_not_called()
                    self.assertEqual(paths, [])
                    continue
                protocol.assert_called_once_with(environment)
                self.assertEqual(len(paths), 1)
                record = json.loads(paths[0].read_text())
                self.assertEqual(record["physics"], type(physics).__name__)
                self.assertEqual(record["requested_physics_backend"], backend)
                self.assertEqual(record["seed"], 42)

    def test_bad_device_fails_before_starting_application(self):
        cfg = OmegaConf.create(dict(
            physics_backend="newton_mjwarp", device="cpu", arena_args=[],
        ))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            make_sim(cfg, 2, 42)


if __name__ == "__main__":
    unittest.main()
