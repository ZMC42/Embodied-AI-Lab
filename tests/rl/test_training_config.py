"""Compose the training recipe with RLinf's registered OmegaConf resolvers."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

import rlinf  # noqa: F401 -- registers the upstream multiply/int_div resolvers


ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigTest(unittest.TestCase):
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
