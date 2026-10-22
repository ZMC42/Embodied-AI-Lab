"""Check statistical endpoints and mutually exclusive failure categories."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from embodied_ai_lab.eval import compare_checkpoints, summary


class BaselineSummaryTest(unittest.TestCase):
    def write_evaluation(self, root, name, engine):
        model = root / name
        directory = model / "gpu"
        directory.mkdir(parents=True)
        protocol = {
            "episodes": 1, "seed": 42, "max_episode_steps": 1500,
            "navigation_threshold_m": 0.5,
        }
        if engine is not None:
            protocol["physics_backend"] = engine
        (directory / "protocol.json").write_text(json.dumps(protocol))
        (directory / "complete.json").write_text(json.dumps({
            "episodes": 1, "successes": 1, "elapsed_s": 10,
        }))
        (directory / "episodes.jsonl").write_text(json.dumps({
            "env_id": 0, "episode_in_env": 0, "seed": 42,
            "episode_length": 100, "success": True, "fall_once": False,
            "termination_terms": {"object_dropped": False}, "progress": {"events": []},
            "min_base_box_xy_distance_m": 1,
        }) + "\n")
        (model / "gpu-gpu.csv").write_text("timestamp, memory.used, utilization.gpu\nnow, 200 MiB, 0 %\n")

    def test_compare_checkpoints_records_engine_and_accepts_legacy_physx(self):
        for sft_engine, rl_engine, expected in (
            (None, None, "isaacsim_physx"),
            (None, "isaacsim_physx", "isaacsim_physx"),
            ("newton_mjwarp", "newton_mjwarp", "newton_mjwarp"),
        ):
            with self.subTest(sft=sft_engine, rl=rl_engine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_evaluation(root, "sft", sft_engine)
                self.write_evaluation(root, "rl", rl_engine)
                with patch("sys.argv", ["compare", str(root), "gpu"]), redirect_stdout(io.StringIO()):
                    compare_checkpoints.main()
                report = (root / "report.md").read_text()
                self.assertIn(f"| {expected} / GPU | SFT |", report)
                self.assertIn(f"| {expected} / GPU | RL |", report)
                comparison = json.loads((root / "comparison.json").read_text())
                self.assertEqual(comparison["gpu"]["physics_backend"], expected)

    def test_compare_checkpoints_rejects_mismatched_actual_physics_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_evaluation(root, "sft", "isaacsim_physx")
            self.write_evaluation(root, "rl", "newton_mjwarp")
            with patch("sys.argv", ["compare", str(root), "gpu"]), \
                 self.assertRaisesRegex(AssertionError, "physics backend mismatch"):
                compare_checkpoints.main()
            self.assertFalse((root / "report.md").exists())
            self.assertFalse((root / "comparison.json").exists())

    def test_falls_are_counted_independently_of_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "gpu"
            backend.mkdir()
            (backend / "protocol.json").write_text(json.dumps({
                "episodes": 3, "seed": 42, "max_episode_steps": 1500,
                "navigation_threshold_m": 0.5,
            }))
            (backend / "complete.json").write_text(json.dumps({
                "episodes": 3, "successes": 1, "elapsed_s": 10,
            }))
            records = [{
                "env_id": env_id, "episode_in_env": 0, "seed": 42,
                "episode_length": 100, "success": env_id == 0,
                "fall_once": env_id != 2,
                "termination_terms": {"object_dropped": False},
                "progress": {"events": []}, "min_base_box_xy_distance_m": 1,
            } for env_id in range(3)]
            (backend / "episodes.jsonl").write_text("\n".join(map(json.dumps, records)))
            (root / "gpu-gpu.csv").write_text("timestamp, memory.used, utilization.gpu\nnow, 200 MiB, 0 %\n")
            result = summary.summarize_backend(backend)
        self.assertEqual(result["falls"], 2)
        self.assertEqual(result["fall_rate"], 2 / 3)
        self.assertEqual(result["categories"], {"success": 1, "not_reached": 2})

    def test_wilson_known_endpoints(self):
        self.assertAlmostEqual(summary.wilson_interval(0, 100)[1], 0.0369935, places=7)
        self.assertAlmostEqual(summary.wilson_interval(100, 100)[0], 0.9630065, places=7)
        lo, hi = summary.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 1 - hi)
        self.assertAlmostEqual(lo, 0.4038315, places=7)

    def test_failure_progress_and_priority(self):
        record = {
            "success": False,
            "termination_terms": {"object_dropped": False},
            "progress": {"events": []},
            "min_base_box_xy_distance_m": 0.6,
        }
        self.assertEqual(summary.failure_category(record), "not_reached")
        record["min_base_box_xy_distance_m"] = 0.4
        self.assertEqual(summary.failure_category(record), "not_lifted")
        record["progress"]["events"].append({"predicate_index": 1, "criteria_name": "pick_and_place"})
        self.assertEqual(summary.failure_category(record), "not_placed")
        record["termination_terms"]["object_dropped"] = True
        self.assertEqual(summary.failure_category(record), "dropped_box")
        record["success"] = True
        self.assertEqual(summary.failure_category(record), "success")


if __name__ == "__main__":
    unittest.main()
