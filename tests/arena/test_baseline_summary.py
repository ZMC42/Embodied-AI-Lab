"""Check statistical endpoints and mutually exclusive failure categories."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_ai_lab.eval import summary


class BaselineSummaryTest(unittest.TestCase):
    def test_diagnostics_and_host_memory_in_generic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "gpu"
            backend.mkdir()
            (backend / "protocol.json").write_text(json.dumps({
                "episodes": 3, "seed": 42, "max_episode_steps": 1500,
                "navigation_threshold_m": 0.5, "num_envs": 3,
                "action_chunk_length": 50, "checkpoint": "custom-checkpoint",
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
            samples = (
                "timestamp, memory.used, utilization.gpu\n"
                "now, 200 MiB, 0 %\nnow, 9000 MiB, 50 %\n"
                "later, 300 MiB, 10 %\nlater, 8000 MiB, 40 %\n"
            )
            (root / "gpu-gpu.csv").write_text(samples)
            result = summary.summarize_backend(backend)
            shutil.copytree(backend, root / "cpu")
            (root / "gpu-cpu.csv").write_text(samples)
            with patch("sys.argv", ["summary", str(root)]), contextlib.redirect_stdout(io.StringIO()):
                summary.main()
            combined = json.loads((root / "summary.json").read_text())
            report = (root / "report.md").read_text()
        self.assertEqual(result["falls"], 2)
        self.assertEqual(result["fall_rate"], 2 / 3)
        self.assertEqual(result["categories"], {"success": 1, "not_reached": 2})
        self.assertEqual(result["peak_host_single_gpu_memory_mib"], 9000)
        self.assertNotIn("peak_total_gpu_memory_mib", result)
        self.assertNotIn("official_checkpoint_meets_le_80_percent_rule", combined)
        self.assertEqual(combined["cpu"]["success_rate"], 1 / 3)
        self.assertEqual(combined["gpu"]["peak_host_single_gpu_memory_mib"], 9000)
        self.assertIn("其他进程", report)
        self.assertIn("9000", report)

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
