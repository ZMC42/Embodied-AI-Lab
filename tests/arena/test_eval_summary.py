"""Check batch statistics and which attempts count as complete evaluations."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_ai_lab.eval import summary


class EvalSummaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = {
            "checkpoints": ["/models/run-a/checkpoint-1000"],
            "seeds": [42, 43],
            "episodes": 10,
            "num_envs": 5,
            "physics": "gpu",
            "inference_runtimes": ["official-flash-attention"],
        }
        self.write_config()

    def write_config(self):
        (self.root / "run.json").write_text(json.dumps(self.config))

    def attempt(self, checkpoint, seed, number, successes=None):
        directory = self.root / f"ckpt-{checkpoint:03d}" / f"seed-{seed}" / f"attempt-{number:04d}"
        directory.mkdir(parents=True)
        if successes is not None:
            (directory / "complete.json").write_text(json.dumps({
                "episodes": 10, "successes": successes, "elapsed_s": 12,
            }))
        return directory

    def test_mean_and_sample_std_use_seed_success_rates(self):
        self.attempt(0, 42, 1, 2)
        self.attempt(0, 43, 1, 8)
        result = summary.summarize_run(self.root)
        checkpoint = result["checkpoints"][0]
        self.assertEqual([job["success_rate"] for job in result["jobs"]], [0.2, 0.8])
        self.assertEqual(checkpoint["mean_success_rate"], 0.5)
        self.assertAlmostEqual(checkpoint["sample_std_success_rate"], math.sqrt(0.18))
        self.assertEqual(checkpoint["completed_seeds"], 2)
        self.assertEqual(checkpoint["status"], "completed")

    def test_single_seed_has_no_sample_std(self):
        self.config["seeds"] = [42]
        self.write_config()
        self.attempt(0, 42, 1, 4)
        checkpoint = summary.summarize_run(self.root)["checkpoints"][0]
        self.assertEqual(checkpoint["mean_success_rate"], 0.4)
        self.assertIsNone(checkpoint["sample_std_success_rate"])

    def test_failed_and_pending_seeds_prevent_checkpoint_aggregate(self):
        self.config["seeds"].append(44)
        self.write_config()
        self.attempt(0, 42, 1, 4)
        failed = self.attempt(0, 43, 1)
        # A worker result without the parent's completion marker is insufficient.
        (failed / "result.json").write_text('{"episodes": 10, "successes": 10, "elapsed_s": 1}')
        result = summary.summarize_run(self.root)
        self.assertEqual([job["status"] for job in result["jobs"]], ["completed", "failed", "pending"])
        self.assertIsNone(result["jobs"][1]["success_rate"])
        self.assertEqual(result["jobs"][1]["log_paths"]["eval"], str(failed / "eval.log"))
        checkpoint = result["checkpoints"][0]
        self.assertEqual(checkpoint["status"], "incomplete")
        self.assertEqual(checkpoint["completed_seeds"], 1)
        self.assertEqual(checkpoint["total_seeds"], 3)
        self.assertIsNone(checkpoint["mean_success_rate"])
        self.assertIsNone(checkpoint["sample_std_success_rate"])

    def test_checkpoint_indices_keep_same_basename_separate(self):
        self.config["checkpoints"].append("/models/run-b/checkpoint-1000")
        self.config["inference_runtimes"].append("rl-reference-sdpa")
        self.config["seeds"] = [42]
        self.write_config()
        self.attempt(0, 42, 1, 0)
        self.attempt(1, 42, 1, 10)
        result = summary.summarize_run(self.root)
        self.assertEqual([checkpoint["mean_success_rate"] for checkpoint in result["checkpoints"]], [0, 1])
        self.assertEqual([job["checkpoint"] for job in result["jobs"]], self.config["checkpoints"])

    def test_only_latest_complete_attempt_counts(self):
        self.attempt(0, 42, 1)
        self.attempt(0, 42, 2, 3)
        latest_complete = self.attempt(0, 42, 3, 7)
        self.attempt(0, 42, 4)
        job = summary.summarize_run(self.root)["jobs"][0]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["successes"], 7)
        self.assertEqual(job["attempt_dir"], str(latest_complete))

    def test_cli_writes_json_and_prints_checkpoint_mapping(self):
        self.attempt(0, 42, 1, 10)
        failed = self.attempt(0, 43, 1)
        output = io.StringIO()
        with patch("sys.argv", ["summary", str(self.root)]), contextlib.redirect_stdout(output):
            summary.main()
        result = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(result["config"], self.config)
        self.assertIn(self.config["checkpoints"][0], output.getvalue())
        self.assertIn(str(failed), output.getvalue())
        self.assertIn("incomplete", output.getvalue())
        self.assertFalse((self.root / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
