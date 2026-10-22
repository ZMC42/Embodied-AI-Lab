"""Reject unsupported baseline combinations before starting inference or simulation."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BackendLaunchTest(unittest.TestCase):
    def test_newton_cpu_is_rejected_before_creating_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            result = subprocess.run(
                [str(ROOT / "scripts/eval/baseline.sh"), str(output),
                 "--physics-backend", "newton_mjwarp", "cpu", "10", "2",
                 str(Path(directory) / "missing-model" / "checkpoint"), "42"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("newton_mjwarp requires gpu", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
