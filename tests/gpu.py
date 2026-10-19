"""Shared subprocess isolation and artifact paths for the GPU regression suite."""

import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models/GR00T-N1.6-G1/checkpoint-20000"
DATASET = ROOT / "data/locomanipulation/generated/lerobot"
REPORTS = ROOT / "logs/gpu-tests"


def artifact(name):
    """Resolve a required artifact path relative to LAB_ROOT."""
    return ROOT / Path(os.environ[f"LAB_GPU_{name}"])


def require(test, *paths):
    for path in paths:
        test.assertTrue(path.exists(), f"Required GPU test artifact is missing: {path}")


def run_worker(test, module, *args, runtime="reference", environment="rl", report=None):
    """Use a fresh process so GR00T variants and Isaac Sim never share imports."""
    env = os.environ.copy()
    groot_root = env[{
        "reference": "GROOT_REF_ROOT", "fork": "GROOT_RL_ROOT", "sft": "GROOT_ROOT",
    }[runtime]]
    env["PYTHONPATH"] = os.pathsep.join((
        groot_root, env["RLINF_ROOT"], str(ROOT), env.get("PYTHONPATH", ""),
    ))
    python = env[f"{environment.upper()}_PY"]
    if environment == "rl":
        purelib = subprocess.check_output(
            [python, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
            env=env, text=True,
        ).strip()
        runtime_lib = Path(purelib) / "nvidia/cuda_runtime/lib"
        env["LD_LIBRARY_PATH"] = str(runtime_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    REPORTS.mkdir(parents=True, exist_ok=True)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.unlink(missing_ok=True)
    subprocess.run(
        [python, "-m", module, *map(str, args)], env=env, cwd=env["ARENA_ROOT"], check=True,
    )
    if report is not None:
        test.assertTrue(report.is_file(), f"GPU worker exited without its report: {report}")
