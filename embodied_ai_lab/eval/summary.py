"""Summarize checkpoint × seed evaluations without comparing checkpoints."""

import argparse
import json
import statistics
from pathlib import Path


def completed_attempt(job_dir):
    """Return the latest successful attempt; partial attempts never count."""
    attempts = sorted(job_dir.glob("attempt-*"), key=lambda path: int(path.name.split("-")[1]))
    return next((path for path in reversed(attempts) if (path / "complete.json").is_file()), None)


def summarize_run(root):
    root = Path(root).resolve()
    config = json.loads((root / "run.json").read_text())
    jobs = []
    checkpoints = []
    for index, checkpoint in enumerate(config["checkpoints"]):
        checkpoint_jobs = []
        for seed in config["seeds"]:
            job_dir = root / f"ckpt-{index:03d}" / f"seed-{seed}"
            complete = completed_attempt(job_dir)
            attempts = sorted(job_dir.glob("attempt-*"), key=lambda path: int(path.name.split("-")[1]))
            attempt = complete or (attempts[-1] if attempts else None)
            result = json.loads((complete / "complete.json").read_text()) if complete else None
            job = {
                "checkpoint_index": index,
                "checkpoint": checkpoint,
                "seed": seed,
                "status": "completed" if complete else "failed" if attempt else "pending",
                "attempt_dir": str(attempt) if attempt else None,
                "log_paths": {name: str(attempt / f"{name}.log") for name in ("server", "eval", "kit", "error")} if attempt else {},
                "episodes": result["episodes"] if result else None,
                "successes": result["successes"] if result else None,
                "success_rate": result["successes"] / result["episodes"] if result else None,
                "elapsed_s": result["elapsed_s"] if result else None,
            }
            jobs.append(job)
            checkpoint_jobs.append(job)
        rates = [job["success_rate"] for job in checkpoint_jobs if job["status"] == "completed"]
        complete = len(rates) == len(config["seeds"])
        checkpoints.append({
            "checkpoint_index": index,
            "checkpoint": checkpoint,
            "status": "completed" if complete else "incomplete",
            "completed_seeds": len(rates),
            "total_seeds": len(config["seeds"]),
            "mean_success_rate": statistics.mean(rates) if complete else None,
            "sample_std_success_rate": statistics.stdev(rates) if complete and len(rates) > 1 else None,
        })
    return {"config": config, "jobs": jobs, "checkpoints": checkpoints}


def write_summary(root):
    result = summarize_run(root)
    (Path(root) / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("CHECKPOINT  SEED  STATUS     SUCCESSES  SUCCESS RATE  ELAPSED (s)")
    for job in result["jobs"]:
        if job["status"] == "completed":
            metrics = f"{job['successes']}/{job['episodes']}  {job['success_rate']:.1%}  {job['elapsed_s']:.1f}"
        else:
            metrics = "-  -  -"
        print(f"ckpt-{job['checkpoint_index']:03d}  {job['seed']}  {job['status']}  {metrics}")
        if job["status"] == "failed":
            print(f"  Logs: {job['attempt_dir']}")
    print("\nCHECKPOINT  SEEDS  STATUS      MEAN SUCCESS RATE  SAMPLE STD")
    for checkpoint in result["checkpoints"]:
        mean = checkpoint["mean_success_rate"]
        std = checkpoint["sample_std_success_rate"]
        metrics = f"{mean:.1%}" if mean is not None else "-"
        metrics += f"  {std:.1%}" if std is not None else "  -"
        print(f"ckpt-{checkpoint['checkpoint_index']:03d}  "
              f"{checkpoint['completed_seeds']}/{checkpoint['total_seeds']}  "
              f"{checkpoint['status']}  {metrics}  {checkpoint['checkpoint']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    write_summary(args.run_dir)


if __name__ == "__main__":
    main()
