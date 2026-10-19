"""在同一 systemd 服务内运行命令，采样整卡显存并保存退出状态。"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    process = subprocess.Popen(args.command)
    peak = 0
    with (args.output / "gpu-memory.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "memory_used_mib"])
        while process.poll() is None:
            sample = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            used = sum(int(value) for value in sample.splitlines())
            peak = max(peak, used)
            writer.writerow([round(time.monotonic() - start, 3), used])
            f.flush()
            time.sleep(args.interval)
    report = dict(
        command=args.command,
        exit_code=process.returncode,
        elapsed_s=time.monotonic() - start,
        peak_gpu_memory_mib=peak,
        sampling_interval_s=args.interval,
    )
    (args.output / "process.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
