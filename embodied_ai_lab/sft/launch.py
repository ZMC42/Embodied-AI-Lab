"""Launch a recipe with an effective batch shared across available GPUs."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml


def build_command(recipe, *, lab_root, python, num_gpus, per_device_batch=None,
                  extra_args=(), debug=False):
    """Translate a recipe into torchrun arguments; keep CLI overrides last."""
    batch = recipe["per_device_batch"] if per_device_batch is None else per_device_batch
    effective_batch = recipe["effective_batch"]
    if num_gpus < 1 or batch < 1 or effective_batch < 1:
        raise ValueError("GPU count and batch sizes must be positive")
    global_batch = num_gpus * batch
    accum, remainder = divmod(effective_batch, global_batch)
    if remainder:
        raise ValueError(
            f"effective_batch {effective_batch} must be divisible by "
            f"num_gpus × per_device_batch ({num_gpus} × {batch})"
        )

    command = [
        str(python), "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={num_gpus}", "--no-python", str(python),
    ]
    if debug:
        command += [
            "-m", "debugpy", "--listen", "127.0.0.1:5678", "--wait-for-client",
            "--configure-subProcess", "false",
        ]
    command.append("gr00t/experiment/launch_finetune.py")
    for name, value in recipe["args"].items():
        command.append(f"--{name}")
        if isinstance(value, bool):
            if not value:
                command[-1] = f"--no-{name}"
        elif isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            if name.endswith(("-path", "-dir")):
                value = Path(lab_root) / value
            command.append(str(value))
    command += [
        "--num-gpus", str(num_gpus), "--global-batch-size", str(global_batch),
        "--gradient-accumulation-steps", str(accum),
    ]
    return command + list(extra_args)


def output_directory(command, cwd):
    """Find the final output-dir, including an upstream CLI override."""
    for index, argument in enumerate(command):
        if argument == "--output-dir":
            output = command[index + 1]
        elif argument.startswith("--output-dir="):
            output = argument.split("=", 1)[1]
    return Path(cwd) / output


def run(command, *, cwd, log_path):
    """Tee combined output and mark successful completion in the actual output."""
    output = output_directory(command, cwd)
    output.mkdir(parents=True, exist_ok=True)
    done = output / "DONE"
    done.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            previous_handlers = {}
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(
                    signum, lambda received, frame: process.send_signal(received)
                )
            try:
                for line in process.stdout:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
    if returncode == 0:
        done.touch()
    return returncode if returncode >= 0 else 128 - returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    extra_args = argv[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", nargs="?", default="smoke")
    parser.add_argument("--per-device-batch", type=int)
    args = parser.parse_args(argv[:separator])

    import torch

    lab_root = Path(os.environ["LAB_ROOT"])
    groot_root = Path(os.environ["GROOT_ROOT"])
    recipe_path = lab_root / "configs/sft" / f"{args.recipe}.yaml"
    with recipe_path.open() as source:
        recipe = yaml.safe_load(source)
    try:
        command = build_command(
            recipe, lab_root=lab_root, python=sys.executable,
            num_gpus=torch.cuda.device_count(), per_device_batch=args.per_device_batch,
            extra_args=extra_args, debug=os.environ.get("TRAIN_DEBUG") == "1",
        )
    except ValueError as error:
        parser.error(str(error))
    return run(command, cwd=groot_root,
               log_path=lab_root / "logs/train-sft" / f"{args.recipe}.log")


if __name__ == "__main__":
    sys.exit(main())
