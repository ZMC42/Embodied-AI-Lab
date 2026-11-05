"""Evaluate checkpoint × seed combinations with retries and run-level resume."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from embodied_ai_lab.eval.summary import completed_attempt, write_summary


PORT = 5555


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--physics", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.num_envs < 1 or args.retries < 0:
        parser.error("episodes and num-envs must be positive; retries must be nonnegative")
    args.checkpoints = [path.resolve() for path in args.checkpoints]
    if any(not path.is_dir() for path in args.checkpoints):
        parser.error("checkpoints must be inference model directories; export RL weights first")
    if len(set(args.checkpoints)) != len(args.checkpoints) or len(set(args.seeds)) != len(args.seeds):
        parser.error("checkpoints and seeds must each be unique")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**32)")
    args.output = args.output.resolve()
    return args


def run_config(args):
    return {
        "checkpoints": [str(path) for path in args.checkpoints],
        "seeds": args.seeds,
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "physics": args.physics,
        "inference_runtimes": [
            "rl-reference-sdpa"
            if (path / "rl_export.json").is_file() or os.environ.get("EVAL_RL_SERVER") == "1"
            else "official-flash-attention"
            for path in args.checkpoints
        ],
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_process(process):
    """Release the process group, including children left behind by its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_command(command, *, cwd, env, log, server):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            returncode = process.poll()
            if server.poll() is not None:
                raise RuntimeError(f"inference server exited with code {server.returncode}; see server.log")
            if returncode is not None:
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                return
            time.sleep(0.1)
    finally:
        stop_process(process)


def evaluate(config, checkpoint_index, seed, attempt):
    checkpoint = config["checkpoints"][checkpoint_index]
    reference = config["inference_runtimes"][checkpoint_index] == "rl-reference-sdpa"
    arena_root = os.environ["ARENA_ROOT"]
    arena_python = os.environ["ARENA_PY"]
    server_source = os.environ["GROOT_REF_ROOT" if reference else "GROOT_ROOT"]
    server_python = os.environ["RL_PY" if reference else "GROOT_PY"]
    env = dict(os.environ, EVAL_SEED=str(seed), EVAL_RL_SERVER="1" if reference else "0")
    server_env = dict(env, PYTHONPATH=f"{server_source}{os.pathsep}{env.get('PYTHONPATH', '')}")
    client_env = dict(env, PYTHONPATH=f"{arena_root}{os.pathsep}{env.get('PYTHONPATH', '')}")
    purelib = subprocess.check_output(
        [arena_python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip()
    cuda_runtime = f"{purelib}/nvidia/cuda_runtime/lib"
    client_env["LD_LIBRARY_PATH"] = cuda_runtime + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    # The readiness probe must connect to this attempt's model, never an older server.
    with socket.socket() as port_check:
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_check.bind(("127.0.0.1", PORT))

    with ExitStack() as stack:
        server_log = stack.enter_context((attempt / "server.log").open("w"))
        eval_log = stack.enter_context((attempt / "eval.log").open("w"))
        server = subprocess.Popen(
            [server_python, "-m", "embodied_ai_lab.eval.server",
             "--modality-config-path", f"{arena_root}/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py",
             "--model-path", checkpoint, "--embodiment-tag", "NEW_EMBODIMENT",
             "--device", "cuda", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=server_source, env=server_env, stdout=server_log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        stack.callback(stop_process, server)
        run_command(
            [os.environ["GROOT_PY"], "-m", "isaaclab_arena_gr00t.utils.wait_for_gr00t_server",
             "--host", "127.0.0.1", "--port", str(PORT), "--timeout-sec", "600",
             "--poll-interval-sec", "5", "--request-timeout-ms", "5000"],
            cwd=arena_root, env=client_env, log=eval_log, server=server,
        )
        run_command(
            [arena_python, "-m", "embodied_ai_lab.eval.rollout",
             "--device", "cuda" if config["physics"] == "gpu" else "cpu",
             "--num_envs", str(config["num_envs"]), "--seed", str(seed), "--enable_cameras",
             "--episodes", str(config["episodes"]), "--output", str(attempt),
             "--model-path", checkpoint, "--port", str(PORT),
             f"--kit_args=--/log/file={attempt}/kit.log",
             "galileo_g1_locomanip_pick_and_place", "--object", "brown_box",
             "--embodiment", "g1_wbc_joint"],
            cwd=arena_root, env=client_env, log=eval_log, server=server,
        )


def complete_attempt(attempt, episodes):
    result = json.loads((attempt / "result.json").read_text())
    records = [json.loads(line) for line in (attempt / "episodes.jsonl").read_text().splitlines()]
    if len(records) != episodes or result["episodes"] != episodes:
        raise ValueError("evaluation did not finish the requested episode count")
    if result["successes"] != sum(record["success"] for record in records):
        raise ValueError("success count does not match episode records")
    write_json(attempt / "complete.json", result)


def run_batch(args):
    config = run_config(args)
    manifest = args.output / "run.json"
    if args.resume:
        if json.loads(manifest.read_text()) != config:
            raise ValueError("resume parameters differ from run.json; use the original parameters or a new output directory")
    else:
        args.output.mkdir(parents=True)
        write_json(manifest, config)

    try:
        for checkpoint_index, checkpoint in enumerate(config["checkpoints"]):
            for seed in config["seeds"]:
                job = args.output / f"ckpt-{checkpoint_index:03d}" / f"seed-{seed}"
                if completed_attempt(job) is not None:
                    print(f"Skip completed: {checkpoint} seed={seed}", flush=True)
                    continue
                previous = max((int(path.name.removeprefix("attempt-"))
                                for path in job.glob("attempt-*")), default=0)
                for retry in range(args.retries + 1):
                    attempt = job / f"attempt-{previous + retry + 1:04d}"
                    attempt.mkdir(parents=True)
                    print(f"Evaluate: {checkpoint} seed={seed} -> {attempt}", flush=True)
                    try:
                        evaluate(config, checkpoint_index, seed, attempt)
                        complete_attempt(attempt, config["episodes"])
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        (attempt / "error.log").write_text(message + "\n")
                        print(f"Failed: {message} (logs: {attempt})", flush=True)
                    else:
                        break
    finally:
        summary = write_summary(args.output)
    return 0 if all(job["status"] == "completed" for job in summary["jobs"]) else 1


def main(argv=None):
    args = parse_args(argv)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return run_batch(args)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
