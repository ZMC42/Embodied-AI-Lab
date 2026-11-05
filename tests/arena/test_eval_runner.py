"""Exercise batch recovery and subprocess isolation without Isaac Sim."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import Mock, patch

from embodied_ai_lab.eval import runner


class EvalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoints = [self.root / run / "checkpoint-100" for run in ("a", "b")]
        for checkpoint in self.checkpoints:
            checkpoint.mkdir(parents=True)
        self.output = self.root / "results"
        self.env = patch.dict(os.environ, {"EVAL_RL_SERVER": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def arguments(self, *extra):
        return ["--checkpoints", *map(str, self.checkpoints), "--seeds", "42", "43",
                "--episodes", "2", "--num-envs", "1", "--output", str(self.output), *extra]

    @staticmethod
    def successful(config, checkpoint_index, seed, attempt):
        records = [{"success": True}, {"success": False}]
        (attempt / "episodes.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
        (attempt / "result.json").write_text(json.dumps({
            "episodes": 2, "successes": 1, "elapsed_s": 3,
        }))

    def test_cartesian_product_and_same_named_checkpoints(self):
        with patch.object(runner, "evaluate", side_effect=self.successful) as evaluate:
            self.assertEqual(runner.main(self.arguments()), 0)
        self.assertEqual([(call.args[1], call.args[2]) for call in evaluate.call_args_list],
                         [(0, 42), (0, 43), (1, 42), (1, 43)])
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(summary["jobs"]), 4)
        self.assertEqual({job["status"] for job in summary["jobs"]}, {"completed"})
        self.assertEqual(len({job["attempt_dir"] for job in summary["jobs"]}), 4)

    def test_retry_then_continue_and_preserve_failed_attempt(self):
        def evaluate(config, index, seed, attempt):
            self.successful(config, index, seed, attempt)
            if index == 0 and seed == 42 and attempt.name == "attempt-0001":
                # Even complete-looking worker files cannot bless a failed process.
                raise subprocess.CalledProcessError(1, ["rollout"])

        with patch.object(runner, "evaluate", side_effect=evaluate) as evaluate:
            self.assertEqual(runner.main(self.arguments("--retries", "1")), 0)
        job = self.output / "ckpt-000/seed-42"
        self.assertEqual(evaluate.call_count, 5)
        self.assertTrue((job / "attempt-0001/error.log").is_file())
        self.assertFalse((job / "attempt-0001/complete.json").exists())
        self.assertTrue((job / "attempt-0002/complete.json").is_file())

    def test_failed_job_does_not_stop_batch_and_resume_only_reruns_it(self):
        def evaluate(config, index, seed, attempt):
            if index == 0 and seed == 42:
                raise RuntimeError("model failed")
            self.successful(config, index, seed, attempt)

        with patch.object(runner, "evaluate", side_effect=evaluate) as evaluate:
            self.assertEqual(runner.main(self.arguments("--retries", "1")), 1)
            self.assertEqual(evaluate.call_count, 5)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertIsNone(summary["checkpoints"][0]["mean_success_rate"])
        self.assertEqual(summary["checkpoints"][1]["mean_success_rate"], 0.5)
        with patch.object(runner, "evaluate", side_effect=self.successful) as evaluate:
            self.assertEqual(runner.main(self.arguments("--resume")), 0)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[3].name, "attempt-0003")
        with patch.object(runner, "evaluate") as evaluate:
            self.assertEqual(runner.main(self.arguments("--resume")), 0)
            evaluate.assert_not_called()

    def test_resume_rejects_changed_evaluation_parameters_and_runtime(self):
        with patch.object(runner, "evaluate", side_effect=self.successful):
            self.assertEqual(runner.main(self.arguments()), 0)
        for extra in (("--episodes", "3"), ("--num-envs", "2"), ("--physics", "cpu"),
                      ("--seeds", "43", "42")):
            with self.subTest(extra=extra), patch.object(runner, "evaluate") as evaluate:
                with self.assertRaisesRegex(ValueError, "resume parameters differ"):
                    runner.main(self.arguments("--resume", *extra))
                evaluate.assert_not_called()
        with patch.dict(os.environ, {"EVAL_RL_SERVER": "1"}):
            with self.assertRaisesRegex(ValueError, "resume parameters differ"):
                runner.main(self.arguments("--resume"))
        with self.assertRaises(FileExistsError):
            runner.main(self.arguments())

    def test_interrupt_stops_without_retry_and_can_resume(self):
        def interrupt(config, index, seed, attempt):
            self.successful(config, index, seed, attempt)
            raise KeyboardInterrupt

        with patch.object(runner, "evaluate", side_effect=interrupt) as evaluate:
            self.assertEqual(runner.main(self.arguments("--retries", "3")), 130)
            self.assertEqual(evaluate.call_count, 1)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual([job["status"] for job in summary["jobs"]],
                         ["failed", "pending", "pending", "pending"])
        self.assertFalse(list(self.output.glob("**/complete.json")))
        with patch.object(runner, "evaluate", side_effect=self.successful):
            self.assertEqual(runner.main(self.arguments("--resume")), 0)

    def test_incomplete_or_inconsistent_records_never_get_completion_marker(self):
        def evaluate(config, index, seed, attempt):
            self.successful(config, index, seed, attempt)
            if seed == 42:
                (attempt / "episodes.jsonl").write_text('{"success": true}\n')
            else:
                result = {"episodes": 2, "successes": 0, "elapsed_s": 1}
                (attempt / "result.json").write_text(json.dumps(result))

        with patch.object(runner, "evaluate", side_effect=evaluate):
            self.assertEqual(runner.main(self.arguments()), 1)
        self.assertEqual(len(list(self.output.glob("**/error.log"))), 4)
        self.assertFalse(list(self.output.glob("**/complete.json")))

    def test_real_subprocesses_receive_model_runtime_and_both_seeds(self):
        interpreter = self.root / "fake-python"
        events = self.root / "events.jsonl"
        interpreter.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
            import json
            import os
            from pathlib import Path
            import sys
            import time

            if sys.argv[1] == "-c":
                print("/fake/site-packages")
                sys.exit(0)
            module = sys.argv[2]
            if module.endswith("wait_for_gr00t_server"):
                sys.exit(0)
            def value(flag):
                return sys.argv[sys.argv.index(flag) + 1]
            event = {"module": module, "checkpoint": value("--model-path"),
                     "seed": os.environ["EVAL_SEED"], "rl": os.environ["EVAL_RL_SERVER"],
                     "cwd": os.getcwd(), "pid": os.getpid()}
            if module.endswith("rollout"):
                event["simulation_seed"] = value("--seed")
                event["device"] = value("--device")
                output = Path(value("--output"))
                (output / "episodes.jsonl").write_text('{"success": true}\\n{"success": false}\\n')
                (output / "result.json").write_text(json.dumps({
                    "episodes": 2, "successes": 1, "elapsed_s": 1,
                }))
            with open(os.environ["EVAL_TEST_EVENTS"], "a") as stream:
                stream.write(json.dumps(event) + "\\n")
            if module.endswith("server"):
                time.sleep(60)
            '''))
        interpreter.chmod(0o755)
        arena = self.root / "arena"
        official = self.root / "official"
        reference = self.root / "reference"
        for path in (arena, official, reference):
            path.mkdir()
        (self.checkpoints[1] / "rl_export.json").write_text("{}")
        env = {"ARENA_ROOT": str(arena), "GROOT_ROOT": str(official),
               "GROOT_REF_ROOT": str(reference), "ARENA_PY": str(interpreter),
               "GROOT_PY": str(interpreter), "RL_PY": str(interpreter),
               "EVAL_TEST_EVENTS": str(events)}
        with patch.dict(os.environ, env), patch.object(runner, "PORT", 0):
            self.assertEqual(runner.main(self.arguments()), 0)
        records = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(len(records), 8)
        for record in records:
            index = self.checkpoints.index(Path(record["checkpoint"]))
            self.assertEqual(record["rl"], str(index))
            if record["module"].endswith("server"):
                self.assertEqual(record["cwd"], str(reference if index else official))
            else:
                self.assertEqual(record["simulation_seed"], record["seed"])
                self.assertEqual(record["device"], "cuda")
            with self.assertRaises(ProcessLookupError):
                os.kill(record["pid"], 0)
        config = json.loads((self.output / "run.json").read_text())
        self.assertEqual(config["inference_runtimes"], ["official-flash-attention", "rl-reference-sdpa"])


class ProcessCleanupTest(unittest.TestCase):
    def test_cleanup_kills_remaining_group_after_leader_exits(self):
        leader = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"],
                                  preexec_fn=os.setpgrp)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 preexec_fn=lambda: os.setpgid(0, leader.pid))
        self.addCleanup(child.wait)
        self.addCleanup(runner.stop_process, leader)
        leader.wait()
        self.assertIsNone(child.poll())
        runner.stop_process(leader)
        self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)

    def test_interrupt_cleans_client_and_its_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid = root / "child.pid"
            command = [sys.executable, "-c", textwrap.dedent('''\
                import signal
                import subprocess
                import sys
                import time
                from pathlib import Path
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                def stop(signum, frame):
                    child.wait()
                    sys.exit(0)
                signal.signal(signal.SIGTERM, stop)
                Path(sys.argv[1]).write_text(str(child.pid))
                while True:
                    time.sleep(1)
                '''), str(child_pid)]
            sleep = time.sleep
            deadline = time.monotonic() + 5
            interrupted = False

            def interrupt_when_ready(_):
                nonlocal interrupted
                if child_pid.exists() and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                if not interrupted and time.monotonic() > deadline:
                    self.fail("child did not start")
                sleep(0.01)

            with (root / "client.log").open("w") as log:
                with patch.object(runner.time, "sleep", side_effect=interrupt_when_ready):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run_command(command, cwd=root, env=os.environ, log=log,
                                           server=Mock(poll=Mock(return_value=None)))
            with self.assertRaises(ProcessLookupError):
                os.kill(int(child_pid.read_text()), 0)

    def test_server_exit_aborts_and_reaps_client(self):
        processes = []
        popen = subprocess.Popen

        def start(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "client.log").open("w") as log:
                with patch.object(runner.subprocess, "Popen", side_effect=start):
                    with self.assertRaisesRegex(RuntimeError, "server exited"):
                        runner.run_command([sys.executable, "-c", "import time; time.sleep(60)"],
                                           cwd=directory, env=os.environ, log=log,
                                           server=Mock(poll=Mock(return_value=7), returncode=7))
        self.assertIsNotNone(processes[0].poll())


if __name__ == "__main__":
    unittest.main()
