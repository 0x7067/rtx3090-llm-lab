"""Exercise observer failure isolation, detection, and evidence retention."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import serve_observer as observer


def metrics(running=1, prompt=10, generated=20, waiting=0):
    return dict(zip(observer.REQUIRED, (running, prompt, generated))) | {
        "vllm:num_requests_waiting": waiting
    }


class ProgressTests(unittest.TestCase):
    def test_label_aggregation_and_missing_metrics(self):
        text = "\n".join(
            f'{name}{{model_name="test"}} {value}' for name, value in metrics().items()
        )
        text += '\nvllm:num_requests_running{model_name="second"} 2\n'
        self.assertEqual(observer.parse_metrics(text)[observer.REQUIRED[0]], 3)
        with self.assertRaises(ValueError):
            observer.parse_metrics("vllm:num_requests_running NaN\n")

    def test_active_stall_recovery_and_counter_reset(self):
        tracker = observer.Progress(0)
        tracker.update(0, metrics())
        self.assertIsNone(tracker.update(239, metrics())[1])
        self.assertEqual(
            tracker.update(240, metrics())[1], "active_without_token_progress"
        )
        self.assertIsNone(tracker.update(250, metrics())[1])
        self.assertIsNone(tracker.update(260, metrics(generated=21))[0]["reason"])
        self.assertEqual(
            tracker.update(500, metrics(generated=21))[1],
            "active_without_token_progress",
        )
        self.assertIsNone(
            tracker.update(510, metrics(prompt=0, generated=0))[0]["reason"]
        )
        self.assertIsNone(tracker.update(1000, metrics(running=0))[0]["reason"])

    def test_missing_endpoint_and_waiting_work(self):
        tracker = observer.Progress(0)
        self.assertIsNone(tracker.update(899, None)[1])
        self.assertEqual(tracker.update(900, None)[1], "startup_metrics_unavailable")
        tracker.update(901, metrics(running=0, waiting=1))
        self.assertIsNone(tracker.update(960, None)[1])
        self.assertEqual(tracker.update(961, None)[1], "metrics_unavailable")
        self.assertIsNone(
            tracker.update(962, metrics(running=0, waiting=1))[0]["reason"]
        )
        self.assertEqual(
            tracker.update(1141, metrics(running=0, waiting=1))[1],
            "active_without_token_progress",
        )


class RecorderTests(unittest.TestCase):
    def test_capture_survives_rotation_as_independent_copy(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(observer, "LOG_BYTES", 64),
        ):
            recorder = observer.Recorder(Path(directory), os.getpid())
            recorder.append({"evidence": "before", "padding": "x" * 80})
            recorder.capture("manual_capture", {}, "metrics")
            captured = recorder.run / "capture-01/samples.jsonl"
            before = captured.read_bytes()
            self.assertNotEqual(
                captured.stat().st_ino, (recorder.run / "samples.jsonl").stat().st_ino
            )
            for index in range(8):
                recorder.append({"evidence": index, "padding": "x" * 80})
            self.assertEqual(captured.read_bytes(), before)
            self.assertEqual(len(list(recorder.run.glob("samples*.jsonl"))), 4)
            with patch.object(observer, "BUDGET_BYTES", 1):
                recorder.capture("second", {}, "")
                self.assertEqual(recorder.state["capture_count"], 1)
                self.assertTrue(recorder.state["capture_limit_reached"])


class LifecycleTests(unittest.TestCase):
    def test_metrics_failure_preserves_model_and_sigterm_reaches_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "nvidia-smi"
            probe.write_text('#!/bin/sh\nprintf "test GPU\\n"\n')
            probe.chmod(0o755)
            pidfile, stopped = root / "model.pid", root / "stopped"
            code = (
                "import os,signal,time; from pathlib import Path; "
                f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(stopped)!r}).touch(), exit(0))); "
                f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                "time.sleep(60)"
            )
            env = dict(
                os.environ,
                SPARSEKV_OBSERVATIONS_DIR=str(root / "records"),
                SPARSEKV_METRICS_URL="http://127.0.0.1:1/metrics",
                PATH=str(root),
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(observer.__file__)),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                status = root / "records/current/status.json"
                deadline = time.monotonic() + 8
                while (not status.exists() or not pidfile.exists()) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIsNone(process.poll())
                self.assertTrue(pidfile.exists())
                self.assertIsNotNone(json.loads(status.read_text())["metrics_error"])
                process.terminate()
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0)
                self.assertTrue(stopped.exists())
                self.assertTrue(json.loads(status.read_text())["target_exited"])
            finally:
                # The child inherits the pipes and owns a process group. Reap it
                # before communicating with a failed wrapper to avoid open pipes.
                if pidfile.exists():
                    try:
                        os.killpg(int(pidfile.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_unwritable_storage_preserves_model_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "file"
            blocker.touch()
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(observer.__file__)),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                env=dict(os.environ, SPARSEKV_OBSERVATIONS_DIR=str(blocker), PATH=""),
                capture_output=True,
                timeout=8,
                check=False,
            )
            self.assertEqual(result.returncode, 7)
            self.assertIn(b"storage unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()
