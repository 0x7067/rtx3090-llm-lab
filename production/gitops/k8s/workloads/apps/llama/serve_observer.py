"""Observe SparseKV progress; preserve bounded evidence without restarting inference."""

import collections
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("SPARSEKV_OBSERVATIONS_DIR", "/cache/sparsekv-observations"))
URL = os.environ.get("SPARSEKV_METRICS_URL", "http://127.0.0.1:5890/metrics")
SAMPLE_SECONDS = 10
BUDGET_BYTES = 64 * 1024**2
LOG_BYTES = 256 * 1024
METRICS_LIMIT = 2 * 1024**2
REQUIRED = (
    "vllm:num_requests_running",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)
SELECTED = set(REQUIRED) | {
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
}


def write_json(path, data):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(data, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_metrics(text):
    values = collections.defaultdict(float)
    for line in text.splitlines():
        match = re.match(r"([^\s{]+)(?:\{.*\})?\s+(\S+)", line)
        if match and match[1] in SELECTED:
            value = float(match[2])
            if math.isfinite(value):
                values[match[1]] += value
    if not all(name in values for name in REQUIRED):
        raise ValueError("required progress metrics unavailable")
    return dict(values)


class Progress:
    """Distinguish idle, active work without progress, and a missing metrics server."""

    def __init__(self, now):
        self.started = self.last_progress = now
        self.last_success = None
        self.tokens = None
        self.reason = None

    def update(self, now, values):
        reason = None
        if values is not None:
            active = values[REQUIRED[0]] + values.get("vllm:num_requests_waiting", 0)
            tokens = values[REQUIRED[1]] + values[REQUIRED[2]]
            if active == 0 or tokens != self.tokens:
                self.last_progress = now
            self.tokens = tokens
            self.last_success = now
            if active > 0 and now - self.last_progress >= 240:
                reason = "active_without_token_progress"
        elif self.last_success is not None and now - self.last_success >= 60:
            reason = "metrics_unavailable"
        elif self.last_success is None and now - self.started >= 900:
            reason = "startup_metrics_unavailable"
        event = reason if reason and reason != self.reason else None
        self.reason = reason
        return {
            "reason": reason,
            "stalled_seconds": round(now - self.last_progress, 1),
            "metrics_ready": self.last_success is not None,
        }, event


class GpuProbe:
    """At most one bounded nvidia-smi child, even if a driver call stops responding."""

    def __init__(self):
        self.child = None

    def read(self):
        if self.child is not None and self.child.poll() is None:
            return {"error": "previous GPU probe remains unresponsive"}
        try:
            self.child = subprocess.Popen(
                [
                    "nvidia-smi",
                    (
                        "--query-gpu=timestamp,power.limit,power.draw,temperature.gpu,"
                        "utilization.gpu,utilization.memory,memory.used,clocks.gr,clocks.mem"
                    ),
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            return {"error": str(error)}
        try:
            out, err = self.child.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self.child.kill()
            try:
                self.child.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return {"error": "GPU probe timed out"}
        return {
            "csv": out.strip(),
            "error": err.strip()[:1000],
            "exit_code": self.child.returncode,
        }


def process_snapshot():
    result = []
    fields = {"Name", "State", "PPid", "VmRSS", "VmHWM", "Threads"}
    for proc in sorted(Path("/proc").glob("[0-9]*"), key=lambda p: int(p.name))[:256]:
        try:
            row = {"pid": int(proc.name)}
            for line in (proc / "status").read_text().splitlines():
                name, _, value = line.partition(":")
                if name in fields:
                    row[name] = value.strip()
            row["wchan"] = (proc / "wchan").read_text().strip()
            result.append(row)
        except (OSError, ValueError):
            continue
    return result


def memory_snapshot():
    result = {}
    for name in (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.pressure",
    ):
        try:
            result[name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError:
            pass
    result["host"] = {
        line.split(":")[0]: line.split(":")[1].strip()
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in {"MemAvailable", "MemFree", "SwapFree", "SwapTotal"}
    }
    return result


class Recorder:
    def __init__(self, root, pid):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.run = root / (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        )
        self.run.mkdir()
        link = root / "current.new"
        link.unlink(missing_ok=True)
        link.symlink_to(self.run.name)
        link.replace(root / "current")
        self.state = {
            "target_pid": pid,
            "capture_count": 0,
            "target_exited": False,
            "sample_seconds": SAMPLE_SECONDS,
            "budget_bytes": BUDGET_BYTES,
        }
        self.used = 0
        self.next_budget_check = 0

    def space(self, now):
        if now >= self.next_budget_check:
            self.used = sum(
                p.stat().st_size for p in self.root.rglob("*") if p.is_file()
            )
            self.next_budget_check = now + 60
        return self.used < BUDGET_BYTES

    def append(self, sample):
        log = self.run / "samples.jsonl"
        if log.exists() and log.stat().st_size >= LOG_BYTES:
            (self.run / "samples.3.jsonl").unlink(missing_ok=True)
            for index in (2, 1):
                source = self.run / f"samples.{index}.jsonl"
                if source.exists():
                    source.replace(self.run / f"samples.{index + 1}.jsonl")
            log.replace(self.run / "samples.1.jsonl")
        with log.open("a") as out:
            out.write(json.dumps(sample, allow_nan=False) + "\n")

    def capture(self, reason, sample, metrics):
        self.next_budget_check = 0
        if self.state["capture_count"] >= 8 or not self.space(time.monotonic()):
            self.state["capture_limit_reached"] = True
            return
        count = self.state["capture_count"] + 1
        event = self.run / f"capture-{count:02d}"
        event.mkdir()
        self.state["capture_count"] = count
        self.state["last_capture"] = {
            "reason": reason,
            "path": str(event),
            "time": time.time(),
        }
        for source in self.run.glob("samples*.jsonl"):
            shutil.copy2(source, event / source.name)
        write_json(
            event / "snapshot.json",
            {
                "reason": reason,
                "sample": sample,
                "processes": process_snapshot(),
                "memory": memory_snapshot(),
            },
        )
        (event / "metrics.txt").write_text(metrics[:METRICS_LIMIT])
        print(
            json.dumps({"sparsekv_observer_capture": self.state["last_capture"]}),
            flush=True,
        )
        self.next_budget_check = 0

    def save(self, **fields):
        self.state.update(fields, updated_at=time.time())
        write_json(self.run / "status.json", self.state)


def main(command):
    os.umask(0o077)
    child = None
    stopping = None

    def terminate(signum, _frame):
        nonlocal stopping
        stopping = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    child = subprocess.Popen(command, start_new_session=True)
    if stopping:
        terminate(stopping, None)
    warmer = None
    warm_script = Path(__file__).with_name("warm_saved_prefixes.py")
    if warm_script.exists():
        try:
            warmer = subprocess.Popen([sys.executable, "-I", "-S", str(warm_script)])
        except OSError as error:
            print(f"sparsekv prefix warmer unavailable: {error}", file=sys.stderr, flush=True)
    recorder = None
    try:
        recorder = Recorder(ROOT, child.pid)
    except OSError as error:
        print(
            f"sparsekv observer storage unavailable: {error}",
            file=sys.stderr,
            flush=True,
        )
    tracker = Progress(time.monotonic())
    gpu = GpuProbe()
    next_sample = 0
    last_error = None
    while child.poll() is None:
        if warmer is not None:
            warmer.poll()  # Reap the short-lived warmer while inference continues.
        if not stopping and time.monotonic() >= next_sample:
            next_sample = time.monotonic() + SAMPLE_SECONDS
            try:
                metrics = ""
                values = None
                error = None
                try:
                    with urllib.request.urlopen(URL, timeout=2) as response:
                        raw = response.read(METRICS_LIMIT + 1)
                    if len(raw) > METRICS_LIMIT:
                        raise ValueError("metrics response exceeded size limit")
                    metrics = raw.decode()
                    values = parse_metrics(metrics)
                except (OSError, ValueError) as failure:
                    error = str(failure)
                progress, event = tracker.update(time.monotonic(), values)
                sample = {
                    "time": time.time(),
                    "metrics": values,
                    "metrics_error": error,
                    "gpu": gpu.read(),
                    "memory": memory_snapshot(),
                    **progress,
                }
                if recorder is not None:
                    space = recorder.space(time.monotonic())
                    if space:
                        recorder.append(sample)
                    marker = recorder.run / "capture"
                    if marker.exists():
                        marker.unlink()
                        event = "manual_capture"
                    if event:
                        recorder.capture(event, sample, metrics)
                    recorder.save(
                        **progress,
                        metrics=values,
                        metrics_error=error,
                        gpu=sample["gpu"],
                        storage_budget_reached=not space,
                    )
                last_error = None
            except Exception as failure:  # noqa: BLE001 - isolate all observation failures
                # Observation failures never terminate or restart inference.
                message = str(failure)
                if message != last_error:
                    print(
                        f"sparsekv observer error: {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_error = message
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    if warmer is not None and warmer.poll() is None:
        warmer.terminate()
        try:
            warmer.wait(timeout=3)
        except subprocess.TimeoutExpired:
            warmer.kill()
            warmer.wait()
    if recorder is not None:
        try:
            recorder.save(target_exited=True, target_exit_code=child.returncode)
        except OSError as error:
            print(
                f"sparsekv observer final status: {error}", file=sys.stderr, flush=True
            )
    return 128 - child.returncode if child.returncode < 0 else child.returncode


if __name__ == "__main__":
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    raise SystemExit(main(command or ["/bin/bash", "/opt/sparsekv/serve-sparse.sh"]))
