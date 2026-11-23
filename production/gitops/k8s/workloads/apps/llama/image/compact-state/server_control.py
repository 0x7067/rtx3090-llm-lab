"""Own one temporary backend; restart it when its task-owned arm file changes."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path("/control")
config = json.loads((root / "launcher.json").read_text())
child = None
previous = None


def stop(*_):
    global child
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    child = None


def end(*_):
    stop()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, end)
signal.signal(signal.SIGINT, end)
while True:
    arm = (root / "arm").read_text().strip()
    if arm != previous:
        stop()
        previous = arm
        if arm != "stopped":
            assert arm in ["baseline-a", "candidate-a", "candidate-b", "baseline-b"]
            env = os.environ.copy()
            env.update(config["env"])
            env["VLLM_DFLASH2_CHAIN"] = "0"
            env["VLLM_COMPACT_GDN"] = "1" if arm.startswith("candidate") else "0"
            env["VLLM_DFLASH2_CHAIN_LOG_SEC"] = "1"
            log = (root / (arm + "-server.log")).open("wb")
            child = subprocess.Popen(
                config["command"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log.close()
        (root / "started.json").write_text(
            json.dumps({"arm": arm, "pid": child.pid if child else None})
        )
    if child and child.poll() is not None:
        raise RuntimeError(f"{previous} backend exited {child.returncode}")
    time.sleep(0.5)
