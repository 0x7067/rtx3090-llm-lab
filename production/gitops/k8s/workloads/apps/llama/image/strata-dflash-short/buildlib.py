"""Host-side helpers shared by run_build_cpu.py and optimization/normal-120/run-cpu-candidates.py."""
import datetime
import json
import subprocess
import time
from pathlib import Path


def run_logged(argv, path, timeout):
    """Run argv with stdout and stderr written to path; raise on a non-zero exit."""
    with path.open("w") as output:
        subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def memory_preflight(out, status):
    """Wait until MemAvailable and memory PSI allow an 8 GiB CPU job, sampling into out/memory-preflight.jsonl."""
    deadline = time.monotonic() + 600
    while True:
        mem = {line.split(":", 1)[0]: int(line.split()[1])
               for line in Path("/proc/meminfo").read_text().splitlines()}
        pressure = Path("/proc/pressure/memory").read_text()
        full = next(line for line in pressure.splitlines() if line.startswith("full "))
        full_avg10 = float(dict(part.split("=") for part in full.split()[1:])["avg10"])
        report = {"mem_available_kib": mem["MemAvailable"], "swap_free_kib": mem["SwapFree"],
                  "memory_psi": pressure, "memory_psi_full_avg10": full_avg10,
                  "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (out / "memory-preflight.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        # Leave at least 4 GiB available beyond the bounded 8 GiB CPU job.
        # Full swap alone is not a failure; active memory pressure is measured.
        if mem["MemAvailable"] >= 12 * 1024 * 1024 and full_avg10 < 1.0:
            return
        status("waiting-memory-headroom", **report)
        if time.monotonic() >= deadline:
            raise RuntimeError("Insufficient memory headroom or sustained memory pressure; no heavy job started")
        time.sleep(45)
