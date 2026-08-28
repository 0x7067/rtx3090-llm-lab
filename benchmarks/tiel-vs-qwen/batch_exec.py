"""Runs candidates and returns real failure output. Executes INSIDE a
network-less container. Reads /work/candidates.jsonl -> /work/results.jsonl.

Unlike sandbox_runner.py this keeps enough stderr to feed back as a repair
prompt, and truncates from the tail where the assertion actually lives.
"""
import json, os, subprocess, sys, tempfile

results = []
for line in open("/work/candidates.jsonl"):
    c = json.loads(line)
    program = (
        "import math, re, itertools, collections, heapq, string, functools\n"
        "from typing import *\n"
        + c["code"] + "\n\n" + c["test"] + f"\n\ncheck({c['entry_point']})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=15, text=True)
        ok = p.returncode == 0
        err = "" if ok else (p.stderr or "")[-1200:]
    except subprocess.TimeoutExpired:
        ok, err = False, "TimeoutError: execution exceeded 15s (likely an infinite loop)"
    finally:
        os.unlink(path)
    results.append({"task_id": c["task_id"], "passed": ok, "error": err})

with open("/work/results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
n = len(results)
print(f"{sum(r['passed'] for r in results)}/{n} passed")
