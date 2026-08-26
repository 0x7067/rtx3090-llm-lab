"""Executes HumanEval candidates. Runs INSIDE a network-less docker container.

Reads /work/candidates.jsonl, runs each candidate + its test in a fresh
subprocess with a 15s timeout, writes /work/results.jsonl.
"""
import json, subprocess, sys, tempfile, os

IN, OUT = "/work/candidates.jsonl", "/work/results.jsonl"

results = []
for line in open(IN):
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
        err = (p.stderr or "")[-300:] if not ok else ""
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    finally:
        os.unlink(path)
    results.append({"task_id": c["task_id"], "passed": ok, "error": err})
    print(f"{c['task_id']}: {'PASS' if ok else 'FAIL'}", flush=True)

with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
n = len(results)
p = sum(r["passed"] for r in results)
print(f"\npass@1: {p}/{n} = {p/n:.3f}")
