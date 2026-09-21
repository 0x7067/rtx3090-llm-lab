#!/usr/bin/env python3
"""Validate archived runs and summarize this pilot; print JSON to stdout."""
import gzip
import hashlib
import json
import shutil
import statistics
import tempfile
from pathlib import Path

from agent_smoke import harness

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def load_run(name):
    plan = harness.load_plan(ARTIFACTS / "plans" / f"plan-{name}.json")
    source = ARTIFACTS / "runs" / name
    with tempfile.TemporaryDirectory(prefix="spark-summary-") as tmp:
        tmp = Path(tmp)
        shutil.copy2(source / "manifest.json", tmp / "manifest.json")
        with gzip.open(source / "results.jsonl.gz", "rb") as inp:
            with (tmp / "results.jsonl").open("wb") as out:
                shutil.copyfileobj(inp, out)
        rows = harness.load_run(tmp, plan)
    compact = {key: {"score": row["score"], "finish_reason": row["result"].get("finish_reason"),
        "wall_s": row["result"]["wall_s"], "output_tokens": row["result"].get("usage", {}).get("completion_tokens"),
        "grade_detail": row["grade_detail"]} for key, row in rows.items()}
    return plan, compact


def summarize(plan, rows):
    result = {}
    for tier in ("coding", "tool_replay"):
        subset = [rows[job["id"]] for job in plan["jobs"] if job["case"]["tier"] == tier]
        result[tier] = {"passed": sum(row["score"] for row in subset), "attempted": len(subset),
            "truncated": sum(row["finish_reason"] == "length" for row in subset),
            "mean_api_wall_s": statistics.mean(row["wall_s"] for row in subset),
            "median_api_wall_s": statistics.median(row["wall_s"] for row in subset)}
    return {"plan_sha256": plan["sha256"], "metrics": result, "cases": rows}


def compare_instruction(control, candidate, instruction):
    a_plan, a_rows = control
    b_plan, b_rows = candidate
    a_jobs = {job["id"]: job for job in a_plan["jobs"]}
    b_jobs = {job["id"]: job for job in b_plan["jobs"]}
    assert a_jobs.keys() == b_jobs.keys(), "different evaluation cases"
    result = {"wins": [], "losses": [], "unchanged": []}
    for key, a_job in a_jobs.items():
        b_job = b_jobs[key]
        for field in ("id", "tier", "cluster", "phase", "grader"):
            assert a_job["case"][field] == b_job["case"][field], "changed grading contract"
        request = json.loads(json.dumps(b_job["request"]))
        # The shared harness prepends an evaluation-ID system message. Only the
        # next message is the deliberate policy change; all other fields match.
        assert request["messages"].pop(1) == {"role": "system", "content": instruction}
        assert request == a_job["request"], "unintended request change"
        delta = b_rows[key]["score"] - a_rows[key]["score"]
        result["wins" if delta > 0 else "losses" if delta < 0 else "unchanged"].append(key)
    return result


def summarize_agents(name):
    source = ARTIFACTS / "runs" / name
    manifest = harness.read(source / "manifest.json")
    with gzip.open(source / "results.jsonl.gz", "rt") as inp:
        rows = [json.loads(line) for line in inp]
    assert len(rows) == len(manifest["suite"]["cases"])
    assert {r["id"] for r in rows} == {c["id"] for c in manifest["suite"]["cases"]}
    return {row["id"]: {"score": row["score"], "workflow_completed": row["workflow_completed"],
        "api_time_sum_s": sum(step["response"]["wall_s"] for step in row["trace"]),
        "requests": len(row["trace"])} for row in rows}


def main():
    names = ["dev-publisher", "dev-focused", "dev-no-thinking", "dev-no-thinking-focused",
             "dev-concise", "holdout-publisher", "holdout-concise"]
    loaded = {name: load_run(name) for name in names}
    instruction = harness.read(ROOT / "coding-profile.json")["system_instruction"]
    report = {"analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": {name: summarize(*value) for name, value in loaded.items()},
        "paired_instruction": {split: compare_instruction(loaded[f"{split}-publisher"],
            loaded[f"{split}-concise"], instruction) for split in ("dev", "holdout")},
        "agents": {name: summarize_agents(name) for name in ("agent-publisher", "agent-no-thinking")}}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
