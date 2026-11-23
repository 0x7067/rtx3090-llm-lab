"""Evaluate the predeclared serving gate using complete ABBA results."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
args = p.parse_args()
arms = ["baseline-a", "candidate-a", "candidate-b", "baseline-b"]
rows = {
    a: [json.loads(x) for x in (args.root / (a + ".jsonl")).read_text().splitlines()]
    for a in arms
}
assert all(len(x) == 43 for x in rows.values()), {a: len(x) for a, x in rows.items()}
quality = {a: json.loads((args.root / (a + "-quality.json")).read_text()) for a in arms}
failures = [
    {k: x[k] for k in ["arm", "name", "phase", "concurrency", "repeat"]}
    for arm in rows.values()
    for x in arm
    if not x["content_passed"] or x["finish_reason"] != "stop"
]
keys = sorted(
    {(x["name"], x["concurrency"]) for x in rows[arms[0]] if x["phase"] == "measure"}
)
cases = {}
for name, c in keys:
    values = {
        arm: [
            x["elapsed_seconds"]
            for x in rr
            if x["phase"] == "measure" and (x["name"], x["concurrency"]) == (name, c)
        ]
        for arm, rr in rows.items()
    }
    med = {a: statistics.median(x) for a, x in values.items()}
    baseline = statistics.median(values["baseline-a"] + values["baseline-b"])
    candidate = statistics.median(values["candidate-a"] + values["candidate-b"])
    cases[f"{name}/C{c}"] = {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "latency_reduction_percent": 100 * (1 - candidate / baseline),
        "baseline_drift_percent": 100 * abs(med["baseline-b"] / med["baseline-a"] - 1),
        "arm_medians": med,
    }
engagement = {
    a: all(
        s in (args.root / (a + "-server.log")).read_text()
        for s in [
            "Compact GDN verification engaged",
            "Compact GDN accepted and boundary states finalized",
        ]
    )
    for a in ["candidate-a", "candidate-b"]
}
# Each fixture/repeat has identical request bodies in every arm.
for arm in arms[1:]:
    assert sorted(
        (x["phase"], x["repeat"], x["concurrency"], x["request_sha256"])
        for x in rows[arm]
    ) == sorted(
        (x["phase"], x["repeat"], x["concurrency"], x["request_sha256"])
        for x in rows[arms[0]]
    )
qualified = (
    not failures
    and all(x["passed"] for x in quality.values())
    and all(engagement.values())
    and all(x["baseline_drift_percent"] <= 5 for x in cases.values())
    and all(
        x["latency_reduction_percent"] >= (-3 if key.endswith("C1") else 3)
        for key, x in cases.items()
    )
)
result = {
    "qualified": qualified,
    "requests": sum(map(len, rows.values())),
    "failures": failures,
    "strict_format_failures": {
        arm: sum(not x["passed"] and x["content_passed"] for x in rr)
        for arm, rr in rows.items()
    },
    "cases": cases,
    "quality": quality,
    "engagement": engagement,
}
(args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
