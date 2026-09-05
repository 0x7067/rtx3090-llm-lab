#!/usr/bin/env python3
"""Frozen, paired evaluations of OpenAI-compatible inference servers (stdlib)."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid

UA = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def validate_suite(suite):
    assert suite["cases"], "empty suite"
    assert 0 < suite["confidence"] < 1
    assert isinstance(suite["bootstrap_samples"], int) and suite["bootstrap_samples"] >= 100
    assert isinstance(suite["min_clusters"], int) and suite["min_clusters"] >= 2
    assert suite["seeds"] and len(set(suite["seeds"])) == len(suite["seeds"])
    assert all(type(s) is int for s in suite["seeds"])
    assert type(suite["repeats"]) is int and suite["repeats"] > 0
    ids = set()
    for case in suite["cases"]:
        assert case["id"] not in ids, "duplicate case id"
        ids.add(case["id"])
        assert case["cluster"] and case["tier"]
        assert case["phase"] in ("quality", "performance")
        request = {**suite["request_defaults"], **case["request"]}
        assert request["messages"] and type(request["max_tokens"]) is int and request["max_tokens"] > 0
        assert not {"model", "stream", "stream_options", "seed"} & request.keys()
        assert request.get("n", 1) == 1, "one completion per request is required"
        assert all(m["role"] in ("system", "user", "assistant", "tool") for m in request["messages"])
        if case["phase"] == "quality":
            assert case["grader"]["kind"] in ("exact", "json", "set_f1", "tool", "python")
            assert 0 <= suite["margins"][case["tier"]] < 1
            grader = case["grader"]
            if grader["kind"] == "python":
                assert suite["code_image"].startswith("sha256:"), "pin the code-test image ID"
                compile(grader["tests"], case["id"], "exec")
            else:
                expected = grader["expected"]
                if grader["kind"] == "exact":
                    assert isinstance(expected, str)
                elif grader["kind"] == "set_f1":
                    assert isinstance(expected, list) and all(isinstance(v, str) for v in expected)
                elif grader["kind"] == "tool":
                    assert isinstance(expected, list)
                    assert all(isinstance(c["name"], str) and isinstance(c["arguments"], dict) for c in expected)


def make_plan(suite, arms, campaign):
    validate_suite(suite)
    names = [a["name"] for a in arms]
    assert names and len(set(names)) == len(names)
    for arm in arms:
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", arm["name"])
        assert arm["base_url"].startswith(("http://", "https://"))
        assert arm["model"] and arm["metadata"], "record runtime/weights/KV provenance"
        assert set(arm) <= {"name", "base_url", "model", "metadata", "api_key_env"}
    jobs = []
    for repeat in range(suite["repeats"]):
        for seed in suite["seeds"]:
            for case in suite["cases"]:
                body = {**suite["request_defaults"], **case["request"]}
                body = json.loads(json.dumps(body))
                # A unique leading prefix reduces reuse across cases/runs. Servers may
                # still cache template tokens; this is not proof of a cold KV cache.
                nonce = digest([campaign, case["id"], repeat, seed])[:20]
                body["messages"].insert(0, {"role": "system", "content": f"Evaluation ID: {nonce}."})
                body.update(seed=seed, stream=True, stream_options={"include_usage": True})
                jobs.append({"id": f"{case['id']}:{repeat}:{seed}", "case": case,
                             "request": body, "request_sha256": digest(body)})
    random.Random(42).shuffle(jobs)
    plan = {"schema": 1, "campaign": campaign, "suite": suite, "arms": arms, "jobs": jobs,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    plan["sha256"] = digest(plan)
    return plan


def load_plan(path):
    plan = read(path)
    saved = plan.pop("sha256")
    assert digest(plan) == saved, "plan changed after freezing"
    plan["sha256"] = saved
    return plan


def sse_events(lines):
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                value = "\n".join(data)
                data = []
                if value == "[DONE]":
                    yield None
                    return
                event = json.loads(value)
                if not isinstance(event, dict):
                    raise ValueError("SSE data must be an object")
                yield event
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        raise ValueError("unfinished SSE event")


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def consume(events, start):
    result = {"status": "protocol_error", "content": "", "reasoning": "", "tool_calls": [],
              "finish_reason": None, "events": [], "ttft_s": None, "usage": {}, "timings": {}}
    calls = {}
    done = False
    for event in events:
        if event is None:
            done = True
            break
        result["events"].append(event)
        if "error" in event:
            result["error"] = event["error"]
            break
        result["usage"].update(event.get("usage") or {})
        result["timings"].update(event.get("timings") or {})
        choices = event.get("choices") or []
        if len(choices) > 1:
            raise ValueError("only one completion per request is supported")
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            tool_parts = delta.get("tool_calls") or []
            if result["ttft_s"] is None and (content or reasoning or any(
                    c.get("function", {}).get("name") or c.get("function", {}).get("arguments")
                    for c in tool_parts)):
                result["ttft_s"] = time.monotonic() - start
            result["content"] += content
            result["reasoning"] += reasoning
            for part in tool_parts:
                call = calls.setdefault(part["index"], {"name": "", "arguments": ""})
                for key in ("name", "arguments"):
                    call[key] += part.get("function", {}).get(key) or ""
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
    result["tool_calls"] = [calls[i] for i in sorted(calls)]
    result["wall_s"] = time.monotonic() - start
    if done and result["finish_reason"] and "error" not in result:
        result["status"] = "ok"
    usage, timings = result["usage"], result["timings"]
    n = usage.get("completion_tokens")
    result["e2e_output_tps"] = n / result["wall_s"] if positive_number(n) else None
    # Never infer tokens from SSE chunks or characters. Client wall time includes
    # prefill/scheduling; only explicit server timing supplies server decode rate.
    # llama.cpp excludes the first output token from its decode interval. A
    # one-token prefill probe has no decode interval at all. Preserve its native
    # rate instead of deriving n/duration and inventing throughput for that case.
    result["server_decode_tps"] = (timings["predicted_per_second"]
        if positive_number(timings.get("predicted_n")) and timings["predicted_n"] > 1
        and positive_number(timings.get("predicted_per_second")) else None)
    result["server_prefill_tps"] = (1000 * timings["prompt_n"] / timings["prompt_ms"]
        if positive_number(timings.get("prompt_n")) and positive_number(timings.get("prompt_ms")) else None)
    return result


def request(arm, body, timeout):
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if arm.get("api_key_env"):
        headers["Authorization"] = "Bearer " + os.environ[arm["api_key_env"]]
    req = urllib.request.Request(arm["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps({**body, "model": arm["model"]}).encode(), headers=headers)
    start = time.monotonic()
    wire = []

    def lines(response):
        for line in response:
            wire.append(line)
            if time.monotonic() - start > timeout:
                raise TimeoutError("request exceeded total time budget")
            yield line

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = consume(sse_events(lines(response)), start)
        return result
    except urllib.error.HTTPError as e:
        return {"status": "http_error", "http_status": e.code,
                "error": e.read().decode("utf-8", "replace"), "wall_s": time.monotonic() - start}
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"status": "transport_or_protocol_error", "error": str(e), "wall_s": time.monotonic() - start,
                "raw_response": b"".join(wire).decode("utf-8", "replace")}


def python_grade(content, tests, image):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
    code = blocks[0] if len(blocks) == 1 else content
    name = "paired-eval-" + uuid.uuid4().hex
    marker = "PAIRED_HARNESS_TESTS_COMPLETED"
    runner = "import json,sys; p=json.load(sys.stdin); ns={}; exec(compile(p['code'],'answer.py','exec'),ns); exec(compile(p['tests'],'tests.py','exec'),ns); print('PAIRED_HARNESS_TESTS_COMPLETED')"
    try:
        p = subprocess.run(["docker", "run", "--pull", "never", "--rm", "-i", "--name", name,
                            "--network", "none", "--read-only", "--cap-drop", "ALL",
                            "--security-opt", "no-new-privileges", "--memory", "256m", "--cpus", "1",
                            "--pids-limit", "64", "--user", "1000:1000", "--tmpfs",
                            "/tmp:rw,noexec,nosuid,size=16m", image, "python", "-I", "-c", runner],
                           input=json.dumps({"code": code, "tests": tests}), text=True,
                           capture_output=True, timeout=30)
        if p.returncode in (125, 126, 127):
            raise RuntimeError("code grader infrastructure failure: " + p.stderr[-1000:])
        return float(p.returncode == 0 and marker in p.stdout.splitlines()), p.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return 0.0, "code execution timed out"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def json_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(json_equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def grade(grader, result, image):
    if result["status"] != "ok" or result.get("finish_reason") not in ("stop", "tool_calls"):
        return 0.0, "request failed or output incomplete"
    content = result["content"].strip()
    kind = grader["kind"]
    try:
        if kind == "exact":
            return float(content == grader["expected"]), "exact answer"
        if kind == "json":
            return float(json_equal(json.loads(content), grader["expected"])), "JSON semantic equality"
        if kind == "set_f1":
            values = json.loads(content)
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                return 0.0, "expected JSON array of strings"
            actual, expected = set(values), set(grader["expected"])
            return (2 * len(actual & expected) / (len(actual) + len(expected))
                    if actual or expected else 1.0), "set micro F1 within item"
        if kind == "tool":
            calls = [{"name": c["name"], "arguments": json.loads(c["arguments"])} for c in result["tool_calls"]]
            return float(json_equal(calls, grader["expected"])), "tool names and parsed arguments"
        if kind == "python":
            return python_grade(content, grader["tests"], image)
    except (ValueError, TypeError, KeyError):
        return 0.0, "invalid answer format"
    raise ValueError("unknown grader")


def run(plan, arm_name, out, timeout):
    assert plan["harness_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "harness changed; create a new plan"
    arm = next(a for a in plan["arms"] if a["name"] == arm_name)
    # Fail before inference if the sandbox required by any case is unavailable.
    if any(j["case"].get("grader", {}).get("kind") == "python" for j in plan["jobs"]):
        subprocess.run(["docker", "image", "inspect", plan["suite"]["code_image"]],
                       check=True, stdout=subprocess.DEVNULL)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", {"plan": plan, "arm": arm_name, "started_unix": time.time(), "timeout_s": timeout})
    warmup = {"messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 32,
              "temperature": 0, "seed": 42, "stream": True, "stream_options": {"include_usage": True}}
    write_new(out / "warmup.json", request(arm, warmup, timeout))
    with (out / "results.jsonl").open("x") as f:
        for job in plan["jobs"]:
            result = request(arm, job["request"], timeout)
            score, detail = (None, "performance probe")
            if job["case"]["phase"] == "quality":
                score, detail = grade(job["case"]["grader"], result, plan["suite"]["code_image"])
            row = {"id": job["id"], "plan_sha256": plan["sha256"], "request_sha256": job["request_sha256"],
                   "arm": arm_name, "score": score, "grade_detail": detail, "result": result}
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            print(f"{arm_name} {job['id']}: {result['status']} score={score}", flush=True)


def interval(values, samples, confidence):
    rng = random.Random(42)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    tail = (1 - confidence) / 2
    return [means[int(tail * (samples - 1))], means[int((1 - tail) * (samples - 1))]]


def load_run(path, plan):
    manifest = read(Path(path) / "manifest.json")
    assert manifest["plan"]["sha256"] == plan["sha256"], "different frozen plans"
    rows = [json.loads(line) for line in (Path(path) / "results.jsonl").read_text().splitlines()]
    mapped = {r["id"]: r for r in rows}
    assert len(mapped) == len(rows), "duplicate observations"
    assert set(mapped) == {j["id"] for j in plan["jobs"]}, "incomplete run; missing observations cannot disappear"
    for job in plan["jobs"]:
        row = mapped[job["id"]]
        assert row["request_sha256"] == job["request_sha256"] and row["plan_sha256"] == plan["sha256"]
        assert row["arm"] == manifest["arm"]
        if job["case"]["phase"] == "quality":
            assert type(row["score"]) in (float, int) and 0 <= row["score"] <= 1
    return mapped


def compare(plan, control_path, candidate_path):
    control, candidate = load_run(control_path, plan), load_run(candidate_path, plan)
    quality = collections.defaultdict(list)
    performance = collections.defaultdict(list)
    for job in plan["jobs"]:
        a, b = control[job["id"]], candidate[job["id"]]
        (quality if job["case"]["phase"] == "quality" else performance)[job["case"]["tier"]].append((job, a, b))
    report = {"plan_sha256": plan["sha256"], "control_arm": next(iter(control.values()))["arm"],
              "candidate_arm": next(iter(candidate.values()))["arm"], "quality": {}, "performance": {},
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "Candidate minus control. Quality weights source/episode clusters equally. Per-tier intervals are exploratory, not multiplicity-adjusted. Identical seeds do not imply identical RNG across engines."}
    settings = plan["suite"]
    for tier, pairs in quality.items():
        clusters = collections.defaultdict(list)
        for job, a, b in pairs:
            clusters[job["case"]["cluster"]].append(b["score"] - a["score"])
        differences = [statistics.mean(v) for v in clusters.values()]
        ci = interval(differences, settings["bootstrap_samples"], settings["confidence"])
        margin = settings["margins"][tier]
        verdict = "inconclusive"
        degenerate = len(set(differences)) < 2
        if len(clusters) >= settings["min_clusters"] and not degenerate:
            if ci[0] > -margin:
                verdict = "non-inferior at registered margin (this tier only)"
            elif ci[1] < -margin:
                verdict = "inferior beyond registered margin"
        arms = {}
        for name, idx in (("control", 1), ("candidate", 2)):
            rows = [p[idx] for p in pairs]
            served = [r for r in rows if r["result"]["status"] == "ok" and r["result"].get("finish_reason") in ("stop", "tool_calls")]
            arms[name] = {"items": len(rows), "mean_score_all_items": statistics.mean(r["score"] for r in rows),
                          "completed": len(served), "mean_score_completed_only": statistics.mean(r["score"] for r in served) if served else None,
                          "statuses": dict(collections.Counter(r["result"]["status"] for r in rows)),
                          "finish_reasons": dict(collections.Counter(str(r["result"].get("finish_reason")) for r in rows))}
        report["quality"][tier] = {"arms": arms, "clusters": len(clusters), "cluster_mean_delta": statistics.mean(differences),
                                    "ci": ci, "degenerate_bootstrap": degenerate, "margin": margin, "verdict": verdict}
    for tier, pairs in performance.items():
        metrics = {}
        for metric in ("wall_s", "ttft_s", "e2e_output_tps", "server_decode_tps", "server_prefill_tps"):
            values = [(a["result"].get(metric), b["result"].get(metric)) for _, a, b in pairs
                      if a["result"]["status"] == b["result"]["status"] == "ok"]
            values = [(a, b) for a, b in values if positive_number(a) and positive_number(b)]
            metrics[metric] = {"paired_n": len(values), "control_median": statistics.median(a for a, _ in values) if values else None,
                               "candidate_median": statistics.median(b for _, b in values) if values else None,
                               "median_candidate_over_control": statistics.median(b / a for a, b in values) if values else None}
        report["performance"][tier] = {"attempted_pairs": len(pairs), "metrics": metrics,
            "control_statuses": dict(collections.Counter(a["result"]["status"] for _, a, _ in pairs)),
            "candidate_statuses": dict(collections.Counter(b["result"]["status"] for _, _, b in pairs)),
            "observations": {name: [{"id": row["id"], "finish_reason": row["result"].get("finish_reason"),
                                      "prompt_tokens": row["result"].get("usage", {}).get("prompt_tokens"),
                                      "completion_tokens": row["result"].get("usage", {}).get("completion_tokens")}
                                     for row in rows]
                             for name, rows in (("control", [a for _, a, _ in pairs]), ("candidate", [b for _, _, b in pairs]))}}
    return report


def summarize(plan, run_path):
    rows = load_run(run_path, plan)
    report = {"plan_sha256": plan["sha256"], "arm": next(iter(rows.values()))["arm"],
              "scope": "Single-arm observations; no quality-equivalence or speedup verdict.",
              "quality": {}, "performance": {}, "failures": []}
    groups = collections.defaultdict(list)
    for job in plan["jobs"]:
        row = rows[job["id"]]
        groups[(job["case"]["phase"], job["case"]["tier"])].append((job, row))
        if row["result"]["status"] != "ok" or (row["score"] is not None and row["score"] < 1):
            report["failures"].append({"id": row["id"], "score": row["score"],
                "status": row["result"]["status"], "finish_reason": row["result"].get("finish_reason"),
                "detail": row["grade_detail"], "error": row["result"].get("error")})
    for (phase, tier), pairs in sorted(groups.items()):
        results = [r["result"] for _, r in pairs]
        tokens = [r.get("usage", {}).get("prompt_tokens") for r in results]
        counts = [n for n in tokens if type(n) is int and n >= 0]
        value = {"attempted": len(pairs), "clusters": len({j["case"]["cluster"] for j, _ in pairs}),
                 "statuses": dict(collections.Counter(r["status"] for r in results)),
                 "finish_reasons": dict(collections.Counter(str(r.get("finish_reason")) for r in results)),
                 "prompt_tokens_observed": len(counts),
                 "prompt_tokens_min": min(counts) if counts else None,
                 "prompt_tokens_max": max(counts) if counts else None}
        if phase == "quality":
            complete = [r for r in results if r["status"] == "ok" and r.get("finish_reason") in ("stop", "tool_calls")]
            value.update(full_score=sum(r["score"] == 1 for _, r in pairs), completed=len(complete),
                         mean_score=statistics.mean(r["score"] for _, r in pairs))
        else:
            value["metrics"] = {}
            for metric in ("wall_s", "ttft_s", "e2e_output_tps", "server_decode_tps", "server_prefill_tps"):
                measurements = [r.get(metric) for r in results if r["status"] == "ok" and positive_number(r.get(metric))]
                value["metrics"][metric] = {"n": len(measurements),
                    "median": statistics.median(measurements) if measurements else None}
        report[phase][tier] = value
    return report


def summary_markdown(report):
    lines = [f"# Baseline: {report['arm']}", "", report["scope"], "",
             f"Frozen plan: `{report['plan_sha256']}`", "",
             "| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for tier, row in report["quality"].items():
        lines.append(f"| {tier} | {row['full_score']}/{row['attempted']} | {row['mean_score']:.3f} | {row['clusters']} | {row['prompt_tokens_min']}–{row['prompt_tokens_max']} |")
    lines += ["", "Mean scores include errors and truncations as zero. Clusters group related cases.", "",
              "| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for tier, row in report["performance"].items():
        cells = []
        for metric in ("ttft_s", "wall_s", "server_decode_tps", "server_prefill_tps"):
            value = row["metrics"][metric]["median"]
            cells.append(f"{value:.3f}" if value is not None else "n/a")
        lines.append(f"| {tier} | " + " | ".join(cells) + " |")
    lines += ["", "These are descriptive observations, not a controlled speed comparison.", "", "## Failed or partial-score observations", ""]
    if not report["failures"]:
        lines.append("None.")
    for row in report["failures"]:
        lines.append(f"- `{row['id']}`: score={row['score']}; status={row['status']}; finish={row['finish_reason']}.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--suite", required=True)
    freeze.add_argument("--arms", required=True)
    freeze.add_argument("--campaign", required=True)
    freeze.add_argument("--out", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--arm", required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--timeout", type=float, default=900)
    report = sub.add_parser("compare")
    report.add_argument("--plan", required=True)
    report.add_argument("--control", required=True)
    report.add_argument("--candidate", required=True)
    report.add_argument("--out", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--run", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--markdown")
    args = parser.parse_args()
    if args.command == "freeze":
        write_new(args.out, make_plan(read(args.suite), read(args.arms), args.campaign))
    elif args.command == "run":
        run(load_plan(args.plan), args.arm, args.out, args.timeout)
    elif args.command == "compare":
        write_new(args.out, compare(load_plan(args.plan), args.control, args.candidate))
    else:
        report = summarize(load_plan(args.plan), args.run)
        write_new(args.out, report)
        if args.markdown:
            with Path(args.markdown).open("x") as f:
                f.write(summary_markdown(report))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O: input and campaign validation must remain enabled.")
    main()
