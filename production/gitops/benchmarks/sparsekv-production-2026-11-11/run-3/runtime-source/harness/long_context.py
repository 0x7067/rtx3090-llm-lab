"""Compare cache reuse and decode across two independent long conversations."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

VALIDATION = Path(__file__).resolve().parents[3] / "k8s/workloads/apps/llama/optimization/validation"
sys.path.insert(0, str(VALIDATION))
import harness


PADDING = "The routine status is unchanged.\n" * 8850
KEYS = (("PINE_7162", "LAKE_2954", "ROSE_8031"), ("FERN_3297", "COVE_4125", "LILY_9840"))
ANSWER_FORMAT = " Reply with only the key."


def conversation(label, keys, trial):
    start, middle, end = keys
    document = (
        f"Start key: {start}\n"
        f"{PADDING}"
        f"\nMiddle key: {middle}\n"
        f"{PADDING}"
        f"\nEnd key: {end}\n"
        f"\nWhat is the start key?{ANSWER_FORMAT}"
    )
    return [
        {"role": "system", "content": f"Trial {trial}, independent archive {label}. Answer from this archive."},
        {"role": "user", "content": document},
    ]


def follow(messages, answer, question):
    # Freeze the transmitted prefix so a bad answer in one arm does not
    # change the next request across arms.
    return [*messages, {"role": "assistant", "content": answer},
            {"role": "user", "content": question}]


def decode_counters(base, key):
    url = base.rstrip("/").removesuffix("/v1") + "/metrics"
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    names = (
        "vllm:request_time_per_output_token_seconds_count",
        "vllm:request_time_per_output_token_seconds_sum",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            lines = response.read().decode().splitlines()
        samples = [[float(line.rsplit(maxsplit=1)[1]) for line in lines
                    if line.startswith(name + "{")] for name in names]
        if all(samples):
            return tuple(sum(values) for values in samples)
    except (OSError, ValueError):
        pass
    return None


def run(base, model, output, trial, api_key_file=None):
    arm = {"base_url": base, "model": model}
    key = api_key_file.read_text().strip() if api_key_file else None
    if key:
        os.environ["SPARSEKV_BENCH_API_TOKEN"] = key
        arm["api_key_env"] = "SPARSEKV_BENCH_API_TOKEN"

    def request(name, messages, *, expected=None, gate=None, measure_decode=False):
        decode = expected is None
        body = {
            "messages": messages, "max_tokens": 256 if decode else 64,
            "temperature": 0, "top_p": 1, "seed": 42, "reasoning_effort": "none",
            "min_p": 0, "repetition_penalty": 1.0,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if gate:
            gate.wait(timeout=10)
        before = decode_counters(base, key) if measure_decode else None
        result = harness.request(arm, body, timeout=480)
        finished = time.monotonic()
        usage = result.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        if decode:
            answer_ok = tokens >= 240
            ttft = result.get("ttft_s")
            if tokens > 1 and ttft is not None and result["wall_s"] > ttft:
                result["client_post_ttft_tps"] = (tokens - 1) / (result["wall_s"] - ttft)
        if before is not None and result["status"] == "ok" and tokens > 1:
            for _ in range(8):
                after = decode_counters(base, key)
                if after is None or after[0] > before[0]:
                    break
                time.sleep(.25)
            if after is not None:
                count, duration = (end - start for start, end in zip(before, after))
                result["server_tpot_count_delta"] = count
                if count == 1 and duration > 0:
                    result["server_decode_tps"] = 1 / duration
                    result["server_decode_method"] = "inverse vLLM request TPOT histogram delta"
        if not decode:
            answer_ok = result.get("content", "").strip() == expected and result.get("finish_reason") == "stop"
        passed = (result["status"] == "ok" and answer_ok
                  and 118_000 <= usage.get("prompt_tokens", 0) < 131_072
                  and result.get("ttft_s") is not None)
        return {
            "name": name, "passed": passed, "expected": expected, "request_sha256": harness.digest(body),
            "result": result,
        }, finished

    a, b = (conversation(label, keys, trial) for label, keys in zip("AB", KEYS))
    a_warm = follow(a, KEYS[0][0], "What is the middle key?" + ANSWER_FORMAT)
    a_return = follow(a_warm, KEYS[0][1], "What is the end key?" + ANSWER_FORMAT)
    explain = (
        "Write at least 600 words explaining how to design a reliable job scheduler with retries, "
        "idempotency, and recovery. Give concrete examples and pseudocode."
    )
    a_solo_decode = follow(a_warm, KEYS[0][1], explain)
    a_decode = follow(a_return, KEYS[0][2], explain)
    b_decode = follow(b, KEYS[1][0], explain)

    rows = []
    with (output / "long-context.jsonl").open("x") as raw:
        def record(row):
            rows.append(row)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            result = row["result"]
            print(json.dumps({
                "name": row["name"], "passed": row["passed"], "status": result["status"],
                "wall_s": result.get("wall_s"), "ttft_s": result.get("ttft_s"), "usage": result.get("usage"),
                "server_decode_tps": result.get("server_decode_tps"),
                "error": result.get("error"),
            }), flush=True)

        for name, messages, expected in (
            ("a-cold", a, KEYS[0][0]),
            ("a-follow-up", a_warm, KEYS[0][1]),
            ("a-warm-solo-decode", a_solo_decode, None),
            ("b-cold", b, KEYS[1][0]),
            ("a-return-after-b", a_return, KEYS[0][2]),
        ):
            row, _ = request(name, messages, expected=expected, measure_decode=expected is None)
            record(row)
            if row["result"]["status"] != "ok":
                raise RuntimeError(f"{name} transport failed, see {raw.name}")

        gate = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(request, name, messages, gate=gate)
                for name, messages in (("a-concurrent-decode", a_decode), ("b-concurrent-decode", b_decode))
            ]
            started = time.monotonic()
            gate.wait(timeout=10)
            completed = []
            for job in as_completed(jobs):
                row, finished = job.result()
                completed.append(finished)
                record(row)
            elapsed = max(completed) - started

    parallel = rows[-2:]
    complete = all(row["result"]["status"] == "ok" for row in parallel)
    completion_tokens = sum(row["result"].get("usage", {}).get("completion_tokens", 0) for row in parallel)
    summary = {
        "passed": sum(row["passed"] for row in rows), "total": len(rows),
        "parallel_wall_s": elapsed, "parallel_completion_tokens": completion_tokens,
        "parallel_e2e_output_tps": completion_tokens / elapsed if complete else None,
        "requests": [
            {
                "name": row["name"], "passed": row["passed"],
                **{field: row["result"].get(field) for field in ("ttft_s", "wall_s", "e2e_output_tps", "server_decode_tps", "client_post_ttft_tps", "usage")},
            }
            for row in rows
        ],
    }
    harness.write_new(output / "long-context-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] == summary["total"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial", required=True, help="Same campaign ID for both arms, different for each rerun")
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return run(args.base, args.model, args.out, args.trial, args.api_key_file)


if __name__ == "__main__":
    raise SystemExit(main())
