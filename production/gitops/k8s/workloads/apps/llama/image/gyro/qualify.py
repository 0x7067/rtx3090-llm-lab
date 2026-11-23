"""Functional release checks; raw responses stay in the local report directory."""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-flash-next-gyro-m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    token = Path(a.key_file).read_text().strip() if a.key_file else ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    def request(path, body=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(a.base + path, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                if body and body.get("stream"):
                    chunks, usage, done = [], {}, False
                    for line in r:
                        line = line.decode().strip()
                        if line == "data: [DONE]":
                            done = True
                            break
                        if line.startswith("data: "):
                            event = json.loads(line[6:])
                            for choice in event.get("choices", []):
                                chunks.append(choice.get("delta", {}).get("content") or "")
                            usage = event.get("usage") or usage
                    data = {"content": "".join(chunks), "usage": usage, "done": done}
                else:
                    data = json.load(r)
                return r.status, data, round(time.monotonic() - start, 3)
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode()[:1000]}, round(time.monotonic() - start, 3)

    def check(name, messages, expected=None, **extra):
        body = {"model": MODEL, "messages": messages, "max_tokens": 2048,
                "temperature": 0, "seed": 42, **extra}
        status, data, seconds = request("/chat/completions", body)
        text = data.get("content") if body.get("stream") else data.get("choices", [{}])[0].get("message", {}).get("content")
        passed = status == 200 and (expected is None or (text or "").strip() == expected)
        if body.get("stream"):
            passed = passed and data.get("done") is True
        report[name] = {"passed": bool(passed), "status": status, "seconds": seconds,
                        "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                        "response": data}
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"check": name, "passed": bool(passed), "seconds": seconds}), flush=True)
        return data

    status, data, _ = request("/models")
    report["catalog"] = {"passed": status == 200 and MODEL in [x["id"] for x in data.get("data", [])]}
    check("streaming_arithmetic", [{"role": "user", "content": "Return only the result of 17+25."}], "42",
          stream=True, stream_options={"include_usage": True})
    check("reasoning", [{"role": "user", "content": "A box has 7 rows of 13 tiles. Remove 19 tiles and add 8. Return only the remaining tile count."}], "80")
    check("code", [{"role": "user", "content": "Return only Python code, no markdown: define unique(xs) that removes duplicates while preserving first occurrence order, for hashable values. Use exactly two lines: def unique(xs): then four spaces and return list(dict.fromkeys(xs))."}], "def unique(xs):\n    return list(dict.fromkeys(xs))")
    messages = [{"role": "user", "content": "Call report_ready with tag GYRO_READY. After its result, reply exactly GYRO_TOOL_DONE."}]
    tools = [{"type": "function", "function": {"name": "report_ready", "description": "Report readiness",
              "parameters": {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}}]
    data = check("tool_call", messages, tools=tools, tool_choice="auto")
    msg = data.get("choices", [{}])[0].get("message", {})
    calls = msg.get("tool_calls") or []
    good = len(calls) == 1 and calls[0]["function"]["name"] == "report_ready" and json.loads(calls[0]["function"]["arguments"]) == {"tag": "GYRO_READY"}
    report["tool_call"]["passed"] &= good
    if good:
        check("tool_followup", messages + [msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"ready":true}'}],
              "GYRO_TOOL_DONE", tools=tools, tool_choice="none")
    if a.long:
        # Count the actual rendered prompt before running the near-limit request.
        def prompt(n):
            filler = "The routine status is unchanged.\n" * n
            return "Start key: PINE_7162\n" + filler + "\nMiddle key: LAKE_2954\n" + filler + "\nEnd key: ROSE_8031\nReturn the start, middle, and end keys separated by commas, with no spaces."
        lo, hi = 1, 20000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            status, count, _ = request("/messages/count_tokens", {"model": MODEL, "messages": [{"role": "user", "content": prompt(mid)}]})
            if status != 200:
                raise RuntimeError("Token-count request failed: " + str(status))
            if count["input_tokens"] <= 190000:
                lo = mid
            else:
                hi = mid - 1
        messages = [{"role": "user", "content": prompt(lo)}]
        check("long_context", messages, "PINE_7162,LAKE_2954,ROSE_8031", stream=True, stream_options={"include_usage": True})
        messages += [{"role": "assistant", "content": "PINE_7162,LAKE_2954,ROSE_8031"},
                     {"role": "user", "content": "Return only the middle key again."}]
        data = check("cached_followup", messages, "LAKE_2954", stream=True, stream_options={"include_usage": True})
        cached = data.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        report["cached_followup"]["cached_tokens"] = cached
        report["cached_followup"]["passed"] &= cached > 0
    passed = all(x["passed"] for x in report.values()) and "tool_followup" in report
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
