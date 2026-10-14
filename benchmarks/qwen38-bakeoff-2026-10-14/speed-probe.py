#!/usr/bin/env python3
"""Speed probes for one OpenAI-compatible arm (stdlib only).

Probes, each with a unique nonce at the start of the prompt so no prefix cache
can serve it:
  short-prose / short-code   single stream, ~300 prompt tokens, 512 output tokens
  ctx-64k / ctx-120k          single stream, filler document then a short task
  concurrent-4                four short-prose requests at once, aggregate rate

Prefill rate = prompt_tokens / TTFT. Decode rate = (completion_tokens - 1) /
(last token time - first token time). Token counts come from server usage;
probes whose usage is missing are recorded as such, never estimated.
"""
import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.request

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
MAX_TOKENS = 512

PROSE_TASK = ("Write a detailed, multi-paragraph explanation of how a city water "
              "distribution network keeps pressure stable through the day. Do not stop early.")
CODE_TASK = ("Write a complete Python module implementing a thread-safe LRU cache with TTL "
             "expiry, type hints, docstrings and a small unittest suite. Output only code.")
DOC_TASK = ("Above is a build log. Summarise the overall pattern of events in several "
            "detailed paragraphs. Do not stop early.")

WORDS = ("alpha bravo cedar delta ember fjord granite harbor island juniper kettle lantern "
         "meadow nectar orchid pepper quartz river saddle timber umber velvet willow xenon "
         "yarrow zephyr").split()


def filler(approx_tokens, seed):
    rng = random.Random(seed)
    lines, n = [], 0
    # ~23 Qwen3.8 tokens per line (measured with the model tokenizer); actual count comes from usage.
    while n < approx_tokens:
        i = len(lines)
        lines.append(f"Event {i}: node {rng.choice(WORDS)}-{rng.randint(0, 999)} reported "
                     f"{rng.choice(WORDS)} {rng.choice(WORDS)} status {rng.randint(0, 99)}.")
        n += 23
    return "\n".join(lines)


def stream(base_url, model, messages, timeout):
    body = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "low", **SAMPLING}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    chunks = 0
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                        chunks += 1
    except Exception as e:  # recorded, not retried
        err = f"{type(e).__name__}: {e}"
    end = time.monotonic()
    out = {"error": err, "wall_s": round(end - t0, 3), "chunks": chunks, "usage": usage}
    if first:
        out["ttft_s"] = round(first - t0, 3)
    if usage and first:
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        out["prompt_tokens"], out["completion_tokens"] = pt, ct
        if pt and first - t0 > 0:
            out["prefill_tok_s"] = round(pt / (first - t0), 1)
        if ct and ct > 1 and last > first:
            out["decode_tok_s"] = round((ct - 1) / (last - first), 2)
        if ct:
            out["e2e_tok_s"] = round(ct / (end - t0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-ctx", type=int, required=True, help="arm context window")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    nonce = lambda tag: f"[probe {a.arm} {tag} {time.time_ns()}]\n"
    results = []

    def run(tag, messages):
        r = {"probe": tag, **stream(a.base_url, a.model, messages, a.timeout)}
        print(json.dumps(r), flush=True)
        results.append(r)

    # Excluded warmup.
    stream(a.base_url, a.model, [{"role": "user", "content": nonce("warmup") + "Say hi."}], a.timeout)
    run("short-prose", [{"role": "user", "content": nonce("sp") + PROSE_TASK}])
    run("short-code", [{"role": "user", "content": nonce("sc") + CODE_TASK}])
    for tag, target in (("ctx-64k", 64000), ("ctx-120k", 120000)):
        if target + MAX_TOKENS + 2048 > a.max_ctx:
            results.append({"probe": tag, "skipped": f"arm context {a.max_ctx} too small"})
            continue
        run(tag, [{"role": "user", "content": nonce(tag) + filler(target, target) + "\n\n" + DOC_TASK}])

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(stream, a.base_url, a.model,
                          [{"role": "user", "content": nonce(f"c4-{i}") + PROSE_TASK}], a.timeout)
                for i in range(4)]
        conc = [f.result() for f in futs]
    wall = time.monotonic() - t0
    total = sum((c.get("completion_tokens") or 0) for c in conc)
    results.append({"probe": "concurrent-4", "wall_s": round(wall, 3), "completion_tokens": total,
                    "aggregate_tok_s": round(total / wall, 2) if total else None,
                    "errors": [c["error"] for c in conc if c["error"]], "streams": conc})
    print(json.dumps(results[-1])[:400], flush=True)

    with open(a.out, "w") as f:
        json.dump({"arm": a.arm, "sampling": SAMPLING, "max_tokens": MAX_TOKENS,
                   "reasoning_effort": "low", "results": results}, f, indent=2)
    return 1 if any(r.get("error") for r in results if "error" in r) else 0


if __name__ == "__main__":
    sys.exit(main())
