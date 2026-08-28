# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Decode-rate sampler for MTP tuning: N samples across mixed content classes,
plus spec-decode acceptance read from /metrics before/after.

Usage: bench_decode_n.py BASE MODEL N [OUT.json]

qwen3.8's decode is bimodal because MTP acceptance depends on content, so this
samples code-gen, prose, and code-with-context prompts separately and reports
per-class medians. Sampling matches production (temp 1.0 top-k 20 top-p 0.95)
unless GREEDY=1.
"""
import asyncio, json, os, random, re, statistics, string, sys, time
import httpx

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
N = int(sys.argv[3])
OUT = sys.argv[4] if len(sys.argv) > 4 else None
GREEDY = os.environ.get("GREEDY") == "1"

def nonce():
    return "s-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=20))

PROMPTS = {
    "code": "Write a Python function that parses an nginx access log line into a dict, "
            "with a regex, type conversion for status and bytes, and a docstring with an example. "
            "Then explain each regex group.",
    "prose": "Explain, for a new team member, how TCP congestion control works: slow start, "
             "congestion avoidance, fast retransmit, and fast recovery. Use concrete numbers.",
    "repair": "This function should merge overlapping intervals but fails on touching intervals:\n"
              "```python\ndef merge(iv):\n    iv.sort()\n    out=[iv[0]]\n    for s,e in iv[1:]:\n"
              "        if s>out[-1][1]:\n            out.append([s,e])\n        else:\n"
              "            out[-1][1]=max(out[-1][1],e)\n    return out\n```\n"
              "Find the bug for input [[1,3],[3,5]], fix it, and add tests.",
}

async def one(client, content, max_tokens=512):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{nonce()}] {content}"}],
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    body |= {"temperature": 0} if GREEDY else {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    t0 = time.perf_counter(); ttft = None; usage = None
    async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "): continue
            d = line[6:]
            if d.strip() == "[DONE]": break
            ch = json.loads(d)
            if ch.get("usage"): usage = ch["usage"]
            for c in ch.get("choices", []):
                dl = c.get("delta", {})
                if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    ct = usage["completion_tokens"]
    return {"gen_tok_s": round((ct - 1) / (total - ttft), 2), "completion_tokens": ct,
            "ttft_s": round(ttft, 3)}

async def metrics(client):
    try:
        r = await client.get(f"{BASE}/metrics", timeout=10)
        vals = {}
        for key in ("num_accepted_tokens", "num_draft_tokens", "num_drafts"):
            m = re.findall(rf'{key}_total{{[^}}]*}} ([0-9.e+]+)', r.text) or \
                re.findall(rf'{key}{{[^}}]*}} ([0-9.e+]+)', r.text)
            if m: vals[key] = sum(float(x) for x in m)
        return vals
    except Exception:
        return {}

async def main():
    results = {"base": BASE, "n_per_class": N, "greedy": GREEDY}
    async with httpx.AsyncClient() as client:
        await one(client, "say hi", 8)  # warmup
        m0 = await metrics(client)
        for cls, prompt in PROMPTS.items():
            runs = [await one(client, prompt) for _ in range(N)]
            rates = sorted(r["gen_tok_s"] for r in runs)
            results[cls] = {"runs": runs, "median": statistics.median(rates),
                            "min": rates[0], "max": rates[-1]}
            print(f"{cls}: median={results[cls]['median']} range=[{rates[0]},{rates[-1]}]", flush=True)
        m1 = await metrics(client)
        if m0 and m1 and "num_draft_tokens" in m1:
            acc = (m1.get("num_accepted_tokens", 0) - m0.get("num_accepted_tokens", 0))
            drf = (m1.get("num_draft_tokens", 0) - m0.get("num_draft_tokens", 0))
            if drf > 0:
                results["draft_acceptance"] = round(acc / drf, 4)
                print(f"draft acceptance: {results['draft_acceptance']}", flush=True)
        allr = sorted(r["gen_tok_s"] for cls in PROMPTS for r in results[cls]["runs"])
        results["overall_median"] = statistics.median(allr)
        print(f"overall median: {results['overall_median']}", flush=True)
    if OUT: json.dump(results, open(OUT, "w"), indent=2)

asyncio.run(main())
