# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Speed benchmark against an OpenAI-compatible /v1/chat/completions endpoint.

Measures, with prefix caching defeated by a unique random preamble per request:
  A. single-stream decode: TTFT + generation tok/s (3 runs, 512 tokens)
  B. prefill: ~8k-token prompt, 16 output tokens -> prompt tok/s (3 runs)
  C. concurrency 4: aggregate generation tok/s (256 tokens each)
"""
import asyncio, json, random, string, sys, time
import httpx

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

def nonce() -> str:
    return "session-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=24))

# ~8k tokens of plausible code-review context for the prefill test
CODE_CHUNK = open(__file__).read()

async def one_stream(client, messages, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    usage = None
    async with client.stream(
        "POST", f"{BASE}/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens,
              "temperature": 0, "stream": True,
              "stream_options": {"include_usage": True}},
        timeout=600,
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                delta = c.get("delta", {})
                if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"ttft_s": ttft, "total_s": total, "usage": usage}

async def main():
    results = {"base_url": BASE, "model": MODEL}
    async with httpx.AsyncClient() as client:
        # warmup
        await one_stream(client, [{"role": "user", "content": f"{nonce()}: say hi"}], 8)

        # A. single-stream decode
        decode_runs = []
        for _ in range(3):
            msgs = [{"role": "user", "content":
                     f"[{nonce()}] Write a Python function that parses an nginx access log line "
                     "into a dict, with a regex, type conversion for status and bytes, and a "
                     "docstring with an example. Then explain each regex group."}]
            r = await one_stream(client, msgs, 512)
            ct = r["usage"]["completion_tokens"]
            gen_s = r["total_s"] - r["ttft_s"]
            decode_runs.append({"ttft_s": round(r["ttft_s"], 3),
                                "completion_tokens": ct,
                                "gen_tok_s": round((ct - 1) / gen_s, 2)})
            print(f"decode: ttft={r['ttft_s']:.3f}s gen={decode_runs[-1]['gen_tok_s']} tok/s", flush=True)
        results["single_stream_decode"] = decode_runs

        # B. prefill throughput (~8k tokens, unique prefix so nothing is cached)
        prefill_runs = []
        big = (CODE_CHUNK * 8)[:24000]  # ~8k tokens of code text
        for _ in range(3):
            msgs = [{"role": "user", "content":
                     f"[{nonce()}]\nHere is a code listing:\n```python\n{big}\n```\nReply with just: OK"}]
            r = await one_stream(client, msgs, 16)
            pt = r["usage"]["prompt_tokens"]
            prefill_runs.append({"prompt_tokens": pt, "ttft_s": round(r["ttft_s"], 3),
                                 "prefill_tok_s": round(pt / r["ttft_s"], 1)})
            print(f"prefill: {pt} tok in {r['ttft_s']:.2f}s = {prefill_runs[-1]['prefill_tok_s']} tok/s", flush=True)
        results["prefill"] = prefill_runs

        # C. concurrency 4
        async def worker(i):
            msgs = [{"role": "user", "content":
                     f"[{nonce()}] Implement task {i}: a thread-safe LRU cache class in Python "
                     "with get/put and maxsize, plus a short usage example."}]
            return await one_stream(client, msgs, 256)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[worker(i) for i in range(4)])
        wall = time.perf_counter() - t0
        toks = sum(r["usage"]["completion_tokens"] for r in rs)
        results["concurrent_4"] = {"wall_s": round(wall, 2), "completion_tokens": toks,
                                   "aggregate_gen_tok_s": round(toks / wall, 2)}
        print(f"concurrent x4: {toks} tok in {wall:.1f}s = {results['concurrent_4']['aggregate_gen_tok_s']} tok/s", flush=True)

    if OUT:
        json.dump(results, open(OUT, "w"), indent=2)
    print(json.dumps(results, indent=2))

asyncio.run(main())
