# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pillow"]
# ///
"""Adversarial stress suite for the qwen3.8 vLLM deployment (bench container).

Phases target the stack's known seams:
  churn      requests finishing while others are mid-generation (the MTP k=4
             crash signature, run hard at k=3), 3x max_num_seqs oversubscribed
  poolstorm  8 concurrent ~40k-token prompts: sum >> KV pool, forces
             preemption + CPU-offload traffic
  aborts     streaming clients that disconnect mid-generation, repeatedly,
             while other requests run
  jsonstorm  concurrent structured-output (json_schema) requests, the path
             that historically livelocked EngineCore, mixed with tools
  edges      >max_model_len prompt (expect clean 400), multi-image (expect
             400), min_p with spec decode, empty/degenerate inputs,
             near-limit ~60k prompt
  visionmix  large image requests concurrent with deep text prompts
  soak       6 minutes of mixed traffic at sustained concurrency

Every phase records errors and is followed (in the driver shell) by health,
log, and VRAM checks. Success = no 5xx storms, no EngineDead, no livelock
(token counters must keep moving), server healthy at the end.

Usage: stress_qwen.py BASE MODEL [OUT.json]
"""
import asyncio, base64, io, json, random, re, string, sys, time
import httpx

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
R = random.Random(7)
report = {}

def nonce(n=16):
    return "".join(R.choices(string.ascii_lowercase + string.digits, k=n))

def doc(tokens_approx, seed):
    lines = [f"[{seed}] rec {i}: v={i*7%9973} t={'ab'[i%2]}{i%97}" for i in range(int(tokens_approx/11))]
    return "\n".join(lines)

def big_image_b64(px_w=3000, px_h=2000):
    from PIL import Image
    img = Image.frombytes("RGB", (px_w, px_h), bytes(R.getrandbits(8) for _ in range(px_w*px_h*3)))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

async def req(client, msgs, max_tokens=128, effort="low", stream=False, abort_after=None,
              tools=None, schema=None, extra=None, timeout=900):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 1.0, "top_k": 20, "top_p": 0.95,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    if tools: body |= {"tools": tools, "tool_choice": "auto"}
    if schema: body |= {"response_format": {"type": "json_schema",
                        "json_schema": {"name": "s", "schema": schema}}}
    if extra: body |= extra
    t0 = time.perf_counter()
    try:
        if stream:
            body["stream"] = True
            n = 0
            async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body,
                                     timeout=timeout) as r:
                if r.status_code != 200:
                    return {"status": r.status_code, "s": round(time.perf_counter()-t0, 2)}
                async for line in r.aiter_lines():
                    n += 1
                    if abort_after and n >= abort_after:
                        return {"status": "aborted", "chunks": n,
                                "s": round(time.perf_counter()-t0, 2)}
            return {"status": 200, "chunks": n, "s": round(time.perf_counter()-t0, 2)}
        r = await client.post(f"{BASE}/v1/chat/completions", json=body, timeout=timeout)
        out = {"status": r.status_code, "s": round(time.perf_counter()-t0, 2)}
        if r.status_code == 200:
            j = r.json(); c = j["choices"][0]
            out |= {"finish": c["finish_reason"],
                    "clen": len(c["message"].get("content") or ""),
                    "ct": j["usage"]["completion_tokens"]}
        else:
            out["body"] = r.text[:120]
        return out
    except Exception as e:
        return {"status": f"EXC:{type(e).__name__}", "err": str(e)[:100],
                "s": round(time.perf_counter()-t0, 2)}

async def gen_tokens_total(client):
    try:
        t = (await client.get(f"{BASE}/metrics", timeout=10)).text
        m = re.findall(r'generation_tokens_total[^ ]* ([0-9.e+]+)', t)
        return sum(float(x) for x in m)
    except Exception:
        return -1

def tally(name, results):
    from collections import Counter
    c = Counter(str(r.get("status")) for r in results)
    errs = [r for r in results if r.get("status") not in (200, "aborted")]
    report[name] = {"statuses": dict(c), "n": len(results),
                    "errors_sample": errs[:5],
                    "max_s": max((r["s"] for r in results), default=0)}
    print(f"== {name}: {dict(c)} max_s={report[name]['max_s']}", flush=True)

async def phase_churn(client):
    # waves of mixed-length requests so completions constantly overlap
    # mid-generation neighbors; 24 in flight against max_num_seqs=8
    async def one(i):
        ln = R.choice([16, 48, 160, 400])
        return await req(client, [{"role": "user", "content":
            f"[{nonce()}] Count from 1 upward, one number per line."}], ln)
    results = []
    for wave in range(4):
        rs = await asyncio.gather(*[one(i) for i in range(24)])
        results += rs
    tally("churn", results)

async def phase_poolstorm(client):
    async def one(i):
        return await req(client, [{"role": "user", "content":
            f"{doc(40000, nonce())}\nSummarize in one sentence."}], 48, timeout=1200)
    results = await asyncio.gather(*[one(i) for i in range(8)])
    tally("poolstorm", results)

async def phase_aborts(client):
    async def victim(i):
        return await req(client, [{"role": "user", "content":
            f"[{nonce()}] Write a long essay about rivers."}], 2048,
            stream=True, abort_after=R.randint(3, 30))
    async def bystander(i):
        return await req(client, [{"role": "user", "content":
            f"[{nonce()}] What is 17*23? Answer briefly."}], 64)
    results = []
    for wave in range(6):
        rs = await asyncio.gather(*[victim(i) for i in range(6)],
                                  *[bystander(i) for i in range(4)])
        results += rs
    tally("aborts", results)

async def phase_jsonstorm(client):
    schema = {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "score": {"type": "number"}},
            "required": ["name", "score"], "additionalProperties": False}},
        "verdict": {"type": "string", "enum": ["good", "bad", "mixed"]}},
        "required": ["items", "verdict"], "additionalProperties": False}
    tools = [{"type": "function", "function": {"name": "lookup",
              "description": "look up a fact",
              "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                             "required": ["q"]}}}]
    g0 = await gen_tokens_total(client)
    async def js(i):
        return await req(client, [{"role": "user", "content":
            f"[{nonce()}] Rate these fruits: apple, kiwi, mango. JSON only."}],
            512, schema=schema, effort=R.choice(["low", "medium"]))
    async def tl(i):
        return await req(client, [{"role": "user", "content":
            f"[{nonce()}] Use the lookup tool to find the boiling point of lead."}],
            256, tools=tools)
    results = []
    for wave in range(4):
        rs = await asyncio.gather(*[js(i) for i in range(6)], *[tl(i) for i in range(2)])
        results += rs
    g1 = await gen_tokens_total(client)
    tally("jsonstorm", results)
    report["jsonstorm"]["gen_tokens_moved"] = g1 > g0
    ok_json = sum(1 for r in results if r.get("status") == 200)
    report["jsonstorm"]["http200"] = ok_json

async def phase_edges(client):
    results = []
    # over max_model_len: ~150k tokens vs 140k limit -> expect 400, not a crash
    results.append(await req(client, [{"role": "user", "content": doc(150000, "over")}],
                             16, timeout=1800))
    results[-1]["case"] = "over-limit-prompt"
    # near-limit prompt ~60k: must succeed
    results.append(await req(client, [{"role": "user", "content":
        f"{doc(60000, 'near')}\nSay OK."}], 16, timeout=1800))
    results[-1]["case"] = "near-limit-60k"
    # multi-image -> expect 400 per the count:1 cap
    img = big_image_b64(800, 600)
    results.append(await req(client, [{"role": "user", "content": [
        {"type": "text", "text": "Compare these."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}], 64))
    results[-1]["case"] = "two-images"
    # min_p with spec decode (documented as won't work -- must not crash)
    results.append(await req(client, [{"role": "user", "content": "hi"}], 32,
                             extra={"min_p": 0.1}))
    results[-1]["case"] = "min_p"
    # degenerate inputs
    results.append(await req(client, [{"role": "user", "content": ""}], 16))
    results[-1]["case"] = "empty-content"
    results.append(await req(client, [{"role": "user", "content": "x"}], 1))
    results[-1]["case"] = "max_tokens-1"
    results.append(await req(client, [{"role": "system", "content": "s"}], 16))
    results[-1]["case"] = "system-only"
    for r in results: print(json.dumps(r), flush=True)
    tally("edges", results)
    report["edges"]["cases"] = results

async def phase_visionmix(client):
    img = big_image_b64(3000, 2000)
    async def vision(i):
        return await req(client, [{"role": "user", "content": [
            {"type": "text", "text": f"[{nonce()}] Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}],
            96, timeout=1200)
    async def deep(i):
        return await req(client, [{"role": "user", "content":
            f"{doc(30000, nonce())}\nSummarize in one sentence."}], 48, timeout=1200)
    results = []
    for wave in range(2):
        rs = await asyncio.gather(*[vision(i) for i in range(3)], *[deep(i) for i in range(3)])
        results += rs
    tally("visionmix", results)

async def phase_soak(client):
    end = time.time() + 360
    results = []
    tools = [{"type": "function", "function": {"name": "calc", "description": "math",
              "parameters": {"type": "object", "properties": {"e": {"type": "string"}},
                             "required": ["e"]}}}]
    async def worker(w):
        while time.time() < end:
            kind = R.random()
            if kind < 0.4:
                r = await req(client, [{"role": "user", "content":
                    f"[{nonce()}] Explain {R.choice(['DNS','TLS','RAID','NAT'])} briefly."}], 192)
            elif kind < 0.6:
                r = await req(client, [{"role": "user", "content":
                    f"{doc(15000, nonce())}\nOne-line summary."}], 48, timeout=1200)
            elif kind < 0.8:
                r = await req(client, [{"role": "user", "content":
                    f"[{nonce()}] Use calc for 91*7."}], 128, tools=tools)
            else:
                r = await req(client, [{"role": "user", "content":
                    f"[{nonce()}] Long story please."}], 1024, stream=True,
                    abort_after=R.choice([None, 10, 40]))
            results.append(r)
    await asyncio.gather(*[worker(w) for w in range(6)])
    tally("soak", results)

async def main():
    async with httpx.AsyncClient() as client:
        await req(client, [{"role": "user", "content": "warmup"}], 8)
        import os
        want = os.environ.get("PHASES", "").split(",") if os.environ.get("PHASES") else None
        for name, fn in [("churn", phase_churn), ("poolstorm", phase_poolstorm),
                         ("aborts", phase_aborts), ("jsonstorm", phase_jsonstorm),
                         ("edges", phase_edges), ("visionmix", phase_visionmix),
                         ("soak", phase_soak)]:
            if want and name not in want:
                continue
            print(f"### phase {name}", flush=True)
            await fn(client)
            h = await client.get(f"{BASE}/health", timeout=15)
            print(f"health after {name}: {h.status_code}", flush=True)
            report.setdefault("health", {})[name] = h.status_code
    if OUT: json.dump(report, open(OUT, "w"), indent=2)
    print("STRESS_DONE", flush=True)

asyncio.run(main())
