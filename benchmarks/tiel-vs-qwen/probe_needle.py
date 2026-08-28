# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Correctness probe for the mamba prefix-cache + MTP read path (#47194 /
#43650 class) and tool-call leakage on cache hits.

Arms, all greedy:
  cold        unique doc, single read (control; no possible cache hit)
  hit-prefill warm the doc with max_tokens=1 (pure prefill, no MTP decode
              tail), then read on the hit
  hit-gen     warm the doc with max_tokens=384 (MTP decode writes the tail),
              then read on the hit  <- the poison arm if #43650 bites
  tools-hit   two-turn tool conversation; turn 2 rides the cache hit; check
              tool_calls parse vs <tool_call> text leaking into content

Needle recall: doc plants 8 key=value needles at spread depths; the read asks
for 3 of them. Reports per-arm recall plus prefix-cache hit counters.

Usage: probe_needle.py BASE MODEL REPS [OUT.json]
"""
import json, random, re, string, sys, httpx

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
REPS = int(sys.argv[3])
OUT = sys.argv[4] if len(sys.argv) > 4 else None

def make_doc(seed):
    rng = random.Random(seed)
    needles = {f"key_{seed}_{i}": "".join(rng.choices(string.ascii_lowercase, k=10))
               for i in range(8)}
    lines = [f"filler {seed} line {j}: " + " ".join(
             rng.choices(["alpha","beta","gamma","delta","omega","sigma"], k=9))
             for j in range(700)]
    ks = list(needles)
    for i, k in enumerate(ks):
        lines.insert(80 + i * 80, f"SECRET {k} = {needles[k]}")
    return "\n".join(lines), needles

def chat(client, msgs, max_tokens, tools=None):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}}
    if tools: body |= {"tools": tools, "tool_choice": "auto"}
    r = client.post(f"{BASE}/v1/chat/completions", json=body, timeout=900)
    j = r.json()
    if "choices" not in j: return {"err": str(j)[:150]}
    c = j["choices"][0]; m = c["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning") or "",
            "tool_calls": m.get("tool_calls") or [], "finish": c["finish_reason"],
            "pt": j["usage"]["prompt_tokens"]}

def metrics_hits(client):
    try:
        t = client.get(f"{BASE}/metrics", timeout=10).text
        vals = re.findall(r'prefix_cache.*hits_total[^ ]*}? ([0-9.e+]+)', t) or \
               re.findall(r'prefix_cache_hits[^ ]* ([0-9.e+]+)', t)
        return sum(float(x) for x in vals)
    except Exception:
        return -1

def ask_needles(doc, needles, which):
    ks = sorted(needles)[3:6] if which else sorted(needles)[:3]
    q = "From the document, output exactly these secret values, one per line, " \
        "as KEY=VALUE with no other text: " + ", ".join(ks)
    return q, {k: needles[k] for k in ks}

def score(reply, expect):
    body = reply.get("content", "") + "\n" + reply.get("reasoning", "")
    return sum(1 for k, v in expect.items() if v in body), len(expect)

TOOLS = [{"type": "function", "function": {"name": "get_secret",
          "description": "fetch a secret by key",
          "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                         "required": ["key"]}}}]

rows = []
with httpx.Client() as client:
    for rep in range(REPS):
        # cold control
        doc, nd = make_doc(f"cold{rep}")
        q, exp = ask_needles(doc, nd, 0)
        r = chat(client, [{"role": "user", "content": f"{doc}\n\n{q}"}], 256)
        got, tot = score(r, exp)
        rows.append({"arm": "cold", "rep": rep, "recall": f"{got}/{tot}", "ok": got == tot,
                     "finish": r.get("finish")})
        # hit-prefill: warm via max_tokens=1, then read
        doc, nd = make_doc(f"hp{rep}")
        h0 = metrics_hits(client)
        chat(client, [{"role": "user", "content": f"{doc}\n\nSay OK."}], 1)
        q, exp = ask_needles(doc, nd, 1)
        r = chat(client, [{"role": "user", "content": f"{doc}\n\n{q}"}], 256)
        h1 = metrics_hits(client)
        got, tot = score(r, exp)
        rows.append({"arm": "hit-prefill", "rep": rep, "recall": f"{got}/{tot}",
                     "ok": got == tot, "hits_moved": h1 > h0, "finish": r.get("finish")})
        # hit-gen: warm via 384-token generation (MTP tail), then read
        doc, nd = make_doc(f"hg{rep}")
        h0 = metrics_hits(client)
        chat(client, [{"role": "user", "content":
             f"{doc}\n\nSummarize the filler lines in 5 sentences."}], 384)
        q, exp = ask_needles(doc, nd, 1)
        r = chat(client, [{"role": "user", "content": f"{doc}\n\n{q}"}], 256)
        h1 = metrics_hits(client)
        got, tot = score(r, exp)
        rows.append({"arm": "hit-gen", "rep": rep, "recall": f"{got}/{tot}",
                     "ok": got == tot, "hits_moved": h1 > h0, "finish": r.get("finish")})
        # tools on a cache hit: turn 2 rides turn 1's prefix
        doc, nd = make_doc(f"tool{rep}")
        msgs = [{"role": "user", "content":
                 f"{doc}\n\nUse the get_secret tool to fetch key_tool{rep}_0."}]
        t1 = chat(client, msgs, 256, TOOLS)
        msgs += [{"role": "assistant", "content": t1.get("content") or "",
                  **({"tool_calls": t1["tool_calls"]} if t1.get("tool_calls") else {})}]
        if t1.get("tool_calls"):
            msgs += [{"role": "tool", "tool_call_id": t1["tool_calls"][0]["id"],
                      "content": "value-abc123"}]
        msgs += [{"role": "user", "content": "Now fetch key_beta with the tool."}]
        t2 = chat(client, msgs, 256, TOOLS)
        leak = "<tool_call>" in (t2.get("content") or "") or "<function=" in (t2.get("content") or "")
        rows.append({"arm": "tools-hit", "rep": rep,
                     "t1_calls": len(t1.get("tool_calls") or []),
                     "t2_calls": len(t2.get("tool_calls") or []),
                     "t2_leak_in_content": leak, "finish": t2.get("finish")})
        for row in rows[-4:]:
            print(json.dumps(row), flush=True)
if OUT: json.dump(rows, open(OUT, "w"), indent=2)
bad = [r for r in rows if r.get("ok") is False or r.get("t2_leak_in_content")]
print(f"defects: {len(bad)}")
