# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Probe defect A: finish_reason=length replies where content AND
reasoning_content come back empty despite thousands of generated tokens.

Sends a reasoning-heavy prompt at several max_tokens, non-streaming and
streaming, with and without tools, greedy. Records exactly what came back.

Usage: probe_truncation.py BASE MODEL [OUT.json]
"""
import json, sys, httpx

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

PROMPT = ("Prove or disprove: every positive integer can be written as the sum of "
          "three palindromic numbers in base 10. Reason carefully step by step, "
          "then give a final verdict and a worked example for 1000000.")
TOOLS = [{"type": "function", "function": {"name": "calc", "description": "evaluate math",
          "parameters": {"type": "object", "properties": {"expr": {"type": "string"}},
                         "required": ["expr"]}}}]

def nonstream(client, max_tokens, tools):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0}
    if tools: body |= {"tools": TOOLS, "tool_choice": "auto"}
    r = client.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    j = r.json()
    if "choices" not in j: return {"http": r.status_code, "error": j}
    c = j["choices"][0]
    m = c["message"]
    return {"mode": "nonstream", "max_tokens": max_tokens, "tools": bool(tools),
            "finish_reason": c.get("finish_reason"),
            "content_len": len(m.get("content") or ""),
            "reasoning_len": len(m.get("reasoning_content") or m.get("reasoning") or ""),
            "tool_calls": len(m.get("tool_calls") or []),
            "completion_tokens": j.get("usage", {}).get("completion_tokens"),
            "content_head": (m.get("content") or "")[:80],
            "reasoning_head": (m.get("reasoning_content") or m.get("reasoning") or "")[:80]}

def stream(client, max_tokens, tools):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools: body |= {"tools": TOOLS, "tool_choice": "auto"}
    content = reasoning = ""
    finish = None; usage = None
    with client.stream("POST", f"{BASE}/v1/chat/completions", json=body, timeout=600) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "): continue
            d = line[6:]
            if d.strip() == "[DONE]": break
            ch = json.loads(d)
            if ch.get("usage"): usage = ch["usage"]
            for c in ch.get("choices", []):
                dl = c.get("delta", {})
                content += dl.get("content") or ""
                reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
                finish = c.get("finish_reason") or finish
    return {"mode": "stream", "max_tokens": max_tokens, "tools": bool(tools),
            "finish_reason": finish, "content_len": len(content),
            "reasoning_len": len(reasoning),
            "completion_tokens": (usage or {}).get("completion_tokens")}

rows = []
with httpx.Client() as client:
    for mt in (64, 512, 2048):
        for tools in (False, True):
            for fn in (nonstream, stream):
                row = fn(client, mt, tools)
                rows.append(row)
                print(json.dumps(row), flush=True)
if OUT: json.dump(rows, open(OUT, "w"), indent=2)
