# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Probe defect B: empty reply with finish_reason=stop on a prefix-cache hit at
a specific templated-prompt-length residue (mod 128) — the bug class the
launcher documents for the KVarN path, tested here on the production path.

Builds a fixed ~2k-token document, then grows the tail one filler word at a
time. Each length is sent twice greedily (first arms the prefix cache, second
must hit it); the second reply's emptiness is the signal. usage.prompt_tokens
records the exact templated length, so residues are known exactly.

Usage: probe_residue.py BASE MODEL STEPS [OUT.json]   (STEPS≈140 covers all 128)
"""
import json, sys, httpx

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
STEPS = int(sys.argv[3])
OUT = sys.argv[4] if len(sys.argv) > 4 else None

DOC = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog number {i}."
                for i in range(220))  # ~2k tokens, stable prefix

rows = []
with httpx.Client() as client:
    for i in range(STEPS):
        tail = " alpha" * i
        msgs = [{"role": "user", "content":
                 f"Document:\n{DOC}\nPadding:{tail}\nReply with exactly: ACK {i}"}]
        body = {"model": MODEL, "messages": msgs, "max_tokens": 24, "temperature": 0}
        out = []
        for attempt in ("arm", "hit"):
            r = client.post(f"{BASE}/v1/chat/completions", json=body, timeout=120)
            j = r.json()
            if "choices" not in j:
                out.append({"attempt": attempt, "http": r.status_code}); continue
            c = j["choices"][0]
            out.append({"attempt": attempt,
                        "prompt_tokens": j["usage"]["prompt_tokens"],
                        "completion_tokens": j["usage"]["completion_tokens"],
                        "finish_reason": c["finish_reason"],
                        "content_len": len(c["message"].get("content") or ""),
                        "reasoning_len": len( (c["message"].get("reasoning") or c["message"].get("reasoning_content")) or "")})
        pt = out[-1].get("prompt_tokens")
        row = {"step": i, "residue": pt % 128 if pt else None, "arm": out[0], "hit": out[1]}
        rows.append(row)
        empty = out[1].get("content_len", 1) == 0 and out[1].get("reasoning_len", 1) == 0
        flag = "  <-- EMPTY ON HIT" if empty and out[1].get("finish_reason") == "stop" else ""
        print(f"step {i:3d} residue {row['residue']} hit: fr={out[1].get('finish_reason')} "
              f"clen={out[1].get('content_len')} ct={out[1].get('completion_tokens')}{flag}", flush=True)
if OUT: json.dump(rows, open(OUT, "w"), indent=2)
seen = {r["residue"] for r in rows if r["residue"] is not None}
print(f"residues covered: {len(seen)}/128")
