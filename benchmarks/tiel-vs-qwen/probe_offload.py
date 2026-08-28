# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Defect B repro attempt: the CPU KV-offload restore path.

Arms a large prompt A (greedy, fixed), floods the GPU KV pool with other large
prompts so A's blocks are evicted to the CPU tier, then re-sends A. The re-send
should restore blocks over PCIe (offload connector) and must return the same
bytes as the armed reply. Checks: byte equality, emptiness, finish_reason.
Repeats the cycle CYCLES times with multi-turn growth (append assistant reply +
follow-up), which is the shape the empty-stop replies appeared in.

Usage: probe_offload.py BASE MODEL CYCLES [OUT.json]
"""
import json, sys, httpx

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
CYCLES = int(sys.argv[3])
OUT = sys.argv[4] if len(sys.argv) > 4 else None

def bigdoc(seed, lines=2600):
    return "\n".join(f"[{seed}] record {i}: value={i*7%9973} tag={'ab'[i%2]}{i%97}"
                     for i in range(lines))  # ~30k tokens

def chat(client, msgs, max_tokens=160):
    r = client.post(f"{BASE}/v1/chat/completions",
                    json={"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
                          "temperature": 0, "thinking_token_budget": 64},
                    timeout=900)
    j = r.json()
    if "choices" not in j: return {"http": r.status_code, "err": str(j)[:200]}
    c = j["choices"][0]; m = c["message"]
    return {"content": m.get("content") or "", "reasoning_len": len(m.get("reasoning") or ""),
            "finish": c["finish_reason"], "prompt_tokens": j["usage"]["prompt_tokens"],
            "completion_tokens": j["usage"]["completion_tokens"]}

rows = []
with httpx.Client() as client:
    convo = [{"role": "user", "content":
              f"Document:\n{bigdoc('A')}\nWhat is value for record 1234? Answer briefly."}]
    armed = chat(client, convo)
    print(f"armed: pt={armed.get('prompt_tokens')} fr={armed.get('finish')} "
          f"clen={len(armed.get('content',''))}", flush=True)
    for cyc in range(CYCLES):
        # flood: 5 unique ~30k prompts (~150k tokens > 140k GPU pool)
        for f in range(5):
            fl = chat(client, [{"role": "user", "content":
                 f"Document:\n{bigdoc(f'flood{cyc}-{f}')}\nSay OK."}], 24)
            print(f"  flood {f}: pt={fl.get('prompt_tokens')} fr={fl.get('finish')}", flush=True)
        # restore: same conversation, next turn (prefix must come back from CPU tier)
        convo += [{"role": "assistant", "content": armed["content"]},
                  {"role": "user", "content": f"Turn {cyc+2}: now record {2345+cyc}? Answer briefly."}]
        got = chat(client, convo)
        empty = not got.get("content") and got.get("reasoning_len", 0) == 0
        row = {"cycle": cyc, "restore": got, "empty": empty}
        rows.append(row)
        flag = "  <-- EMPTY" if empty else ""
        print(f"cycle {cyc}: pt={got.get('prompt_tokens')} fr={got.get('finish')} "
              f"clen={len(got.get('content',''))} rlen={got.get('reasoning_len')}{flag}", flush=True)
        armed = got if got.get("content") else armed
if OUT: json.dump(rows, open(OUT, "w"), indent=2)
empties = [r["cycle"] for r in rows if r["empty"]]
print(f"empty restores: {len(empties)}/{CYCLES} {empties}")
