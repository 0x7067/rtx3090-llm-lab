#!/usr/bin/env python3
"""Build the regeneration input set: mined agent contexts first, then public
coding prompts, all as rows shaped like extract_claude_sessions.py output
(`id`, `conversations` ending on a user/tool turn, `tools`).

Public rows are truncated to their first user turn, so K2 answers from
scratch rather than continuing another model's text.

Usage: build_regen_prompts.py <out.jsonl> [public_n]
"""
import json
import sys

W = "/data/buttercup_6tb/specforge-work"
out_path = sys.argv[1]
public_n = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

rows = []

# 1. Mined agent contexts (highest value: real tools, real repos, multi-turn).
mined = f"{W}/sessions/denguinho-server/claude-code/extracted.jsonl"
n_mined = 0
try:
    for line in open(mined):
        if not line.strip():
            continue
        row = json.loads(line)
        row.pop("reference_assistant", None)
        rows.append(row)
        n_mined += 1
except FileNotFoundError:
    pass

# 2. Public coding prompts: keep the first user turn only.
n_pub = 0
for line in open(f"{W}/cache/dataset/k2-eagle3-train.jsonl"):
    if n_pub >= public_n:
        break
    if not line.strip():
        continue
    row = json.loads(line)
    conv = row.get("conversations") or []
    first_user = next((m for m in conv if m.get("role") in ("user", "human")), None)
    if not first_user or not (first_user.get("content") or "").strip():
        continue
    rows.append({
        "id": f"pub-{row.get('id', n_pub)}",
        "conversations": [{"role": "user", "content": first_user["content"]}],
        "tools": [],
    })
    n_pub += 1

with open(out_path, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows: {n_mined} mined agent contexts + {n_pub} public prompts -> {out_path}")
