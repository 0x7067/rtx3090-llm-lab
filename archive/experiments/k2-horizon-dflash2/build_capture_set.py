#!/usr/bin/env python3
"""Turn regen-output.jsonl into the capture training set.

Drops rows whose assistant turn was cut off at the token cap: a reasoning
block that stops mid-thought teaches the drafter to predict a truncation
that the target would never produce. Keeps `reasoning_content` and
`tool_calls` so the K2 thinking template renders production-shaped text.

Usage: build_capture_set.py <regen-output.jsonl> <out.jsonl>
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
kept = dropped_trunc = dropped_empty = 0
with open(dst, "w") as out:
    for line in open(src):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("finish_reason") == "length":
            dropped_trunc += 1
            continue
        assistant = row["conversations"][-1]
        if not (assistant.get("content") or "").strip() and not assistant.get("tool_calls"):
            dropped_empty += 1
            continue
        out.write(json.dumps({
            "id": row["id"],
            "conversations": row["conversations"],
            "tools": row.get("tools", []),
        }, ensure_ascii=False) + "\n")
        kept += 1
print(f"kept {kept}; dropped {dropped_trunc} truncated, {dropped_empty} empty -> {dst}")
