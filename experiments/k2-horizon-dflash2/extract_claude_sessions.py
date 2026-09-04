#!/usr/bin/env python3
"""Mine Claude Code transcripts (~/.claude/projects/**/*.jsonl) into
regeneration-ready agent contexts for drafter training.

Each output row is a ShareGPT-style conversation ending right before an
assistant turn, plus the original assistant turn under `reference_assistant`
(not used for training: the target model regenerates it). Tool calls become
OpenAI-style `tool_calls`; tool results become role=tool messages. Tool schemas
are inferred from observed arguments so the K2 chat template can render the
tool block. Secret-looking strings are redacted.

Usage: extract_claude_sessions.py <out.jsonl> [max_chars_per_sample=24000]
"""
import glob
import json
import os
import re
import sys
import uuid

OUT = sys.argv[1]
MAX_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
TOOL_RESULT_MAX = 6000

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\"?\s*[:=]\s*\"?)([^\s\"',;]{8,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"),  # JWT
]


def scrub(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pat in SECRET_PATTERNS:
        if pat.groups >= 3:
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = pat.sub("[REDACTED]", text)
    return text


def user_text(content):
    if isinstance(content, str):
        t = content
    else:
        t = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    # drop harness noise
    t = re.sub(r"<system-reminder>[\s\S]*?</system-reminder>", "", t)
    t = re.sub(r"<local-command-caveat>[\s\S]*?</local-command-caveat>", "", t)
    t = re.sub(r"<command-[a-z]+>[\s\S]*?</command-[a-z]+>", "", t)
    t = re.sub(r"<local-command-stdout>[\s\S]*?</local-command-stdout>", "", t)
    return t.strip()


def infer_schema(args: dict) -> dict:
    props = {}
    for k, v in (args or {}).items():
        if isinstance(v, bool):
            props[k] = {"type": "boolean"}
        elif isinstance(v, int):
            props[k] = {"type": "integer"}
        elif isinstance(v, float):
            props[k] = {"type": "number"}
        elif isinstance(v, list):
            props[k] = {"type": "array"}
        elif isinstance(v, dict):
            props[k] = {"type": "object"}
        else:
            props[k] = {"type": "string"}
    return {"type": "object", "properties": props}


def merge_schema(a: dict, b: dict) -> dict:
    out = {"type": "object", "properties": dict(a.get("properties", {}))}
    out["properties"].update(b.get("properties", {}))
    return out


def convert_file(path):
    msgs, tools = [], {}
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t not in ("user", "assistant") or d.get("isSidechain"):
            continue
        m = d.get("message") or {}
        c = m.get("content")
        if t == "user":
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                for b in c:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        continue
                    rc = b.get("content")
                    if isinstance(rc, list):
                        rc = "\n".join(x.get("text", "") for x in rc if isinstance(x, dict))
                    rc = scrub(str(rc or ""))
                    if len(rc) > TOOL_RESULT_MAX:
                        rc = rc[: TOOL_RESULT_MAX // 2] + "\n...[truncated]...\n" + rc[-TOOL_RESULT_MAX // 2 :]
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": rc})
            else:
                if d.get("isMeta"):
                    continue
                txt = user_text(c)
                if txt:
                    msgs.append({"role": "user", "content": scrub(txt)})
        else:
            text_parts, calls = [], []
            for b in c if isinstance(c, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    name = b.get("name", "tool")
                    args = b.get("input") or {}
                    tools[name] = merge_schema(tools.get(name, {"type": "object", "properties": {}}), infer_schema(args))
                    calls.append({"id": b.get("id", ""), "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
            entry = {"role": "assistant", "content": scrub("\n".join(text_parts).strip())}
            if calls:
                entry["tool_calls"] = [{**cl, "function": {**cl["function"], "arguments": scrub(cl["function"]["arguments"])}} for cl in calls]
            # merge consecutive assistant records (streamed blocks arrive as separate lines)
            if msgs and msgs[-1]["role"] == "assistant":
                prev = msgs[-1]
                prev["content"] = (prev["content"] + "\n" + entry["content"]).strip()
                if "tool_calls" in entry:
                    prev.setdefault("tool_calls", []).extend(entry["tool_calls"])
            else:
                msgs.append(entry)
    tool_defs = [{"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": s}} for n, s in sorted(tools.items())]
    return msgs, tool_defs


def windows(msgs, tool_defs, session_id):
    """One sample per assistant turn: context up to that turn, trimmed to MAX_CHARS from the end."""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if i == 0 or msgs[i - 1]["role"] not in ("user", "tool"):
            continue
        ctx = msgs[:i]
        # trim from the front by whole messages, never starting on a tool message
        while ctx and sum(len(json.dumps(x, ensure_ascii=False)) for x in ctx) > MAX_CHARS:
            ctx = ctx[1:]
        while ctx and ctx[0]["role"] != "user":
            ctx = ctx[1:]
        if not ctx:
            continue
        yield {
            "id": f"{session_id}-{i}",
            "conversations": ctx,
            "tools": tool_defs,
            "reference_assistant": m,
        }


def main():
    files = sorted(glob.glob(os.path.expanduser("~/.claude/projects/**/*.jsonl"), recursive=True))
    n_files = n_rows = 0
    chars = 0
    with open(OUT, "w") as out:
        for f in files:
            msgs, tool_defs = convert_file(f)
            if len(msgs) < 3:
                continue
            n_files += 1
            sid = os.path.basename(f).split(".")[0][:8]
            for row in windows(msgs, tool_defs, sid):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                chars += sum(len(json.dumps(x, ensure_ascii=False)) for x in row["conversations"])
    print(f"sessions used: {n_files}/{len(files)}; samples: {n_rows}; avg context chars: {chars // max(1, n_rows)}; est tokens total: {chars // 4:,}")


if __name__ == "__main__":
    main()
