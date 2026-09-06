#!/usr/bin/env python3
"""Regenerate assistant turns for mined agent contexts with the target model.

Input: extracted.jsonl from extract_claude_sessions.py (rows with
`conversations` ending on a user/tool turn, `tools`, `reference_assistant`).
Output: ShareGPT-style rows whose final assistant turn was produced by the
target through an OpenAI-compatible endpoint, keeping `reasoning_content` and
`tool_calls` so the K2 chat template renders exactly what the model emitted.

Usage:
  regenerate_sessions.py in.jsonl out.jsonl --base-url http://100.64.0.2/v1 \
      --model k2-horizon-7b --api-key "$TOK" --effort medium --concurrency 2
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def call(base_url, api_key, body, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


THINK_TAG = re.compile(r"</?ifm\|think(?:_fast|_faster)?>")


def strip_think_tags(text: str) -> str:
    """Drop stray thinking tags the server's parser leaves in the payload.

    When the model emits no reasoning, SGLang's k2_horizon parser hands back a
    lone closing tag as the reasoning field; the chat template adds the tags
    itself at render time, so any tag inside the text is duplication that
    would teach the drafter to emit them mid-stream.
    """
    return THINK_TAG.sub("", text).strip()


def to_openai_messages(conv):
    out = []
    for m in conv:
        mm = {"role": m["role"], "content": m.get("content", "")}
        if m["role"] == "assistant":
            mm["reasoning_content"] = m.get("reasoning_content", "")
            if m.get("tool_calls"):
                mm["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool":
            mm["tool_call_id"] = m.get("tool_call_id", "")
        out.append(mm)
    return out


def regen_one(args, row):
    body = {
        "model": args.model,
        "messages": to_openai_messages(row["conversations"]),
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.effort,
    }
    if row.get("tools"):
        body["tools"] = row["tools"]
    for attempt in range(3):
        try:
            resp = call(args.base_url, args.api_key, body, args.timeout)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return None, f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(5 * (attempt + 1))
    choice = resp["choices"][0]
    msg = choice["message"]
    assistant = {
        "role": "assistant",
        "content": strip_think_tags(msg.get("content") or ""),
        "reasoning_content": strip_think_tags(
            msg.get("reasoning_content") or msg.get("reasoning") or ""
        ),
    }
    if msg.get("tool_calls"):
        assistant["tool_calls"] = [
            {"id": tc.get("id", ""), "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
            for tc in msg["tool_calls"]
        ]
    if choice.get("finish_reason") != "length" and not (
        assistant["content"] or assistant.get("tool_calls")
    ):
        return None, "Completed response has no answer or tool call; inspect the reasoning parser"
    out = {
        "id": row["id"],
        "conversations": row["conversations"] + [assistant],
        "tools": row.get("tools", []),
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
        "reasoning_effort": args.effort,
        # Retain the parser output before normalization so a delimiter bug
        # can be diagnosed without losing the original response again.
        "source_message": msg,
    }
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="k2-horizon-7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    # A run of ~10k rows reliably produces a handful of unparseable responses.
    # Failing the whole job on those aborts the downstream filtering step and
    # discards hours of usable generation, so only a broad failure is fatal.
    ap.add_argument("--max-error-rate", type=float, default=0.01)
    args = ap.parse_args()

    with open(args.inp) as source:
        rows = [json.loads(l) for l in source if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done_ids = set()
    try:
        with open(args.out) as existing:
            for l in existing:
                done_ids.add(json.loads(l)["id"])
    except FileNotFoundError:
        pass
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"{len(rows)} rows, {len(done_ids)} already done, {len(todo)} to go", file=sys.stderr)

    ok = err = 0
    t0 = time.time()
    with open(args.out, "a") as f, ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(regen_one, args, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out, e = fut.result()
            if out is None:
                err += 1
                print(f"[{i}] error: {e}", file=sys.stderr)
                continue
            ok += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={ok} err={err} {el/60:.1f} min, "
                      f"{ok/el*60:.1f} rows/min", file=sys.stderr)
    print(f"done ok={ok} err={err}", file=sys.stderr)
    if err > len(todo) * args.max_error_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
