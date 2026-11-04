#!/usr/bin/env python3
"""Frozen long-context benchmark for the MTP + suffix hillclimb.

The six tasks match the existing LABD benchmark. The primary score is the
geometric mean of per-task streamed decode rates. Repetitions alternate task
order so thermal drift does not always favor the same task.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path


TASKS = (
    ("copy", "Gengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."),
    ("code", "Gengiv alle bash-kommandoer fra teksten i en samlet liste, ordret."),
    ("edit", "Omskriv afsnittet om Docker sa det er kortere, men behold alle kommandoer ordret."),
    ("quote", "Citer ordret de afsnit der handler om hukommelse og KV-cache, og forklar dem."),
    ("summary", "Skriv en grundig, struktureret opsummering pa dansk."),
    ("qa", "Svar udførligt: hvilke optimeringer gav mest, og hvorfor?"),
)


def build_corpus(chars: int) -> str:
    roots = (
        "/data/buttercup_6tb/k3s/vllm-trial/df2-repo/README.md",
        "/data/buttercup_6tb/k3s/vllm-trial/df2-repo/docs/*.md",
        "/data/buttercup_6tb/k3s/vllm-trial/venv/lib/python3.12/site-packages/vllm/v1/**/*.py",
    )
    paths: list[str] = []
    for pattern in roots:
        paths.extend(glob.glob(pattern, recursive=True))
    parts: list[str] = []
    total = 0
    for path in dict.fromkeys(sorted(paths)):
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        parts.append(f"\n\nFILE {path}\n\n{text}")
        total += len(parts[-1])
        if total >= chars:
            break
    corpus = "".join(parts)
    if len(corpus) < chars:
        raise RuntimeError(f"corpus has {len(corpus)} chars, need {chars}")
    return corpus[:chars]


class Client:
    def __init__(self, base: str, model: str, api_key: str):
        self.base = base.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def metrics(self) -> tuple[float, float]:
        req = urllib.request.Request(f"{self.base}/metrics", headers=self.headers)
        values: dict[str, float] = {}
        for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines():
            for key in (
                "vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_accepted_tokens_total",
            ):
                if line.startswith(key + " ") or line.startswith(key + "{"):
                    values[key] = values.get(key, 0.0) + float(line.split()[-1])
        return (
            values.get("vllm:spec_decode_num_drafts_total", 0.0),
            values.get("vllm:spec_decode_num_accepted_tokens_total", 0.0),
        )

    def complete(self, prompt: str, max_tokens: int) -> dict[str, float | int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self.headers,
        )
        start = time.perf_counter()
        first = None
        usage: dict[str, int] = {}
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                if first is None:
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content") or delta.get("reasoning"):
                            first = time.perf_counter()
                            break
        end = time.perf_counter()
        first = first or end
        output_tokens = usage.get("completion_tokens", 0)
        decode_seconds = max(end - first, 1e-6)
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": output_tokens,
            "ttft_s": first - start,
            "decode_s": decode_seconds,
            "decode_tok_s": max(output_tokens - 1, 0) / decode_seconds,
        }


def geometric_mean(values: list[float]) -> float:
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base", default=os.getenv("QWEN_BASE", "http://127.0.0.1:8094"))
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--ctx", type=int, default=100_000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    document = build_corpus(int(args.ctx * 3.6))
    client = Client(args.base, args.model, os.getenv("VLLM_API_KEY", ""))

    for _ in range(2):
        client.complete(
            "Dokument:\n\n" + document[:4000] + "\n\nGengiv ordret de første 10 linjer.",
            64,
        )

    rows: list[dict[str, object]] = []
    for rep in range(args.reps):
        ordered = TASKS if rep % 2 == 0 else tuple(reversed(TASKS))
        for name, question in ordered:
            before_steps, before_accepted = client.metrics()
            result = client.complete(f"Dokument:\n\n{document}\n\n{question}", args.max_tokens)
            after_steps, after_accepted = client.metrics()
            steps = after_steps - before_steps
            accepted = after_accepted - before_accepted
            row = {
                "rep": rep + 1,
                "task": name,
                **result,
                "steps": steps,
                "tokens_per_step": 1 + accepted / max(steps, 1),
            }
            rows.append(row)
            print(
                f"{args.tag} rep={rep + 1} task={name} "
                f"prompt={result['prompt_tokens']} decode={result['decode_tok_s']:.2f} "
                f"tok/step={row['tokens_per_step']:.3f} ttft={result['ttft_s']:.2f}s",
                flush=True,
            )

    medians = {
        name: statistics.median(
            float(row["decode_tok_s"]) for row in rows if row["task"] == name
        )
        for name, _ in TASKS
    }
    score = geometric_mean(list(medians.values()))
    output = {
        "tag": args.tag,
        "base": args.base,
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "task_median_decode_tok_s": medians,
        "geomean_decode_tok_s": score,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"{args.tag} GEOMEAN={score:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
