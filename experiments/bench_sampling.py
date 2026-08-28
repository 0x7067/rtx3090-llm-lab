#!/usr/bin/env python3
"""Deterministic probabilistic-sampling benchmark for MTP draft tuning."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


PROMPTS = {
    "code": (
        "Write a complete TypeScript implementation of a bounded async work queue "
        "with cancellation, backpressure, typed errors, and tests. Continue until you "
        "have emitted at least 700 tokens. Return code only."
    ),
    "prose": (
        "Write a precise technical essay explaining how speculative decoding remains "
        "distribution-exact under rejection sampling. Include derivation, examples, "
        "failure modes, and operational guidance. Continue for at least 700 tokens."
    ),
    "repair": (
        "Repair and fully rewrite this intentionally broken Python module. Preserve its "
        "API, add concurrency safety and tests, and continue for at least 700 tokens.\n\n"
        "class Cache:\n"
        " def __init__(self): self.data = {}\n"
        " async def get(self, key, loader):\n"
        "  if key not in self.data: self.data[key] = await loader()\n"
        "  return self.data[key]\n"
        " def clear(self): self.data = {}\n"
    ),
}


def counters(base: str) -> tuple[float, float]:
    request = urllib.request.Request(f"{base}/metrics", headers={"User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"})
    text = urllib.request.urlopen(request, timeout=30).read().decode()
    values: dict[str, float] = {}
    for line in text.splitlines():
        for key in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                values[key] = values.get(key, 0.0) + float(line.split()[-1])
    return values.get("vllm:spec_decode_num_draft_tokens_total", 0.0), values.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)


def complete(base: str, prompt: str, seed: int) -> dict[str, float | int]:
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"},
    )
    started = time.perf_counter()
    first = None
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
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
    finished = time.perf_counter()
    first = first or finished
    tokens = int(usage.get("completion_tokens", 0))
    return {
        "gen_tok_s": max(tokens - 1, 0) / max(finished - first, 1e-9),
        "completion_tokens": tokens,
        "ttft_s": first - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8094")
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before_drafts, before_accepted = counters(args.base)
    groups: dict[str, dict[str, object]] = {}
    all_rates: list[float] = []
    for class_index, (name, prompt) in enumerate(PROMPTS.items()):
        runs = []
        for rep in range(args.reps):
            result = complete(args.base, prompt, 1000 + class_index * 100 + rep)
            runs.append(result)
            all_rates.append(float(result["gen_tok_s"]))
            print(f"{args.tag} class={name} rep={rep + 1} decode={result['gen_tok_s']:.2f} ttft={result['ttft_s']:.3f}s tokens={result['completion_tokens']}", flush=True)
        rates = [float(run["gen_tok_s"]) for run in runs]
        groups[name] = {"runs": runs, "median": statistics.median(rates), "min": min(rates), "max": max(rates)}
    after_drafts, after_accepted = counters(args.base)
    drafts = after_drafts - before_drafts
    result = {
        "tag": args.tag,
        "base": args.base,
        "reps": args.reps,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        **groups,
        "draft_acceptance": (after_accepted - before_accepted) / max(drafts, 1.0),
        "overall_geomean": math.exp(sum(math.log(rate) for rate in all_rates) / len(all_rates)),
        "overall_median": statistics.median(all_rates),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.tag} MEDIAN={result['overall_median']:.2f} GEOMEAN={result['overall_geomean']:.2f} ACCEPTANCE={result['draft_acceptance']:.4f}")


if __name__ == "__main__":
    main()
