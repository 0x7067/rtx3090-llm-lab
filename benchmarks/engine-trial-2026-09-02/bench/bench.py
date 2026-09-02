#!/usr/bin/env python3
"""Engine-agnostic benchmark harness for OpenAI-compatible LLM servers.

Compares vLLM / llama.cpp llama-server / SGLang (or any other OpenAI-compatible
chat-completions server) serving the same model on the same GPU. Python 3.12
stdlib only: urllib, json, time, argparse, threading/concurrent.futures,
statistics, random, uuid.

Subcommands:
  decode      N single streaming decode requests of a fixed ~1k-token coding
              prompt (max_tokens 512, temperature 0). Reports TTFT + decode
              tok/s, median and per-run.
  prefill     Cold prefill throughput: a unique --tokens-token prompt (nonced
              so prefix caching cannot hit), max_tokens 1. Reports
              prompt_tokens (from usage, else /tokenize, else a char-based
              estimate honestly labelled) and TTFT.
  session     Cumulative multi-turn file-editing session (the agentic
              workload): starts from an embedded ~120-line Python file, each
              turn asks for the full file back with one small named change,
              growing the conversation. Reports per-turn TTFT/tok/s/applied
              and cumulative session tok/s.
  concurrent  N parallel copies of the decode prompt (different nonces),
              aggregate tok/s.
  sustained   One streaming request for a very long output (max_tokens 6000
              by default). Reports decode tok/s per --window-tokens-token
              window from first token to end, to surface drafters/engines
              whose throughput decays over a long generation, plus overall
              tok/s and finish_reason.
  quality     The 4-task quality battery ported from
              scripts/evaluate-qwen38-profiles.py (exact/json/normalized/tool
              grading), run once at the given --reasoning setting.
  report      Read one or more results.jsonl files, print a markdown
              comparison table grouped by tag and subcommand (medians).
  selftest    Offline test of the SSE parsing and grading logic against
              fixture data. No network access.

Every measurement subcommand appends one JSON record per measurement to
--out (default results.jsonl, JSONL: one record per line).

Reasoning control (--reasoning {default,medium,low,off}):
  medium  -> body["reasoning_effort"] = "medium"
  low     -> body["reasoning_effort"] = "low"
  off     -> body["chat_template_kwargs"] = {"enable_thinking": false}
  default -> no reasoning fields added (whatever the server defaults to)

Example:
  python3 bench.py selftest
  python3 bench.py decode --base-url http://127.0.0.1:18080/v1 --tag vllm
  python3 bench.py report results.jsonl
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent import futures as cf

DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class BenchError(Exception):
    pass


class BenchHTTPError(BenchError):
    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {(detail or '')[:200]}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def build_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def apply_reasoning(payload, reasoning):
    """Mutate payload in place per the harness's reasoning convention."""
    if reasoning == "medium":
        payload["reasoning_effort"] = "medium"
    elif reasoning == "low":
        payload["reasoning_effort"] = "low"
    elif reasoning == "off":
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    # "default": no-op, let the server use its own default.


def derive_root(base_url):
    """Strip a trailing /v1 (with or without slash) to reach engine-root
    endpoints like /tokenize that live outside the OpenAI-compatible prefix."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def try_tokenize(base_url, api_key, model, text, timeout=30):
    """Best-effort exact token count via a server /tokenize endpoint.
    Tries vLLM's shape ({"model","prompt"} -> {"tokens":[...]}/{"count":N})
    and llama.cpp llama-server's shape ({"content"} -> {"tokens":[...]}).
    Returns None if neither works (caller should fall back to an estimate)."""
    root = derive_root(base_url)
    url = root + "/tokenize"
    attempts = [
        {"model": model, "prompt": text},
        {"content": text},
    ]
    for payload in attempts:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=build_headers(api_key),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            continue
        tokens = body.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        count = body.get("count")
        if isinstance(count, int):
            return count
    return None


def count_tokens_estimate(text):
    """Honest fallback: ~3.5 chars/token, used only when no server tokenizer
    endpoint is reachable and the response carries no usage field."""
    return max(1, round(len(text) / 3.5))


# ---------------------------------------------------------------------------
# SSE parsing (network-agnostic: operates on an iterable of decoded text lines
# so it can be unit tested offline against fixture data in `selftest`).
# ---------------------------------------------------------------------------

def iter_sse_json(line_iter):
    """Yield parsed JSON dicts from an iterable of SSE text lines. Stops at
    'data: [DONE]'. Skips blank lines and ':'-prefixed comments/keepalives."""
    for raw in line_iter:
        line = raw.rstrip("\r\n") if isinstance(raw, str) else raw
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def consume_stream(json_event_iter, t_send):
    """Accumulate one streamed chat-completion response.

    TTFT is measured to the first event carrying a *non-empty* content or
    reasoning_content delta -- not to the first SSE event overall, since some
    engines emit an initial empty {"role": "assistant"} delta immediately on
    connect, before prefill/decode has produced anything.

    completion_tokens prefers usage.completion_tokens (present when the
    request set stream_options.include_usage and the server honours it,
    whether as a dedicated trailing empty-choices event, vLLM-style, or
    co-located with the final content-bearing chunk, llama.cpp-style).
    Falls back to counting non-empty delta events (content + reasoning_content)
    when no server ever reports usage in-stream.

    token_times is a list of elapsed seconds (since t_send) at each non-empty
    content/reasoning delta, in arrival order -- one entry per counted token
    in the same chunk-count fallback sense as completion_tokens. It lets a
    caller bucket a long generation into fixed-size windows and see whether
    decode throughput decays over the output (see `sustained`).
    """
    ttft = None
    content_parts = []
    reasoning_parts = []
    finish_reason = None
    usage_completion = None
    usage_prompt = None
    usage_seen = False
    chunk_count = 0
    token_times = []

    for event in json_event_iter:
        now = time.monotonic()
        usage = event.get("usage")
        if usage:
            usage_seen = True
            if usage.get("completion_tokens") is not None:
                usage_completion = usage.get("completion_tokens")
            if usage.get("prompt_tokens") is not None:
                usage_prompt = usage.get("prompt_tokens")
        choices = event.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            content_piece = delta.get("content")
            if content_piece:
                if ttft is None:
                    ttft = now - t_send
                content_parts.append(content_piece)
                chunk_count += 1
                token_times.append(now - t_send)
            # vLLM >= 0.11 streams thinking as "reasoning"; older servers and llama.cpp use "reasoning_content"
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning_piece:
                if ttft is None:
                    ttft = now - t_send
                reasoning_parts.append(reasoning_piece)
                chunk_count += 1
                token_times.append(now - t_send)
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    t_end = time.monotonic()
    total_time = t_end - t_send
    if ttft is None:
        ttft = total_time
    completion_tokens = usage_completion if usage_completion is not None else chunk_count

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts) or None,
        "ttft": ttft,
        "total_time": total_time,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage_prompt,
        "usage_seen": usage_seen,
        "finish_reason": finish_reason,
        "token_times": token_times,
    }


def stream_chat(base_url, api_key, model, messages, max_tokens, temperature, reasoning,
                 timeout, extra_body=None, top_p=None, top_k=None, presence_penalty=None):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if extra_body:
        payload.update(extra_body)
    apply_reasoning(payload, reasoning)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_headers(api_key),
        method="POST",
    )
    t_send = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise BenchHTTPError(e.code, detail) from e
    except urllib.error.URLError as e:
        raise BenchError(f"connection error calling {url}: {e}") from e

    try:
        line_iter = (raw.decode("utf-8", "replace") for raw in resp)
        result = consume_stream(iter_sse_json(line_iter), t_send)
    finally:
        resp.close()
    return result


def chat_completion_once(base_url, api_key, model, messages, max_tokens, temperature, reasoning,
                          timeout, extra_body=None, tools=None, tool_choice=None,
                          top_p=None, top_k=None, presence_penalty=None):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if extra_body:
        payload.update(extra_body)
    apply_reasoning(payload, reasoning)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise BenchHTTPError(e.code, detail) from e
    except urllib.error.URLError as e:
        raise BenchError(f"connection error calling {url}: {e}") from e


# ---------------------------------------------------------------------------
# Deterministic prompt builders
# ---------------------------------------------------------------------------

DECODE_REQUIREMENTS = [
    'Provide a `RateLimiter` class supporting both token-bucket and sliding-window-counter algorithms, selectable via a constructor argument `algorithm`.',
    'The constructor accepts `capacity: int`, `refill_rate: float` (tokens per second), and `algorithm: str = "token_bucket"`.',
    'Support per-key rate limiting via `allow(key: str, cost: int = 1) -> bool`, tracking independent state for each key.',
    'Keys must be lazily created on first use and never pre-allocated.',
    'Provide `remaining(key: str) -> float` returning the current number of available tokens for a key without consuming any.',
    "Provide `reset(key: str) -> None` to clear a single key's state.",
    'Provide `reset_all() -> None` to clear all tracked keys.',
    'Internally track the last-refill timestamp per key using `time.monotonic()`, never wall-clock time.',
    'Refill logic must be lazy: computed on each `allow()` call based on elapsed time, not via a background thread.',
    'The class must be thread-safe; guard all mutable state with a single `threading.Lock`.',
    'Support an optional `max_keys: int` eviction cap; when exceeded, evict the least-recently-used key.',
    'Implement LRU eviction using `collections.OrderedDict` for O(1) move-to-end and popitem.',
    'Provide a `stats() -> dict` method reporting `allowed_count`, `denied_count`, and `active_keys` cumulatively.',
    "Track `allowed_count` and `denied_count` as instance-level integers, incremented on every `allow()` call.",
    "Provide a context manager `__enter__`/`__exit__` that is a no-op but allows `with RateLimiter(...) as rl:` usage.",
    'Add a `__repr__` that shows algorithm, capacity, and refill_rate.',
    "Validate constructor arguments: `capacity` and `refill_rate` must be positive, else raise `ValueError` with a descriptive message.",
    'Validate `algorithm` is one of "token_bucket" or "sliding_window", else raise `ValueError`.',
    'For the sliding-window algorithm, track a deque of timestamps per key and prune entries older than the window.',
    'The sliding window length in seconds is `capacity / refill_rate`.',
    'Provide a module-level function `make_limiter(config: dict) -> RateLimiter` that builds an instance from a plain dict of the constructor kwargs.',
    'Add type hints to every method signature, including return types.',
    'Add a one-line docstring to every public method.',
    "Raise `KeyError` from `remaining()` and `reset()` if the key does not exist and `create_if_missing=False` is passed.",
    "Default `create_if_missing=True` for `remaining()` so unknown keys report full capacity.",
    'Support a `cost` greater than the bucket capacity by always denying and never partially consuming.',
    '`allow()` must never leave partial state mutations behind when it denies a request.',
    'Provide a small `RateLimiterError(Exception)` base class used for all library-raised errors.',
    'Provide an `AsyncRateLimiter` wrapper class exposing an async `allow()` coroutine that delegates to the sync limiter under an `asyncio.Lock`.',
    'Do not import any third-party packages; standard library only.',
    'Keep the module importable under Python 3.9+ (avoid syntax newer than that).',
    'Add an `__all__` list exporting `RateLimiter`, `AsyncRateLimiter`, `RateLimiterError`, and `make_limiter`.',
    'Add a small `if __name__ == "__main__":` block demonstrating basic usage with print statements.',
    'Log denied requests at DEBUG level via a module-level `logging.getLogger(__name__)` logger.',
    '`stats()` must return a shallow copy so callers cannot mutate internal counters.',
]


def build_decode_prompt(nonce=None):
    header = (
        "You are a senior Python engineer. Implement a production-quality "
        "in-memory rate limiter library in Python 3.11. Output ONLY the code "
        "in a single fenced ```python block, with no commentary before or after.\n\n"
        "Requirements:\n"
    )
    lines = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(DECODE_REQUIREMENTS))
    footer = "\n\nImplement the `RateLimiter` class and any helper classes now."
    nonce_line = f"# bench-nonce: {nonce}\n" if nonce is not None else ""
    return nonce_line + header + lines + footer


FILLER_VOCAB = [
    "system", "cache", "kernel", "vector", "cluster", "latency", "token",
    "buffer", "socket", "thread", "mutex", "queue", "registry", "runtime",
    "schema", "payload", "cursor", "manifest", "checksum", "pipeline",
    "shard", "replica", "quorum", "ledger", "topology", "adapter",
]


def build_prefill_prompt(n_tokens, nonce):
    """Deterministic body (fixed seed) + a per-run random-ish nonce prefix,
    so the text differs across runs and cannot hit prefix/prompt caching,
    while the bulk of the token count is reproducible."""
    target_chars = max(1, int(n_tokens * 3.5))
    rng = random.Random(20260101)
    words = []
    length = 0
    while length < target_chars:
        w = rng.choice(FILLER_VOCAB)
        words.append(w)
        length += len(w) + 1
    body = " ".join(words)
    prefix = f"BENCH-PREFILL-NONCE:{nonce}\n"
    text = prefix + body
    return (
        "Read the reference text below in full, then reply with exactly the "
        "single word DONE and nothing else.\n\nReference text:\n" + text
    )


def build_preamble(n_tokens):
    """Deterministic large fake-codebase dump to simulate a deep-context
    system-message preamble."""
    target_chars = max(1, int(n_tokens * 3.5))
    rng = random.Random(7654321)
    lines = []
    length = 0
    i = 0
    while length < target_chars:
        i += 1
        line = f"def helper_function_{i}(a, b, c={rng.randint(0, 999)}):\n    return a + b - c  # generated context line {i}\n"
        lines.append(line)
        length += len(line)
    return (
        "The following is a large existing codebase for context. Do not modify "
        "it; it is provided only as background.\n\n```python\n" + "".join(lines) + "```\n"
    )


SUSTAINED_PROMPT_HEADER = (
    "You are a senior Python engineer. Write a complete, self-contained Python "
    "module of approximately 3000 lines implementing a full in-process "
    "job-scheduling and worker-pool framework, including: a priority queue "
    "with delayed execution, a thread-pool executor with configurable "
    "concurrency, retry with exponential backoff and jitter, per-job timeout "
    "enforcement, a pluggable persistence backend interface with an in-memory "
    "reference implementation, structured logging, a small CLI for submitting "
    "and inspecting jobs, comprehensive type hints, and thorough docstrings on "
    "every public class and method. Output ONLY the code in a single fenced "
    "```python block. Do not include any commentary, explanation, or summary "
    "before or after the code block, and do not truncate: continue the module "
    "until it is complete."
)


def build_sustained_prompt(nonce=None):
    nonce_line = f"# bench-nonce: {nonce}\n" if nonce is not None else ""
    return nonce_line + SUSTAINED_PROMPT_HEADER


def windowed_tok_s(token_times, ttft, window_size, total_tokens=None):
    """Bucket a token-arrival-time trace into fixed-size, non-overlapping
    windows and report tok/s per window. token_times and ttft are elapsed
    seconds since the request was sent (as returned by consume_stream); the
    first window's clock starts at ttft (the first token), not at request
    send, so windows measure pure decode throughput, not prefill. The final
    window may be shorter than window_size if the total token count isn't an
    exact multiple."""
    # Speculative decoders emit several tokens per SSE chunk, so token_times has one
    # entry per CHUNK. When the server reported the true completion_tokens, scale
    # chunk counts by tokens-per-chunk so windows are sized and rated in tokens.
    n_chunks = len(token_times)
    scale = (total_tokens / n_chunks) if (total_tokens and n_chunks) else 1.0
    chunks_per_window = max(1, int(round(window_size / scale)))
    windows = []
    prev_boundary = ttft
    idx = 0
    for start in range(0, n_chunks, chunks_per_window):
        end = min(start + chunks_per_window, n_chunks) - 1
        if end < start:
            break
        idx += 1
        window_tokens = (end - start + 1) * scale
        window_end_time = token_times[end]
        duration = max(window_end_time - prev_boundary, 1e-6)
        windows.append({
            "window": idx,
            "tokens": round(window_tokens),
            "chunks": end - start + 1,
            "elapsed_s": round(window_end_time, 2),
            "tok_s": round(window_tokens / duration, 2),
        })
        prev_boundary = window_end_time
    return windows


SESSION_BASE_FILE = '''import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("taskqueue")


@dataclass
class Task:
    task_id: str
    title: str
    priority: int
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_terminal(self) -> bool:
        return self.status in ("done", "cancelled")


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "task_id TEXT PRIMARY KEY, title TEXT NOT NULL, priority INTEGER NOT NULL,"
            "status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        self.conn.commit()

    def upsert(self, task: Task) -> None:
        self.conn.execute(
            "INSERT INTO tasks (task_id, title, priority, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET"
            " title=excluded.title, priority=excluded.priority,"
            " status=excluded.status, updated_at=excluded.updated_at",
            (task.task_id, task.title, task.priority, task.status,
             task.created_at, task.updated_at),
        )
        self.conn.commit()

    def get(self, task_id: str) -> Optional[Task]:
        row = self.conn.execute(
            "SELECT task_id, title, priority, status, created_at, updated_at"
            " FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        return Task(*row) if row else None

    def all_tasks(self) -> Iterable[Task]:
        for row in self.conn.execute(
            "SELECT task_id, title, priority, status, created_at, updated_at"
            " FROM tasks ORDER BY priority DESC, created_at ASC"
        ):
            yield Task(*row)

    def pending_count(self) -> int:
        return sum(1 for t in self.all_tasks() if t.status == "pending")

    def mark_status(self, task_id: str, status: str) -> bool:
        task = self.get(task_id)
        if task is None:
            return False
        task.status = status
        task.updated_at = datetime.now(timezone.utc).isoformat()
        self.upsert(task)
        return True


def export_json(store: TaskStore, out_path: Path) -> int:
    tasks = [t.__dict__ for t in store.all_tasks()]
    out_path.write_text(json.dumps(tasks, indent=2))
    return len(tasks)


def import_json(store: TaskStore, in_path: Path) -> int:
    data = json.loads(in_path.read_text())
    n = 0
    for row in data:
        store.upsert(Task(**row))
        n += 1
    return n


def next_task(store: TaskStore) -> Optional[Task]:
    for task in store.all_tasks():
        if task.status == "pending":
            return task
    return None


def summarize(store: TaskStore) -> dict:
    tasks = list(store.all_tasks())
    by_status: dict = {}
    for t in tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
    return {"total": len(tasks), "by_status": by_status}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tiny task queue CLI")
    parser.add_argument("--db", type=Path, default=Path("tasks.db"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--import-from", type=Path, dest="import_from")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    store = TaskStore(args.db)
    if args.import_from:
        n = import_json(store, args.import_from)
        log.info("imported %d tasks", n)
    if args.export:
        n = export_json(store, args.export)
        log.info("exported %d tasks", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

SESSION_EDITS = [
    ("Rename the class TaskStore to SqliteTaskStore everywhere.", "SqliteTaskStore"),
    ("Add a method `delete(self, task_id: str) -> bool` to the store class that removes a row and returns whether it existed.", "def delete(self, task_id: str) -> bool"),
    ("Add a `--verbose` flag to the argument parser that sets logging level to DEBUG.", "--verbose"),
    ("Add a one-line docstring to every public method on the store class.", '"""'),
    ("Add a `high_priority(self, threshold: int)` method returning tasks with priority above threshold.", "def high_priority(self, threshold: int)"),
    ("Change export_json to also write a `pending_count` field at the top level (wrap tasks in an object with keys `tasks` and `pending_count`).", "pending_count"),
    ("Add type hint `-> None` to `TaskStore.__init__` and make db_path accept str or Path via `os.fspath`.", "os.fspath"),
    ("Add a `requeue(self, task_id: str) -> bool` method that sets a task's status back to pending and updates updated_at.", "def requeue(self, task_id: str) -> bool"),
    ("Add a `--status` filter flag to the CLI that only exports tasks with a matching status.", "--status"),
    ("Add a module-level constant `SCHEMA_VERSION = 2` near the top of the file.", "SCHEMA_VERSION"),
]


# ---------------------------------------------------------------------------
# Quality battery, ported verbatim (tasks + grading) from
# scripts/evaluate-qwen38-profiles.py. The reasoning/profile selection is
# replaced by this harness's unified --reasoning flag.
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

TASKS = [
    {
        "name": "arithmetic_exact",
        "prompt": "Compute (37 * 19) - 48. Reply with exactly the integer and nothing else.",
        "grade": "exact",
        "expected": "655",
    },
    {
        "name": "portuguese_json",
        "prompt": (
            "Extraia os dados a seguir. Responda somente JSON válido, sem markdown: "
            "'O servidor atlas tem 48 GB de RAM e está em Recife.' "
            "Use exatamente as chaves servidor, ram_gb e cidade."
        ),
        "grade": "json",
        "expected": {"servidor": "atlas", "ram_gb": 48, "cidade": "Recife"},
    },
    {
        "name": "python_fix",
        "prompt": (
            "A Python counter contains `counts[key] = counts.get(key) + 1` and crashes "
            "for a new key. Reply with only the corrected assignment line."
        ),
        "grade": "normalized",
        "expected": "counts[key] = counts.get(key, 0) + 1",
    },
    {
        "name": "required_tool",
        "prompt": "What is the weather in Sao Paulo? Use the provided tool.",
        "grade": "tool",
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "expected": {"name": "get_weather", "city": "Sao Paulo"},
    },
]


def normalize(value):
    return " ".join((value or "").strip().replace("`", "").split())


def grade(task, message):
    content = (message.get("content") or "").strip()
    if task["grade"] == "exact":
        return content == task["expected"], content
    if task["grade"] == "normalized":
        return normalize(content) == normalize(task["expected"]), content
    if task["grade"] == "json":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return False, content
        return parsed == task["expected"], parsed
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return False, calls
    function = calls[0].get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, function
    observed = {"name": function.get("name"), "city": arguments.get("city")}
    return observed == task["expected"], observed


# ---------------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------------

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_result(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_decode(args):
    rows = []
    for i in range(args.n):
        nonce = f"{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:8]}"
        prompt = build_decode_prompt(nonce)
        messages = [{"role": "user", "content": prompt}]
        result = stream_chat(args.base_url, args.api_key, args.model, messages,
                              max_tokens=args.max_tokens, temperature=0,
                              reasoning=args.reasoning, timeout=args.timeout)
        decode_time = max(result["total_time"] - result["ttft"], 1e-6)
        tok_s = result["completion_tokens"] / decode_time if result["completion_tokens"] else 0.0
        row = {
            "run": i, "ttft_s": round(result["ttft"], 3), "total_s": round(result["total_time"], 3),
            "completion_tokens": result["completion_tokens"], "prompt_tokens": result["prompt_tokens"],
            "tok_s": round(tok_s, 2), "finish_reason": result["finish_reason"],
            "usage_seen": result["usage_seen"],
        }
        rows.append(row)
        print(f"[decode] run {i + 1}/{args.n} ttft={row['ttft_s']}s tok_s={row['tok_s']} "
              f"tokens={row['completion_tokens']}", flush=True)

    tok_s_values = [r["tok_s"] for r in rows]
    ttft_values = [r["ttft_s"] for r in rows]
    record = {
        "tag": args.tag, "subcommand": "decode", "timestamp": now_iso(),
        "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
        "n": args.n, "max_tokens": args.max_tokens,
        "median_tok_s": round(statistics.median(tok_s_values), 2) if tok_s_values else None,
        "median_ttft_s": round(statistics.median(ttft_values), 3) if ttft_values else None,
        "runs": rows,
    }
    append_result(args.out, record)
    print(f"[decode] SUMMARY median_tok_s={record['median_tok_s']} "
          f"median_ttft_s={record['median_ttft_s']}")


def cmd_prefill(args):
    nonce = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    prompt = build_prefill_prompt(args.tokens, nonce)
    messages = [{"role": "user", "content": prompt}]
    result = stream_chat(args.base_url, args.api_key, args.model, messages,
                          max_tokens=1, temperature=0, reasoning=args.reasoning,
                          timeout=args.timeout)
    prompt_tokens = result["prompt_tokens"]
    token_source = "usage"
    if prompt_tokens is None:
        prompt_tokens = try_tokenize(args.base_url, args.api_key, args.model, prompt,
                                      timeout=min(args.timeout, 60))
        token_source = "tokenize_endpoint" if prompt_tokens is not None else "char_estimate"
        if prompt_tokens is None:
            prompt_tokens = count_tokens_estimate(prompt)
    ttft = result["ttft"]
    tok_s = prompt_tokens / ttft if ttft else None
    record = {
        "tag": args.tag, "subcommand": "prefill", "timestamp": now_iso(),
        "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
        "requested_tokens": args.tokens, "prompt_chars": len(prompt),
        "prompt_tokens": prompt_tokens, "token_source": token_source,
        "ttft_s": round(ttft, 3), "prefill_tok_s": round(tok_s, 1) if tok_s else None,
        "finish_reason": result["finish_reason"],
    }
    append_result(args.out, record)
    print(f"[prefill] tokens={prompt_tokens} ({token_source}) ttft={record['ttft_s']}s "
          f"prefill_tok_s={record['prefill_tok_s']}")


def cmd_session(args):
    messages = []
    if args.preamble_tokens > 0:
        messages.append({"role": "system", "content": build_preamble(args.preamble_tokens)})

    first_user_prefix = (
        "You are a precise code editor. Here is a Python file:\n\n```python\n"
        + SESSION_BASE_FILE + "\n```\n\n"
    )

    turns_data = []
    session_start = time.monotonic()
    total_tokens = 0
    total_decode_time = 0.0

    for turn in range(1, args.turns + 1):
        instruction, marker = SESSION_EDITS[(turn - 1) % len(SESSION_EDITS)]
        task_text = (
            f"Task {turn}: {instruction}\nOutput the COMPLETE updated file in a single "
            "```python code block. No commentary before or after."
        )
        user_content = (first_user_prefix + task_text) if turn == 1 else task_text
        messages.append({"role": "user", "content": user_content})

        result = stream_chat(args.base_url, args.api_key, args.model, messages,
                              max_tokens=args.max_tokens, temperature=0,
                              reasoning=args.reasoning, timeout=args.timeout)

        applied = marker in result["content"]
        decode_time = max(result["total_time"] - result["ttft"], 1e-6)
        tok_s = result["completion_tokens"] / decode_time if result["completion_tokens"] else 0.0
        row = {
            "turn": turn, "instruction": instruction, "applied": applied,
            "ttft_s": round(result["ttft"], 3), "total_s": round(result["total_time"], 3),
            "completion_tokens": result["completion_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "tok_s": round(tok_s, 2), "finish_reason": result["finish_reason"],
            "usage_seen": result["usage_seen"],
        }
        turns_data.append(row)
        print(f"[session] turn {turn}/{args.turns} applied={applied} "
              f"ttft={row['ttft_s']}s tok_s={row['tok_s']} tokens={row['completion_tokens']}",
              flush=True)

        messages.append({"role": "assistant", "content": result["content"]})
        total_tokens += result["completion_tokens"] or 0
        total_decode_time += decode_time

    session_wall = time.monotonic() - session_start
    cumulative_tok_s = total_tokens / total_decode_time if total_decode_time else 0.0
    applied_rate = sum(1 for r in turns_data if r["applied"]) / len(turns_data) if turns_data else 0.0

    record = {
        "tag": args.tag, "subcommand": "session", "timestamp": now_iso(),
        "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
        "turns": args.turns, "preamble_tokens": args.preamble_tokens,
        "max_tokens": args.max_tokens,
        "cumulative_tok_s": round(cumulative_tok_s, 2),
        "applied_rate": round(applied_rate, 3),
        "session_wall_s": round(session_wall, 2),
        "total_completion_tokens": total_tokens,
        "per_turn": turns_data,
    }
    append_result(args.out, record)
    print(f"[session] SUMMARY cumulative_tok_s={record['cumulative_tok_s']} "
          f"applied_rate={record['applied_rate']} wall={record['session_wall_s']}s")


def cmd_concurrent(args):
    def one_run(i):
        nonce = f"{int(time.time() * 1000)}-{i}-{uuid.uuid4().hex[:8]}"
        prompt = build_decode_prompt(nonce)
        messages = [{"role": "user", "content": prompt}]
        result = stream_chat(args.base_url, args.api_key, args.model, messages,
                              max_tokens=args.max_tokens, temperature=0,
                              reasoning=args.reasoning, timeout=args.timeout)
        return i, result

    batch_start = time.monotonic()
    rows = [None] * args.n
    with cf.ThreadPoolExecutor(max_workers=args.n) as ex:
        futures = [ex.submit(one_run, i) for i in range(args.n)]
        for fut in cf.as_completed(futures):
            i, result = fut.result()
            decode_time = max(result["total_time"] - result["ttft"], 1e-6)
            tok_s = result["completion_tokens"] / decode_time if result["completion_tokens"] else 0.0
            rows[i] = {
                "run": i, "ttft_s": round(result["ttft"], 3),
                "total_s": round(result["total_time"], 3),
                "completion_tokens": result["completion_tokens"],
                "tok_s": round(tok_s, 2), "finish_reason": result["finish_reason"],
            }
    batch_wall = time.monotonic() - batch_start
    total_tokens = sum((r["completion_tokens"] or 0) for r in rows)
    aggregate_tok_s = total_tokens / batch_wall if batch_wall else 0.0

    record = {
        "tag": args.tag, "subcommand": "concurrent", "timestamp": now_iso(),
        "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
        "n": args.n, "max_tokens": args.max_tokens,
        "batch_wall_s": round(batch_wall, 2),
        "total_completion_tokens": total_tokens,
        "aggregate_tok_s": round(aggregate_tok_s, 2),
        "runs": rows,
    }
    append_result(args.out, record)
    print(f"[concurrent] n={args.n} aggregate_tok_s={record['aggregate_tok_s']} "
          f"wall={record['batch_wall_s']}s")


def cmd_sustained(args):
    nonce = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    prompt = build_sustained_prompt(nonce)
    messages = [{"role": "user", "content": prompt}]
    result = stream_chat(args.base_url, args.api_key, args.model, messages,
                          max_tokens=args.max_tokens, temperature=0,
                          reasoning=args.reasoning, timeout=args.timeout)

    windows = windowed_tok_s(result.get("token_times") or [], result["ttft"], args.window_tokens,
                             total_tokens=result["completion_tokens"] if result["usage_seen"] else None)

    decode_time = max(result["total_time"] - result["ttft"], 1e-6)
    overall_tok_s = result["completion_tokens"] / decode_time if result["completion_tokens"] else 0.0

    record = {
        "tag": args.tag, "subcommand": "sustained", "timestamp": now_iso(),
        "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
        "max_tokens": args.max_tokens, "window_tokens": args.window_tokens,
        "ttft_s": round(result["ttft"], 3), "total_s": round(result["total_time"], 3),
        "completion_tokens": result["completion_tokens"], "prompt_tokens": result["prompt_tokens"],
        "overall_tok_s": round(overall_tok_s, 2), "finish_reason": result["finish_reason"],
        "usage_seen": result["usage_seen"], "windows": windows,
    }
    append_result(args.out, record)
    print(f"[sustained] tokens={record['completion_tokens']} finish={record['finish_reason']} "
          f"overall_tok_s={record['overall_tok_s']} ttft={record['ttft_s']}s")
    if windows:
        first_w, last_w = windows[0]["tok_s"], windows[-1]["tok_s"]
        decay_pct = round(100.0 * (first_w - last_w) / first_w, 1) if first_w else None
        print(f"[sustained] windows={len(windows)} first_window_tok_s={first_w} "
              f"last_window_tok_s={last_w} decay={decay_pct}%")
        for w in windows:
            print(f"[sustained]   window {w['window']}: {w['tokens']} tokens @ "
                  f"{w['elapsed_s']}s -> {w['tok_s']} tok/s")


def run_quality_task(args, task):
    thinking = args.reasoning != "off"
    messages = [{"role": "user", "content": task["prompt"]}]
    max_tokens = task.get("max_tokens", 512)
    temperature = 1.0 if thinking else 0.7
    top_p = 0.95 if thinking else 0.8
    top_k = 20
    presence_penalty = 0.0 if thinking else 1.5

    t0 = time.monotonic()
    try:
        body = chat_completion_once(
            args.base_url, args.api_key, args.model, messages,
            max_tokens=max_tokens, temperature=temperature, reasoning=args.reasoning,
            timeout=args.timeout, tools=task.get("tools"), tool_choice=task.get("tool_choice"),
            top_p=top_p, top_k=top_k, presence_penalty=presence_penalty,
        )
    except BenchHTTPError as e:
        return {"ok": False, "http_error": e.code, "detail": e.detail}
    wall = time.monotonic() - t0
    choice = body["choices"][0]
    message = choice["message"]
    passed, observed = grade(task, message)
    usage = body.get("usage") or {}
    return {
        "ok": True, "passed": passed, "observed": observed,
        "finish_reason": choice.get("finish_reason"), "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def cmd_quality(args):
    records = []
    for task in TASKS:
        result = run_quality_task(args, task)
        record = {
            "tag": args.tag, "subcommand": "quality", "timestamp": now_iso(),
            "base_url": args.base_url, "model": args.model, "reasoning": args.reasoning,
            "task": task["name"], **result,
        }
        records.append(record)
        append_result(args.out, record)
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"[quality] {status:4} {task['name']:18} finish={result.get('finish_reason')} "
              f"wall={result.get('wall_s')} tokens={result.get('completion_tokens')}")
    passed = sum(1 for r in records if r.get("passed"))
    print(f"[quality] SUMMARY {passed}/{len(records)} passed")


def group_by_tag(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r.get("tag", "unknown"), []).append(r)
    return dict(sorted(groups.items()))


def median_fmt(values):
    if not values:
        return "n/a"
    return f"{statistics.median(values):.2f}"


def cmd_report(args):
    records = []
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    by_sub = {}
    for r in records:
        by_sub.setdefault(r.get("subcommand"), []).append(r)

    print("# Benchmark comparison\n")

    if "decode" in by_sub:
        print("## decode (single-request, streaming)\n")
        print("| tag | runs | median tok/s | median ttft (s) |")
        print("|---|---|---|---|")
        for tag, rows in group_by_tag(by_sub["decode"]).items():
            tok_s_vals = [r["median_tok_s"] for r in rows if r.get("median_tok_s") is not None]
            ttft_vals = [r["median_ttft_s"] for r in rows if r.get("median_ttft_s") is not None]
            n_runs = sum(r.get("n", 0) for r in rows)
            print(f"| {tag} | {n_runs} | {median_fmt(tok_s_vals)} | {median_fmt(ttft_vals)} |")
        print()

    if "prefill" in by_sub:
        print("## prefill (cold, unique nonced prompt)\n")
        print("| tag | median prompt tokens | median ttft (s) | median tok/s |")
        print("|---|---|---|---|")
        for tag, rows in group_by_tag(by_sub["prefill"]).items():
            pt = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
            ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
            tps = [r["prefill_tok_s"] for r in rows if r.get("prefill_tok_s") is not None]
            print(f"| {tag} | {median_fmt(pt)} | {median_fmt(ttft)} | {median_fmt(tps)} |")
        print()

    if "session" in by_sub:
        print("## session (cumulative multi-turn file-editing)\n")
        print("| tag | configs (turns/preamble-tokens) | median cumulative tok/s | median applied rate |")
        print("|---|---|---|---|")
        for tag, rows in group_by_tag(by_sub["session"]).items():
            configs = sorted({f"{r.get('turns')}t/{r.get('preamble_tokens')}p" for r in rows})
            cts = [r["cumulative_tok_s"] for r in rows if r.get("cumulative_tok_s") is not None]
            ars = [r["applied_rate"] for r in rows if r.get("applied_rate") is not None]
            print(f"| {tag} | {', '.join(configs)} | {median_fmt(cts)} | {median_fmt(ars)} |")
        print()

    if "concurrent" in by_sub:
        print("## concurrent (parallel decode)\n")
        print("| tag | n | median aggregate tok/s |")
        print("|---|---|---|")
        for tag, rows in group_by_tag(by_sub["concurrent"]).items():
            ns = sorted({r.get("n") for r in rows})
            agg = [r["aggregate_tok_s"] for r in rows if r.get("aggregate_tok_s") is not None]
            print(f"| {tag} | {', '.join(str(n) for n in ns)} | {median_fmt(agg)} |")
        print()

    if "sustained" in by_sub:
        print("## sustained (long single generation, windowed decode tok/s)\n")
        print("| tag | median overall tok/s | median first-window tok/s | median last-window tok/s |")
        print("|---|---|---|---|")
        for tag, rows in group_by_tag(by_sub["sustained"]).items():
            overall = [r["overall_tok_s"] for r in rows if r.get("overall_tok_s") is not None]
            first_w = [r["windows"][0]["tok_s"] for r in rows if r.get("windows")]
            last_w = [r["windows"][-1]["tok_s"] for r in rows if r.get("windows")]
            print(f"| {tag} | {median_fmt(overall)} | {median_fmt(first_w)} | {median_fmt(last_w)} |")
        print()

    if "quality" in by_sub:
        print("## quality (4-task battery)\n")
        print("| tag | passed/total | median wall (s) |")
        print("|---|---|---|")
        for tag, rows in group_by_tag(by_sub["quality"]).items():
            passed = sum(1 for r in rows if r.get("passed"))
            total = len(rows)
            walls = [r["wall_s"] for r in rows if r.get("wall_s") is not None]
            print(f"| {tag} | {passed}/{total} | {median_fmt(walls)} |")
        print()


# ---------------------------------------------------------------------------
# Selftest: offline SSE-parsing + grading fixtures, no network
# ---------------------------------------------------------------------------

FIXTURE_VLLM_STYLE = [
    'data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    '',
    'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
    '',
    'data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
    '',
    'data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    '',
    'data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}',
    '',
    'data: [DONE]',
    '',
]

FIXTURE_LLAMACPP_STYLE = [
    'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
    'data: [DONE]',
]

FIXTURE_NO_USAGE = [
    'data: {"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":"C"},"finish_reason":"stop"}]}',
    'data: [DONE]',
]

FIXTURE_REASONING = [
    'data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"reasoning_content":" some more"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":"Answer: 4"},"finish_reason":"stop"}]}',
    'data: [DONE]',
]


def cmd_selftest(args):
    failures = []

    def check(name, cond, detail=""):
        if not cond:
            failures.append(f"{name}: {detail}")
        print(f"{'PASS' if cond else 'FAIL'} {name}")

    r = consume_stream(iter_sse_json(iter(FIXTURE_VLLM_STYLE)), time.monotonic())
    check("vllm-style content", r["content"] == "Hello world", r["content"])
    check("vllm-style usage_seen", r["usage_seen"] is True)
    check("vllm-style completion_tokens (from usage)", r["completion_tokens"] == 2, r["completion_tokens"])
    check("vllm-style finish_reason", r["finish_reason"] == "stop")

    r = consume_stream(iter_sse_json(iter(FIXTURE_LLAMACPP_STYLE)), time.monotonic())
    check("llamacpp-style content", r["content"] == "Hi there", r["content"])
    check("llamacpp-style usage co-located with final chunk", r["usage_seen"] is True)
    check("llamacpp-style completion_tokens", r["completion_tokens"] == 2, r["completion_tokens"])

    r = consume_stream(iter_sse_json(iter(FIXTURE_NO_USAGE)), time.monotonic())
    check("no-usage content", r["content"] == "ABC", r["content"])
    check("no-usage usage_seen false", r["usage_seen"] is False)
    check("no-usage fallback completion_tokens (chunk count)", r["completion_tokens"] == 3, r["completion_tokens"])

    r = consume_stream(iter_sse_json(iter(FIXTURE_REASONING)), time.monotonic())
    check("reasoning captured separately", r["reasoning_content"] == "Let me think some more", r["reasoning_content"])
    check("reasoning content field", r["content"] == "Answer: 4", r["content"])
    check("reasoning fallback tokens include reasoning deltas", r["completion_tokens"] == 3, r["completion_tokens"])

    p = {}
    apply_reasoning(p, "medium")
    check("apply_reasoning medium", p.get("reasoning_effort") == "medium", p)
    p = {}
    apply_reasoning(p, "low")
    check("apply_reasoning low", p.get("reasoning_effort") == "low", p)
    p = {}
    apply_reasoning(p, "off")
    check("apply_reasoning off", p.get("chat_template_kwargs", {}).get("enable_thinking") is False, p)
    p = {}
    apply_reasoning(p, "default")
    check("apply_reasoning default is no-op", p == {}, p)

    dp = build_decode_prompt("nonce123")
    check("decode prompt contains nonce", "nonce123" in dp)
    check("decode prompt is roughly 1k tokens (chars in [2500,6000))", 2500 < len(dp) < 6000, len(dp))

    pf = build_prefill_prompt(1000, "abc")
    est = count_tokens_estimate(pf)
    check("prefill prompt token estimate near target", 700 < est < 1300, est)
    pf2 = build_prefill_prompt(1000, "xyz")
    check("prefill prompt differs across nonces", pf != pf2)

    pre = build_preamble(2000)
    check("preamble builds nonzero content", len(pre) > 1000, len(pre))

    sp = build_sustained_prompt("nonce456")
    check("sustained prompt contains nonce", "nonce456" in sp)
    check("sustained prompt mentions 3000-line module", "3000" in sp)

    # windowed_tok_s: fabricate a token_times trace with a clear decay -- the
    # first 500 tokens arrive over 5s (100 tok/s), the next 500 over 10s (50
    # tok/s), and confirm the windows reflect that decay.
    ttft = 0.1
    fake_times = [ttft + (i + 1) * (5.0 / 500) for i in range(500)]
    fake_times += [fake_times[-1] + (i + 1) * (10.0 / 500) for i in range(500)]
    windows = windowed_tok_s(fake_times, ttft, 500)
    check("windowed_tok_s produces 2 windows", len(windows) == 2, windows)
    if len(windows) == 2:
        check("windowed_tok_s window 1 ~100 tok/s", 95 < windows[0]["tok_s"] < 105, windows[0])
        check("windowed_tok_s window 2 ~50 tok/s", 45 < windows[1]["tok_s"] < 55, windows[1])
        check("windowed_tok_s detects decay", windows[1]["tok_s"] < windows[0]["tok_s"])
    # a short trace (< window_size) still yields one partial window, not zero
    short_windows = windowed_tok_s(fake_times[:120], ttft, 500)
    check("windowed_tok_s handles a short/partial trace", len(short_windows) == 1 and short_windows[0]["tokens"] == 120, short_windows)
    check("windowed_tok_s handles an empty trace", windowed_tok_s([], ttft, 500) == [])

    # consume_stream now also exposes token_times for the fixtures above
    r = consume_stream(iter_sse_json(iter(FIXTURE_VLLM_STYLE)), time.monotonic())
    check("consume_stream token_times length matches completion_tokens fallback shape",
          len(r["token_times"]) == 2, r["token_times"])

    check("derive_root strips /v1", derive_root("http://x:1/v1") == "http://x:1")
    check("derive_root strips /v1/", derive_root("http://x:1/v1/") == "http://x:1")
    check("derive_root passthrough (no /v1 suffix)", derive_root("http://x:1") == "http://x:1")

    ok, _ = grade(TASKS[0], {"content": "655"})
    check("grade: arithmetic_exact", ok is True)
    ok, _ = grade(TASKS[1], {"content": '{"servidor": "atlas", "ram_gb": 48, "cidade": "Recife"}'})
    check("grade: portuguese_json", ok is True)
    ok, _ = grade(TASKS[2], {"content": "counts[key] = counts.get(key, 0) + 1"})
    check("grade: python_fix (normalized)", ok is True)
    ok, _ = grade(TASKS[3], {"tool_calls": [{"function": {
        "name": "get_weather", "arguments": '{"city": "Sao Paulo"}'}}]})
    check("grade: required_tool", ok is True)
    ok, _ = grade(TASKS[0], {"content": "wrong"})
    check("grade: arithmetic_exact rejects wrong answer", ok is False)

    print()
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} failure(s)")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("SELFTEST OK")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def add_common_args(sp, need_tag=True):
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL,
                     help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    sp.add_argument("--api-key", default=None, help="optional bearer token")
    sp.add_argument("--model", default=DEFAULT_MODEL, help=f"served model name (default {DEFAULT_MODEL})")
    if need_tag:
        sp.add_argument("--tag", required=True, help="arm name, e.g. vllm / llama-cpp / sglang")
    sp.add_argument("--out", default="results.jsonl", help="JSONL file to append results to")
    sp.add_argument("--reasoning", choices=["default", "medium", "low", "off"], default="default")
    sp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-request timeout, seconds")


def build_parser():
    p = argparse.ArgumentParser(
        description="Engine-agnostic benchmark harness for OpenAI-compatible LLM servers")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("decode", help="N single streaming decode requests")
    add_common_args(sp)
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.set_defaults(func=cmd_decode)

    sp = sub.add_parser("prefill", help="cold prefill throughput on a unique nonced prompt")
    add_common_args(sp)
    sp.add_argument("--tokens", type=int, default=30000)
    sp.set_defaults(func=cmd_prefill)

    sp = sub.add_parser("session", help="cumulative multi-turn file-editing session")
    add_common_args(sp)
    sp.add_argument("--turns", type=int, default=8)
    sp.add_argument("--preamble-tokens", type=int, default=0)
    sp.add_argument("--max-tokens", type=int, default=1536)
    sp.set_defaults(func=cmd_session)

    sp = sub.add_parser("concurrent", help="N parallel decode requests, aggregate tok/s")
    add_common_args(sp)
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.set_defaults(func=cmd_concurrent)

    sp = sub.add_parser("sustained", help="one long streaming generation, windowed decode tok/s")
    add_common_args(sp)
    sp.add_argument("--max-tokens", type=int, default=6000)
    sp.add_argument("--window-tokens", type=int, default=500)
    sp.set_defaults(func=cmd_sustained)

    sp = sub.add_parser("quality", help="4-task quality battery (ported from evaluate-qwen38-profiles.py)")
    add_common_args(sp)
    sp.set_defaults(func=cmd_quality)

    sp = sub.add_parser("report", help="markdown comparison table from one or more results.jsonl files")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("selftest", help="offline SSE-parsing + grading self-test, no network")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
