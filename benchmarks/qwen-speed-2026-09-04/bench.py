#!/usr/bin/env python3
"""Run the existing engine trial benchmark with the project HTTP user agent."""
import importlib.util
import ast
import hashlib
import json
import os
import re
import threading
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("engine_bench", Path(__file__).resolve().parents[1] / "engine-trial-2026-09-02/bench/bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
original_headers = bench.build_headers
def headers(api_key):
    return {**original_headers(api_key), "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"}
bench.build_headers = headers
original_stream = bench.stream_chat
output_lock = threading.Lock()
def stream(*args, **kwargs):
    result = original_stream(*args, **kwargs)
    path = os.environ.get("QWEN_OUTPUTS")
    if path:
        content = result["content"]
        code = re.search(r"```python\s*\n(.*?)```", content, re.S)
        syntax_ok = None
        if code:
            try:
                ast.parse(code.group(1))
                syntax_ok = True
            except SyntaxError:
                syntax_ok = False
        record = {"content": content, "reasoning": result.get("reasoning_content"),
                  "sha256": hashlib.sha256(content.encode()).hexdigest(),
                  "finish_reason": result["finish_reason"], "python_syntax_ok": syntax_ok}
        with output_lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return result
bench.stream_chat = stream
CONCURRENT_TASKS = [
    "Implement a Python TTL cache with a size bound, LRU eviction, an injected monotonic clock, and thread-safe get/set/delete operations. Include unit tests for expiration boundaries and concurrent access.",
    "Implement a TypeScript parser for server-sent events from arbitrary UTF-8 byte chunks. Handle multiline data, event names, IDs, comments, reconnect delays, and incomplete final events. Include tests for split multibyte characters.",
    "Implement a Rust directed dependency graph with deterministic topological sorting and informative cycle detection. Include tests for disconnected vertices, duplicate edges, self loops, and cycles.",
    "Implement a Go HTTP retry transport with capped exponential backoff and jitter, cancellation, Retry-After support, and safe handling of non-idempotent methods. Include tests with a fake server and injected clock.",
]
concurrent_index = 0
def diverse_prompt(nonce=None):
    global concurrent_index
    with output_lock:
        task = CONCURRENT_TASKS[concurrent_index % len(CONCURRENT_TASKS)]
        concurrent_index += 1
    return "Write complete code and explain the edge cases. " + task
if __name__ == "__main__":
    if "concurrent" in sys.argv:
        bench.build_decode_prompt = diverse_prompt
    bench.main()
