#!/usr/bin/env python3
"""End the parent window at the completed Orca stage, invoking its restoration."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
relative = 'rtx3090-llm-lab/benchmarks/orca-thinkingcap-2026-10-07/run.py'
while True:
    if (ROOT / 'state-restored.json').exists():
        print('Parent window already restored.', flush=True)
        break
    path = ROOT / 'orca_vllm-concurrency.jsonl'
    if path.exists():
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        except json.JSONDecodeError:
            rows = []
        if len(rows) == 14:
            candidates = []
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    argv = (entry / 'cmdline').read_bytes().split(b'\0')
                except (OSError, PermissionError):
                    continue
                if relative.encode() in argv and argv[0].endswith(b'python3'):
                    candidates.append(int(entry.name))
            if len(candidates) != 1:
                raise RuntimeError('Expected exactly one owned parent Python runner')
            os.kill(candidates[0], signal.SIGTERM)
            print('Completed Orca stage preserved; parent restoration requested.', flush=True)
            break
    time.sleep(0.5)
