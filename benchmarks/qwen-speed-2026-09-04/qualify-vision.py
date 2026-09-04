#!/usr/bin/env python3
"""Reuse the repository's synthetic vision fixtures on the test endpoint."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
label = sys.argv[1]
source = (ROOT.parents[2] / 'scripts/benchmark-llama-vision.sh').read_text()
source = source.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
source = source.replace('headers={"Content-Type": "application/json"}', 'headers={"Content-Type": "application/json", "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"}')
source = source.replace('"temperature": 0.1,', '"temperature": 0.1, "reasoning_effort": "medium",')
sys.argv = ['vision', 'http://127.0.0.1:18089', 'qwen3.8-27b', '192', '1024',
            '180', '1', 'qualify', str(ROOT), label, str(ROOT / (label+'-vision.jsonl'))]
exec(compile(source, 'existing-vision-fixtures', 'exec'), {'__name__':'__main__'})
