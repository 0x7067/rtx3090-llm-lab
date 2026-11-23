"""Resolve baseline provenance and verify it against its saved request hash."""
import json
from pathlib import Path
import sys

import long_context


def resolve(run, seen=()):
    run = Path(run).resolve()
    if run in seen:
        raise ValueError(f"Baseline provenance cycle at {run}")
    if (run / "baseline-exit").read_text().strip() != "0":
        raise ValueError(f"Baseline failed: {run}")
    metadata = run / "baseline/trial.json"
    source_file = run / "baseline-source.txt"
    if metadata.exists():
        trial = json.loads(metadata.read_text())["trial_id"]
    elif source_file.exists():
        source = Path(source_file.read_text().strip())
        # Historical receipts contain their original worktree's absolute path.
        # Archived runs keep the same sibling directory names and request hashes.
        if not source.is_dir():
            source = run.parent / source.name
        trial = resolve(source, (*seen, run))
    else:
        trial = json.loads((run / "claim.json").read_text())["owner"]
    if not isinstance(trial, str) or not trial:
        raise ValueError(f"Missing baseline trial identifier: {run}")
    messages = long_context.conversation("A", long_context.KEYS[0], trial)
    expected = long_context.harness.digest(long_context.request_body(messages, long_context.KEYS[0][0]))
    rows = [json.loads(line) for line in (run / "baseline/long-context.jsonl").read_text().splitlines()]
    hashes = [row["request_sha256"] for row in rows if row["name"] == "a-cold"]
    if hashes != [expected]:
        raise ValueError(f"Baseline trial does not match its saved request hash: {run}")
    return trial


if __name__ == "__main__":
    print(resolve(Path(sys.argv[1])))
