#!/usr/bin/env python3
"""Freeze the bake-off plan: the 2026-09-30 reset suite (58 cases, 16,384-token
output cap) unchanged except for the code-test image, which must be re-pinned
because the original python:3.12-slim image is no longer on this host."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / "paired-harness" / "harness.py"
SOURCE_PLAN = ROOT.parent / "local-llm-reset-2026-09-30" / "plan-16384.json"
CODE_IMAGE_TAG = "python:3.12-slim"


def main():
    suite = json.loads(SOURCE_PLAN.read_text())["suite"]
    image_id = subprocess.run(
        ["docker", "image", "inspect", CODE_IMAGE_TAG, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True).stdout.strip()
    suite["provenance"]["code_image_note"] = (
        f"re-pinned from {suite['code_image']} (no longer present) to {CODE_IMAGE_TAG} {image_id}")
    suite["code_image"] = image_id
    suite["description"] += " Re-used unchanged for the 2026-10-14 Qwen3.8 engine bake-off."
    (ROOT / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    plan = ROOT / "plan.json"
    plan.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(HARNESS), "freeze", "--suite", str(ROOT / "suite.json"),
                    "--arms", str(ROOT / "arms.json"), "--campaign", "qwen38-bakeoff-2026-10-14",
                    "--out", str(plan)], check=True)
    p = json.loads(plan.read_text())
    print(f"plan {plan} sha256={p['sha256']} jobs={len(p['jobs'])} arms={[a['name'] for a in p['arms']]}")


if __name__ == "__main__":
    main()
