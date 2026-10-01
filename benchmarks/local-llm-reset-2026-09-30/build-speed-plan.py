#!/usr/bin/env python3
import copy
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / "paired-harness" / "harness.py"
ARM_NAMES = {"strata_flash_next", "orca_vllm"}


def main():
    source = json.loads((ROOT / "suite-post-upgrade.json").read_text())
    suite = copy.deepcopy(source)
    suite["description"] = (
        "Matched three-seed short and medium prefill/decode calibration for Strata and OrcaSAQ2."
    )
    suite["repeats"] = 1
    suite["seeds"] = [42, 314, 2718]
    suite["cases"] = [case for case in suite["cases"] if case["phase"] == "performance"]
    arms = [
        arm
        for arm in json.loads((ROOT / "arms-post-upgrade.json").read_text())
        if arm["name"] in ARM_NAMES
    ]
    out = ROOT / "speed-plans"
    out.mkdir(exist_ok=True)
    suite_path = out / "suite-speed.json"
    arms_path = out / "arms-speed.json"
    plan_path = out / "plan-speed.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    arms_path.write_text(json.dumps(arms, indent=2) + "\n")
    plan_path.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "freeze",
            "--suite",
            str(suite_path),
            "--arms",
            str(arms_path),
            "--campaign",
            "local-llm-speed-tuning-2026-10-01",
            "--out",
            str(plan_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
