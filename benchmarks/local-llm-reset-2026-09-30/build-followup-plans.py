#!/usr/bin/env python3
import copy
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / "paired-harness" / "harness.py"
CASE_IDS = {
    "code-diff-keys",
    "code-lru",
    "code-merge-config",
    "code-redact",
    "code-topological",
    "episode-cache-first",
    "episode-cache-next",
    "extract-missing",
    "extract-pt-correction",
    "extract-quoted",
    "support-reason-boundary",
    "support-relevant-db",
}
POLICIES = ("control", "budget-512", "budget-1024", "budget-2048", "nonthinking-simple")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def corrected_cases(source):
    cases = [copy.deepcopy(case) for case in source["cases"] if case["id"] in CASE_IDS]
    if {case["id"] for case in cases} != CASE_IDS:
        raise ValueError("source suite is missing a selected case")
    by_id = {case["id"]: case for case in cases}
    by_id["code-redact"]["request"]["messages"][0]["content"] = (
        "Implement redact(value, sensitive), recursively copying JSON-like dictionaries/lists. "
        "sensitive is a list of strings. Replace any dictionary value whose key "
        "case-insensitively matches any sensitive string with \"[REDACTED]\". Preserve unrelated "
        "scalars and do not mutate input. Sensitive matching is exact, not substring. Return only "
        "complete Python code using the standard library."
    )
    by_id["extract-quoted"]["request"]["messages"][0]["content"] = (
        "Extract the text between the delimiters into a JSON object whose key is literally "
        "\"value\": <<<a\"b\\c>>>. Preserve the extracted text exactly. Return only JSON."
    )
    return cases


def make_suite(source, policy):
    suite = copy.deepcopy(source)
    suite["description"] = (
        "Targeted Strata quality follow-up with corrected contracts, three fixed seeds, and one "
        f"declared request policy: {policy}."
    )
    suite["min_clusters"] = 2
    suite["repeats"] = 1
    suite["seeds"] = [42, 314, 2718]
    suite["request_defaults"]["max_tokens"] = 4096
    suite["cases"] = corrected_cases(source)
    for case in suite["cases"]:
        request = case["request"]
        if policy.startswith("budget-") and case["tier"] == "coding":
            request["chat_template_kwargs"] = {"enable_thinking": True}
            request["reasoning_budget_tokens"] = int(policy.removeprefix("budget-"))
        if policy == "nonthinking-simple" and case["tier"] in {"extraction", "relevance"}:
            request.pop("reasoning_effort", None)
            request.update(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                presence_penalty=1.5,
                chat_template_kwargs={"enable_thinking": False},
            )
    return suite


def main():
    source = read(ROOT / "suite.json")
    arms = read(ROOT / "arms-post-upgrade.json")
    strata = next(arm for arm in arms if arm["name"] == "strata_flash_next")
    out = ROOT / "followup-plans"
    out.mkdir(exist_ok=True)
    write(out / "arms-strata.json", [strata])
    for policy in POLICIES:
        suite_path = out / f"suite-{policy}.json"
        plan_path = out / f"plan-{policy}.json"
        write(suite_path, make_suite(source, policy))
        if plan_path.exists():
            plan_path.unlink()
        subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "freeze",
                "--suite",
                str(suite_path),
                "--arms",
                str(out / "arms-strata.json"),
                "--campaign",
                f"local-llm-reset-strata-{policy}-2026-10-01",
                "--out",
                str(plan_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
