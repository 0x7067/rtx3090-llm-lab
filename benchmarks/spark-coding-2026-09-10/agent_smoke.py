#!/usr/bin/env python3
"""Synthetic file-edit/test episodes; generated code executes only in Docker."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HARNESS = Path(__file__).parents[1] / "paired-harness" / "harness.py"
spec = importlib.util.spec_from_file_location("paired_harness", HARNESS)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

TOOLS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": {"type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False}}}
    for name, description, properties in [
        ("read_file", "Read a file from the working directory.", {"path": {"type": "string"}}),
        ("write_file", "Replace the source file with complete Python code.",
         {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("run_tests", "Run the visible tests against the current source.", {}),
    ]
]


def execute(call, files, case, image):
    """Only the named in-memory source is writable; test fixtures are immutable."""
    name = call["name"]
    args = json.loads(call["arguments"])
    required = {"read_file": {"path"}, "write_file": {"path", "content"}, "run_tests": set()}
    if name not in required or not isinstance(args, dict) or set(args) != required[name]:
        raise ValueError("unknown tool or incorrect arguments")
    if any(not isinstance(value, str) for value in args.values()):
        raise ValueError("arguments must be strings")
    if name == "run_tests":
        score, detail = harness.python_grade(files[case["path"]], case["visible_tests"], image)
        return {"passed": bool(score), "detail": detail}
    path = args["path"]
    if path not in files:
        raise ValueError("unknown path; available files: " + ", ".join(files))
    if name == "read_file":
        return {"content": files[path]}
    if path != case["path"] or len(args["content"].encode()) > 65536:
        raise ValueError("only the source file is writable, with a 64 KiB limit")
    files[path] = args["content"]
    return {"written": path}


def episode(case, arm, defaults, image):
    files = {case["path"]: case["source"], "tests.py": case["visible_tests"]}
    messages = [
        {"role": "system", "content": "Fix the reported bug using the file tools. Read the source and tests, write the fix, then run_tests. Tests execute in the same Python namespace as the source. Only the source is writable. Finish with a brief summary."},
        {"role": "user", "content": case["issue"] + " Files: " + ", ".join(files)},
    ]
    trace = []
    for turn in range(8):
        result = harness.request(arm, {**defaults, "messages": messages, "tools": TOOLS,
            "tool_choice": "auto", "stream": True, "stream_options": {"include_usage": True}}, 180)
        trace.append({"response": result, "tools": []})
        if result["status"] != "ok" or result.get("finish_reason") not in ("stop", "tool_calls"):
            break
        calls = result["tool_calls"]
        if not calls:
            break
        # The existing streaming adapter returns names/arguments without IDs.
        # Assign stable IDs to reconstruct the next request's tool transcript.
        messages.append({"role": "assistant", "content": result["content"] or None,
            "tool_calls": [{"id": f"call_{turn}_{i}", "type": "function", "function": call}
                           for i, call in enumerate(calls)]})
        for i, call in enumerate(calls):
            try:
                output = execute(call, files, case, image)
            except (ValueError, TypeError, KeyError) as error:
                output = {"error": str(error)}
            trace[-1]["tools"].append({"call": call, "output": output})
            messages.append({"role": "tool", "tool_call_id": f"call_{turn}_{i}",
                             "content": json.dumps(output)})
    score, detail = harness.python_grade(files[case["path"]], case["hidden_tests"], image)
    exercised = {item["call"]["name"] for step in trace for item in step["tools"]
                 if "error" not in item["output"]}
    workflow = {"read_file", "write_file", "run_tests"} <= exercised
    return {"id": case["id"], "score": score, "workflow_completed": workflow,
            "grade_detail": detail, "final_source": files[case["path"]], "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    arm = json.loads(args.arms.read_text())[0]
    defaults = {"temperature": args.temperature, "top_p": .95, "top_k": 0, "seed": 42, "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": not args.no_thinking}, "reasoning_budget_tokens": args.budget}
    args.out.mkdir(parents=True, exist_ok=False)
    harness.write_new(args.out / "manifest.json", {"suite": suite, "arm": arm, "defaults": defaults,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest()})
    with (args.out / "results.jsonl").open("x") as output:
        for case in suite["cases"]:
            result = episode(case, arm, defaults, suite["code_image"])
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(case["id"], "score", result["score"], "workflow", result["workflow_completed"], flush=True)


if __name__ == "__main__":
    main()
