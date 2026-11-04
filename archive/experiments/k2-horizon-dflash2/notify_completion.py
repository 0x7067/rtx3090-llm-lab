#!/usr/bin/env python3
"""Notify the originating Codex chat when one systemd job invocation ends."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def command(*args):
    return subprocess.run(args, text=True, capture_output=True, timeout=60)


def service_status(unit):
    result = command(
        "systemctl", "--user", "show", unit, "--no-pager",
        "-p", "LoadState", "-p", "ActiveState", "-p", "SubState",
        "-p", "InvocationID", "-p", "Result", "-p", "ExecMainStatus",
    )
    status = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode and status.get("LoadState") != "not-found":
        raise RuntimeError(f"Cannot inspect {unit}: {result.stderr.strip()}")
    if not status.get("ActiveState"):
        raise RuntimeError(f"No service state returned for {unit}")
    return status


def completion_result(status, invocation, journal):
    # A transient unit may disappear after success; its invocation-scoped
    # journal remains authoritative about the job and its API restoration.
    # capture/regeneration print "Regeneration finished"; jobs wrapped in
    # with-local-api-paused.sh print "Command finished" instead.
    endings = re.findall(r"(?:Regeneration|Command) finished with exit=(\d+);", journal)
    if endings:
        return "succeeded" if endings[-1] == "0" else f"failed (exit {endings[-1]})"
    current = status.get("InvocationID")
    if current and current != invocation:
        return "was replaced by another invocation; inspect the original job logs"
    if status.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}:
        return None
    if status.get("ActiveState") == "failed":
        return f"failed ({status.get('Result', 'unknown')}, exit {status.get('ExecMainStatus', '?')})"
    return "stopped without a recorded completion; inspect the job logs"


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--codex", default="/home/denguinho/.local/bin/codex")
    parser.add_argument("--subject", default="Regeneration",
                        help="What finished, as the message's sentence subject")
    args = parser.parse_args()
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if (state["unit"], state["thread"]) != (args.unit, args.thread):
            raise ValueError("Notification state belongs to a different job or chat")
        if state.get("delivered"):
            print("Completion notification was already queued", flush=True)
            return
    else:
        status = service_status(args.unit)
        invocation = status.get("InvocationID")
        if not invocation:
            raise RuntimeError("Start monitoring while the target invocation is still identifiable")
        state = {"unit": args.unit, "thread": args.thread, "invocation": invocation, "delivered": False}
        args.state.parent.mkdir(parents=True, exist_ok=True)
        save_state(args.state, state)
    invocation = state["invocation"]
    print(f"Monitoring {args.unit}, invocation {invocation}, for this chat", flush=True)
    while True:
        status = service_status(args.unit)
        journal = command(
            "journalctl", "--user", f"_SYSTEMD_INVOCATION_ID={invocation}",
            "--no-pager", "-o", "cat", "-n", "30",
        )
        if journal.returncode:
            raise RuntimeError("Cannot read the watched job's journal")
        result = completion_result(status, invocation, journal.stdout)
        if result is None:
            time.sleep(30)
            continue
        message = (
            "Automated K2 job notification requested by the user in this chat. "
            f"{args.subject} {result}. Service: {args.unit}; invocation: {invocation}. "
            "Check the current service result, the job's output artifacts and local API "
            "restoration, then report the outcome here. Continue the existing K2 task "
            "under the user's recorded maintenance authorization. "
            "Resume details: /data/docker-services/rtx3090-llm-lab/experiments/k2-horizon-dflash2/RUNBOOK.md."
        )
        delivered = command(args.codex, "queue", "--thread", args.thread, "--message", message)
        if delivered.returncode:
            raise RuntimeError(f"Cannot queue chat notification: {delivered.stderr.strip()}")
        state.update(delivered=True, result=result, receipt=delivered.stdout.strip())
        save_state(args.state, state)
        print(delivered.stdout.strip(), flush=True)
        return


if __name__ == "__main__":
    main()
