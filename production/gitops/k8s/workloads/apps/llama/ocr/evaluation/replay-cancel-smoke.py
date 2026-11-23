"""Explicit live smoke: one synthetic OCR job, one queued cancellation, exact replays."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from evaluate import (Client, EvaluationError, ROOT, TERMINAL, check_identity,
                      multipart, read_json, require, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9417")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private directory; existing records are never replaced")
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    require(30 <= args.timeout <= 600, "Timeout must be 30..600 seconds")
    runtime = read_json(args.runtime_config, 65_536)
    require(set(runtime) == {"pipeline", "workerConfig"}, "Expected exact preflight runtime snapshot")
    pipeline = runtime["pipeline"]
    require(pipeline.get("executionMode") == "real", "Expected a real pinned pipeline")
    case = next(item for item in read_json(ROOT / "groundtruth.json")["cases"] if item["id"] == "clean-receipt-png")
    original = (ROOT / "fixtures" / "clean-receipt.png").read_bytes()
    require(0 < len(original) <= 10_485_760 and len(original) == case["sizeBytes"]
            and hashlib.sha256(original).hexdigest() == case["originalSha256"], "Synthetic fixture changed")
    require(not args.output.exists(), "Use a new output directory; preserve existing uncertain request keys")
    args.output.mkdir(parents=True, mode=0o700)
    client = Client(args.base_url, args.token_file)
    require(client.port == 9417, "Smoke is restricted to the loopback service on port 9417")
    records = {}
    for name in ("a", "b"):
        request = {"contractVersion": 1, "source": {"documentId": "synthetic-replay-cancel-" + name,
            "originalSha256": case["originalSha256"], "mime": "image/png", "sizeBytes": len(original),
            "originalName": "clean-receipt.png"}, "profile": "receipt-core-v1", "pipelineId": pipeline["id"],
            "localeHints": ["pt-BR"], "retryOf": None}
        records[name] = {"requestKey": str(uuid.uuid4()), "request": request, "pipeline": pipeline}
        write_json(args.output / (name + ".request.json"), records[name])
    cancel = {"requestKey": str(uuid.uuid4())}
    write_json(args.output / "cancel-key.json", cancel)
    write_json(args.output / "runtime-config.json", runtime)
    deadline = time.monotonic() + args.timeout

    def save(name, job, record):
        write_json(args.output / (name + ".job.json"), job)
        write_json(args.output / (name + ".request.json"), record)

    def submit(name, previous=None):
        record = records[name]
        body, content_type = multipart(record["request"], original, "image/png")
        status, job, _ = client.request("POST", "/v1/ocr/jobs", deadline, body,
            {"Content-Type": content_type, "Idempotency-Key": record["requestKey"]})
        require(status == (200 if previous else 202), "Unexpected submission/replay status; preserve recorded keys")
        check_identity(job, record, pipeline, previous)
        write_json(args.output / (name + ("-submit-replay" if previous else "-accepted") + ".envelope.json"), job)
        return job

    def wait(job, name, target):
        while job["state"] not in target:
            require(job["state"] not in TERMINAL, "Job ended before the queued-cancellation setup was ready")
            require(time.monotonic() + 1.1 < deadline, "Smoke deadline expired; preserve recorded keys")
            time.sleep(1.1)
            status, fresh, _ = client.request("GET", "/v1/ocr/jobs/" + job["jobId"], deadline)
            if status in (429, 503):
                continue
            require(status == 200, "Job polling failed; preserve recorded keys")
            check_identity(fresh, records[name], pipeline, job)
            job = fresh
        return job

    a = wait(submit("a"), "a", {"running"})
    b = submit("b")
    require(b["state"] == "queued", "B was not queued; no cancellation was sent")
    cancel.update({"jobId": b["jobId"], "command": {"version": b["version"]}})
    write_json(args.output / "cancel-command.json", cancel)  # Persist exact CAS command before POST.
    cancel_body = json.dumps(cancel["command"], separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json", "Idempotency-Key": cancel["requestKey"]}
    path = "/v1/ocr/jobs/" + b["jobId"] + "/cancel"
    status, cancelled, _ = client.request("POST", path, deadline, cancel_body, headers)
    write_json(args.output / "cancel-response.envelope.json", cancelled)
    require(status == 200 and cancelled.get("state") == "cancelled", "Queued cancellation lost its version race; no running cancellation will be attempted")
    check_identity(cancelled, records["b"], pipeline, b)
    save("b", cancelled, records["b"])
    status, replay, _ = client.request("POST", path, deadline, cancel_body, headers)
    require(status == 200 and replay == cancelled, "Exact cancel replay changed its immutable response")
    save("b-cancel-replay", replay, records["b"])
    a = wait(a, "a", TERMINAL)
    require(a["state"] == "completed", "Synthetic job A did not complete successfully")
    save("a", a, records["a"])
    replay = submit("a", a)
    require(replay == a, "Same-key terminal submission replay changed the completed job")
    save("a-submit-replay", replay, records["a"])
    for name, terminal in (("a", a), ("b", cancelled)):
        # Respect per-job read polling limit after the final running-job poll.
        require(time.monotonic() + 1.1 < deadline, "Smoke deadline expired before final lookup")
        time.sleep(1.1)
        status, lookup, _ = client.request("GET", "/v1/ocr/requests/" + records[name]["requestKey"], deadline)
        require(status == 200 and lookup == terminal, "Request-key lookup changed the immutable terminal job")
        save(name + "-lookup", lookup, records[name])
    write_json(args.output / "report.json", {"suite": "same-key-replay-and-queued-cancel", "passed": True,
        "jobsSubmitted": 2, "runningJobCancellationAttempted": False, "fullContractValidation": "pending"})
    print("Replay and queued-cancellation smoke passed; saved six terminal envelopes for shared-schema validation.")


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as error:
        print(str(error))
        raise SystemExit(2)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        print("Smoke stopped; inspect saved records. Do not replace uncertain request/cancel keys.")
        raise SystemExit(2)
