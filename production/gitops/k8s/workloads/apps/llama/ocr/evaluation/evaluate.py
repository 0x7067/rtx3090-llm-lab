"""Bounded live smoke evaluation; ONLY explicit invocation sends synthetic files.

HTTP uses only the standard library, literal loopback, and no proxy/redirects.
Expected field answers stay local and are never sent to the OCR service.
Run the companion validate-results.ts for complete shared-contract validation.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import threading
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parent
MAX_INPUT = 10_485_760
MAX_OUTPUT = 10_485_760
TERMINAL = {"completed", "partial", "failed", "cancelled"}


class EvaluationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def read_json(path: Path, limit=MAX_OUTPUT):
    require(path.is_file() and path.stat().st_size <= limit, "Missing or oversized evaluation JSON file")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def write_json(path: Path, data):
    # Atomic local records permit a resumed invocation to reuse its exact key.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Client:
    def __init__(self, base_url, token_file: Path):
        parsed = urlsplit(base_url)
        require(parsed.scheme == "http" and parsed.path in ("", "/")
                and not parsed.query and not parsed.fragment and parsed.username is None,
                "Evaluation requires an HTTP loopback origin without credentials or path")
        try:
            require(ipaddress.ip_address(parsed.hostname).is_loopback, "Only literal loopback addresses are authorized")
            self.host, self.port = parsed.hostname, parsed.port or 80
        except ValueError:
            raise EvaluationError("Only literal loopback addresses are authorized") from None
        info = token_file.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and not info.st_mode & 0o137 and info.st_size <= 4097,
                "Token must be an owned regular file; no execute, group-write, or other permissions")
        self.token = token_file.read_text().strip()
        require(24 <= len(self.token) <= 4096 and self.token.isascii() and not re.search(r"\s", self.token),
                "Invalid service token")

    def request(self, method, path, deadline, body=None, headers=None):
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Evaluation deadline expired; accepted job may still be running")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=min(30, remaining))
        def interrupt():
            sock = connection.sock
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
        watchdog = threading.Timer(min(30, remaining), interrupt)
        watchdog.daemon = True
        watchdog.start()
        try:
            connection.request(method, path, body=body, headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/json",
                **(headers or {}),
            })
            response = connection.getresponse()
            length = response.getheader("Content-Length")
            require(length is None or length.isdigit() and int(length) <= MAX_OUTPUT,
                    "Response exceeds the result limit")
            chunks, size = [], 0
            while True:
                require(time.monotonic() < deadline, "Evaluation response deadline expired")
                if connection.sock:
                    connection.sock.settimeout(min(10, max(0.01, deadline-time.monotonic())))
                chunk = response.read(min(64 * 1024, MAX_OUTPUT + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                require(size <= MAX_OUTPUT, "Response exceeds the result limit")
            require(response.getheader("Cache-Control") == "no-store", "Service omitted no-store")
            require((response.getheader("Content-Type") or "").split(";")[0] == "application/json",
                    "Expected a JSON service response")
            retry = response.getheader("Retry-After")
            retry_seconds = min(10, max(1, int(retry))) if retry and retry.isdigit() else 1
            data = json.loads(b"".join(chunks).decode("utf-8"))
            return response.status, data, retry_seconds
        except (OSError, http.client.HTTPException):
            raise EvaluationError("Transport interrupted; resume with the recorded idempotency key") from None
        finally:
            watchdog.cancel()
            connection.close()


def multipart(metadata, original, mime):
    boundary = "ocr-eval-" + uuid.uuid4().hex
    prefix = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n"
        "Content-Type: application/json\r\n\r\n"
    ).encode()
    middle = (
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"synthetic-original\"\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    data = prefix + json.dumps(metadata, separators=(",", ":")).encode() + middle + original + f"\r\n--{boundary}--\r\n".encode()
    return data, "multipart/form-data; boundary=" + boundary


def check_identity(job, record, pipeline, previous=None):
    require(isinstance(job, dict) and job.get("contractVersion") == 1, "Invalid job response")
    require(job.get("source") == record["request"]["source"] and job.get("pipeline") == pipeline,
            "Source or pinned pipeline identity changed")
    require(job.get("requestKey") == record["requestKey"] and job.get("profile") == "receipt-core-v1"
            and job.get("retryOf") is None and job.get("attempt") == 1, "Job request identity changed")
    try:
        require(str(uuid.UUID(job["jobId"])) == job["jobId"], "Invalid job ID")
    except (KeyError, ValueError, AttributeError):
        raise EvaluationError("Invalid job ID") from None
    require(type(job.get("version")) is int and job["version"] >= 1, "Invalid job version")
    require(job.get("state") in TERMINAL | {"queued", "running", "cancelling"}, "Invalid job state")
    if previous:
        require(job["jobId"] == previous["jobId"] and job["version"] >= previous["version"], "Job identity/version regressed")


def compare(case, job):
    expected = case["expected"]
    checks = []
    def check(name, actual, wanted):
        checks.append({"check": name, "actual": actual, "expected": wanted, "matches": actual == wanted})
    check("terminalState", job["state"], expected["state"])
    result = job.get("result")
    check("realExecution", result.get("mode") if isinstance(result, dict) else None, "real")
    check("ocrStageReported", "ocr" in result.get("methodsUsed", []) if isinstance(result, dict) else False, True)
    if isinstance(result, dict):
        check("pageCount", result.get("pageCount"), len(expected["pageDimensions"]))
        check("referenceDimensions", [[page.get("width"), page.get("height")] for page in result.get("pages", [])],
              expected["pageDimensions"])
        for name, wanted in expected["fields"].items():
            actual = result.get("fields", {}).get(name, {})
            check(name + ".status", actual.get("status"), wanted["status"])
            check(name + ".values", [item.get("value") for item in actual.get("candidates", [])], wanted["values"])
        warnings = {item.get("code") for item in result.get("warnings", [])}
        for warning in expected["requiredWarnings"]:
            check("warning." + warning, warning in warnings, True)
    return {"caseId": case["id"], "jobId": job["jobId"], "state": job["state"],
            "expectedExtractionMatches": all(check["matches"] for check in checks), "checks": checks,
            "serviceErrorCode": (job.get("error") or {}).get("code"),
            "fullContractValidation": "Pending companion validate-results.ts; this comparison is not schema validation."}


def run_case(client, case, pipeline, output, timeout):
    path = ROOT / case["file"]
    require(path.resolve().parent == (ROOT / "fixtures").resolve(), "Fixture escaped its approved directory")
    original = path.read_bytes()
    require(0 < len(original) <= MAX_INPUT and len(original) == case["sizeBytes"]
            and hashlib.sha256(original).hexdigest() == case["originalSha256"], "Synthetic original changed after ground truth was written")
    request = {"contractVersion": 1, "source": {
        "documentId": "synthetic-eval-" + case["id"], "originalSha256": case["originalSha256"],
        "mime": case["mime"], "sizeBytes": len(original), "originalName": path.name,
    }, "profile": "receipt-core-v1", "pipelineId": pipeline["id"], "localeHints": ["pt-BR"], "retryOf": None}
    record_path, envelope_path = output / (case["id"] + ".request.json"), output / (case["id"] + ".job.json")
    if record_path.exists():
        record = read_json(record_path)
        require(record.get("request") == request and record.get("pipeline") == pipeline, "Existing request differs; do not reuse or replace its uncertain key")
    else:
        record = {"requestKey": str(uuid.uuid4()), "request": request, "pipeline": pipeline}
        write_json(record_path, record)
    if envelope_path.exists():
        saved = read_json(envelope_path)
        check_identity(saved, record, pipeline)
        if saved["state"] in TERMINAL:
            return compare(case, saved)
    deadline, started = time.monotonic() + timeout, time.monotonic()
    status, job, delay = client.request("GET", "/v1/ocr/requests/" + record["requestKey"], deadline)
    if status == 404:
        body, content_type = multipart(request, original, case["mime"])
        # Same-key acceptance is retried only after bounded explicit rejections.
        for attempt in range(4):
            status, job, delay = client.request("POST", "/v1/ocr/jobs", deadline, body, {
                "Content-Type": content_type, "Idempotency-Key": record["requestKey"],
            })
            if status in (200, 202):
                break
            require(status in (429, 503) and attempt < 3, "Service did not accept the synthetic request")
            time.sleep(min(delay, max(0, deadline - time.monotonic())))
    require(status in (200, 202), "Request lookup/acceptance failed; preserve the recorded key")
    check_identity(job, record, pipeline)
    write_json(envelope_path, job)
    polls = 0
    while job["state"] not in TERMINAL:
        delay = max(delay, min(10, 1 + polls // 5))
        require(time.monotonic() + delay < deadline, "Evaluation deadline expired; accepted job may still be running")
        time.sleep(delay)
        status, fresh, delay = client.request("GET", "/v1/ocr/jobs/" + job["jobId"], deadline)
        if status in (429, 503):
            continue
        require(status == 200, "Job polling failed; preserve the recorded key")
        check_identity(fresh, record, pipeline, job)
        job = fresh
        write_json(envelope_path, job)
        polls += 1
    result = compare(case, job)
    result["observedSecondsThisInvocation"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True,
                        help="Exact JSON emitted by the deployed container's successful preflight --emit-runtime")
    parser.add_argument("--output", type=Path, required=True, help="Private output directory; reuse it to resume the SAME recorded keys")
    parser.add_argument("--case", action="append", default=[], help="Run selected case IDs; default runs all six sequentially")
    parser.add_argument("--job-timeout", type=int, default=360)
    args = parser.parse_args()
    require(30 <= args.job_timeout <= 600, "Job evaluation timeout must be 30..600 seconds")
    runtime = read_json(args.runtime_config, 65_536)
    require(set(runtime) == {"pipeline", "workerConfig"}, "Expected an exact preflight runtime snapshot")
    pipeline = runtime["pipeline"]
    require(pipeline.get("executionMode") == "real" and pipeline.get("ocrModel", {}).get("id") == "PaddlePaddle/PaddleOCR-VL-1.6"
            and pipeline.get("inferenceEngine", {}).get("name") == "vllm"
            and re.fullmatch(r"[a-f0-9]{64}", pipeline.get("fingerprint", "")) is not None
            and pipeline.get("id") == "paddleocr-vl-1.6-" + pipeline["fingerprint"][:16], "Expected pinned real OCR pipeline")
    groundtruth = read_json(ROOT / "groundtruth.json")
    case_ids = {case["id"] for case in groundtruth["cases"]}
    require(len(case_ids) == len(groundtruth["cases"]) and set(args.case) <= case_ids, "Unknown/duplicate evaluation case")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not args.output.is_symlink() and not args.output.stat().st_mode & 0o077, "Output directory must be private (mode 0700)")
    client = Client(args.base_url, args.token_file)
    write_json(args.output / "runtime-config.json", runtime)
    results = []
    selected_ids = [case["id"] for case in groundtruth["cases"] if not args.case or case["id"] in args.case]
    for case in groundtruth["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        print("Evaluating synthetic case: " + case["id"], flush=True)
        result = run_case(client, case, pipeline, args.output, args.job_timeout)
        results.append(result)
        report = {"suite": "six-case-synthetic-deployment-smoke", "isAccuracyBenchmark": False,
                  "requestedCaseIds": selected_ids, "evaluatedCaseCount": len(results),
                  "allSixCasesEvaluated": len(results) == len(groundtruth["cases"]),
                  "results": results, "allExpectedExtractionsMatch": all(item["expectedExtractionMatches"] for item in results),
                  "fullContractValidation": "pending", "humanEvidenceOverlayReview": "pending"}
        write_json(args.output / "report.json", report)
        print(json.dumps({"caseId": case["id"], "state": result["state"],
                          "expectedExtractionMatches": result["expectedExtractionMatches"]}), flush=True)
    return 0 if all(result["expectedExtractionMatches"] for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        print(str(error))
        print("No replacement key was submitted. Resume the same output directory if acceptance is uncertain.")
        raise SystemExit(2)
    except (ValueError, OSError, KeyError, TypeError):
        # No token, document text, provider body, or untrusted exception details.
        print("Evaluation stopped; inspect saved records and resume the same output directory. No replacement key was submitted.")
        raise SystemExit(2)
