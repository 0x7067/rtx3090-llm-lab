import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import harness as h


def stream(events):
    return [line for event in events for line in
            [("data: " + (json.dumps(event) if event is not None else "[DONE]") + "\n").encode(), b"\n"]]


def result(content="OK", finish="stop"):
    return h.consume(h.sse_events(stream([
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": content}, "finish_reason": finish}]},
        {"usage": {"prompt_tokens": 10, "completion_tokens": 20},
         "timings": {"predicted_n": 20, "predicted_ms": 500, "predicted_per_second": 38}}, None])), time.monotonic())


def fixture():
    suite = h.read(Path(__file__).with_name("suite-smoke.json"))
    suite["cases"] = [{"id": "one", "tier": "reasoning", "cluster": "source-one", "phase": "quality",
                       "request": {"messages": [{"role": "user", "content": "Reply OK."}]},
                       "grader": {"kind": "exact", "expected": "OK"}}]
    suite["bootstrap_samples"] = 100
    arms = [{"name": name, "model": name, "base_url": "http://localhost:1/v1", "metadata": {"engine": "fixture"}}
            for name in ("control", "candidate")]
    return suite, arms


class Tests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PAIRED_DOCKER_TESTS") == "1", "set PAIRED_DOCKER_TESTS=1 for sandbox checks")
    def test_generated_python_behavior_in_sandbox(self):
        image = h.read(Path(__file__).with_name("suite-smoke.json"))["code_image"]
        for code, expected in (("def double(x): return x*2", 1),
                               ("```python\ndef double(x): return x*2\n```", 1),
                               ("def double(x): return x", 0),
                               ("import sys; sys.exit(0)", 0), ("def double(:", 0)):
            grader = {"kind": "python", "tests": "assert double(3) == 6"}
            score, detail = h.grade(grader, result(code), image)
            self.assertEqual(score, expected, detail)

    def test_fragmented_tools_reasoning_and_usage(self):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "I should read it."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th":"a.py"}'}}]}, "finish_reason": "tool_calls"}]},
            {"usage": {"completion_tokens": 17, "prompt_tokens": 50}}, None]
        r = h.consume(h.sse_events(stream(events)), time.monotonic())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["usage"]["completion_tokens"], 17)
        self.assertIsNone(r["server_decode_tps"])
        self.assertEqual(h.grade({"kind": "tool", "expected": [{"name": "read_file", "arguments": {"path": "a.py"}}]}, r, "")[0], 1)
        self.assertGreater(r["ttft_s"], 0)

    def test_no_token_estimates_or_role_only_ttft(self):
        r = h.consume(h.sse_events(stream([{"choices": [{"delta": {"role": "assistant"}, "finish_reason": "stop"}]}, None])), time.monotonic())
        self.assertIsNone(r["ttft_s"])
        self.assertIsNone(r["e2e_output_tps"])
        self.assertIsNone(r["server_decode_tps"])
        self.assertEqual(result()["server_decode_tps"], 38)

    def test_one_token_prefill_has_no_decode_rate(self):
        r = h.consume(iter([{"timings": {"predicted_n": 1, "predicted_ms": 0.001, "predicted_per_second": 1000000},
                            "choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}, None]), time.monotonic())
        self.assertIsNone(r["server_decode_tps"])

    def test_multiline_sse_and_malformed_stream(self):
        self.assertEqual(list(h.sse_events([b": comment\n", b'data: {"choices":\n', b'data: []}\n', b"\n", b"data: [DONE]\n", b"\n"])), [{"choices": []}, None])
        with self.assertRaises(ValueError):
            list(h.sse_events([b"data: broken\n", b"\n"]))
        r = h.consume(iter([{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}]), time.monotonic())
        self.assertEqual(r["status"], "protocol_error")

    def test_error_and_truncation_are_zero(self):
        grader = {"kind": "exact", "expected": "OK"}
        for r in ({"status": "http_error"}, result(finish="length")):
            self.assertEqual(h.grade(grader, r, "")[0], 0)
        self.assertEqual(h.grade(grader, result(), "")[0], 1)

    def test_extraction_penalizes_hallucinations_and_not_duplicates(self):
        grader = {"kind": "set_f1", "expected": ["a", "b"]}
        self.assertEqual(h.grade(grader, result('["a","a","b"]'), "")[0], 1)
        self.assertEqual(h.grade(grader, result('["a","invented"]'), "")[0], 0.5)
        self.assertEqual(h.grade(grader, result('```json\n["a","b"]\n```'), "")[0], 0)

    def test_json_boolean_is_not_an_integer(self):
        self.assertFalse(h.json_equal({"ok": True}, {"ok": 1}))
        self.assertTrue(h.json_equal({"n": 1.0}, {"n": 1}))

    def test_bad_gold_and_multiple_completions_fail_before_inference(self):
        suite, arms = fixture()
        del suite["cases"][0]["grader"]["expected"]
        with self.assertRaises(KeyError):
            h.make_plan(suite, arms, "invalid")
        suite, arms = fixture()
        suite["request_defaults"]["n"] = 2
        with self.assertRaises(AssertionError):
            h.make_plan(suite, arms, "invalid")

    def test_frozen_requests_are_paired_and_distinct(self):
        suite, arms = fixture()
        suite["repeats"] = 2
        plan = h.make_plan(suite, arms, "test")
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertNotEqual(plan["jobs"][0]["request_sha256"], plan["jobs"][1]["request_sha256"])
        self.assertNotIn("model", plan["jobs"][0]["request"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.json"
            h.write_new(path, plan)
            self.assertEqual(h.load_plan(path), plan)
            plan["suite"]["margins"]["reasoning"] = 0.99
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(AssertionError, "plan changed"):
                h.load_plan(path)

    def test_paired_report_failure_denominators_clusters_and_missing_rows(self):
        suite, arms = fixture()
        suite["repeats"] = 3
        plan = h.make_plan(suite, arms, "report")
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for arm in arms:
                path = Path(d) / arm["name"]
                path.mkdir()
                paths.append(path)
                h.write_new(path / "manifest.json", {"plan": plan, "arm": arm["name"]})
                rows = [{"id": j["id"], "arm": arm["name"], "plan_sha256": plan["sha256"],
                         "request_sha256": j["request_sha256"], "score": 1 if arm["name"] == "control" else 0,
                         "result": result() if arm["name"] == "control" else {"status": "http_error", "http_status": 400}}
                        for j in plan["jobs"]]
                (path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            r = h.compare(plan, *paths)["quality"]["reasoning"]
            self.assertEqual(r["clusters"], 1)
            self.assertEqual(r["cluster_mean_delta"], -1)
            self.assertEqual(r["verdict"], "inconclusive")
            self.assertEqual(r["arms"]["candidate"]["items"], 3)
            self.assertEqual(r["arms"]["candidate"]["completed"], 0)
            self.assertIsNone(r["arms"]["candidate"]["mean_score_completed_only"])
            (paths[1] / "results.jsonl").write_text("")
            with self.assertRaisesRegex(AssertionError, "incomplete run"):
                h.compare(plan, *paths)

    def test_http_integration_preserves_request_and_errors(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append((dict(self.headers), payload))
                if payload["model"] == "unsupported":
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"unsupported reasoning_effort"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.writelines(stream([{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}, None]))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            suite, arms = fixture()
            for arm in arms:
                arm["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
            plan = h.make_plan(suite, arms, "http")
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                h.run(plan, "control", Path(d) / "run", 5)
                rows = h.load_run(Path(d) / "run", plan)
                self.assertEqual(next(iter(rows.values()))["score"], 1)
                h.run(plan, "candidate", Path(d) / "candidate", 5)
                report = h.compare(plan, Path(d) / "run", Path(d) / "candidate")
                self.assertEqual(report["quality"]["reasoning"]["verdict"], "inconclusive")
                self.assertEqual(report["candidate_arm"], "candidate")
            self.assertEqual(captured[-1][0]["User-Agent"], h.UA)
            sent = captured[-1][1]
            self.assertEqual(sent["seed"], 42)
            self.assertEqual(sent["reasoning_effort"], "medium")
            self.assertNotIn("cache_prompt", sent)
            error = h.request(dict(arms[0], model="unsupported"), plan["jobs"][0]["request"], 5)
            self.assertEqual(error["http_status"], 400)
            self.assertIn("unsupported reasoning_effort", error["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_identical_clusters_do_not_prove_noninferiority(self):
        suite, arms = fixture()
        template = suite["cases"][0]
        suite["cases"] = [dict(template, id=f"case-{i}", cluster=f"source-{i}") for i in range(20)]
        plan = h.make_plan(suite, arms, "degenerate")
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for arm in arms:
                path = Path(d) / arm["name"]
                path.mkdir()
                paths.append(path)
                h.write_new(path / "manifest.json", {"plan": plan, "arm": arm["name"]})
                rows = [{"id": j["id"], "arm": arm["name"], "plan_sha256": plan["sha256"],
                         "request_sha256": j["request_sha256"], "score": 1.0, "result": result()}
                        for j in plan["jobs"]]
                (path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            tier = h.compare(plan, *paths)["quality"]["reasoning"]
            self.assertEqual(tier["clusters"], 20)
            self.assertEqual(tier["ci"], [0, 0])
            self.assertTrue(tier["degenerate_bootstrap"])
            self.assertEqual(tier["verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
