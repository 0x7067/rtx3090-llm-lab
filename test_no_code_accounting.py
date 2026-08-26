# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Check that bench_multiturn.py accounts for replies that contain no code.

This is the behaviour the multi-turn rows in REPORT.md turned on: a reply with
no fenced block costs the task a turn, and until it recorded a finish_reason
there was no way to tell "ran out of token budget" from "chose to write prose"
after the fact.

The model API is the only thing stubbed; the harness itself runs for real. No
reply ever contains code, so no candidate is executed and no container starts.
httpx is declared here only so the harness runs in this script's interpreter.

Usage: uv run test_no_code_accounting.py
"""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8199
REPLY = "Could you clarify which behaviour is intended?"

MUTANTS = [
    {"task_id": "Fixture/0", "code": "def f(x):\n    return x + 1\n",
     "test": "def check(f):\n    assert f(1) == 2\n", "entry_point": "f",
     "initial_error": "AssertionError"},
    {"task_id": "Fixture/1", "code": "def g(x):\n    return x * 3\n",
     "test": "def check(g):\n    assert g(2) == 4\n", "entry_point": "g",
     "initial_error": "AssertionError"},
]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{
            "finish_reason": "length",
            "message": {"content": REPLY}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mutants = os.path.join(tmp, "mutants.jsonl")
            with open(mutants, "w") as f:
                for m in MUTANTS:
                    f.write(json.dumps(m) + "\n")
            out = os.path.join(tmp, "out.json")
            subprocess.run(
                [sys.executable, os.path.join(here, "bench_multiturn.py"),
                 f"http://127.0.0.1:{PORT}", "stub", mutants, out,
                 os.path.join(tmp, "work"), "2"],
                check=True, capture_output=True, cwd=here)
            summary = json.load(open(out))
    finally:
        server.shutdown()

    n = summary["n_tasks"]
    assert n == len(MUTANTS), n
    assert summary["solved_total"] == 0, summary["solved_total"]
    assert len(summary["unsolved"]) == n, summary["unsolved"]
    # Every task burns all three turns on no-code replies.
    assert summary["no_code_replies"] == n * 3, summary["no_code_replies"]
    assert summary["tasks_with_no_code"] == n, summary["tasks_with_no_code"]
    for task_id, turns in summary["no_code_detail"].items():
        assert [t["turn"] for t in turns] == [1, 2, 3], (task_id, turns)
        assert all(t["finish_reason"] == "length" for t in turns), (task_id, turns)
        assert all(t["raw_len"] == len(REPLY) for t in turns), (task_id, turns)
        # The text itself, not just its length: that is what tells a truncated
        # reply from a model that chose to answer in prose.
        assert all(t["reply"] == REPLY for t in turns), (task_id, turns)
    print(f"ok: {n} tasks, {summary['no_code_replies']} no-code turns recorded "
          "with finish_reason and reply length")


main()
