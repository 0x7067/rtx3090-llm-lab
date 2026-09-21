import json
import unittest
from pathlib import Path

import agent_smoke as agent


class ToolBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.case = {"path": "answer.py", "visible_tests": "assert value() == 3"}
        self.files = {"answer.py": "def value(): return 2", "tests.py": self.case["visible_tests"]}
        self.image = "sha256:613c5454f914638dc2bfcab71c8630de0089f7ebff0a7239b54f34866c4b6051"

    def call(self, name, arguments):
        return agent.execute({"name": name, "arguments": json.dumps(arguments)},
                             self.files, self.case, self.image)

    def test_reads_and_edits_the_source(self):
        self.assertEqual(self.call("read_file", {"path": "answer.py"})["content"],
                         "def value(): return 2")
        self.call("write_file", {"path": "answer.py", "content": "def value(): return 3"})
        self.assertEqual(self.files["answer.py"], "def value(): return 3")

    def test_rejects_unknown_paths_and_test_edits(self):
        for path in ("../answer.py", "/etc/passwd", "missing.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.call("read_file", {"path": path})
        with self.assertRaises(ValueError):
            self.call("write_file", {"path": "tests.py", "content": ""})
        self.assertEqual(self.files["tests.py"], self.case["visible_tests"])

    def test_rejects_invalid_tool_arguments(self):
        for name, args in (("shell", {}), ("run_tests", {"extra": 1}),
                           ("write_file", {"path": "answer.py", "content": 3}),
                           ("write_file", {"path": "answer.py", "content": "x" * 65537})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.call(name, args)

    def test_real_sandbox_reports_failure_then_success(self):
        self.assertFalse(self.call("run_tests", {})["passed"])
        self.call("write_file", {"path": "answer.py", "content": "def value(): return 3"})
        self.assertTrue(self.call("run_tests", {})["passed"])

    def test_episode_fixtures_reject_bugs_and_accept_reference_fixes(self):
        root = Path(__file__).parent
        suite = json.loads((root / "agent-suite.json").read_text())
        references = json.loads((root / "artifacts/agent-references.json").read_text())
        for case in suite["cases"]:
            for code, expected in ((case["source"], 0), (references[case["id"]], 1)):
                for tests in (case["visible_tests"], case["hidden_tests"]):
                    with self.subTest(case=case["id"], expected=expected):
                        score, detail = agent.harness.python_grade(code, tests, suite["code_image"])
                        self.assertEqual(score, expected, detail)


if __name__ == "__main__":
    unittest.main()
