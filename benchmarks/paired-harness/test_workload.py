import copy
import tempfile
from pathlib import Path
import unittest

import build_workload as workload
import harness as h
from test_harness import fixture, result


class WorkloadTests(unittest.TestCase):
    def test_reference_solutions_satisfy_authored_contracts(self):
        self.assertEqual(workload.validate_references(), 20)

    def test_gold_checks_reject_plausible_boundary_regressions(self):
        mutations = {
            "ttl": ("v[1]>now", "v[1]>=now"),
            "range-coalesce": ("a<=out[-1][1]", "a<out[-1][1]"),
            "pagination": ("x>after", "x>=after"),
            "batch-bytes": ("len(item.encode('utf-8'))", "len(item)"),
            "rate-limit": ("q[0]<=t-window", "q[0]<t-window"),
        }
        for id, _, reference, tests in workload.CODING:
            if id not in mutations:
                continue
            with self.subTest(contract=id):
                old, new = mutations[id]
                self.assertIn(old, reference)
                namespace = {}
                exec(compile(reference.replace(old, new), id, "exec"), namespace)
                with self.assertRaises(AssertionError):
                    exec(compile(tests, id + "-tests", "exec"), namespace)

    def test_expansion_preserves_related_clusters_and_hides_solutions(self):
        suite = workload.build()
        self.assertEqual(len(suite["cases"]), 58)
        tools = [c for c in suite["cases"] if c["tier"] == "tool_replay"]
        self.assertEqual(len(tools), 16)
        self.assertEqual(len({c["cluster"] for c in tools}), 8)
        retrieval = [c for c in suite["cases"] if c["tier"] == "retrieval"]
        self.assertEqual(len(retrieval), 6)
        self.assertEqual(len({c["cluster"] for c in retrieval}), 2)
        for case in suite["cases"]:
            self.assertNotIn("reference", case)
        for id, _, reference, tests in workload.CODING:
            case = next(c for c in suite["cases"] if c["id"] == "code-" + id)
            prompt = h.json.dumps(case["request"])
            self.assertNotIn(reference, prompt)
            self.assertNotIn(tests, prompt)

    def test_long_document_gold_facts_are_present(self):
        suite = workload.build(depths=(16, 128))
        for case in suite["cases"]:
            if case["tier"] != "retrieval":
                continue
            prompt = case["request"]["messages"][0]["content"]
            for value in case["grader"]["expected"].values():
                self.assertIn(str(value), prompt)
            if "revision" in case["id"]:
                self.assertIn("revision=10; status=proposed", prompt)
                self.assertIn("revision=9; status=approved", prompt)

    def test_single_arm_summary_keeps_failure_denominator(self):
        suite, arms = fixture()
        second = copy.deepcopy(suite["cases"][0])
        second["id"] = "two"
        suite["cases"].append(second)
        plan = h.make_plan(suite, arms, "summary")
        with tempfile.TemporaryDirectory() as d:
            h.write_new(Path(d) / "manifest.json", {"plan": plan, "arm": "control"})
            rows = []
            for i, job in enumerate(plan["jobs"]):
                rows.append({"id": job["id"], "plan_sha256": plan["sha256"], "request_sha256": job["request_sha256"],
                             "arm": "control", "score": float(i == 0), "grade_detail": "test",
                             "result": result() if i == 0 else {"status": "http_error", "http_status": 400}})
            (Path(d) / "results.jsonl").write_text("\n".join(h.json.dumps(r) for r in rows))
            report = h.summarize(plan, d)
            tier = report["quality"]["reasoning"]
            self.assertEqual(tier["attempted"], 2)
            self.assertEqual(tier["full_score"], 1)
            self.assertEqual(tier["completed"], 1)
            self.assertEqual(tier["mean_score"], 0.5)
            self.assertEqual(tier["clusters"], 1)
            self.assertEqual(len(report["failures"]), 1)
            self.assertIn("1/2", h.summary_markdown(report))


if __name__ == "__main__":
    unittest.main()
