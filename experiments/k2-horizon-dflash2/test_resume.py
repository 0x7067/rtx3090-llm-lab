"""Behavior checks for the local K2 regeneration pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import regenerate_sessions as regen


class RegenerationTests(unittest.TestCase):
    def test_swallowed_answer_is_rejected_but_truncation_is_recorded(self):
        args = SimpleNamespace(model="k2", max_tokens=4096, effort="medium", base_url="http://localhost/v1", api_key="", timeout=1)
        row = {"id": "one", "conversations": [{"role": "user", "content": "Compute 2+2."}]}
        message = {"content": "", "reasoning_content": "Reasoning and a swallowed answer."}
        for finish in ["stop", "length"]:
            with self.subTest(finish=finish), patch.object(regen, "call", return_value={
                "choices": [{"finish_reason": finish, "message": message}]
            }):
                result, error = regen.regen_one(args, row)
            if finish == "stop":
                self.assertIsNone(result)
                self.assertIn("no answer", error)
            else:
                self.assertIsNone(error)
                self.assertEqual(result["finish_reason"], "length")
                self.assertEqual(result["source_message"], message)
                self.assertEqual(result["reasoning_effort"], "medium")

    def test_completed_rows_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            inp.write_text('\n'.join(json.dumps({"id": i}) for i in ["done", "next"]) + '\n')
            original = '{"id":"done"}\n'
            out.write_text(original)
            with patch("sys.argv", ["regen", str(inp), str(out), "--base-url", "http://localhost/v1"]), patch.object(
                regen, "regen_one", return_value=({"id": "next"}, None)
            ) as call:
                regen.main()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[1]["id"], "next")
            self.assertTrue(out.read_text().startswith(original))
            self.assertEqual([json.loads(line)["id"] for line in out.read_text().splitlines()], ["done", "next"])

    def test_failed_requests_do_not_mark_dataset_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            inp.write_text('{"id":"retry-me"}\n')
            with patch("sys.argv", ["regen", str(inp), str(out), "--base-url", "http://localhost/v1"]), patch.object(
                regen, "regen_one", return_value=(None, "server unavailable")
            ), self.assertRaises(SystemExit) as error:
                regen.main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(out.read_text(), "")


class K2ParserTests(unittest.TestCase):
    def test_nonstream_generic_close_preserves_answer_at_every_effort(self):
        from sglang.srt.parser.reasoning_parser import K2V3Detector

        for effort in ["high", "medium", "low"]:
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                    "Two plus two is four.</ifm|think>4"
                )
                self.assertEqual(result.reasoning_text, "Two plus two is four.")
                self.assertEqual(result.normal_text, "4")

    def test_nonstream_canonical_close_still_works(self):
        from sglang.srt.parser.reasoning_parser import K2V3Detector

        for effort, tag in [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")]:
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                    f"Reasoning.</ifm|{tag}>Answer."
                )
                self.assertEqual(result.reasoning_text, "Reasoning.")
                self.assertEqual(result.normal_text, "Answer.")


if __name__ == "__main__":
    unittest.main()
