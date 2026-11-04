import unittest

from notify_completion import completion_result


class CompletionTests(unittest.TestCase):
    def test_running_job_does_not_notify_despite_zero_exit_status(self):
        self.assertIsNone(completion_result({
            "InvocationID": "abc", "ActiveState": "active", "ExecMainStatus": "0", "Result": "success",
        }, "abc", ""))

    def test_unloaded_successful_job_uses_recorded_completion(self):
        self.assertEqual(completion_result({
            "LoadState": "not-found", "ActiveState": "inactive",
        }, "abc", "Regeneration finished with exit=0; local API restoration attempted"), "succeeded")

    def test_restoration_failure_is_reported_as_failure(self):
        self.assertEqual(completion_result({
            "ActiveState": "failed",
        }, "abc", "REGENERATION-DONE\nRegeneration finished with exit=1; local API restoration attempted"), "failed (exit 1)")

    def test_crash_without_cleanup_still_notifies(self):
        self.assertIn("failed", completion_result({
            "ActiveState": "failed", "Result": "signal", "ExecMainStatus": "9",
        }, "abc", ""))

    def test_replacement_is_not_confused_with_original_job(self):
        self.assertIn("replaced", completion_result({
            "ActiveState": "active", "InvocationID": "other",
        }, "abc", ""))

    def test_disappearance_without_proof_is_not_reported_as_success(self):
        self.assertIn("without a recorded completion", completion_result({
            "LoadState": "not-found", "ActiveState": "inactive",
        }, "abc", ""))


if __name__ == "__main__":
    unittest.main()
