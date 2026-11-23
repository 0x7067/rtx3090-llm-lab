import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cuda_trace as trace


class TraceRetentionTests(unittest.TestCase):
    def test_recent_windows_survive_and_frozen_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'current-run'
            run.mkdir()
            old = root / 'earlier-hang'
            old.mkdir()
            (old / 'frozen').touch()
            evidence = old / 'window-000001.nsys-rep'
            evidence.write_bytes(b'x' * 900)
            for i in range(5):
                (run / f'window-{i:06d}.nsys-rep').write_bytes(b'x' * 100)
            with patch.object(trace, 'ROOT', root), patch.object(trace, 'BUDGET_BYTES', 1000):
                used = trace.prune(run)
            self.assertEqual(evidence.stat().st_size, 900)
            self.assertEqual([p.name for p in sorted(run.glob('*.nsys-rep'))],
                             ['window-000003.nsys-rep', 'window-000004.nsys-rep'])
            self.assertGreaterEqual(used, 1000)  # Caller freezes instead of deleting hang evidence.

    def test_progress_combines_engines_and_rejects_missing_counters(self):
        metrics = b'''vllm:num_requests_running{engine="0"} 1
vllm:num_requests_running{engine="1"} 2
vllm:prompt_tokens_total{engine="0"} 500
vllm:generation_tokens_total{engine="0"} 20
'''
        with patch.object(trace.urllib.request, 'urlopen', return_value=io.BytesIO(metrics)):
            self.assertEqual(trace.progress(), (3, 520))
        with patch.object(trace.urllib.request, 'urlopen', return_value=io.BytesIO(b'')):
            with self.assertRaises(ValueError):
                trace.progress()


if __name__ == '__main__':
    unittest.main()
