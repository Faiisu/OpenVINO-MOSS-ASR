"""Verify CPU thread settings reach the isolated worker before it imports PyTorch."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_to_text.isolated import transcribe_isolated


class CpuThreadTests(unittest.TestCase):
    def test_worker_receives_thread_environment_and_request(self):
        def worker(command, *, env, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text())
            self.assertEqual(request["cpu_threads"], 8)
            self.assertEqual(request["interop_threads"], 2)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                self.assertEqual(env[name], "8")
            request_path.with_name("result.json").write_text(json.dumps({
                "raw": "text", "segments": [], "metrics": {},
            }))
            return 0

        with patch("speech_to_text.isolated.run_model_worker", side_effect=worker):
            raw, segments, _ = transcribe_isolated("unused.wav", "test", "xpu:0", 100,
                                                  cpu_threads=8, interop_threads=2)
        self.assertEqual(raw, "text")
        self.assertEqual(segments, [])

    def test_invalid_thread_count_rejected_before_launch(self):
        with patch("speech_to_text.isolated.run_model_worker") as worker:
            for settings in ({"cpu_threads": 0}, {"interop_threads": 0}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    transcribe_isolated("unused.wav", "test", "xpu:0", 100, **settings)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
