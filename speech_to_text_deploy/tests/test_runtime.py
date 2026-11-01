"""Verify resource limits reap real worker processes before returning control."""
import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

from speech_to_text_deploy.runtime import run_model_worker


class WorkerLimitTests(unittest.TestCase):
    def setUp(self):
        self.workers = []
        self.original_popen = subprocess.Popen

    def launch(self, *args, **kwargs):
        process = self.original_popen(*args, **kwargs)
        self.workers.append(process)
        return process

    def test_timeout_reaps_worker(self):
        with patch("speech_to_text_deploy.runtime.subprocess.Popen", side_effect=self.launch):
            with self.assertRaises(TimeoutError):
                run_model_worker([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=os.environ.copy(), timeout_seconds=0.2,
                                 min_available_memory_bytes=0)
        self.assertIsNotNone(self.workers[0].poll())

    def test_unlimited_worker_can_exit_normally_and_still_enforces_memory_reserve(self):
        code = [sys.executable, "-c", "import time; time.sleep(.05)"]
        self.assertEqual(run_model_worker(code, env=os.environ.copy(), timeout_seconds=None,
                                          min_available_memory_bytes=0), 0)
        with patch("speech_to_text_deploy.runtime.subprocess.Popen", side_effect=self.launch):
            with self.assertRaises(MemoryError):
                run_model_worker([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=os.environ.copy(), timeout_seconds=None,
                                 min_available_memory_bytes=2**63)
        self.assertIsNotNone(self.workers[-1].poll())

    def test_invalid_deadline_does_not_launch_a_worker(self):
        with patch("speech_to_text_deploy.runtime.subprocess.Popen") as launch:
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(timeout=value), self.assertRaises(ValueError):
                    run_model_worker([sys.executable], env=os.environ.copy(), timeout_seconds=value)
            launch.assert_not_called()

    def test_memory_reserve_reaps_worker(self):
        with patch("speech_to_text_deploy.runtime.subprocess.Popen", side_effect=self.launch):
            with self.assertRaises(MemoryError):
                run_model_worker([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=os.environ.copy(), min_available_memory_bytes=2**63)
        self.assertIsNotNone(self.workers[0].poll())

    def test_interrupt_reaps_worker(self):
        code = "import os,signal,time; time.sleep(.2); os.kill(os.getppid(),signal.SIGINT); time.sleep(60)"
        with patch("speech_to_text_deploy.runtime.subprocess.Popen", side_effect=self.launch):
            with self.assertRaises(KeyboardInterrupt):
                run_model_worker([sys.executable, "-c", code], env=os.environ.copy(),
                                 min_available_memory_bytes=0)
        self.assertIsNotNone(self.workers[0].poll())

    def test_configured_interrupt_grace_allows_drain_or_enforces_deadline(self):
        code = ("import signal,time,sys\n"
                "def stop(*args):\n"
                "    time.sleep(.3)\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGINT, stop)\n"
                "time.sleep(60)\n")
        for grace, exit_code in ((1, 0), (.05, -signal.SIGKILL)):
            with self.subTest(grace=grace), patch(
                    "speech_to_text_deploy.runtime.subprocess.Popen", side_effect=self.launch):
                with self.assertRaises(TimeoutError):
                    run_model_worker([sys.executable, "-c", code], env=os.environ.copy(),
                                     timeout_seconds=.2, min_available_memory_bytes=0,
                                     interrupt_grace_seconds=grace)
                self.assertEqual(self.workers[-1].returncode, exit_code)


if __name__ == "__main__":
    unittest.main()
