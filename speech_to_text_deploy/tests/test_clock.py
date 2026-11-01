"""Test Internet time anchoring and computer-clock fallback."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from speech_to_text_deploy.audio.clock import reference_clock


class ClockTests(unittest.TestCase):
    def test_uses_internet_date_header_with_round_trip_compensation(self):
        class Response:
            headers = {"Date": "Sun, 01 Nov 2026 00:00:00 GMT"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (patch("speech_to_text_deploy.audio.clock.urlopen", return_value=Response()),
              patch("speech_to_text_deploy.audio.clock.perf_counter", side_effect=[10.0, 10.4])):
            utc, monotonic_sample, source = reference_clock()
        self.assertEqual(source, "internet")
        self.assertEqual(utc, datetime(2026, 11, 1, 0, 0, 0, 200000, tzinfo=timezone.utc))
        self.assertEqual(monotonic_sample, 10.2)

    def test_falls_back_to_computer_clock_when_internet_is_unavailable(self):
        with patch("speech_to_text_deploy.audio.clock.urlopen", side_effect=OSError("offline")):
            utc, monotonic_sample, source = reference_clock(timeout_seconds=0.01)
        self.assertEqual(source, "computer")
        self.assertEqual(utc.tzinfo, timezone.utc)
        self.assertGreater(monotonic_sample, 0)
