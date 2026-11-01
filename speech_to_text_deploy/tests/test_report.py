"""Check weighted timing, escaped transcripts and exact embedded clip playback."""

import base64
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from speech_to_text_deploy.model import Segment
from speech_to_text_deploy.config import REPORT_COLUMNS
from speech_to_text_deploy.config import ROOT
from speech_to_text_deploy.outputs.report import gathering_summary_html
from speech_to_text_deploy.outputs.clips import export_timed_clips


class GatheringReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "reports").mkdir()
        (self.root / "results").mkdir()

    def window(self, index, seconds, latency, segments):
        source = self.root / f"source-{index}.wav"
        sf.write(source, np.arange(round(seconds * 10), dtype=np.int16), 10, subtype="PCM_16")
        folder = self.root / "results" / f"segment-{index:06d}"
        export_timed_clips(source, folder, "raw", segments,
                           {"offset_seconds": (index - 1) * 30,
                            "capture_start_utc": "2026-11-01T00:00:00+00:00"}, invalid_segment_policy="retain")
        rejected = len(json.loads((folder / "transcript.json").read_text())["rejected_segments"])
        return {"segment_index": index, "status": "partial" if rejected else "success", "audio_seconds": seconds,
                "latency_seconds": latency, "rtf": latency / seconds,
                "rejected_count": rejected,
                "output_dir": str(folder)}

    def report(self, rows, *, status="success", errors=()):
        with (self.root / "reports" / "segments.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (self.root / "reports" / "runtime.json").write_text(
            json.dumps({"status": status, "errors": list(errors)}), encoding="utf-8")

    def test_weighted_rtf_and_exact_embedded_audio_with_escaped_text(self):
        first = self.window(1, 30, 3, [Segment(.2, .7, "S01", "สวัสดี <script>alert(1)</script>")])
        second = self.window(2, 5, 5, [Segment(.2, .7, "S01", "Second")])
        self.report([second, first])
        html = gathering_summary_html(self.root)
        self.assertIn("Completed windows: 2/2", html)
        self.assertIn("Duration-weighted RTF: 0.229", html)
        self.assertIn("30.200–30.700", html)
        self.assertIn("สวัสดี &lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>", html)
        players = re.findall(r"data:audio/wav;base64,([^']+)", html)
        self.assertEqual(len(players), 2)
        for index, encoded in enumerate(players, 1):
            clip = self.root / "results" / f"segment-{index:06d}" / "utterance-0001.wav"
            self.assertEqual(base64.b64decode(encoded), clip.read_bytes())
            np.testing.assert_array_equal(sf.read(clip, dtype="int16")[0], np.arange(2, 7))

    def test_failed_and_empty_windows_remain_visible_without_success_rtf(self):
        empty = self.window(1, 30, 3, [])
        failed = {"segment_index": 2, "status": "failed", "audio_seconds": 5,
                  "error": "Invalid <timestamp>", "output_dir": str(self.root / "results/segment-000002")}
        folder = self.root / "results/segment-000002"
        folder.mkdir()
        (folder / "raw_transcript.txt").write_text("raw <model output>", encoding="utf-8")
        self.report([empty, failed], status="failed",
                    errors=[{"worker": "inference", "error": "Invalid <timestamp>"}])
        html = gathering_summary_html(self.root)
        self.assertIn("Status: failed", html)
        self.assertIn("Completed windows: 1/2", html)
        self.assertIn("Duration-weighted RTF: 0.100", html)
        self.assertIn("Window 1: no parsed utterances (no_segments)", html)
        self.assertIn("Invalid &lt;timestamp&gt;", html)
        self.assertIn("raw &lt;model output&gt;", html)
        self.assertIn("<td>2</td><td>failed</td>", html)
        self.assertNotIn("<audio", html)

    def test_missing_clip_and_incomplete_run_are_explicit(self):
        row = self.window(1, 5, 2, [Segment(.2, .7, "S01", "Text")])
        self.report([row])
        (self.root / "results/segment-000001/utterance-0001.wav").unlink()
        (self.root / "reports/runtime.json").unlink()
        html = gathering_summary_html(self.root)
        self.assertIn("Status: incomplete", html)
        self.assertIn("Audio file missing: utterance-0001.wav", html)
        self.assertIn("Text", html)
        self.assertNotIn("<audio", html)

    def test_partial_window_shows_rejected_text_without_inventing_audio(self):
        row = self.window(1, 30, 3, [Segment(.2, .7, "S01", "Valid"),
                                    Segment(31, 32, "S02", "Rejected <text>")])
        self.report([row], status="success_with_warnings")
        html = gathering_summary_html(self.root)
        self.assertIn("Completed windows: 1/1", html)
        self.assertIn("Duration-weighted RTF: 0.100", html)
        self.assertIn("Rejected utterances: 1", html)
        self.assertIn("Rejected &lt;text&gt;", html)
        self.assertIn("31–32 (rejected model time)", html)
        self.assertIn("No audio clip was created", html)
        self.assertEqual(html.count("<audio controls"), 1)

    def test_summary_cli_exports_portable_html(self):
        self.report([self.window(1, 5, 2, [Segment(.2, .7, "S01", "Text")])])
        result = subprocess.run([sys.executable, "-m", "speech_to_text_deploy", "summary", str(self.root)],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        target = self.root / "reports" / "summary.html"
        self.assertIn(str(target), result.stdout)
        html = target.read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", html)
        self.assertIn("Text", html)
        self.assertEqual(html.count("<audio controls"), 1)
