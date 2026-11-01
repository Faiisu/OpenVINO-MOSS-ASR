"""Validate sample preservation, paired storage, queue timing and failure recovery."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import time
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from speech_to_text_deploy.model import Segment
from speech_to_text_deploy.audio.capture import capture_segments
from speech_to_text_deploy.config import GatheringConfig
from speech_to_text_deploy.pipeline import run_gathering
from speech_to_text_deploy.outputs.clips import export_timed_clips


class GatheringTests(unittest.TestCase):
    @staticmethod
    def computer_clock_sample(timeout_seconds):
        return datetime.now(timezone.utc), perf_counter(), "computer"

    def setUp(self):
        self.clock_patch = patch(
            "speech_to_text_deploy.audio.capture.reference_clock",
            side_effect=self.computer_clock_sample,
        )
        self.clock_patch.start()

    def tearDown(self):
        self.clock_patch.stop()

    def test_live_input_buffer_preserves_windows_and_closes_on_overflow(self):
        for overflow in (False, True):
            with self.subTest(overflow=overflow), tempfile.TemporaryDirectory() as temporary:
                with patch("sounddevice.InputStream") as factory:
                    stream = factory.return_value.__enter__.return_value
                    stream.samplerate = 10
                    stream.read.side_effect = [
                        (np.array([[index]], dtype=np.int16), overflow and index == 1)
                        for index in range(5)
                    ]
                    items = capture_segments(temporary, threading.Event(), input_device=5,
                                             samplerate=10, segment_seconds=.3,
                                             max_audio_seconds=.5)
                    if overflow:
                        with self.assertRaisesRegex(RuntimeError, "input overflow"):
                            list(items)
                    else:
                        captured = list(items)
                        self.assertEqual([item.duration_seconds for item in captured], [.3, .2])
                        samples = np.concatenate([sf.read(item.path, dtype="int16")[0]
                                                  for item in captured])
                        np.testing.assert_array_equal(samples, np.arange(5))
                    factory.assert_called_once_with(device=5, samplerate=10, channels=1,
                                                    dtype="int16", latency=1.0)
                    factory.return_value.__exit__.assert_called_once()

    def test_input_latency_must_be_positive_and_finite(self):
        for latency in (0, -1, float("nan"), float("inf")):
            with self.subTest(latency=latency), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    list(capture_segments(temporary, threading.Event(),
                                          input_latency_seconds=latency))
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    GatheringConfig(output_dir=temporary, input_latency_seconds=latency).validate()

    def test_native_stereo_precision_and_final_partial_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = (np.arange(1300, dtype=np.int32).reshape(-1, 2) - 650) * 256
            source = root / "source.wav"
            sf.write(source, samples, 10, subtype="PCM_24")
            forwarded_frames = []
            items = list(capture_segments(root / "recordings", threading.Event(),
                                         source_audio=source, replay_realtime=False,
                                         on_audio_chunk=lambda data, rate: forwarded_frames.append((len(data), rate))))
            self.assertEqual([item.duration_seconds for item in items], [30, 30, 5])
            self.assertEqual([item.offset_seconds for item in items], [0, 30, 60])
            recovered = np.concatenate([sf.read(item.path, dtype="int32", always_2d=True)[0]
                                        for item in items])
            np.testing.assert_array_equal(recovered, samples)
            self.assertTrue(all(sf.info(item.path).subtype == "PCM_24" for item in items))
            self.assertEqual(sum(count for count, rate in forwarded_frames), len(samples))
            self.assertTrue(all(rate == 10 for count, rate in forwarded_frames))

    def test_clip_pairing_uses_effective_offsets_and_preserves_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.arange(20, dtype=np.int16), 10, subtype="PCM_16")
            capture = {"offset_seconds": 30, "capture_start_utc": "2026-11-01T00:00:30+00:00"}
            folder = root / "result"
            count = export_timed_clips(source, folder, "raw",
                                      [Segment(.2, .7, "S1", "First"), Segment(.5, 3, "S2", "Second")], capture)
            self.assertEqual(count, 2)
            np.testing.assert_array_equal(sf.read(folder / "utterance-0001.wav", dtype="int16")[0], range(2, 7))
            np.testing.assert_array_equal(sf.read(folder / "utterance-0002.wav", dtype="int16")[0], range(5, 20))
            paired = json.loads((folder / "utterance-0002.json").read_text())
            self.assertEqual((paired["recording_start_seconds"], paired["recording_end_seconds"]), (30.5, 32))
            self.assertEqual(paired["model_end_seconds"], 3)
            self.assertEqual(paired["end_utc"], "2026-11-01T00:00:32+00:00")
            self.assertEqual(paired["text"], "Second")
            self.assertNotIn("rtf", paired)

    def test_greeting_clips_include_two_seconds_of_preceding_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            samples = np.arange(100, dtype=np.int16)
            sf.write(source, samples, 10, subtype="PCM_16")
            capture = {"offset_seconds": 60,
                       "capture_start_utc": "2026-11-01T00:00:00+00:00"}
            folder = root / "result"
            segments = [Segment(3, 3.5, "S1", "สวัสดีค่ะ"),
                        Segment(1, 1.5, "S2", "สวัสดีครับ")]
            self.assertEqual(export_timed_clips(
                source, folder, "raw", segments, capture,
                greeting_preroll_keyword="สวัสดี", greeting_preroll_seconds=2.0), 2)
            first = json.loads((folder / "utterance-0001.json").read_text())
            second = json.loads((folder / "utterance-0002.json").read_text())
            self.assertEqual(first["model_start_seconds"], 3)
            self.assertEqual(first["start_seconds"], 1)
            self.assertEqual(first["pre_roll_seconds"], 2)
            self.assertEqual(first["recording_start_seconds"], 61)
            self.assertEqual(second["start_seconds"], 0)
            self.assertEqual(second["pre_roll_seconds"], 1)
            np.testing.assert_array_equal(
                sf.read(folder / "utterance-0001.wav", dtype="int16")[0], np.arange(10, 35))
            np.testing.assert_array_equal(
                sf.read(folder / "utterance-0002.wav", dtype="int16")[0], np.arange(0, 15))

    def test_invalid_range_retains_raw_without_completion_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(20, dtype=np.int16), 10)
            capture = {"offset_seconds": 0, "capture_start_utc": "2026-11-01T00:00:00+00:00"}
            with self.assertRaises(ValueError):
                export_timed_clips(source, root / "result", "raw",
                                  [Segment(3, 4, "S1", "Invalid")], capture)
            self.assertEqual((root / "result/raw_transcript.txt").read_text(), "raw")
            self.assertFalse((root / "result/transcript.json").exists())

    def test_empty_output_has_explicit_no_segments_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(20, dtype=np.int16), 10)
            capture = {"offset_seconds": 0, "capture_start_utc": "2026-11-01T00:00:00+00:00"}
            self.assertEqual(export_timed_clips(source, root / "result", "raw", [], capture), 0)
            self.assertEqual(json.loads((root / "result/transcript.json").read_text())["status"], "no_segments")

    def test_retained_rejections_preserve_valid_clips_and_original_model_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.arange(20, dtype=np.int16), 10, subtype="PCM_16")
            segments = [Segment(.2, .7, "S1", "Valid"), Segment(3, 4, "S2", "Outside"),
                        Segment(1.5, 3, "S3", "Clamped")]
            capture = {"offset_seconds": 30, "capture_start_utc": "2026-11-01T00:00:30+00:00"}
            folder = root / "result"
            self.assertEqual(export_timed_clips(source, folder, "original raw", segments, capture,
                                               invalid_segment_policy="retain"), 2)
            manifest = json.loads((folder / "transcript.json").read_text())
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["rejected_segments"][0]["text"], "Outside")
            self.assertEqual(manifest["rejected_segments"][0]["model_start_seconds"], 3)
            self.assertEqual([entry["model_segment_index"] for entry in manifest["segments"]], [1, 3])
            self.assertFalse((folder / "utterance-0002.wav").exists())
            np.testing.assert_array_equal(sf.read(folder / "utterance-0001.wav", dtype="int16")[0], range(2, 7))
            np.testing.assert_array_equal(sf.read(folder / "utterance-0003.wav", dtype="int16")[0], range(15, 20))
            self.assertEqual((folder / "raw_transcript.txt").read_text(), "original raw")
            self.assertNotIn("Outside", (folder / "transcript.txt").read_text())

    def test_all_invalid_intervals_have_a_rejected_manifest_without_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(20, dtype=np.int16), 10)
            capture = {"offset_seconds": 0, "capture_start_utc": "2026-11-01T00:00:00+00:00"}
            segments = [Segment(3, 4, "S1", "Outside"), Segment(float("nan"), 1, "S1", "Invalid")]
            self.assertEqual(export_timed_clips(source, root / "result", "raw", segments, capture,
                                               invalid_segment_policy="retain"), 0)
            manifest = json.loads((root / "result/transcript.json").read_text())
            self.assertEqual(manifest["status"], "rejected")
            self.assertEqual(len(manifest["rejected_segments"]), 2)
            self.assertEqual(manifest["rejected_segments"][1]["model_start_seconds"], "nan")
            self.assertFalse(list((root / "result").glob("*.wav")))

    def exercise_queue(self, fail_second=False, reject_second=False, no_speech_first=False):
        # This fake isolates scheduling/recovery assertions; real GPU inference
        # is exercised separately with BY-V and the cached full MOSS model.
        class Session:
            closed = False
            calls = 0

            def __init__(self, *args, **kwargs):
                self.device = torch.device("cpu")
                self.count = 0
                self.runtime_metrics = {"encoder_execution_devices": ["GPU.test"],
                                        "decoder_execution_devices": ["GPU.test"]}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                Session.closed = True
                self.runtime_metrics["ov_memory_after_unload"] = {"usm_device": 0}

            def transcribe(self, *args):
                self.count += 1
                Session.calls += 1
                time.sleep(0.1)
                if fail_second and self.count == 2:
                    raise RuntimeError("Deliberate inference failure")
                if reject_second and self.count == 2:
                    return "raw", [Segment(0, .2, "S1", "Valid"), Segment(31, 32, "S2", "Outside")]
                return "raw", [Segment(0, .2, "S1", "Text")]

        class VAD:
            def __init__(self, *, threshold, min_speech_duration_ms,
                         min_silence_duration_ms, speech_pad_ms):
                self.threshold = threshold
                self.min_speech_duration_ms = min_speech_duration_ms

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def detect(self, path):
                if no_speech_first and Path(path).name == "segment-000001.wav":
                    return []
                return [(0.1, 0.6)]

        class Publisher:
            messages = []

            def __init__(self, **kwargs):
                self.messages = type(self).messages

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def publish_json(self, message):
                self.messages.append(message)
                return len(self.messages)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(6500, dtype=np.int16), 100)
            model = root / "model"
            model.mkdir()
            (model / "manifest.json").write_text("{}")
            config = GatheringConfig(output_dir=str(root / "run"), model_dir=str(model),
                                     source_audio=str(source), replay_realtime=False,
                                     live_audio_enabled=False)
            Publisher.messages = []
            with (patch("speech_to_text_deploy.model.session.OpenVinoSession", Session),
                  patch("speech_to_text_deploy.pipeline.SileroVAD", VAD),
                  patch("speech_to_text_deploy.pipeline.MqttJsonPublisher", Publisher)):
                if fail_second:
                    with self.assertRaisesRegex(RuntimeError, "Collection failed"):
                        run_gathering(config)
                else:
                    run_gathering(config)
            self.assertTrue(Session.closed)
            with (root / "run/reports/segments.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual([float(row["audio_seconds"]) for row in rows], [30, 30, 5])
            if fail_second:
                self.assertEqual([row["status"] for row in rows], ["success", "failed", "not_processed"])
                self.assertTrue((root / "run/recordings/segment-000003.wav").is_file())
                self.assertEqual(rows[-1]["rtf"], "")
            else:
                if no_speech_first:
                    self.assertEqual([row["status"] for row in rows], ["no_speech", "success", "success"])
                    self.assertEqual(Session.calls, 2)
                    runtime = json.loads((root / "run/reports/runtime.json").read_text())
                    self.assertEqual(runtime["no_speech_count"], 1)
                    self.assertEqual(runtime["processed_count"], 3)
                    self.assertEqual(runtime["mqtt"]["published_count"], 2)
                    self.assertEqual(Publisher.messages[0]["transcript"], "Text")
                    self.assertEqual(set(Publisher.messages[0]), {"transcript", "start_time", "end_time"})
                    manifest = json.loads((root / "run/results/segment-000001/transcript.json").read_text())
                    self.assertEqual(manifest["vad"]["threshold"], 0.9)
                    self.assertEqual(manifest["vad"]["min_speech_duration_ms"], 1000)
                if reject_second:
                    self.assertEqual([row["status"] for row in rows], ["success", "partial", "success"])
                    self.assertEqual(rows[1]["rejected_count"], "1")
                    runtime = json.loads((root / "run/reports/runtime.json").read_text())
                    self.assertEqual(runtime["status"], "success_with_warnings")
                    self.assertEqual(runtime["processed_count"], 3)
                    self.assertEqual(runtime["partial_count"], 1)
                self.assertGreater(float(rows[-1]["queue_wait_seconds"]), .1)
                for row in rows:
                    latency = float(row["latency_seconds"])
                    self.assertAlmostEqual(float(row["rtf"]), latency / float(row["audio_seconds"]))
                    self.assertAlmostEqual(latency, sum(float(row[key]) for key in (
                        "prepare_seconds", "queue_wait_seconds", "inference_seconds", "trim_and_store_seconds")))

    def test_queue_latency_and_storage_timing(self):
        self.exercise_queue()

    def test_failed_consumer_retains_unprocessed_inputs_and_unloads(self):
        self.exercise_queue(fail_second=True)

    def test_rejected_interval_does_not_stop_later_windows(self):
        self.exercise_queue(reject_second=True)

    def test_no_speech_window_skips_model_and_continues_queue(self):
        self.exercise_queue(no_speech_first=True)
