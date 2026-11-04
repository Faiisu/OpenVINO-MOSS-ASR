"""Verify live indexing, safe read-only HTTP access, audio ranges and bounded history."""

import csv
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from speech_to_text_deploy.config import REPORT_COLUMNS
from speech_to_text_deploy.model import Segment
from speech_to_text_deploy.monitor.data import MonitorStore, RunIndex
from speech_to_text_deploy.monitor.server import MonitorServer
from speech_to_text_deploy.outputs.clips import export_timed_clips
from speech_to_text_deploy.audio.live_audio import LiveAudioForwarder
from speech_to_text_deploy.monitor.mqtt import LiveTranscriptHub


class MonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "run-20261101T000000Z-test"
        for name in ("reports", "recordings", "results"):
            (self.root / name).mkdir(parents=True)
        (self.root / "config.json").write_text(json.dumps({"segment_seconds": 30, "ov_device": "GPU"}))
        self.csv = self.root / "reports/segments.csv"
        with self.csv.open("w", newline="") as file:
            csv.writer(file).writerow(REPORT_COLUMNS)

    def row(self, index=1, *, rejected=False):
        source = self.root / "recordings" / f"segment-{index:06d}.wav"
        sf.write(source, np.arange(300, dtype=np.int16), 10, subtype="PCM_16")
        capture = {"offset_seconds": (index - 1) * 30, "capture_start_utc": "2026-11-01T00:00:00+00:00"}
        segments = [Segment(.2, .7, "S01", "สวัสดี <script>alert(1)</script>")]
        if rejected:
            segments.append(Segment(31, 32, "S02", "Rejected"))
        export_timed_clips(source, self.root / "results" / source.stem, "raw model output", segments,
                           capture, invalid_segment_policy="retain")
        source.with_suffix(".json").write_text(json.dumps(capture))
        return {"segment_index": index, "status": "partial" if rejected else "success",
                "offset_seconds": (index - 1) * 30, "audio_seconds": 30, "queue_wait_seconds": .2,
                "inference_seconds": 2.5, "trim_and_store_seconds": .3, "latency_seconds": 3,
                "rtf": .1, "clip_count": 1, "rejected_count": int(rejected)}

    def append(self, row):
        with self.csv.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=REPORT_COLUMNS).writerow(row)

    def test_incremental_append_retries_partial_and_multiline_csv_without_double_counting(self):
        self.append(self.row(rejected=True))
        store = MonitorStore(self.base)
        first = store.snapshot(self.root.name)
        self.assertEqual(first["stats"]["processed_windows"], 1)
        self.assertEqual(first["stats"]["rejected_count"], 1)
        self.assertEqual(len(first["windows"][0]["utterances"]), 1)
        self.assertEqual(first["windows"][0]["rejected"][0]["model_start_seconds"], 31)
        failed = {"segment_index": 2, "status": "failed", "audio_seconds": 30,
                  "error": "An error\nwith a second line and สวัสดี"}
        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=REPORT_COLUMNS).writerow(failed)
        encoded = buffer.getvalue().encode("utf-8")
        split = encoded.index(b"\n") + 1
        with self.csv.open("ab") as file:
            file.write(encoded[:split])
        self.assertEqual(store.snapshot(self.root.name)["stats"]["reported_windows"], 1)
        with self.csv.open("ab") as file:
            file.write(encoded[split:])
        snapshot = store.snapshot(self.root.name)
        self.assertEqual(snapshot["stats"]["reported_windows"], 2)
        self.assertEqual(snapshot["stats"]["failed_windows"], 1)
        self.assertEqual(snapshot["stats"]["weighted_rtf"], .1)
        self.assertEqual(snapshot["windows"][0]["error"], failed["error"])
        self.assertEqual(store.snapshot(self.root.name)["stats"]["reported_windows"], 2)

    def test_live_mqtt_hub_does_not_replay_events_before_new_page_connection(self):
        hub = LiveTranscriptHub()
        hub.publish({"transcript": "Earlier", "start_time": "t1", "end_time": "t2"})
        page_cursor = hub.latest_sequence
        hub.publish({"transcript": "Live now", "start_time": "t3", "end_time": "t4"})
        events = hub.after(page_cursor, timeout=0)
        self.assertEqual([event["transcript"] for event in events], ["Live now"])
        with self.assertRaisesRegex(ValueError, "must contain text"):
            hub.publish({"transcript": "Missing timestamps"})

    def test_recent_memory_is_bounded_and_csv_replacement_resets_totals(self):
        index = RunIndex(self.root)
        for value in range(1000):
            index.add_row({"segment_index": str(value + 1), "status": "success", "audio_seconds": "30",
                           "latency_seconds": "3", "clip_count": "1"})
        self.assertEqual(len(index.rows), 200)
        self.assertEqual(index.totals["processed_windows"], 1000)
        store = MonitorStore(self.base)
        self.append(self.row())
        self.assertEqual(store.snapshot(self.root.name)["stats"]["reported_windows"], 1)
        self.csv.unlink()
        with self.csv.open("w", newline="") as file:
            csv.writer(file).writerow(REPORT_COLUMNS)
        self.assertEqual(store.snapshot(self.root.name)["stats"]["reported_windows"], 0)

    def test_older_pages_are_cumulative_and_sparse_indices_do_not_create_phantom_pages(self):
        for index in range(2, 48):
            self.append({
                "segment_index": index, "status": "success", "offset_seconds": index * 30,
                "audio_seconds": 30, "queue_wait_seconds": 0, "inference_seconds": 2,
                "trim_and_store_seconds": 0, "latency_seconds": 2, "rtf": .1,
                "clip_count": 0, "rejected_count": 0,
            })
        store = MonitorStore(self.base)
        newest = store.snapshot(self.root.name, limit=15)
        self.assertEqual([window["index"] for window in newest["windows"]], list(range(47, 32, -1)))
        self.assertTrue(newest["has_older"])
        middle = store.snapshot(self.root.name, limit=15, before=33)
        self.assertEqual([window["index"] for window in middle["windows"]], list(range(32, 17, -1)))
        self.assertTrue(middle["has_older"])
        oldest = store.snapshot(self.root.name, limit=15, before=18)
        self.assertEqual([window["index"] for window in oldest["windows"]], list(range(17, 2, -1)))
        self.assertTrue(oldest["has_older"])
        final = store.snapshot(self.root.name, limit=15, before=3)
        self.assertEqual([window["index"] for window in final["windows"]], [2])
        self.assertFalse(final["has_older"])
        empty = store.snapshot(self.root.name, limit=15, before=2)
        self.assertEqual(empty["windows"], [])
        self.assertFalse(empty["has_older"])

    def test_history_ignores_an_unflushed_partial_csv_row(self):
        self.append(self.row(2))
        partial = io.StringIO(newline="")
        csv.DictWriter(partial, fieldnames=REPORT_COLUMNS).writerow({
            "segment_index": 3, "status": "success", "offset_seconds": 60,
            "audio_seconds": 30, "clip_count": 1,
        })
        self.csv.write_bytes(self.csv.read_bytes() + partial.getvalue().encode("utf-8").rstrip(b"\n"))
        store = MonitorStore(self.base)
        snapshot = store.snapshot(self.root.name, limit=15, before=4)
        self.assertEqual([window["index"] for window in snapshot["windows"]], [2])

    def test_history_keeps_complete_multiline_csv_rows(self):
        self.append({"segment_index": 2, "status": "failed", "error": "failure\ncontinued"})
        self.append({"segment_index": 3, "status": "success", "audio_seconds": 30,
                     "latency_seconds": 2, "clip_count": 0, "rejected_count": 0})
        store = MonitorStore(self.base)

        snapshot = store.snapshot(self.root.name, limit=15, before=4)

        self.assertEqual([window["index"] for window in snapshot["windows"]], [3, 2])
        self.assertEqual(snapshot["windows"][1]["error"], "failure\ncontinued")

    def test_http_serves_no_auth_api_and_byte_exact_seekable_audio(self):
        self.append(self.row(rejected=True))
        with MonitorServer(("127.0.0.1", 0), self.base) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/") as response:
                    self.assertIn(b"Live speech monitor", response.read())
                with urlopen(url + "/api/runs") as response:
                    self.assertEqual(json.load(response)["runs"][0]["id"], self.root.name)
                with urlopen(url + "/api/runs/" + self.root.name) as response:
                    snapshot = json.load(response)
                audio_url = snapshot["windows"][0]["utterances"][0]["audio_url"]
                clip = self.root / "results/segment-000001/utterance-0001.wav"
                with urlopen(url + audio_url) as response:
                    self.assertEqual(response.read(), clip.read_bytes())
                with urlopen(Request(url + audio_url, headers={"Range": "bytes=2-12"})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), clip.read_bytes()[2:13])
                with urlopen(Request(url + audio_url, method="HEAD")) as response:
                    self.assertEqual(int(response.headers["Content-Length"]), clip.stat().st_size)
                    self.assertEqual(response.read(), b"")
                for path in ("/api/runs/%2e%2e", f"/api/runs/{self.root.name}/audio/segment-000001/../../config.json"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(url + path)
                    self.assertEqual(error.exception.code, 404)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url + audio_url, headers={"Range": "bytes=999999-"}))
                self.assertEqual(error.exception.code, 416)
            finally:
                server.shutdown()
                thread.join()

    def test_live_audio_forwarder_reaches_browser_sse_stream(self):
        with MonitorServer(("127.0.0.1", 0), self.base) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            stream = urlopen(url + "/api/live-audio/stream", timeout=3)
            samples = np.array([16384, -16384, 32767, -32768], dtype=np.int16).reshape(-1, 1)
            try:
                with LiveAudioForwarder(publish_url=url + "/api/live-audio/publish") as forwarder:
                    forwarder.submit(samples, 48000)
                    lines = []
                    while True:
                        line = stream.readline()
                        lines.append(line)
                        if line == b"\n":
                            break
                event_id = next(line for line in lines if line.startswith(b"id: "))
                payload_line = next(line for line in lines if line.startswith(b"data: "))
                payload = json.loads(payload_line.removeprefix(b"data: ").decode().strip())
                self.assertEqual(event_id, b"id: 1\n")
                self.assertEqual(payload["sample_rate"], 48000)
                self.assertEqual(base64.b64decode(payload["audio"]), samples.astype("<i2").tobytes())
            finally:
                stream.close()
                server.shutdown()
                thread.join()

    def test_live_audio_forwarder_reaches_browser_sse_stream(self):
        with MonitorServer(("127.0.0.1", 0), self.base) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            stream = urlopen(url + "/api/live-audio/stream", timeout=3)
            try:
                with LiveAudioForwarder(publish_url=url + "/api/live-audio/publish") as forwarder:
                    forwarder.submit(np.zeros((4, 1), dtype=np.int16), 48000)
                    lines = []
                    while True:
                        line = stream.readline()
                        lines.append(line)
                        if line == b"\n":
                            break
                event_id = next(line for line in lines if line.startswith(b"id: "))
                payload_line = next(line for line in lines if line.startswith(b"data: "))
                payload = json.loads(payload_line.removeprefix(b"data: ").decode().strip())
                self.assertEqual(event_id, b"id: 1\n")
                self.assertEqual(payload["sample_rate"], 48000)
                self.assertEqual(base64.b64decode(payload["audio"]), bytes(8))
            finally:
                stream.close()
                server.shutdown()
                thread.join()

    def test_audio_symlink_cannot_leave_the_output_run(self):
        self.append(self.row())
        outside = self.base / "private.wav"
        outside.write_bytes(b"private data")
        clip = self.root / "results/segment-000001/utterance-0001.wav"
        clip.unlink()
        clip.symlink_to(outside)
        with self.assertRaises(FileNotFoundError):
            MonitorStore(self.base).audio_path(self.root.name, "segment-000001", "utterance-0001.wav")
