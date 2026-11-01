"""Validate the minimal MQTT JSON shape for each utterance."""

import json
import unittest

from speech_to_text_deploy.outputs.mqtt import transcript_messages


class MqttReportTests(unittest.TestCase):
    def test_message_contains_only_transcript_and_real_start_end(self):
        manifest = {"segments": [
            {"text": "สวัสดีค่ะ", "model_start_utc": "2026-11-01T00:00:01+00:00",
             "model_end_utc": "2026-11-01T00:00:02+00:00"},
            {"text": "รับอะไรดีคะ", "model_start_utc": "2026-11-01T00:00:03+00:00",
             "model_end_utc": "2026-11-01T00:00:04+00:00"},
        ]}
        messages = [json.loads(json.dumps(item, ensure_ascii=False))
                    for item in transcript_messages(manifest)]
        self.assertEqual(len(messages), 2)
        self.assertEqual(set(messages[0]), {"transcript", "start_time", "end_time"})
        self.assertEqual(messages[0]["transcript"], "สวัสดีค่ะ")
        self.assertEqual(messages[0]["start_time"], "2026-11-01T00:00:01+00:00")
        self.assertEqual(messages[0]["end_time"], "2026-11-01T00:00:02+00:00")

    def test_no_transcripts_produce_no_mqtt_events(self):
        self.assertEqual(transcript_messages({"segments": []}), [])
