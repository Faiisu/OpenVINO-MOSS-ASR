"""Check unequal prompt padding, independent EOS and submission timing."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers.feature_extraction_utils import BatchFeature

from speech_to_text.batched import generate_batch
from speech_to_text.backend import MossSession


class BatchedTests(unittest.TestCase):
    def test_left_padding_and_independent_end_tokens(self):
        processor = Mock()
        processor.feature_extractor.sampling_rate = 16000
        processor.return_value = BatchFeature({
            "input_ids": torch.tensor([[10, 11, 12], [20, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        })
        processor.tokenizer.decode.side_effect = ["first", "second"]
        model = Mock()
        model.generation_config = SimpleNamespace(eos_token_id=[99, 98])
        model.generate.return_value = torch.tensor([[10, 11, 12, 30, 99, 0], [0, 0, 20, 40, 41, 98]])
        with patch("moss_transcribe_diarize.inference_utils.process_audio_info", return_value=[[]]):
            result = generate_batch(model, processor, ["one.wav", "two.wav"], "prompt", 3, "cpu")
        args = model.generate.call_args.kwargs
        self.assertEqual(args["input_ids"].tolist(), [[10, 11, 12], [0, 0, 20]])
        self.assertEqual(args["attention_mask"].tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual([r["generated_tokens"] for r in result], [2, 3])
        self.assertEqual([r["text"] for r in result], ["first", "second"])

    def test_timer_stops_before_cleanup_and_cleanup_runs_on_failure(self):
        session = object.__new__(MossSession)
        session.device = "xpu:0"
        session.model = session.processor = object()
        events = []
        def tick():
            events.append("timer")
            return len(events)
        with patch("torch.xpu.synchronize"), \
                patch("speech_to_text.backend.perf_counter", side_effect=tick), \
                patch("speech_to_text.batched.generate_batch", return_value=[{"text": "raw", "generated_tokens": 2}]), \
                patch("moss_transcribe_diarize.parse_transcript", return_value=[]), \
                patch("speech_to_text.backend.release_model_memory", side_effect=lambda _: events.append("cleanup")):
            result, elapsed = session.transcribe_batch(["one.wav"], 10)
        self.assertEqual(result, [("raw", [])])
        self.assertEqual(elapsed, 1)
        self.assertEqual(events, ["timer", "timer", "cleanup"])
        with patch("torch.xpu.synchronize"), \
                patch("speech_to_text.batched.generate_batch", side_effect=RuntimeError("generation failed")), \
                patch("speech_to_text.backend.release_model_memory") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                session.transcribe_batch(["one.wav"], 10)
            cleanup.assert_called_once_with("xpu:0")
