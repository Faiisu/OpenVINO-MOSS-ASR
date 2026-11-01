"""Verify nested attribution, partial reports, and generation-hook cleanup."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from speech_to_text.profiling import StageProfiler
from speech_to_text.runtime import release_model_memory
from speech_to_text.backend import MossSession


class ProfilingTests(unittest.TestCase):
    def test_profiling_does_not_prevent_cleanup_if_device_synchronization_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            session = object.__new__(MossSession)
            session.device = "xpu:0"
            session.model = session.processor = object()
            session.profiler = StageProfiler(folder)
            with patch.object(session.profiler, "synchronize", side_effect=RuntimeError("device failed")), \
                    patch("speech_to_text.backend.release_model_memory") as cleanup:
                session.close()
            self.assertIsNone(session.model)
            self.assertIsNone(session.processor)
            cleanup.assert_called_once_with("xpu:0")

    def test_nested_times_are_not_counted_twice_and_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            profiler = StageProfiler(folder)
            with self.assertRaises(ValueError):
                with profiler.stage("parent"):
                    with profiler.stage("child"):
                        raise ValueError("interrupted stage")
            profiler.save()
            data = json.loads((Path(folder) / "profile.json").read_text())
            parent, child = data["events"]
            self.assertEqual(child["parent_id"], parent["id"])
            self.assertAlmostEqual(parent["seconds"], parent["exclusive_seconds"] + child["seconds"])
            self.assertEqual(profiler.stack, [])
            self.assertTrue((Path(folder) / "stage_summary.csv").is_file())

    def test_forward_hooks_are_removed_after_model_failure(self):
        class FailingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(whisper_encoder=torch.nn.Identity(),
                                             vq_adaptor=torch.nn.Identity(),
                                             language_model=torch.nn.Identity())

            def forward(self, **kwargs):
                raise RuntimeError("model failure")

        model = inputs = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                profiler = StageProfiler(folder)
                model = FailingModel()
                inputs = {"input_ids": torch.ones((1, 2), dtype=torch.long)}
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    with profiler.generation(model, inputs, True):
                        model(**inputs)
                for module in (model, model.model.whisper_encoder, model.model.vq_adaptor,
                               model.model.language_model):
                    self.assertFalse(module._forward_pre_hooks)
                    self.assertFalse(module._forward_hooks)
                self.assertEqual(profiler.stack, [])
                self.assertEqual(profiler.batches[0]["prefill_calls"], 1)
        finally:
            model = inputs = None
            release_model_memory(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
