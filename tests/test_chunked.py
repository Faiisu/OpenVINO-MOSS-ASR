"""Check complete audio coverage and independent chunk speaker/timestamp mappings."""
import tempfile
import unittest
import wave
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech_to_text.backend import Segment
from speech_to_text.chunked import transcribe_chunked, transcribe_chunks


class ChunkedTests(unittest.TestCase):
    def test_all_samples_and_final_partial_window_are_processed(self):
        frame_counts = []

        def predict(path, *args, **kwargs):
            with wave.open(str(path), "rb") as audio:
                frame_counts.append(audio.getnframes())
            return "raw", [Segment(.1, .4, "S01", "speech")]

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 40000)
            latencies = iter([2, 3, 1])
            def predict_batch(paths, *args):
                return [predict(path) for path in paths], next(latencies)
            session = SimpleNamespace(device="xpu:0", transcribe_batch=predict_batch)
            with patch("torch.xpu.memory_allocated", return_value=100), \
                    patch("torch.xpu.synchronize"), \
                    patch("torch.xpu.memory_reserved", return_value=100):
                raw, segments, result = transcribe_chunks(source, session, 100, chunk_seconds=1, batch_size=1)
        self.assertEqual(frame_counts, [16000, 16000, 8000])
        self.assertEqual([s.start for s in segments], [.1, 1.1, 2.1])
        self.assertEqual([s.speaker for s in segments], ["S01", "S02", "S03"])
        self.assertEqual(result["audio_seconds"], 2.5)
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual([chunk["elapsed_seconds"] for chunk in result["chunks"]], [2, 3, 1])
        self.assertEqual([chunk["rtf"] for chunk in result["chunks"]], [2, 3, 2])
        self.assertEqual(result["chunk_processing_seconds"], 6)
        self.assertEqual(result["chunk_processing_rtf"], 6 / 2.5)
        self.assertIn("offset 2.00s", raw)

    def test_batch_latency_is_not_counted_twice_and_partial_batch_is_kept(self):
        submitted = []
        def predict(paths, *args):
            submitted.append(len(paths))
            return [("raw", [Segment(.1, .4, "S01", "speech")]) for _ in paths], 4.0
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 40000)
            session = SimpleNamespace(device="xpu:0", transcribe_batch=predict)
            with patch("torch.xpu.memory_allocated", return_value=100), \
                    patch("torch.xpu.memory_reserved", return_value=100):
                _, segments, result = transcribe_chunks(source, session, 100, chunk_seconds=1, batch_size=2)
        self.assertEqual(submitted, [2, 1])
        self.assertEqual(result["chunk_processing_seconds"], 8)
        self.assertEqual([b["rtf"] for b in result["batches"]], [2, 8])
        self.assertEqual([c["rtf"] for c in result["chunks"]], [4, 4, 8])
        self.assertEqual([c["batch_index"] for c in result["chunks"]], [1, 1, 2])
        self.assertEqual([s.start for s in segments], [.1, 1.1, 2.1])
        self.assertEqual([s.speaker for s in segments], ["S01", "S02", "S03"])

    def test_invalid_batch_size_rejected_before_loading(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                transcribe_chunked("missing.wav", "test", "xpu:0", 100, batch_size=size)

    def test_invalid_chunk_duration_rejected(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                transcribe_chunked("missing.wav", "test", "xpu:0", 100,
                                   chunk_seconds=duration)

    def test_notebook_refreshes_cached_worker_api(self):
        import speech_to_text.isolated as isolated_module

        def old_worker(audio_path, model_id, device, max_new_tokens, languages=None):
            raise AssertionError("The notebook must replace the cached worker API.")

        notebook_path = Path(__file__).resolve().parents[1] / "research" / "research_iGPU.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        setup = "".join(notebook["cells"][8]["source"]).split("if not audio_files:", 1)[0]
        with patch.object(isolated_module, "transcribe_isolated", old_worker):
            self.assertNotIn("chunk_seconds", inspect.signature(isolated_module.transcribe_isolated).parameters)
            exec(compile(setup, str(notebook_path), "exec"), {})
            self.assertIn("chunk_seconds", inspect.signature(isolated_module.transcribe_isolated).parameters)


if __name__ == "__main__":
    unittest.main()
