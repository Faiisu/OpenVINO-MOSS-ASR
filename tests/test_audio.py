"""Verify native-rate, multichannel, high-precision slicing and export."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from speech_to_text.backend import Segment
from speech_to_text.chunked import transcribe_chunks
from speech_to_text.outputs import export_results


class AudioPreservationTests(unittest.TestCase):
    def test_speaker_exports_preserve_pcm_and_float_samples(self):
        for subtype in ("PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
            with self.subTest(subtype=subtype), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                path = folder / "source.wav"
                samples = np.linspace(-.95, .95, 9600).reshape(-1, 2)
                sf.write(path, samples, 48000, subtype=subtype)
                original, rate = sf.read(path, dtype="float64", always_2d=True)
                export_results(path, folder, "raw", [Segment(.01, .03, "S01", "speech"),
                                                    Segment(.05, .0625, "S01", "speech")])
                exported = folder / "S01.wav"
                info = sf.info(exported)
                self.assertEqual((info.samplerate, info.channels, info.subtype), (rate, 2, subtype))
                actual, _ = sf.read(exported, dtype="float64", always_2d=True)
                np.testing.assert_array_equal(actual, np.concatenate((original[480:1440], original[2400:3000])))

    def test_chunks_preserve_native_samples_including_partial_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            samples = np.linspace(-.95, .95, 24000).reshape(-1, 2)
            sf.write(path, samples, 48000, subtype="PCM_24")
            original, _ = sf.read(path, dtype="int32", always_2d=True)
            windows = []

            def predict(paths, *args):
                for chunk in paths:
                    info = sf.info(chunk)
                    self.assertEqual((info.samplerate, info.channels, info.subtype), (48000, 2, "PCM_24"))
                    windows.append(sf.read(chunk, dtype="int32", always_2d=True)[0])
                return [("raw", [Segment(.01, .04, "S01", "speech")]) for _ in paths], 1.

            session = SimpleNamespace(device="xpu:0", transcribe_batch=predict)
            with patch("torch.xpu.memory_allocated", return_value=100), \
                    patch("torch.xpu.memory_reserved", return_value=100):
                _, _, metrics = transcribe_chunks(path, session, 128, chunk_seconds=.1, batch_size=2)
            self.assertEqual([len(item) for item in windows], [4800, 4800, 2400])
            np.testing.assert_array_equal(np.concatenate(windows), original)
            self.assertEqual(metrics["source_audio"]["sample_rate"], 48000)


if __name__ == "__main__":
    unittest.main()
