"""Exercise real WAV interval extraction and the CLI without model downloads."""
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from speech_to_text.backend import Segment
from speech_to_text.cli import main
from speech_to_text.outputs import export_speakers, export_results


class WorkflowTests(unittest.TestCase):
    def test_speaker_intervals_merge_and_preserve_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "input.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 10, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack("<20h", *range(20)))
            segments = [Segment(.2, .7, "S01", "A"), Segment(.5, 1., "S01", "B"),
                        Segment(1.5, 2.5, "[S02]", "C"), Segment(1.2, 1.4, "S01", "D")]
            export_speakers(audio, folder, segments)
            for speaker, expected in [("S01", list(range(2, 10)) + [12, 13]),
                                      ("S02", list(range(15, 20)))]:
                with wave.open(str(folder / f"{speaker}.wav"), "rb") as result:
                    data = result.readframes(result.getnframes())
                    self.assertEqual(list(struct.unpack(f"<{len(data)//2}h", data)), expected)

    def test_invalid_segments_rejected_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_results(Path("missing.wav"), Path(directory), "raw",
                               [Segment(float("nan"), 1., "S01", "text")])

    def test_empty_result_preserves_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(ValueError):
                export_results(Path("missing.wav"), folder, "unparsed", [])
            self.assertEqual((folder / "raw_transcript.txt").read_text(), "unparsed")

    def test_cancel_does_not_load_model(self):
        with patch("speech_to_text.cli.select_audio", return_value=None), \
                patch("speech_to_text.cli.transcribe") as inference:
            self.assertEqual(main([]), 0)
            inference.assert_not_called()

    def test_missing_file_does_not_load_model(self):
        with patch("speech_to_text.cli.transcribe") as inference:
            self.assertEqual(main(["/nonexistent/audio.wav"]), 1)
            inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
