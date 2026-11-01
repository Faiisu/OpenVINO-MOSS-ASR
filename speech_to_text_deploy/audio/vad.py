"""CPU voice-activity detection for captured audio windows."""

from pathlib import Path

import numpy as np
import soundfile as sf

from ..runtime import release_model_memory


class SileroVAD:
    """Detect speech regions and release the TorchScript VAD model on exit."""

    def __init__(self, *, threshold=0.9, min_speech_duration_ms=1000,
                 min_silence_duration_ms=100, speech_pad_ms=30):
        if not 0 < threshold <= 1:
            raise ValueError("VAD threshold must be in (0, 1].")
        if min_speech_duration_ms < 1:
            raise ValueError("Minimum VAD speech duration must be positive.")
        self.threshold = float(threshold)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.model = None

    def __enter__(self):
        from silero_vad import load_silero_vad

        try:
            self.model = load_silero_vad()
            self.model.eval()
            return self
        except BaseException:
            self.close()
            raise

    def detect(self, audio_path):
        """Return speech regions in seconds for a WAV window."""
        import torch
        import torchaudio.functional
        from silero_vad import get_speech_timestamps

        if self.model is None:
            raise RuntimeError("Silero VAD is not open.")
        samples = waveform = None
        try:
            samples, sample_rate = sf.read(Path(audio_path), dtype="float32", always_2d=True)
            mono = np.ascontiguousarray(samples.mean(axis=1))
            waveform = torch.from_numpy(mono)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            with torch.inference_mode():
                regions = get_speech_timestamps(
                    waveform,
                    self.model,
                    threshold=self.threshold,
                    sampling_rate=16000,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=True,
                )
            return [(float(region["start"]), float(region["end"])) for region in regions]
        finally:
            samples = waveform = None
            release_model_memory(torch.device("cpu"))

    def close(self):
        model, self.model = self.model, None
        del model
        import torch

        release_model_memory(torch.device("cpu"))

    def __exit__(self, *exc):
        self.close()
