"""Bounded in-memory fan-out buffer for live PCM audio chunks."""

from collections import deque
from dataclasses import dataclass
from threading import Condition


@dataclass(frozen=True)
class AudioChunk:
    sequence: int
    sample_rate: int
    pcm16: bytes


class LiveAudioHub:
    """Keep a short audio tail and wake every SSE client when new chunks arrive."""

    def __init__(self, max_chunks=50):
        if max_chunks < 1:
            raise ValueError("Live audio buffer must retain at least one chunk.")
        self._chunks = deque(maxlen=max_chunks)
        self._sequence = 0
        self._condition = Condition()

    @property
    def latest_sequence(self):
        with self._condition:
            return self._sequence

    def publish(self, pcm16, sample_rate):
        """Append mono little-endian PCM16 and return its monotonically increasing ID."""
        if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
            raise ValueError("Live audio must contain nonempty PCM16 samples.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
            raise ValueError("Live audio sample rate must be between 8000 and 96000 Hz.")
        with self._condition:
            self._sequence += 1
            self._chunks.append(AudioChunk(self._sequence, sample_rate, pcm16))
            self._condition.notify_all()
            return self._sequence

    def after(self, sequence, timeout=10.0):
        """Wait for chunks newer than sequence and return the currently buffered tail."""
        with self._condition:
            if self._sequence <= sequence:
                self._condition.wait(timeout)
            return tuple(chunk for chunk in self._chunks if chunk.sequence > sequence)
