"""Forward captured PCM chunks to the local monitor without blocking capture."""

from queue import Empty, Full, Queue
import threading
from time import monotonic
from urllib.request import Request, urlopen

import numpy as np


class LiveAudioForwarder:
    """Convert audio to mono PCM16 and POST chunks on a bounded background queue."""

    def __init__(self, *, enabled=True, publish_url="http://127.0.0.1:18000/api/live-audio/publish",
                 queue_chunks=20):
        self.enabled = enabled
        self.publish_url = publish_url
        self.queue = Queue(maxsize=queue_chunks)
        self.stop_event = threading.Event()
        self.thread = None
        self.warned = False

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._send, name="LiveAudioForwarder", daemon=True)
            self.thread.start()
        return self

    @staticmethod
    def _pcm16(samples):
        audio = np.asarray(samples)
        if audio.dtype == np.int16:
            if audio.ndim == 2:
                audio = (audio[:, 0] if audio.shape[1] == 1 else
                         (audio.astype(np.float32) / 32768.0).mean(axis=1))
            if audio.dtype == np.int16:
                return np.ascontiguousarray(audio.astype("<i2", copy=False)).tobytes()
        if audio.ndim == 2:
            if np.issubdtype(audio.dtype, np.integer):
                bounds = np.iinfo(audio.dtype)
                scale = float(max(abs(bounds.min), bounds.max))
                audio = audio.astype(np.float32) / scale
            else:
                audio = audio.astype(np.float32)
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            bounds = np.iinfo(audio.dtype)
            scale = float(max(abs(bounds.min), bounds.max))
            audio = audio.astype(np.float32) / scale
        else:
            audio = audio.astype(np.float32, copy=False)
        audio = np.clip(audio, -1.0, 1.0)
        return np.rint(audio * 32767).astype("<i2").tobytes()

    def submit(self, samples, sample_rate):
        """Queue one capture chunk; drop the oldest chunk rather than stall ASR."""
        if not self.enabled or self.thread is None or self.stop_event.is_set():
            return
        try:
            chunk = (int(sample_rate), self._pcm16(samples))
            try:
                self.queue.put_nowait(chunk)
            except Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Empty:
                    pass
                self.queue.put_nowait(chunk)
        except (TypeError, ValueError, OverflowError):
            # An unsupported live preview must never stop recording or inference.
            return

    def _send(self):
        retry_after = 0.0
        while not self.stop_event.is_set():
            try:
                sample_rate, pcm16 = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if monotonic() < retry_after:
                    continue
                request = Request(
                    self.publish_url,
                    data=pcm16,
                    headers={"Content-Type": "application/octet-stream",
                             "X-Audio-Sample-Rate": str(sample_rate)},
                    method="POST",
                )
                with urlopen(request, timeout=0.25) as response:
                    if not 200 <= response.status < 300:
                        raise OSError(f"Live audio endpoint returned HTTP {response.status}.")
                retry_after = 0.0
                self.warned = False
            except OSError as error:
                retry_after = monotonic() + 3.0
                if not self.warned:
                    print(f"Live audio monitor unavailable; ASR continues ({error}).", flush=True)
                    self.warned = True
            finally:
                self.queue.task_done()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def __exit__(self, *exc):
        self.close()
