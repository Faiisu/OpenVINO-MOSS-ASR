"""Capture fixed-duration audio files without blocking on ASR inference."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import soundfile as sf

from .audio import open_audio_source, open_audio_target


@dataclass
class CapturedSegment:
    index: int
    path: Path
    offset_seconds: float
    duration_seconds: float
    capture_start_utc: str
    capture_end_utc: str
    ready_at: float
    queued_at: float = 0.0

    def metadata(self):
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def resolve_microphone(device=None):
    """Select BY-V by PortAudio index/name, rejecting absent or ambiguous devices."""
    import sounddevice as sd

    if device is not None:
        sd.query_devices(device, "input")
        return device
    matches = [i for i, info in enumerate(sd.query_devices())
               if "BY-V" in info["name"] and info["max_input_channels"] > 0]
    if len(matches) != 1:
        raise ValueError("Select a microphone explicitly: expected exactly one BY-V input.")
    return matches[0]


def capture_segments(folder, stop, *, segment_seconds=30, samplerate=48000,
                     input_device=None, input_latency_seconds=1.0,
                     max_audio_seconds=None, source_audio=None,
                     replay_realtime=True, min_free_bytes=256 * 1024**2):
    """Yield closed WAVs; preserve a final partial segment on stop or file EOF.

    UTC times estimate capture from stream start; offsets are sample accurate.
    Replay uses the file's native rate/channels/precision and optional pacing.
    Input overflow is fatal because the sample timeline would no longer be valid.
    Live input reserves a buffer for scheduling and inference-cleanup pauses.
    """
    import sounddevice as sd

    if segment_seconds <= 0 or (max_audio_seconds is not None and max_audio_seconds <= 0):
        raise ValueError("Segment length and optional capture duration must be positive.")
    if not math.isfinite(input_latency_seconds) or input_latency_seconds <= 0:
        raise ValueError("Input latency must be positive and finite.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if source_audio is not None:
            reader = stack.enter_context(open_audio_source(source_audio))
            rate, channels = reader.samplerate, reader.channels
            read = lambda frames: reader.read(frames, dtype="float64", always_2d=True)
            target = lambda path: open_audio_target(path, reader)
        else:
            device = resolve_microphone(input_device)
            reader = stack.enter_context(sd.InputStream(
                device=device, samplerate=samplerate, channels=1, dtype="int16",
                latency=input_latency_seconds))
            rate, channels = int(reader.samplerate), 1

            def read(frames):
                data, overflow = reader.read(frames)
                if overflow:
                    raise RuntimeError("BY-V input overflow: recording stopped to preserve timestamp integrity.")
                return data

            target = lambda path: sf.SoundFile(path, mode="w", samplerate=rate,
                                               channels=channels, subtype="PCM_16", format="WAV")
        chunk_frames = round(segment_seconds * rate)
        if chunk_frames < 1:
            raise ValueError("Segment length is shorter than one audio frame.")
        limit = round(max_audio_seconds * rate) if max_audio_seconds is not None else None
        origin_utc, origin_clock = datetime.now(timezone.utc), perf_counter()
        total_frames, index = 0, 0
        eof = False
        while not stop.is_set() and not eof and (limit is None or total_frames < limit):
            if shutil.disk_usage(folder).free < min_free_bytes:
                raise OSError("Capture stopped: insufficient free storage space.")
            index += 1
            path = folder / f"segment-{index:06d}.wav"
            first_frame, written = total_frames, 0
            ready_at = perf_counter()
            with target(path) as output:
                while written < chunk_frames and not stop.is_set():
                    count = min(max(1, rate // 10), chunk_frames - written)
                    if limit is not None:
                        count = min(count, limit - total_frames)
                    if count <= 0:
                        break
                    data = read(count)
                    if len(data) == 0:
                        eof = True
                        break
                    output.write(data)
                    written += len(data)
                    total_frames += len(data)
                    if source_audio is not None and replay_realtime:
                        stop.wait(max(0, origin_clock + total_frames / rate - perf_counter()))
                    ready_at = perf_counter()
            if not written:
                path.unlink()
                break
            item = CapturedSegment(
                index, path, first_frame / rate, written / rate,
                (origin_utc + timedelta(seconds=first_frame / rate)).isoformat(),
                (origin_utc + timedelta(seconds=total_frames / rate)).isoformat(), ready_at,
            )
            path.with_suffix(".json").write_text(
                json.dumps(item.metadata(), indent=2), encoding="utf-8")
            yield item
