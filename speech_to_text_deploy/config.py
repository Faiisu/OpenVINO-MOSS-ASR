"""Configuration, project paths and CSV schema."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]


COMPLETED_STATUSES = ("success", "partial")


REPORT_COLUMNS = (
    "segment_index", "status", "offset_seconds", "audio_seconds",
    "capture_start_utc", "capture_end_utc", "ready_monotonic_seconds",
    "queued_monotonic_seconds", "processing_start_monotonic_seconds",
    "output_ready_monotonic_seconds", "prepare_seconds", "queue_wait_seconds",
    "inference_seconds", "trim_and_store_seconds", "latency_seconds", "rtf",
    "clip_count", "rejected_count", "source_audio", "output_dir", "error",
)


@dataclass(frozen=True)
class GatheringConfig:
    output_dir: str
    model_dir: str = str(ROOT / ".tools" / "moss-openvino-fp16")
    model_id: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    ov_device: str = "GPU"
    input_device: int | str | None = None
    samplerate: int = 48000
    input_latency_seconds: float = 1.0
    segment_seconds: float = 30.0
    max_audio_seconds: float | None = None
    max_new_tokens: int = 1024
    languages: tuple[str, ...] = ("Thai", "English")
    cpu_threads: int = 1
    source_audio: str | None = None
    replay_realtime: bool = True
    min_free_bytes: int = 256 * 1024**2

    def validate(self):
        for value in (self.segment_seconds, self.samplerate, self.max_new_tokens,
                      self.cpu_threads, self.input_latency_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Audio settings, token limit and thread count must be positive and finite.")
        if self.max_audio_seconds is not None and (
                not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0):
            raise ValueError("Optional recording duration must be positive and finite.")
        if self.min_free_bytes < 0:
            raise ValueError("Free-storage reserve cannot be negative.")
        if not (Path(self.model_dir) / "manifest.json").is_file():
            raise FileNotFoundError("Export the MOSS OpenVINO model before collecting audio.")


DEFAULT_OUTPUT_BASE = ROOT / "research_output_gathering" / "output"


def load_settings(path=None):
    """Share JSON settings between CLI/notebook; resolve data paths from project root."""
    source = Path(path) if path is not None else Path(__file__).with_name("settings.json")
    values = json.loads(source.read_text(encoding="utf-8"))
    values.setdefault("output_base", str(DEFAULT_OUTPUT_BASE))
    for field in ("model_dir", "source_audio", "output_base"):
        if values.get(field):
            value = Path(values[field])
            values[field] = str(value if value.is_absolute() else ROOT / value)
    return values


def new_run_dir(output_base=DEFAULT_OUTPUT_BASE):
    """Choose a unique directory; the worker creates it atomically."""
    name = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    return Path(output_base).resolve() / name
