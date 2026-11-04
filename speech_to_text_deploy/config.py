"""Configuration, project paths and CSV schema."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid


DEPLOY_ROOT = Path(__file__).resolve().parent
ROOT = DEPLOY_ROOT.parent


COMPLETED_STATUSES = ("success", "partial", "no_speech")


REPORT_COLUMNS = (
    "segment_index", "status", "offset_seconds", "audio_seconds",
    "capture_start_utc", "capture_end_utc", "timestamp_source", "ready_monotonic_seconds",
    "queued_monotonic_seconds", "processing_start_monotonic_seconds",
    "output_ready_monotonic_seconds", "prepare_seconds", "queue_wait_seconds",
    "inference_seconds", "vad_speech_seconds", "trim_and_store_seconds", "latency_seconds", "rtf",
    "clip_count", "rejected_count", "source_audio", "output_dir", "error",
)


@dataclass(frozen=True)
class GatheringConfig:
    output_dir: str
    model_dir: str = str(ROOT / ".tools" / "moss-subway-openvino-fp16")
    model_id: str = "faiisu/moss-asr-th-finetuned-subway"
    ov_device: str = "GPU"
    input_device: int | str | None = None
    samplerate: int = 48000
    input_latency_seconds: float = 1.0
    internet_time_timeout_seconds: float = 1.0
    live_audio_enabled: bool = True
    live_audio_url: str = "http://127.0.0.1:18000/api/live-audio/publish"
    segment_seconds: float = 30.0
    max_audio_seconds: float | None = None
    max_new_tokens: int = 5120
    hotwords: tuple[str, ...] = ("Subway", "ซัพเวย์", "ซับเวย์")
    vad_enabled: bool = True
    vad_threshold: float = 0.9
    vad_min_speech_duration_ms: int = 1000
    vad_min_silence_duration_ms: int = 100
    vad_speech_pad_ms: int = 30
    languages: tuple[str, ...] = ("Thai", "English")
    cpu_threads: int = 1
    source_audio: str | None = None
    replay_realtime: bool = True
    min_free_bytes: int = 256 * 1024**2
    save_audio: bool = True
    save_transcript: bool = True
    mqtt_enabled: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = "ASR/v1/report"
    mqtt_qos: int = 1
    mqtt_keepalive_seconds: int = 30
    mqtt_connect_timeout_seconds: float = 5.0
    mqtt_publish_timeout_seconds: float = 5.0
    greeting_preroll_keyword: str = "สวัสดี"
    greeting_preroll_seconds: float = 2.0

    def validate(self):
        for value in (self.segment_seconds, self.samplerate, self.max_new_tokens,
                      self.cpu_threads, self.input_latency_seconds,
                      self.internet_time_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Audio settings, token limit and thread count must be positive and finite.")
        if self.max_audio_seconds is not None and (
                not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0):
            raise ValueError("Optional recording duration must be positive and finite.")
        if self.min_free_bytes < 0:
            raise ValueError("Free-storage reserve cannot be negative.")
        if not isinstance(self.vad_enabled, bool):
            raise ValueError("VAD enabled must be a boolean.")
        if not math.isfinite(self.vad_threshold) or not 0 < self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be finite and in (0, 1].")
        if (isinstance(self.vad_min_speech_duration_ms, bool)
                or not isinstance(self.vad_min_speech_duration_ms, int)
                or self.vad_min_speech_duration_ms < 1):
            raise ValueError("Minimum VAD speech duration must be a positive integer.")
        if not isinstance(self.live_audio_enabled, bool):
            raise ValueError("Live audio enabled must be a boolean.")
        if (not isinstance(self.live_audio_url, str)
                or not self.live_audio_url.startswith(("http://", "https://"))):
            raise ValueError("Live audio URL must use HTTP or HTTPS.")
        for name, value in (("minimum VAD silence duration", self.vad_min_silence_duration_ms),
                            ("VAD speech padding", self.vad_speech_pad_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if any(not isinstance(word, str) or not word.strip() for word in self.hotwords):
            raise ValueError("Hotwords must be nonempty strings.")
        if not isinstance(self.mqtt_enabled, bool):
            raise ValueError("MQTT enabled must be a boolean.")
        if not isinstance(self.save_audio, bool) or not isinstance(self.save_transcript, bool):
            raise ValueError("Audio and transcript save settings must be booleans.")
        if not self.mqtt_enabled and (not self.save_audio or not self.save_transcript):
            raise ValueError("Enable MQTT when audio or transcript persistence is disabled.")
        if not self.mqtt_host.strip() or not self.mqtt_topic.strip():
            raise ValueError("MQTT host and topic must be nonempty.")
        if not 1 <= self.mqtt_port <= 65535:
            raise ValueError("MQTT port must be between 1 and 65535.")
        if self.mqtt_qos not in (0, 1, 2):
            raise ValueError("MQTT QoS must be 0, 1, or 2.")
        if self.mqtt_keepalive_seconds < 1:
            raise ValueError("MQTT keepalive must be a positive integer.")
        for value in (self.mqtt_connect_timeout_seconds, self.mqtt_publish_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("MQTT timeouts must be positive and finite.")
        if not isinstance(self.greeting_preroll_keyword, str):
            raise ValueError("Greeting pre-roll keyword must be a string.")
        if not math.isfinite(self.greeting_preroll_seconds) or self.greeting_preroll_seconds < 0:
            raise ValueError("Greeting pre-roll duration must be finite and nonnegative.")
        if not (Path(self.model_dir) / "manifest.json").is_file():
            raise FileNotFoundError("Export the MOSS OpenVINO model before collecting audio.")


DEFAULT_OUTPUT_BASE = ROOT / "research_output_gathering" / "output"


def load_settings(path=None):
    """Share JSON settings between CLI/notebook; resolve data paths from project root."""
    source = Path(path) if path is not None else Path(__file__).with_name("settings.json")
    values = json.loads(source.read_text(encoding="utf-8"))
    values.setdefault("output_base", str(DEFAULT_OUTPUT_BASE))
    values.setdefault("internet_time_timeout_seconds", 1.0)
    values.setdefault("live_audio_enabled", True)
    values.setdefault("live_audio_url", "http://127.0.0.1:18000/api/live-audio/publish")
    values.setdefault("hotwords", ["Subway", "ซัพเวย์", "ซับเวย์"])
    values.setdefault("vad_threshold", 0.9)
    values.setdefault("vad_enabled", True)
    values.setdefault("vad_min_speech_duration_ms", 1000)
    values.setdefault("vad_min_silence_duration_ms", 100)
    values.setdefault("vad_speech_pad_ms", 30)
    values.setdefault("bounded_worker_timeout_seconds", 3600)
    values.setdefault("continuous_worker_timeout_seconds", None)
    values.setdefault("drain_timeout_seconds", 600)
    values.setdefault("mqtt_enabled", True)
    values.setdefault("save_audio", True)
    values.setdefault("save_transcript", True)
    values.setdefault("mqtt_host", "127.0.0.1")
    values.setdefault("mqtt_port", 1883)
    values.setdefault("mqtt_topic", "ASR/v1/report")
    values.setdefault("mqtt_qos", 1)
    values.setdefault("mqtt_keepalive_seconds", 30)
    values.setdefault("mqtt_connect_timeout_seconds", 5.0)
    values.setdefault("mqtt_publish_timeout_seconds", 5.0)
    values.setdefault("greeting_preroll_keyword", "สวัสดี")
    values.setdefault("greeting_preroll_seconds", 2.0)
    values["hotwords"] = tuple(values["hotwords"])
    values["languages"] = tuple(values.get("languages", ("Thai", "English")))
    for field in ("model_dir", "source_audio", "output_base"):
        if values.get(field):
            value = Path(values[field])
            values[field] = str(value if value.is_absolute() else ROOT / value)
    return values


def new_run_dir(output_base=DEFAULT_OUTPUT_BASE):
    """Choose a unique directory; the worker creates it atomically."""
    name = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    return Path(output_base).resolve() / name
