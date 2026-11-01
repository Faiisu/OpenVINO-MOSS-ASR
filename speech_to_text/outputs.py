"""Validate diarization and export transcripts and bounded-memory speaker WAVs."""
import json
import math
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta

from .audio import copy_audio_frames, open_audio_source, open_audio_target


def resolve_ffmpeg():
    """Use FFmpeg on PATH or the binary bundled in the local environment."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "FFmpeg unavailable. Install imageio-ffmpeg into the project .venv using uv."
        ) from error


def validate_segments(segments):
    """Reject malformed model intervals and speaker identifiers before export."""
    for segment in segments:
        if (not math.isfinite(segment.start) or not math.isfinite(segment.end)
                or segment.start < 0 or segment.end <= segment.start
                or not re.fullmatch(r"\[?S\d+\]?", segment.speaker)):
            raise ValueError(f"Invalid MOSS segment: {segment}")


def export_results(audio_path, output_dir, raw, segments):
    if not segments:
        (output_dir / "raw_transcript.txt").write_text(raw, encoding="utf-8")
        raise ValueError("MOSS returned no speaker segments. Raw output was saved for inspection.")
    validate_segments(segments)
    export_speakers(audio_path, output_dir, segments)
    payload = {"source": str(audio_path), "segments": [asdict(s) for s in segments]}
    (output_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / "raw_transcript.txt").write_text(raw, encoding="utf-8")
    (output_dir / "transcript.txt").write_text(
        "\n".join(f"[{s.start:.2f}-{s.end:.2f}] {s.speaker}: {s.text}" for s in segments)
        + "\n", encoding="utf-8",
    )


def export_timed_clips(audio_path, output_dir, raw, segments, capture, *, invalid_segment_policy="raise"):
    """Store each utterance WAV beside its text and local/recording/UTC times.

    Preserve overlapping utterances independently. Speaker labels apply only to
    this input window. Empty model output is retained explicitly without clips.
    The gathering workflow can retain invalid model intervals as rejected records
    while exporting valid clips. File/storage errors always propagate.
    """
    if invalid_segment_policy not in {"raise", "retain"}:
        raise ValueError("Invalid-segment policy must be 'raise' or 'retain'.")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw_transcript.txt").write_text(raw, encoding="utf-8")
    entries, rejected = [], []
    capture_start = datetime.fromisoformat(capture["capture_start_utc"])
    with open_audio_source(audio_path) as source:
        rate, frames = source.samplerate, source.frames
        # Validate every range before writing any clips.
        ranges = []
        for index, segment in enumerate(segments, 1):
            try:
                validate_segments([segment])
                start = math.floor(segment.start * rate)
                end = min(frames, math.ceil(segment.end * rate))
                if start >= end:
                    raise ValueError("A model interval falls outside the captured audio.")
            except ValueError as error:
                if invalid_segment_policy == "raise":
                    raise
                rejected.append({
                    "model_segment_index": index,
                    "model_start_seconds": segment.start if math.isfinite(segment.start) else str(segment.start),
                    "model_end_seconds": segment.end if math.isfinite(segment.end) else str(segment.end),
                    "speaker": segment.speaker, "text": segment.text, "reason": str(error),
                })
                continue
            ranges.append((index, segment, start, end))
        for index, segment, start, end in ranges:
            name = f"utterance-{index:04d}"
            with open_audio_target(output_dir / f"{name}.wav", source) as target:
                copy_audio_frames(source, target, start, end - start)
            entry = {
                "audio_file": f"{name}.wav", "text": segment.text,
                "model_segment_index": index,
                "speaker": segment.speaker,
                "model_start_seconds": segment.start, "model_end_seconds": segment.end,
                "start_seconds": start / rate, "end_seconds": end / rate,
                "recording_start_seconds": capture["offset_seconds"] + start / rate,
                "recording_end_seconds": capture["offset_seconds"] + end / rate,
                "start_utc": (capture_start + timedelta(seconds=start / rate)).isoformat(),
                "end_utc": (capture_start + timedelta(seconds=end / rate)).isoformat(),
                "sample_rate": rate, "channels": source.channels, "subtype": source.subtype,
                "frames": end - start,
            }
            (output_dir / f"{name}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            entries.append(entry)
    # The manifest is the completion marker; partial exports have no manifest.
    payload = {
        "source_audio": str(audio_path),
        "capture": {key: value for key, value in capture.items()
                    if key not in {"ready_at", "queued_at"}},
        "model": "MOSS-Transcribe-Diarize", "runtime": "OpenVINO",
        "speaker_identity_scope": "Independent within each captured segment.",
        "utc_timing_scope": "Estimated from capture start; relative offsets are sample accurate.",
        "status": ("partial" if entries else "rejected") if rejected else ("complete" if entries else "no_segments"),
        "segments": entries,
        "rejected_segments": rejected,
    }
    (output_dir / "transcript.txt").write_text(
        "\n".join(f"[{entry['model_start_seconds']:.2f}-{entry['model_end_seconds']:.2f}] "
                  f"{entry['speaker']}: {entry['text']}" for entry in entries),
        encoding="utf-8")
    (output_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(entries)


def export_speakers(decoded, output_dir, segments):
    """Concatenate each speaker's intervals, merging overlaps within that speaker."""
    with open_audio_source(decoded) as source:
        rate, frames = source.samplerate, source.frames
        groups = {}
        for segment in segments:
            start, end = int(segment.start * rate), min(frames, math.ceil(segment.end * rate))
            if start >= end:
                raise ValueError("A segment falls outside the audio duration.")
            speaker = segment.speaker.strip("[]")
            groups.setdefault(speaker, []).append((start, end))
        for speaker, intervals in groups.items():
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            with open_audio_target(output_dir / f"{speaker}.wav", source) as target:
                for start, end in merged:
                    copy_audio_frames(source, target, start, end - start)
