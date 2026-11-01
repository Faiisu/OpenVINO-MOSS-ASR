"""Validate intervals and save paired timestamped WAV/JSON clips."""
from datetime import datetime, timedelta
import json
import math
import re
from ..audio.io import copy_audio_frames, open_audio_source, open_audio_target


def validate_segments(segments):
    """Reject malformed model intervals and speaker identifiers before export."""
    for segment in segments:
        if (not math.isfinite(segment.start) or not math.isfinite(segment.end)
                or segment.start < 0 or segment.end <= segment.start
                or not re.fullmatch(r"\[?S\d+\]?", segment.speaker)):
            raise ValueError(f"Invalid MOSS segment: {segment}")


def export_timed_clips(audio_path, output_dir, raw, segments, capture, *,
                       invalid_segment_policy="raise", vad=None,
                       greeting_preroll_keyword=None, greeting_preroll_seconds=0.0,
                       save_audio=True, save_transcript=True, return_manifest=False):
    """Store each utterance WAV beside its text and local/recording/UTC times.

    Preserve overlapping utterances independently. Speaker labels apply only to
    this input window. Empty model output is retained explicitly without clips.
    The gathering workflow can retain invalid model intervals as rejected records
    while exporting valid clips. File/storage errors always propagate.
    Set save flags independently to retain only audio or transcript artifacts.
    ``return_manifest`` exposes in-memory metadata for MQTT when transcripts
    must not be written to disk.
    """
    if invalid_segment_policy not in {"raise", "retain"}:
        raise ValueError("Invalid-segment policy must be 'raise' or 'retain'.")
    if greeting_preroll_keyword is not None and not isinstance(greeting_preroll_keyword, str):
        raise ValueError("Greeting pre-roll keyword must be a string or None.")
    if not math.isfinite(greeting_preroll_seconds) or greeting_preroll_seconds < 0:
        raise ValueError("Greeting pre-roll duration must be finite and nonnegative.")
    if not isinstance(save_audio, bool) or not isinstance(save_transcript, bool):
        raise ValueError("Audio and transcript save settings must be booleans.")
    output_dir.mkdir(parents=True, exist_ok=False)
    if save_transcript:
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
                pre_roll = (greeting_preroll_seconds
                            if greeting_preroll_keyword
                            and greeting_preroll_keyword in segment.text else 0.0)
                start = max(0, math.floor((segment.start - pre_roll) * rate))
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
            if save_audio:
                with open_audio_target(output_dir / f"{name}.wav", source) as target:
                    copy_audio_frames(source, target, start, end - start)
            entry = {
                "audio_file": f"{name}.wav", "text": segment.text,
                "model_segment_index": index,
                "speaker": segment.speaker,
                "model_start_seconds": segment.start, "model_end_seconds": segment.end,
                "model_start_utc": (capture_start + timedelta(seconds=segment.start)).isoformat(),
                "model_end_utc": (capture_start + timedelta(seconds=segment.end)).isoformat(),
                "start_seconds": start / rate, "end_seconds": end / rate,
                "pre_roll_seconds": max(0.0, segment.start - start / rate),
                "recording_start_seconds": capture["offset_seconds"] + start / rate,
                "recording_end_seconds": capture["offset_seconds"] + end / rate,
                "start_utc": (capture_start + timedelta(seconds=start / rate)).isoformat(),
                "end_utc": (capture_start + timedelta(seconds=end / rate)).isoformat(),
                "timestamp_source": capture.get("timestamp_source", "computer"),
                "sample_rate": rate, "channels": source.channels, "subtype": source.subtype,
                "frames": end - start,
            }
            if save_transcript:
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
    if vad is not None:
        payload["vad"] = vad
    if save_transcript:
        (output_dir / "transcript.txt").write_text(
            "\n".join(f"[{entry['model_start_seconds']:.2f}-{entry['model_end_seconds']:.2f}] "
                      f"{entry['speaker']}: {entry['text']}" for entry in entries),
            encoding="utf-8")
        (output_dir / "transcript.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if return_manifest:
        return len(entries), payload
    return len(entries)
