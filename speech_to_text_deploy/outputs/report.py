"""Render saved gathering transcripts, timestamped audio and timing in notebooks."""

import base64
import csv
from html import escape
import json
from pathlib import Path

from ..config import COMPLETED_STATUSES


def save_gathering_summary(run_dir):
    """Save a portable HTML report with embedded, unchanged utterance WAVs."""
    root = Path(run_dir).resolve()
    html = gathering_summary_html(root)
    target = root / "reports" / "summary.html"
    target.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Gathering summary</title><style>body{font:16px system-ui;margin:24px}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:8px;vertical-align:top}"
        "pre{white-space:pre-wrap}</style></head><body>" + html + "</body></html>",
        encoding="utf-8",
    )
    return target


def _number(value):
    return f"{float(value):.3f}" if value not in (None, "") else "N/A"


def _table(headers, rows):
    """Render pre-escaped cells; escape column labels here."""
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                   for row in rows)
    return f"<div style='overflow-x:auto'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def gathering_summary_html(run_dir):
    """Summarize one saved run without loading a model or changing audio samples.

    Embed the exported WAV bytes for playback over Remote SSH. RTF belongs to
    each input window; its aggregate includes completed windows with rejections.
    Missing/failed outputs remain visible instead of disappearing from the report.
    """
    root = Path(run_dir).resolve()
    with (root / "reports" / "segments.csv").open(newline="", encoding="utf-8") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["segment_index"]))
    runtime_path = root / "reports" / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    completed = [row for row in rows if row["status"] in COMPLETED_STATUSES]
    audio_seconds = sum(float(row["audio_seconds"]) for row in completed)
    latency_seconds = sum(float(row["latency_seconds"]) for row in completed)
    weighted_rtf = latency_seconds / audio_seconds if audio_seconds else None
    timing_rows, transcript_rows, notes = [], [], []
    rejected_count = 0
    for row in rows:
        index = escape(row["segment_index"])
        rtf = _number(row["rtf"]) if row["status"] in COMPLETED_STATUSES else "N/A"
        timing_rows.append([
            index, escape(row["status"]), _number(row["audio_seconds"]),
            _number(row["queue_wait_seconds"]), _number(row.get("vad_speech_seconds")),
            _number(row["inference_seconds"]),
            _number(row["trim_and_store_seconds"]), rtf, escape(row.get("rejected_count") or "0"),
        ])
        # Resolve against this run, so a copied run remains viewable even when
        # its CSV still contains absolute paths from the original location.
        folder = root / "results" / f"segment-{int(row['segment_index']):06d}"
        manifest_path = folder / "transcript.json"
        if row.get("error"):
            notes.append(f"<p>Window {index}: {escape(row['error'])}</p>")
        if not manifest_path.is_file():
            notes.append(f"<p>Window {index}: no completed transcript manifest.</p>")
            raw_path = folder / "raw_transcript.txt"
            if raw_path.is_file():
                raw = escape(raw_path.read_text(encoding="utf-8"))
                notes.append(f"<details><summary>Window {index} raw model output</summary><pre>{raw}</pre></details>")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest["segments"] and not manifest.get("rejected_segments"):
            vad = manifest.get("vad", {})
            if vad and not vad.get("speech_regions"):
                notes.append(
                    f"<p>Window {index}: skipped by VAD; no speech met threshold "
                    f"{_number(vad.get('threshold'))} and minimum duration "
                    f"{escape(str(vad.get('min_speech_duration_ms')))} ms.</p>")
            else:
                notes.append(f"<p>Window {index}: no parsed utterances ({escape(manifest['status'])}).</p>")
        for entry in manifest["segments"]:
            audio_path = folder / entry["audio_file"]
            if audio_path.is_file():
                encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
                player = (f"<audio controls preload='none' aria-label='Window {index} utterance' "
                          f"src='data:audio/wav;base64,{encoded}'>Audio playback is unavailable.</audio>")
            else:
                player = f"Audio file missing: {escape(entry['audio_file'])}"
            transcript_rows.append([
                index, escape(entry["speaker"]),
                f"{_number(entry['recording_start_seconds'])}–{_number(entry['recording_end_seconds'])}",
                f"{_number(entry['start_seconds'])}–{_number(entry['end_seconds'])}",
                f"<span style='white-space:pre-wrap'>{escape(entry['text'])}</span>", player, rtf,
            ])
        for entry in manifest.get("rejected_segments", []):
            rejected_count += 1
            model_time = f"{escape(str(entry['model_start_seconds']))}–{escape(str(entry['model_end_seconds']))}"
            transcript_rows.append([
                index, escape(entry["speaker"]), "N/A", f"{model_time} (rejected model time)",
                f"<span style='white-space:pre-wrap'>{escape(entry['text'])}</span>",
                f"Rejected: {escape(entry['reason'])} No audio clip was created.", rtf,
            ])
    for error in runtime.get("errors", []):
        notes.append(f"<p>{escape(error['worker'])}: {escape(error['error'])}</p>")
    overview = (
        f"<h3>Run summary</h3><p>{escape(str(root))}</p>"
        f"<p>Status: {escape(runtime.get('status', 'incomplete'))} | "
        f"Completed windows: {len(completed)}/{len(rows)} | Utterances: {len(transcript_rows)} | "
        f"VAD-skipped windows: {sum(row['status'] == 'no_speech' for row in completed)} | "
        f"Rejected utterances: {rejected_count} | "
        f"Processed audio: {_number(audio_seconds)} s | Duration-weighted RTF: {_number(weighted_rtf)}</p>"
        "<p>RTF = window-ready-to-output latency / window audio duration. It includes "
        "preparation, queue wait, inference and local storage. Duration-weighted RTF = "
        "sum(latency) / sum(audio duration) for completed windows (success, partial or no speech); "
        "it is not wall-clock throughput. Partial means some model intervals were rejected. "
        "RTF is measured per window, not per utterance. Speaker IDs are independent per window.</p>"
    )
    timing = _table(["Window", "Status", "Audio (s)", "Queue wait (s)", "VAD speech (s)", "Inference (s)",
                     "Trim/store (s)", "RTF", "Rejected"], timing_rows)
    transcripts = (_table(["Window", "Speaker", "Recording time (s)", "Clip time in window (s)",
                           "Transcript", "Timestamped audio", "Window RTF"], transcript_rows)
                   if transcript_rows else "<p>No completed utterance clips are available.</p>")
    return (overview + "<h3>RTF by window</h3>" + timing + "".join(notes)
            + "<h3>Transcript and timestamped audio</h3>"
            + "<p>Each player contains the saved clip at the effective timestamps shown; "
              "its playback counter starts at zero. Original model timestamps remain in the clip JSON.</p>"
            + transcripts)
