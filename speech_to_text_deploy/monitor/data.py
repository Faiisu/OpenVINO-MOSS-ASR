"""Incrementally index flushed CSV rows without loading models or entire runs."""

from collections import OrderedDict, deque
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote

from ..config import COMPLETED_STATUSES


def read_json(path):
    """Retry files still being written on the next poll, including final manifests."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


class RunIndex:
    """Keep bounded recent rows/manifests plus lifetime counters for a saved run."""

    def __init__(self, root):
        self.root = root
        self.config = {}
        self.captured = 0
        self.manifests = OrderedDict()
        self.reset_csv()

    def reset_csv(self):
        self.identity = None
        self.offset = 0
        self.header = None
        self.rows = deque(maxlen=200)
        self.minimum_index = None
        self.totals = {"reported_windows": 0, "processed_windows": 0, "failed_windows": 0,
                       "partial_windows": 0, "clip_count": 0, "rejected_count": 0,
                       "audio_seconds": 0.0, "latency_seconds": 0.0}

    def update(self):
        self.config = read_json(self.root / "config.json") or self.config
        path = self.root / "reports" / "segments.csv"
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.reset_csv()
                self.identity = identity
            with path.open("rb") as file:
                file.seek(self.offset)
                record = b""
                while line := file.readline():
                    record += line
                    if not line.endswith(b"\n"):
                        break
                    try:
                        parsed = list(csv.reader(io.StringIO(record.decode("utf-8")), strict=True))
                    except (UnicodeDecodeError, csv.Error):
                        continue
                    if len(parsed) != 1:
                        break
                    values = parsed[0]
                    if self.header is None:
                        self.header = values
                    elif len(values) != len(self.header):
                        break
                    else:
                        row = dict(zip(self.header, values))
                        self.add_row(row)
                    self.offset = file.tell()
                    record = b""
        except FileNotFoundError:
            pass
        # Metadata is written only after a recording window is closed. Probe
        # sequential names instead of repeatedly scanning days of recordings.
        while (self.root / "recordings" / f"segment-{self.captured + 1:06d}.json").is_file():
            self.captured += 1

    def add_row(self, row):
        self.rows.append(row)
        index = number(row.get("segment_index"))
        if index is not None:
            self.minimum_index = index if self.minimum_index is None else min(self.minimum_index, index)
        totals = self.totals
        totals["reported_windows"] += 1
        totals["failed_windows"] += row.get("status") == "failed"
        totals["partial_windows"] += row.get("status") == "partial"
        totals["clip_count"] += int(number(row.get("clip_count")) or 0)
        totals["rejected_count"] += int(number(row.get("rejected_count")) or 0)
        if row.get("status") in COMPLETED_STATUSES:
            totals["processed_windows"] += 1
            totals["audio_seconds"] += number(row.get("audio_seconds")) or 0
            totals["latency_seconds"] += number(row.get("latency_seconds")) or 0

    def manifest(self, folder):
        path = folder / "transcript.json"
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self.manifests:
            result = read_json(path)
            if result is None:
                return None
            self.manifests[key] = result
            if len(self.manifests) > 200:
                self.manifests.popitem(last=False)
        return self.manifests[key]

    def window(self, row):
        index = int(row["segment_index"])
        folder = self.root / "results" / f"segment-{index:06d}"
        manifest = self.manifest(folder) or {}
        labels = read_json(folder / "labels.json") or {}
        if not isinstance(labels, dict):
            labels = {}
        entries = []
        for entry in manifest.get("segments", []):
            item = dict(entry)
            label = labels.get(entry.get("audio_file"), {})
            item["label_text"] = label.get("text", item.get("text", ""))
            item["label_speaker"] = label.get("speaker", item.get("speaker", "S01"))
            item["selected"] = label.get("selected", False)
            item["label_updated_at"] = label.get("updated_at")
            audio = folder / entry["audio_file"]
            item["audio_url"] = (f"/api/runs/{quote(self.root.name)}/audio/{folder.name}/"
                                 f"{quote(audio.name)}" if audio.is_file() else None)
            entries.append(item)
        raw_path = folder / "raw_transcript.txt"
        try:
            with raw_path.open(encoding="utf-8") as file:
                raw = file.read(64000)
        except (OSError, UnicodeError):
            raw = ""
        numeric = ("offset_seconds", "audio_seconds", "queue_wait_seconds", "inference_seconds",
                   "vad_speech_seconds", "trim_and_store_seconds", "latency_seconds", "rtf",
                   "clip_count", "rejected_count")
        return {"index": index, "status": row["status"], "error": row.get("error", ""),
                **{key: number(row.get(key)) for key in numeric}, "utterances": entries,
                "manifest_status": manifest.get("status"), "raw_transcript": raw,
                "rejected": manifest.get("rejected_segments", [])}

    def snapshot(self, limit=15, before=None):
        self.update()
        runtime = read_json(self.root / "reports" / "runtime.json")
        active = self.root / "recordings" / f"segment-{self.captured + 1:06d}.wav"
        candidates = [self.root / "config.json", self.root / "reports" / "segments.csv", active]
        if self.captured:
            candidates.append(self.root / "recordings" / f"segment-{self.captured:06d}.json")
        latest = max((path.stat().st_mtime for path in candidates if path.is_file()), default=0)
        recent = time.time() - latest < max(90, (self.config.get("segment_seconds") or 30) * 2)
        if runtime:
            status = runtime.get("status", "finished")
        elif (self.root / "STOP").exists():
            status = "draining" if recent else "inactive"
        else:
            status = "collecting" if recent and active.is_file() else ("starting" if recent else "inactive")
        rows = list(self.rows)
        if before is not None:
            # Historical paging is explicit, not part of the 1-second live poll.
            rows = self.history(before, limit)
        else:
            rows = rows[-limit:]
        stats = dict(self.totals)
        stats.update(captured_windows=self.captured,
                     pending_windows=max(0, self.captured - stats["reported_windows"]),
                     weighted_rtf=(stats["latency_seconds"] / stats["audio_seconds"]
                                   if stats["audio_seconds"] else None))
        source = self.config.get("source_audio")
        oldest = number(rows[0].get("segment_index")) if rows else None
        return {"id": self.root.name, "status": status, "stats": stats,
                "source": Path(source).name if source else "BY-V microphone",
                "device": self.config.get("ov_device", "GPU"),
                "window_seconds": self.config.get("segment_seconds", 30),
                "current_window": self.captured + 1 if active.is_file() and not runtime else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_activity_at": datetime.fromtimestamp(latest, timezone.utc).isoformat(),
                "errors": runtime.get("errors", []) if runtime else [],
                "windows": [self.window(row) for row in reversed(rows)],
                "has_older": bool(oldest is not None and self.minimum_index is not None
                                   and self.minimum_index < oldest)}

    def history(self, before, limit):
        rows = deque(maxlen=limit)
        path = self.root / "reports" / "segments.csv"
        if path.is_file():
            with path.open("rb") as file:
                header = None
                record = b""
                while line := file.readline():
                    record += line
                    if not line.endswith(b"\n"):
                        break
                    try:
                        parsed = list(csv.reader(io.StringIO(record.decode("utf-8")), strict=True))
                    except (UnicodeDecodeError, csv.Error):
                        continue
                    if len(parsed) != 1:
                        break
                    values = parsed[0]
                    if header is None:
                        header = values
                    elif len(values) != len(header):
                        break
                    else:
                        row = dict(zip(header, values))
                        index = number(row.get("segment_index"))
                        if index is not None and index.is_integer() and index < before and row.get("status"):
                            rows.append(row)
                    record = b""
        return list(rows)

class MonitorStore:
    """Serialize index updates shared by all read-only HTTP clients."""

    def __init__(self, output_base):
        self.root = Path(output_base).resolve()
        self.lock = threading.Lock()
        self.indices = OrderedDict()

    def run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise FileNotFoundError("Invalid run ID")
        path = (self.root / run_id).resolve()
        if path.parent != self.root or not (path / "config.json").is_file():
            raise FileNotFoundError("Run not found")
        return path

    def runs(self):
        if not self.root.is_dir():
            return []
        runs = []
        for path in self.root.iterdir():
            try:
                path = self.run_path(path.name)
                created = (path / "config.json").stat().st_mtime
                runtime = read_json(path / "reports" / "runtime.json")
                runs.append({"id": path.name, "created_at": created,
                             "status": runtime.get("status") if runtime else "open"})
            except (FileNotFoundError, OSError):
                continue
        return sorted(runs, key=lambda row: row["created_at"], reverse=True)

    def snapshot(self, run_id, limit=15, before=None):
        path = self.run_path(run_id)
        with self.lock:
            if run_id not in self.indices:
                self.indices[run_id] = RunIndex(path)
            self.indices.move_to_end(run_id)
            if len(self.indices) > 32:
                self.indices.popitem(last=False)
            return self.indices[run_id].snapshot(limit, before)

    def audio_path(self, run_id, segment, filename):
        root = self.run_path(run_id)
        if not re.fullmatch(r"segment-\d{6,}", segment) or not re.fullmatch(r"utterance-\d{4,}\.wav", filename):
            raise FileNotFoundError("Audio not found")
        path = (root / "results" / segment / filename).resolve()
        if not path.is_relative_to(root / "results") or not path.is_file():
            raise FileNotFoundError("Audio not found")
        return path
