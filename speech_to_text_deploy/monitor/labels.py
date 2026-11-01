"""Persist reviewed utterance labels and export a portable MOSS JSONL dataset."""

import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

from ..config import DEPLOY_ROOT
from .data import read_json


RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SEGMENT_ID = re.compile(r"segment-\d{6,}")
UTTERANCE_ID = re.compile(r"utterance-\d{4,}\.json")
SPEAKER_ID = re.compile(r"\[?S\d+\]?")
EXPORTED_AUDIO_ID = re.compile(r"(?P<run>.+)__(?P<segment>segment-\d{6,})__(?P<utterance>utterance-\d{4,})\.wav")


def dataset_identity(record):
    """Return a stable utterance key for current and legacy dataset rows."""
    source = record.get("source")
    if isinstance(source, dict):
        values = (source.get("run"), source.get("segment"), source.get("utterance"))
        if all(values):
            return tuple(str(value) for value in values)
    values = (record.get("source_run"), record.get("source_segment"),
              record.get("source_utterance"))
    if all(values):
        return tuple(str(value) for value in values)

    conversation = record.get("conversation") or []
    audio = (conversation[1].get("content") if len(conversation) > 1
             and isinstance(conversation[1], dict) else record.get("audio"))
    if isinstance(audio, str):
        match = EXPORTED_AUDIO_ID.fullmatch(Path(audio).name)
        if match:
            return match.group("run"), match.group("segment"), match.group("utterance")
        return ("audio", Path(audio).name)
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return ("record", hashlib.sha256(serialized.encode("utf-8")).hexdigest())


def read_existing_dataset(path, dataset_root):
    """Read existing identities without changing existing JSONL bytes."""
    if not path.is_file():
        return set(), set(), 0, True
    identities, audio_hashes, count = set(), set(), 0
    dataset_root = dataset_root.resolve()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            identities.add(dataset_identity(record))
            conversation = record.get("conversation") or []
            audio = (conversation[1].get("content") if len(conversation) > 1
                     and isinstance(conversation[1], dict) else record.get("audio"))
            if isinstance(audio, str):
                audio_path = Path(audio)
                audio_path = audio_path if audio_path.is_absolute() else dataset_root / audio_path
                audio_path = audio_path.resolve()
                if audio_path.is_relative_to(dataset_root) and audio_path.is_file():
                    audio_hashes.add(file_sha256(audio_path))
            count += 1
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(-1, os.SEEK_END) if size else None
        has_trailing_newline = not size or stream.read(1) == b"\n"
    return identities, audio_hashes, count, has_trailing_newline


def file_sha256(path):
    """Hash audio with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_audio_without_overwrite(source, audio_root, target_name, source_digest=None):
    """Reuse identical saved WAVs and choose a digest name on a collision."""
    target = audio_root / target_name
    digest = source_digest or file_sha256(source)
    if target.is_symlink():
        raise OSError(f"Dataset audio path cannot be a symbolic link: {target}")
    if target.is_file():
        if file_sha256(target) == digest:
            return target.name
        target = audio_root / f"{Path(target_name).stem}__{digest[:10]}.wav"
        if target.is_symlink():
            raise OSError(f"Dataset audio path cannot be a symbolic link: {target}")
        if target.is_file():
            if file_sha256(target) == digest:
                return target.name
            raise FileExistsError(f"Dataset audio name collision: {target}")
    shutil.copyfile(source, target)
    return target.name


def write_json_atomic(path, value):
    """Replace one JSON file atomically so concurrent polls never read a partial edit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class LabelStore:
    """Store reviewed text beside a run and export selected audio/text pairs."""

    def __init__(self, output_root, dataset_root=None):
        self.output_root = Path(output_root).resolve()
        self.dataset_root = Path(dataset_root or DEPLOY_ROOT / "labelled_dataset").resolve()
        if not self.dataset_root.is_relative_to(DEPLOY_ROOT.resolve()):
            raise ValueError("Dataset export directory must stay within speech_to_text_deploy")

    def _result_dir(self, run_id, segment_id):
        if not RUN_ID.fullmatch(run_id) or not SEGMENT_ID.fullmatch(segment_id):
            raise FileNotFoundError("Invalid run or segment identifier")
        run = (self.output_root / run_id).resolve()
        if run.parent != self.output_root or not (run / "config.json").is_file():
            raise FileNotFoundError("Run not found")
        results = (run / "results").resolve()
        if not results.is_relative_to(run):
            raise FileNotFoundError("Segment not found")
        result = (results / segment_id).resolve()
        if not result.is_relative_to(results):
            raise FileNotFoundError("Segment not found")
        manifest = read_json(result / "transcript.json")
        if manifest is None:
            raise FileNotFoundError("Transcript not found")
        return run, result, manifest

    def save(self, run_id, segment_id, utterance_id, payload):
        """Validate and persist one selected/editable model utterance."""
        if not UTTERANCE_ID.fullmatch(utterance_id) or not isinstance(payload, dict):
            raise ValueError("Invalid utterance label")
        text = payload.get("text")
        speaker = payload.get("speaker")
        selected = payload.get("selected")
        if (not isinstance(text, str) or len(text) > 10000 or "\x00" in text
                or not isinstance(speaker, str) or len(speaker) > 32
                or not SPEAKER_ID.fullmatch(speaker) or not isinstance(selected, bool)):
            raise ValueError("Text, speaker, or selected value is invalid")
        if selected and not text.strip():
            raise ValueError("A selected dataset transcript cannot be empty")

        _, result, manifest = self._result_dir(run_id, segment_id)
        filename = utterance_id.removesuffix(".json") + ".wav"
        entry = next((item for item in manifest.get("segments", [])
                      if item.get("audio_file") == filename), None)
        if entry is None or not (result / filename).is_file():
            raise FileNotFoundError("Saved utterance audio not found")

        path = result / "labels.json"
        labels = read_json(path) or {}
        labels[filename] = {"text": text, "speaker": speaker.strip("[]"),
                            "selected": selected,
                            "updated_at": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(path, labels)
        return labels[filename]

    def export(self):
        """Append saved selections that are not already in the dataset."""
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        audio_root = self.dataset_root / "audio"
        if audio_root.is_symlink():
            raise OSError("Dataset audio directory cannot be a symbolic link")
        audio_root.mkdir(exist_ok=True)
        if audio_root.resolve().parent != self.dataset_root:
            raise OSError("Dataset audio directory resolves outside its destination")
        manifest_path = self.dataset_root / "dataset.jsonl"
        existing, existing_audio, existing_count, has_trailing_newline = read_existing_dataset(
            manifest_path, self.dataset_root)
        records = []
        skipped_existing = 0
        if self.output_root.is_dir():
            for run_path in sorted(self.output_root.iterdir()):
                if not RUN_ID.fullmatch(run_path.name):
                    continue
                run = run_path.resolve()
                if run.parent != self.output_root or not (run / "config.json").is_file():
                    continue
                results = (run / "results").resolve()
                if not results.is_relative_to(run) or not results.is_dir():
                    continue
                for result_path in sorted(results.iterdir()):
                    if not SEGMENT_ID.fullmatch(result_path.name):
                        continue
                    result = result_path.resolve()
                    if not result.is_relative_to(results.resolve()):
                        continue
                    manifest = read_json(result / "transcript.json")
                    labels = read_json(result / "labels.json") or {}
                    if not manifest or not isinstance(labels, dict):
                        continue
                    entries = {entry.get("audio_file"): entry
                               for entry in manifest.get("segments", [])}
                    for filename, label in labels.items():
                        if not isinstance(label, dict) or not label.get("selected"):
                            continue
                        entry = entries.get(filename)
                        if (entry is None or not isinstance(entry.get("text"), str)
                                or not isinstance(label.get("text"), str)
                                or not SPEAKER_ID.fullmatch(str(label.get("speaker", "")))):
                            continue
                        source = (result / filename).resolve()
                        if (not source.is_relative_to(result) or not source.is_file()
                                or not filename.endswith(".wav")):
                            continue
                        identity = (run.name, result.name, filename)
                        audio_digest = file_sha256(source)
                        if identity in existing or audio_digest in existing_audio:
                            skipped_existing += 1
                            continue
                        duration = float(entry["end_seconds"]) - float(entry["start_seconds"])
                        if not math.isfinite(duration) or duration <= 0:
                            continue
                        speaker = f"[{label['speaker'].strip('[]')}]"
                        target_name = (f"{run.name}__{result.name}__"
                                       f"{Path(filename).stem}.wav")
                        target_name = copy_audio_without_overwrite(
                            source, audio_root, target_name, audio_digest)
                        record = {
                            "conversation": [
                                {"role": "user", "message_type": "text",
                                 "content": "Transcribe the audio with timestamps and speaker labels."},
                                {"role": "user", "message_type": "audio",
                                 "content": f"audio/{target_name}"},
                                {"role": "assistant", "message_type": "text",
                                 "content": (f"[0.00]{speaker}{label['text'].strip()}"
                                             f"[{duration:.2f}]")},
                            ],
                            "source": {
                                "run": run.name,
                                "segment": result.name,
                                "utterance": filename,
                                "recording_start_seconds": entry.get("recording_start_seconds"),
                                "recording_end_seconds": entry.get("recording_end_seconds"),
                                "start_utc": entry.get("start_utc"),
                                "end_utc": entry.get("end_utc"),
                                "audio_sha256": audio_digest,
                            },
                        }
                        records.append(record)
                        existing.add(identity)
                        existing_audio.add(audio_digest)

        if records:
            with manifest_path.open("ab") as stream:
                if existing_count and not has_trailing_newline:
                    stream.write(b"\n")
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        return {"count": len(records), "added": len(records),
                "skipped_existing": skipped_existing,
                "existing_count": existing_count,
                "manifest": str(manifest_path),
                "audio_dir": str(audio_root)}
