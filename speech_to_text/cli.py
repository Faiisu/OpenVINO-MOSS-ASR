"""File selection, command-line configuration, and workflow orchestration."""
import argparse
import sys
import tempfile
from pathlib import Path

from .backend import transcribe
from .outputs import export_results, resolve_ffmpeg


def select_audio():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(
                title="Select an audio file",
                filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.opus *.wma"),
                           ("All files", "*.*")],
            )
        finally:
            root.destroy()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("File picker unavailable. Pass an audio path on the command line.") from error
    except Exception as error:
        # TclError can occur on headless systems before a Tk root exists.
        raise RuntimeError("Cannot open file picker. Use: python transcribe.py /path/to/audio.wav") from error
    return Path(selected) if selected else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe audio and export speaker WAV files using MOSS.")
    parser.add_argument("audio", nargs="?", type=Path, help="Omit to open a desktop file picker.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--languages", nargs="+", default=None, help="Language hints, for example: Thai English")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "xpu", "xpu:0"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    try:
        audio = args.audio if args.audio is not None else select_audio()
        if audio is None:
            print("Selection cancelled.")
            return 0
        audio = audio.expanduser().resolve()
        if not audio.is_file():
            raise ValueError(f"Audio file does not exist: {audio}")
        resolve_ffmpeg()
        print("Loading MOSS and transcribing. The first run downloads model weights.", flush=True)
        raw, segments = transcribe(audio, args.model, args.device, args.max_new_tokens, languages=args.languages)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="transcription-", dir=args.output_dir.resolve()))
        export_results(audio, output, raw, segments)
        print(f"Exported {len(segments)} segments to {output}")
        print("Speaker WAVs contain concatenated speech intervals; overlapping voices remain audible.")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
