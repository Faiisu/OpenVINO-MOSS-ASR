"""Continuously collect BY-V audio, timestamped transcripts and RTF until stopped.

Run with the project-local interpreter through uv. Settings come from the shared
settings.json; continuous mode overrides its capture duration and worker deadline.
Use --duration for a bounded run or --help to see the shared CLI options.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    # Also support direct execution by absolute path, from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech_to_text_deploy.__main__ import main as cli_main


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    cli_main(["run", *arguments], continuous_default=True)


if __name__ == "__main__":
    main()
