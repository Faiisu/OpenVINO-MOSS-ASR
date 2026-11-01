"""Private supervised model worker entry point."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
from .config import ROOT, GatheringConfig
from .pipeline import run_gathering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON GatheringConfig")
    arguments = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Run with the project-local .venv using uv.")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    run_gathering(GatheringConfig(**json.loads(arguments.request.read_text())), stop_event=stop)


if __name__ == "__main__":
    main()
