"""Run live/replay collection or render a saved run without a notebook."""

import argparse
from pathlib import Path
import signal

from .config import GatheringConfig, load_settings, new_run_dir
from .outputs.report import save_gathering_summary
from .pipeline import gather_isolated


def _request_stop(signum, frame):
    """Use the existing supervisor interrupt/drain path for service termination."""
    raise KeyboardInterrupt


def main(argv=None, *, continuous_default=False):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Capture BY-V audio or replay a file with MOSS.")
    run.add_argument("--config", type=Path, default=Path(__file__).with_name("settings.json"))
    run.add_argument("--source-audio", type=Path, help="Replay this file instead of recording live audio.")
    limit = run.add_mutually_exclusive_group()
    limit.add_argument("--duration", type=float, help="Override the maximum audio duration in seconds.")
    limit.add_argument("--continuous", action="store_true", help="Capture until stopped using the continuous timeout setting.")
    run.add_argument("--fast-replay", action="store_true", help="Replay without acquisition pacing.")
    run.add_argument("--output-dir", type=Path, help="Use this new run directory instead of an automatic name.")
    run.add_argument("--timeout", type=float, help="Override the configured worker deadline for this run.")
    run.add_argument("--drain-timeout", type=float, help="Override the configured shutdown drain timeout.")
    summary = commands.add_parser("summary", help="Save a transcript/audio/RTF HTML report for a run.")
    summary.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(save_gathering_summary(args.run_dir))
        return

    values = load_settings(args.config)
    continuous = args.continuous or (continuous_default and args.duration is None)
    bounded_timeout = values.pop("bounded_worker_timeout_seconds")
    continuous_timeout = values.pop("continuous_worker_timeout_seconds")
    drain_timeout = values.pop("drain_timeout_seconds")
    timeout_default = continuous_timeout if continuous else bounded_timeout
    timeout = args.timeout if args.timeout is not None else timeout_default
    drain_timeout = args.drain_timeout if args.drain_timeout is not None else drain_timeout
    output_base = Path(values.pop("output_base"))
    if args.source_audio is not None:
        values["source_audio"] = str(args.source_audio.resolve())
    if args.duration is not None:
        values["max_audio_seconds"] = args.duration
    elif continuous:
        values["max_audio_seconds"] = None
    if args.fast_replay:
        values["replay_realtime"] = False
    run_dir = args.output_dir.resolve() if args.output_dir is not None else new_run_dir(output_base)
    values["output_dir"] = str(run_dir)
    config = GatheringConfig(**values)
    print("Run:", run_dir, flush=True)
    print("Mode:", "continuous" if continuous else "bounded", "; worker timeout:", timeout, flush=True)
    print("Request stop: touch", run_dir / "STOP", flush=True)
    previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    try:
        runtime = gather_isolated(config, timeout_seconds=timeout,
                                  drain_timeout_seconds=drain_timeout)
        print("Status:", runtime["status"], flush=True)
        print("Processed windows:", runtime["processed_count"], flush=True)
        print("Rejected model intervals:", runtime["rejected_segment_count"], flush=True)
        print("OpenVINO memory after unload:", runtime.get("ov_memory_after_unload"), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if (run_dir / "reports" / "segments.csv").is_file():
            if continuous:
                print("Reports:", run_dir / "reports", flush=True)
                print("Create HTML summary: python -m speech_to_text_deploy summary", run_dir, flush=True)
            elif config.save_transcript:
                print("Summary:", save_gathering_summary(run_dir), flush=True)
            else:
                print("Summary skipped because save_transcript is disabled.", flush=True)


if __name__ == "__main__":
    main()
