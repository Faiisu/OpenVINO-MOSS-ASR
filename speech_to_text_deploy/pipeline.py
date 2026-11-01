"""Two-worker queue and isolated-process orchestration."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import tempfile
import threading
from time import perf_counter
from .audio.capture import capture_segments
from .config import ROOT, REPORT_COLUMNS, COMPLETED_STATUSES
from .outputs.clips import export_timed_clips
from .runtime import openvino_environment, release_model_memory, run_model_worker, tensor_memory


def run_gathering(config, *, stop_event=None):
    """Run two logical workers, drain captured files on stop, and always unload.

    The queue holds paths, not audio arrays. Raw windows remain on disk even if
    processing fails. A STOP file in the run folder requests stop and drain.
    RTF starts when each window is complete, excluding its acquisition time and
    initial model compilation; it includes spool finalization, queue wait,
    inference, trimming and local result storage.
    """
    import torch
    from .model.session import OpenVinoSession

    config.validate()
    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    recordings, results, reports = (root / name for name in ("recordings", "results", "reports"))
    for folder in (recordings, results, reports):
        folder.mkdir()
    (root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    stop = stop_event if stop_event is not None else threading.Event()
    ready, producer_done, consumer_done = (threading.Event() for _ in range(3))
    pending = Queue()
    errors = []
    counts = {"segment_count": 0, "success_count": 0, "partial_count": 0,
              "processed_count": 0, "rejected_segment_count": 0}
    runtime = {"backend": "openvino", "model_id": config.model_id,
               "rtf_definition": "(output_ready - window_complete) / actual_window_audio_seconds",
               "model_lifecycle": "Loaded once before capture; unloaded after queue drain or failure."}
    error_lock = threading.Lock()

    def fail(worker, error):
        # Store text only; traceback objects can retain model references.
        with error_lock:
            errors.append({"worker": worker, "error": f"{type(error).__name__}: {error}"})
        stop.set()

    def produce():
        try:
            while not ready.wait(0.1):
                if consumer_done.is_set() or stop.is_set():
                    return
            for item in capture_segments(
                    recordings, stop, segment_seconds=config.segment_seconds,
                    samplerate=config.samplerate, input_device=config.input_device,
                    input_latency_seconds=config.input_latency_seconds,
                    max_audio_seconds=config.max_audio_seconds, source_audio=config.source_audio,
                    replay_realtime=config.replay_realtime, min_free_bytes=config.min_free_bytes):
                item.queued_at = perf_counter()
                item.path.with_suffix(".json").write_text(
                    json.dumps(item.metadata(), indent=2), encoding="utf-8")
                pending.put(item)
                print(f"Captured segment {item.index}: {item.duration_seconds:.3f}s; queued={pending.qsize()}", flush=True)
        except BaseException as error:
            fail("capture", error)
        finally:
            producer_done.set()

    def base_row(item):
        return {"segment_index": item.index, "offset_seconds": item.offset_seconds,
                "audio_seconds": item.duration_seconds,
                "capture_start_utc": item.capture_start_utc, "capture_end_utc": item.capture_end_utc,
                "ready_monotonic_seconds": item.ready_at, "queued_monotonic_seconds": item.queued_at,
                "prepare_seconds": item.queued_at - item.ready_at,
                "source_audio": str(item.path), "output_dir": str(results / item.path.stem)}

    with (reports / "segments.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        csv_file.flush()

        def report(row):
            writer.writerow(row)
            csv_file.flush()
            # The CSV retains every row; keep only counters in RAM for long runs.
            counts["segment_count"] += 1
            counts["success_count"] += row["status"] == "success"
            counts["partial_count"] += row["status"] == "partial"
            counts["processed_count"] += row["status"] in COMPLETED_STATUSES
            counts["rejected_segment_count"] += row.get("rejected_count", 0)

        def consume():
            session = None
            try:
                torch.set_num_threads(config.cpu_threads)
                with OpenVinoSession(config.model_id, config.model_dir, ov_device=config.ov_device,
                                     audio_window_seconds=config.segment_seconds) as session:
                    devices = (session.runtime_metrics["encoder_execution_devices"]
                               + session.runtime_metrics["decoder_execution_devices"])
                    if config.ov_device.startswith("GPU") and not all(str(d).startswith("GPU") for d in devices):
                        raise RuntimeError(f"Unexpected execution devices: {devices}")
                    runtime["torch_allocated_before_inference"], _ = tensor_memory(session.device)
                    ready.set()
                    while not producer_done.is_set() or not pending.empty():
                        try:
                            item = pending.get(timeout=0.1)
                        except Empty:
                            continue
                        row = base_row(item)
                        started = perf_counter()
                        row.update(processing_start_monotonic_seconds=started,
                                   queue_wait_seconds=started - item.queued_at)
                        try:
                            raw, segments = session.transcribe(item.path, config.max_new_tokens, config.languages)
                            inferred = perf_counter()
                            clip_count = export_timed_clips(
                                item.path, Path(row["output_dir"]), raw, segments, item.metadata(),
                                invalid_segment_policy="retain")
                            manifest = json.loads((Path(row["output_dir"]) / "transcript.json").read_text())
                            rejected_count = len(manifest["rejected_segments"])
                            completed = perf_counter()
                            allocated, reserved = tensor_memory(session.device)
                            if allocated != runtime["torch_allocated_before_inference"]:
                                raise RuntimeError("Preprocessing tensor allocations did not return to baseline.")
                            runtime["torch_allocated_after_inference"] = allocated
                            runtime["torch_reserved_after_inference"] = reserved
                            row.update(status="partial" if rejected_count else "success",
                                       output_ready_monotonic_seconds=completed,
                                       inference_seconds=inferred - started,
                                       trim_and_store_seconds=completed - inferred,
                                       latency_seconds=completed - item.ready_at,
                                       rtf=(completed - item.ready_at) / item.duration_seconds,
                                       clip_count=clip_count, rejected_count=rejected_count)
                            report(row)
                            print(f"Saved segment {item.index}: {clip_count} clips; "
                                  f"rejected={rejected_count}; RTF={row['rtf']:.3f}", flush=True)
                        except BaseException as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                            report(row)
                            raise
                        finally:
                            pending.task_done()
            except BaseException as error:
                fail("inference", error)
            finally:
                try:
                    if session is not None:
                        runtime.update(session.runtime_metrics)
                    session = None
                    release_model_memory(torch.device("cpu"))
                    after = runtime.get("ov_memory_after_unload")
                    runtime["openvino_gpu_memory_released"] = (
                        all(value == 0 for value in after.values()) if after is not None else None)
                    if runtime["openvino_gpu_memory_released"] is False:
                        fail("cleanup", RuntimeError("OpenVINO still reports device allocations after unloading."))
                except BaseException as error:
                    fail("cleanup", error)
                finally:
                    consumer_done.set()

        consumer = threading.Thread(target=consume, name="Thread2-ASR")
        producer = threading.Thread(target=produce, name="Thread1-capture")
        started = perf_counter()
        try:
            consumer.start()
            producer.start()
            while producer.is_alive() or consumer.is_alive():
                if (root / "STOP").exists():
                    stop.set()
                producer.join(timeout=0.1)
                consumer.join(timeout=0.1)
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
            if producer.ident is not None:
                producer.join()
            if consumer.ident is not None:
                consumer.join()
            while not pending.empty():
                item = pending.get_nowait()
                row = base_row(item)
                row.update(status="not_processed", error="Consumer stopped; original audio is retained.")
                report(row)
                pending.task_done()
            runtime.update(status="failed" if errors else (
                               "success_with_warnings" if counts["partial_count"] else "success"),
                           errors=errors, **counts,
                           workflow_seconds=perf_counter() - started)
            (reports / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"Collection failed; inspect {reports / 'runtime.json'} and retained recordings: {errors}")
    return runtime


def gather_isolated(config, *, timeout_seconds=3600, drain_timeout_seconds=600):
    """Configure the OpenCL environment before import and release the child on exit.

    Interrupt requests graceful capture stop and queue drain, with a bounded
    shutdown deadline. The process boundary also releases driver context memory.
    """
    config.validate()
    with tempfile.TemporaryDirectory(prefix="moss-gathering-") as folder:
        request = Path(folder) / "request.json"
        request.write_text(json.dumps(asdict(config)), encoding="utf-8")
        try:
            code = run_model_worker(
                [sys.executable, "-m", "speech_to_text_deploy.worker", str(request)],
                env=openvino_environment(), cwd=ROOT, timeout_seconds=timeout_seconds,
                interrupt_grace_seconds=drain_timeout_seconds)
        except KeyboardInterrupt:
            report_path = Path(config.output_dir) / "reports" / "runtime.json"
            if not report_path.is_file():
                raise
            code = 0
        report_path = Path(config.output_dir) / "reports" / "runtime.json"
        runtime = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if code or runtime.get("status") not in ("success", "success_with_warnings"):
            raise RuntimeError(f"Gathering worker failed (exit {code}); inspect {report_path}.")
        return runtime
