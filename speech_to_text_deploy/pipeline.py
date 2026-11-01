"""Two-worker queue and isolated-process orchestration."""
import csv
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import tempfile
import threading
from time import perf_counter
from .audio.capture import capture_segments
from .audio.live_audio import LiveAudioForwarder
from .audio.vad import SileroVAD
from .config import ROOT, REPORT_COLUMNS, COMPLETED_STATUSES
from .outputs.clips import export_timed_clips
from .outputs.mqtt import MqttJsonPublisher, transcript_messages
from .runtime import openvino_environment, release_model_memory, run_model_worker, tensor_memory


def _write_progress(path, row):
    """Atomically publish the active section for runtime inspection."""
    payload = {"segment_index": row["segment_index"],
               "audio_seconds": row["audio_seconds"],
               "started_at_monotonic": row["ready_monotonic_seconds"],
               "report_row": row}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    progress_path = reports / "processing.json"
    stop = stop_event if stop_event is not None else threading.Event()
    ready, producer_done, consumer_done = (threading.Event() for _ in range(3))
    pending = Queue()
    errors = []
    counts = {"segment_count": 0, "success_count": 0, "partial_count": 0, "no_speech_count": 0,
              "processed_count": 0, "rejected_segment_count": 0}
    runtime = {"backend": "openvino", "model_id": config.model_id,
               "rtf_definition": "(output_ready - window_complete) / actual_window_audio_seconds",
               "mqtt": {"enabled": config.mqtt_enabled, "host": config.mqtt_host,
                        "port": config.mqtt_port, "topic": config.mqtt_topic,
                        "qos": config.mqtt_qos, "published_count": 0},
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
            with LiveAudioForwarder(enabled=config.live_audio_enabled,
                                    publish_url=config.live_audio_url) as audio_forwarder:
                for item in capture_segments(
                        recordings, stop, segment_seconds=config.segment_seconds,
                        samplerate=config.samplerate, input_device=config.input_device,
                        input_latency_seconds=config.input_latency_seconds,
                        internet_time_timeout_seconds=config.internet_time_timeout_seconds,
                        max_audio_seconds=config.max_audio_seconds, source_audio=config.source_audio,
                        replay_realtime=config.replay_realtime, min_free_bytes=config.min_free_bytes,
                        on_audio_chunk=audio_forwarder.submit):
                    item.queued_at = perf_counter()
                    if "timestamp_source" not in runtime:
                        runtime["timestamp_source"] = item.timestamp_source
                        print(f"Timestamp reference: {item.timestamp_source}", flush=True)
                    item.path.with_suffix(".json").write_text(
                        json.dumps(item.metadata(), indent=2), encoding="utf-8")
                    pending.put(item)
                    print(f"Captured segment {item.index}: {item.duration_seconds:.3f}s; "
                          f"queued={pending.qsize()}", flush=True)
        except BaseException as error:
            fail("capture", error)
        finally:
            producer_done.set()

    def base_row(item):
        return {"segment_index": item.index, "offset_seconds": item.offset_seconds,
                "audio_seconds": item.duration_seconds,
                "capture_start_utc": item.capture_start_utc, "capture_end_utc": item.capture_end_utc,
                "timestamp_source": item.timestamp_source,
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
            counts["no_speech_count"] += row["status"] == "no_speech"
            counts["processed_count"] += row["status"] in COMPLETED_STATUSES
            counts["rejected_segment_count"] += row.get("rejected_count", 0)

        def consume():
            publisher = vad = session = None
            try:
                torch.set_num_threads(config.cpu_threads)
                with ExitStack() as stack:
                    if config.mqtt_enabled:
                        publisher = stack.enter_context(MqttJsonPublisher(
                            host=config.mqtt_host, port=config.mqtt_port, topic=config.mqtt_topic,
                            qos=config.mqtt_qos, keepalive_seconds=config.mqtt_keepalive_seconds,
                            connect_timeout_seconds=config.mqtt_connect_timeout_seconds,
                            publish_timeout_seconds=config.mqtt_publish_timeout_seconds))
                    vad = stack.enter_context(SileroVAD(
                        threshold=config.vad_threshold,
                        min_speech_duration_ms=config.vad_min_speech_duration_ms,
                        min_silence_duration_ms=config.vad_min_silence_duration_ms,
                        speech_pad_ms=config.vad_speech_pad_ms))
                    session = stack.enter_context(OpenVinoSession(
                        config.model_id, config.model_dir, ov_device=config.ov_device,
                        audio_window_seconds=config.segment_seconds))
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
                            _write_progress(progress_path, row)
                            speech_regions = vad.detect(item.path)
                            speech_seconds = sum(end - start for start, end in speech_regions)
                            if speech_regions:
                                raw, segments = session.transcribe(
                                    item.path, config.max_new_tokens, config.languages,
                                    config.hotwords)
                            else:
                                raw, segments = "", []
                            inferred = perf_counter()
                            clip_count = export_timed_clips(
                                item.path, Path(row["output_dir"]), raw, segments, item.metadata(),
                                invalid_segment_policy="retain",
                                greeting_preroll_keyword=config.greeting_preroll_keyword,
                                greeting_preroll_seconds=config.greeting_preroll_seconds,
                                vad={"engine": "Silero VAD", "threshold": config.vad_threshold,
                                     "min_speech_duration_ms": config.vad_min_speech_duration_ms,
                                     "min_silence_duration_ms": config.vad_min_silence_duration_ms,
                                     "speech_pad_ms": config.vad_speech_pad_ms,
                                     "speech_regions": [{"start_seconds": start, "end_seconds": end}
                                                        for start, end in speech_regions],
                                     "speech_seconds": speech_seconds})
                            manifest = json.loads((Path(row["output_dir"]) / "transcript.json").read_text())
                            rejected_count = len(manifest["rejected_segments"])
                            completed = perf_counter()
                            allocated, reserved = tensor_memory(session.device)
                            if allocated != runtime["torch_allocated_before_inference"]:
                                raise RuntimeError("Preprocessing tensor allocations did not return to baseline.")
                            runtime["torch_allocated_after_inference"] = allocated
                            runtime["torch_reserved_after_inference"] = reserved
                            status = ("no_speech" if not speech_regions else
                                      "partial" if rejected_count else "success")
                            row.update(status=status,
                                       output_ready_monotonic_seconds=completed,
                                       inference_seconds=inferred - started,
                                       vad_speech_seconds=speech_seconds,
                                       trim_and_store_seconds=completed - inferred,
                                       latency_seconds=completed - item.ready_at,
                                       rtf=(completed - item.ready_at) / item.duration_seconds,
                                       clip_count=clip_count, rejected_count=rejected_count)
                            if publisher is not None:
                                for message in transcript_messages(manifest):
                                    message_id = publisher.publish_json(message)
                                    runtime["mqtt"]["published_count"] += 1
                                    runtime["mqtt"]["last_message_id"] = message_id
                            report(row)
                            if status == "no_speech":
                                print(f"Skipped segment {item.index}: VAD found no speech "
                                      f"(threshold={config.vad_threshold:.2f}, "
                                      f"minimum={config.vad_min_speech_duration_ms}ms)", flush=True)
                            else:
                                print(f"Saved segment {item.index}: {clip_count} clips; "
                                      f"rejected={rejected_count}; RTF={row['rtf']:.3f}", flush=True)
                        except BaseException as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                            report(row)
                            raise
                        finally:
                            progress_path.unlink(missing_ok=True)
                            pending.task_done()
            except BaseException as error:
                fail("inference", error)
            finally:
                try:
                    if session is not None:
                        runtime.update(session.runtime_metrics)
                    publisher = vad = session = None
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
