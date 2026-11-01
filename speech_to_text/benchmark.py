"""Isolated CPU/GPU workflow benchmarks reusing the existing model adapters."""
import argparse
import csv
import json
import sys
import traceback
from .runtime import igpu_environment, run_model_worker
from pathlib import Path
from time import perf_counter


def run_benchmarks(audio_files, output_dir, *, cpu_threads=8, repeats=1,
                   moss_tokens=32768, languages=None, typhoon_language="thai", cases=None,
                   timeout_seconds=600, min_available_memory_bytes=1024**3):
    """Run independent processes, preserving each result and streaming progress."""
    if cpu_threads < 1 or repeats < 1 or moss_tokens < 1:
        raise ValueError("Thread count, repeats, and token limit must be positive.")
    if not audio_files:
        raise ValueError("No active audio files found.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if cases is None:
        cases = [("moss", "cpu"), ("moss", "cuda:0"), ("typhoon", "cpu"), ("typhoon", "cuda:1")]
    if not cases or any(model not in {"moss", "typhoon"} for model, _ in cases):
        raise ValueError("Provide at least one case using a supported model.")
    for index, source in enumerate(audio_files):
        for model, device in cases:
            for repeat in range(repeats):
                case_dir = output_dir / f"{index:03d}-{model}-{device.replace(':', '-')}-{repeat + 1}"
                case_dir.mkdir(exist_ok=False)
                request = {
                    "source": str(Path(source).resolve()), "model": model, "device": device,
                    "repeat": repeat + 1, "cpu_threads": cpu_threads,
                    "moss_tokens": moss_tokens, "languages": languages,
                    "typhoon_language": typhoon_language,
                }
                request_path = case_dir / "request.json"
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                environment = {**igpu_environment(), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                               "OMP_NUM_THREADS": str(cpu_threads), "MKL_NUM_THREADS": str(cpu_threads)}
                if device == "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = ""
                print(f"Running {source.name}: {model} on {device}, repeat {repeat + 1}", flush=True)
                # Never start another trial until this worker has exited and its
                # address space (including shared iGPU RAM) has been released.
                worker_error = None
                try:
                    returncode = run_model_worker(
                        [sys.executable, "-u", "-m", "speech_to_text.benchmark", str(request_path)],
                        env=environment, timeout_seconds=timeout_seconds,
                        min_available_memory_bytes=min_available_memory_bytes,
                    )
                except (TimeoutError, MemoryError) as error:
                    returncode = -1
                    worker_error = str(error)
                result_path = case_dir / "result.json"
                row = json.loads(result_path.read_text()) if result_path.exists() else {
                    **request, "status": "failed", "error": f"Worker exited {returncode} without a result",
                }
                if returncode != 0:
                    row["status"] = "failed"
                if worker_error:
                    row["error"] = worker_error
                rows.append(row)
                (output_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                fields = ["source", "model", "device", "repeat", "status", "elapsed_seconds", "audio_seconds", "rtf", "dtype", "cpu_threads", "peak_allocated_bytes", "allocated_bytes_after", "reserved_bytes_after", "error"]
                with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                if worker_error:
                    raise RuntimeError(f"Benchmark stopped: {worker_error} Results saved in {output_dir}")
    return rows


def run_worker(request_path):
    """Measure adapter latency after runtime initialization; exclude output saving."""
    import platform
    import torch
    from moss_transcribe_diarize.inference_utils import load_audio_av
    from .backend import transcribe

    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    result = {**request, "python": sys.version, "torch": torch.__version__, "cpu": platform.processor(),
              "timing_scope": "adapter call: imports, model loading, audio preprocessing, transcription, adapter cleanup; excludes process startup, runtime initialization, duration probing, and result saving"}
    xpu_ready = False
    try:
        torch.set_num_threads(request["cpu_threads"])
        device = torch.device(request["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
                raise RuntimeError(f"Requested GPU unavailable: {device}")
            result["gpu"] = torch.cuda.get_device_name(device)
            torch.cuda.synchronize(device)
        elif device.type == "xpu":
            if not torch.xpu.is_available() or (device.index or 0) >= torch.xpu.device_count():
                raise RuntimeError(f"Requested Intel GPU unavailable: {device}")
            xpu_ready = True
            result["gpu"] = torch.xpu.get_device_name(device)
            torch.xpu.synchronize(device)
            torch.xpu.reset_peak_memory_stats(device)
            result["baseline_allocated_bytes"] = torch.xpu.memory_allocated(device)
        duration = len(load_audio_av(request["source"], sampling_rate=16000)) / 16000
        started = perf_counter()
        if request["model"] == "moss":
            raw, segments = transcribe(request["source"], "OpenMOSS-Team/MOSS-Transcribe-Diarize",
                                       str(device), request["moss_tokens"], languages=request["languages"])
            text = " ".join(s.text for s in segments)
            result["segment_count"] = len(segments)
            result["dtype"] = "bfloat16" if device.type in {"cuda", "xpu"} else "float32"
            result["model_id"] = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        else:
            from .typhoon import transcribe_typhoon
            prediction = transcribe_typhoon(request["source"], "typhoon-ai/typhoon-whisper-large-v3",
                                            str(device), language=request["typhoon_language"], cpu_offload=False)
            raw = text = prediction["text"]
            result["inference_config"] = {key: prediction[key] for key in ("chunk_length_s", "stride_length_s", "batch_size", "return_timestamps", "max_length", "max_new_tokens", "language", "num_beams")}
            result["dtype"] = prediction["dtype"]
            result["model_id"] = "typhoon-ai/typhoon-whisper-large-v3"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu":
            torch.xpu.synchronize(device)
        elapsed = perf_counter() - started
        if not text.strip():
            raise ValueError("Model returned an empty transcript.")
        result.update(status="success", elapsed_seconds=elapsed, audio_seconds=duration,
                      rtf=elapsed / duration if duration > 0 else None, text=text)
        (request_path.parent / "transcript.txt").write_text(text, encoding="utf-8")
        (request_path.parent / "raw_transcript.txt").write_text(raw, encoding="utf-8")
        print(f"PASS {request['model']} {device}: {elapsed:.2f}s; RTF {result['rtf']:.3f}", flush=True)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if xpu_ready:
            from .runtime import release_model_memory

            release_model_memory(device)
            result.update(
                peak_allocated_bytes=torch.xpu.max_memory_allocated(device),
                allocated_bytes_after=torch.xpu.memory_allocated(device),
                reserved_bytes_after=torch.xpu.memory_reserved(device),
            )
            if result["allocated_bytes_after"] != result["baseline_allocated_bytes"]:
                result.update(status="failed", error="XPU tensor allocations did not return to baseline.")
    (request_path.parent / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one isolated benchmark worker.")
    parser.add_argument("request_path", type=Path)
    raise SystemExit(run_worker(parser.parse_args().request_path))
