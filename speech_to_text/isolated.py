"""Run research inference in a disposable process with the Intel runtime configured."""
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .runtime import igpu_environment, run_model_worker, tensor_memory
from .profiling import StageProfiler, stage


def transcribe_isolated(audio_path, model_id, device, max_new_tokens, languages=None,
                        *, timeout_seconds=600, min_available_memory_bytes=1024**3,
                        chunk_seconds=None, cpu_threads=None, interop_threads=1, batch_size=2,
                        quantization="none", profile_output_dir=None, use_cache=None, trace_batch=None,
                        backend="pytorch", ov_model_dir=None, ov_device="GPU"):
    """Return CPU-only results after the model worker has exited and released RAM."""
    from .backend import Segment

    if backend not in {"pytorch", "openvino"}:
        raise ValueError("Backend must be 'pytorch' or 'openvino'.")
    if backend == "openvino" and (ov_model_dir is None or quantization != "none" or use_cache is False or trace_batch is not None):
        raise ValueError("OpenVINO requires an export directory, no quantization, stateful cache, and no PyTorch operator trace.")
    if quantization not in {"int8", "none"}:
        raise ValueError("Quantization must be 'int8' or 'none'.")
    if use_cache is not None and not isinstance(use_cache, bool):
        raise ValueError("use_cache must be a boolean or None.")
    if (use_cache is not None or profile_output_dir is not None) and chunk_seconds is None:
        raise ValueError("Detailed profiling and cache experiments require chunk_seconds.")
    if trace_batch is not None and (isinstance(trace_batch, bool) or not isinstance(trace_batch, int) or trace_batch < 1):
        raise ValueError("trace_batch must be a positive batch index or None.")
    if trace_batch is not None and profile_output_dir is None:
        raise ValueError("Operator tracing requires profile_output_dir.")
    if cpu_threads is None:
        cpu_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if cpu_threads < 1 or interop_threads < 1:
        raise ValueError("CPU and inter-op thread counts must be positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    with tempfile.TemporaryDirectory(prefix="moss-research-") as folder:
        request_path = Path(folder) / "request.json"
        result_path = Path(folder) / "result.json"
        request = {
            "audio_path": str(audio_path), "model_id": model_id, "device": device,
            "max_new_tokens": max_new_tokens, "languages": languages,
            "cpu_threads": cpu_threads, "interop_threads": interop_threads,
            "quantization": quantization,
            "backend": backend,
        }
        if backend == "openvino":
            request.update(ov_model_dir=str(Path(ov_model_dir).resolve()), ov_device=ov_device)
        if chunk_seconds is not None:
            request["chunk_seconds"] = chunk_seconds
            request["batch_size"] = batch_size
        if profile_output_dir is not None:
            request["profile_output_dir"] = str(Path(profile_output_dir).resolve())
            request["trace_batch"] = trace_batch
        if use_cache is not None:
            request["use_cache"] = use_cache
        request_path.write_text(json.dumps(request), encoding="utf-8")
        if backend == "openvino":
            from .openvino_backend import openvino_environment
            worker_environment = openvino_environment()
        else:
            worker_environment = igpu_environment()
        environment = {**worker_environment, "OMP_NUM_THREADS": str(cpu_threads),
                       "MKL_NUM_THREADS": str(cpu_threads), "OPENBLAS_NUM_THREADS": str(cpu_threads)}
        started = perf_counter()
        returncode = run_model_worker(
            [sys.executable, "-u", "-m", "speech_to_text.isolated", str(request_path)],
            env=environment, cwd=Path(__file__).resolve().parents[1],
            timeout_seconds=timeout_seconds,
            min_available_memory_bytes=min_available_memory_bytes,
        )
        if returncode:
            raise RuntimeError(f"Research worker exited with code {returncode}; see its traceback above.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"]["isolated_worker_wall_seconds"] = perf_counter() - started
        return result["raw"], [Segment(**item) for item in result["segments"]], result["metrics"]


def run_worker(request_path):
    """Load one model, verify cleanup, and serialize tensor-free results."""
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    profiler = (StageProfiler(request["profile_output_dir"], trace_batch=request.get("trace_batch"))
                if "profile_output_dir" in request else None)
    with stage(profiler, "worker_imports"):
        import torch
        from .backend import MossSession
        from .runtime import release_model_memory
    with stage(profiler, "worker_runtime_initialization"):
        torch.set_num_threads(request.get("cpu_threads", 4))
        torch.set_num_interop_threads(request.get("interop_threads", 1))
        print(f"CPU threads: {torch.get_num_threads()}; inter-op threads: {torch.get_num_interop_threads()}", flush=True)
        is_openvino = request.get("backend", "pytorch") == "openvino"
        requested_device = request.pop("device")
        device = torch.device("cpu" if is_openvino else requested_device)
        if not is_openvino and (device.type != "xpu" or not torch.xpu.is_available()):
            raise RuntimeError("Intel XPU unavailable in worker. Run uv run --no-project --python .venv setup_iGPU.py first.")
        if not is_openvino and request.get("quantization", "none") == "int8":
            # bitsandbytes owns small device codebooks for the lifetime of its runtime.
            # Include them in the pre-model baseline; the worker exits after each use.
            import bitsandbytes
        if device.type == "xpu":
            torch.xpu.synchronize(device)
            torch.xpu.reset_peak_memory_stats(device)
        if profiler is not None and device.type == "xpu":
            profiler.device = device
        baseline, _ = tensor_memory(device)
    chunk_metrics = {}
    try:
        if is_openvino:
            from .openvino_backend import OpenVinoSession
            session_context = OpenVinoSession(request["model_id"], request["ov_model_dir"],
                                              ov_device=request["ov_device"], profiler=profiler,
                                              use_cache=request.get("use_cache", True))
        else:
            session_context = MossSession(request["model_id"], str(device),
                                          quantization=request.get("quantization", "none"),
                                          profiler=profiler, use_cache=request.get("use_cache"))
        with session_context as session:
            quantization_metrics = {
                "quantization": session.quantization,
                "dtype": "int8" if session.quantization == "int8" else str(session.dtype).removeprefix("torch."),
                "compute_dtype": str(session.dtype).removeprefix("torch."),
                "quantized_linear_count": session.quantized_linear_count,
                "attention_backend": "openvino" if is_openvino else "math_sdpa",
                "backend": request.get("backend", "pytorch"),
            }
            if "chunk_seconds" in request:
                from .chunked import transcribe_chunks
                raw, segments, chunk_metrics = transcribe_chunks(
                    request["audio_path"], session, request["max_new_tokens"],
                    request["languages"], chunk_seconds=request["chunk_seconds"],
                    batch_size=request.get("batch_size", 2),
                )
            else:
                raw, segments = session.transcribe(
                    request["audio_path"], request["max_new_tokens"], request["languages"],
                )
    finally:
        try:
            with stage(profiler, "worker_final_cleanup", synchronize=False):
                release_model_memory(device)
        finally:
            if profiler is not None:
                profiler.save()
    metrics = {
        **quantization_metrics, "torch_version": torch.__version__,
        "device_name": session.runtime_metrics["ov_device_name"] if is_openvino else torch.xpu.get_device_name(device),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "allowed_logical_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": torch.xpu.max_memory_allocated(device) if device.type == "xpu" else 0,
        "allocated_bytes_after": tensor_memory(device)[0],
        "reserved_bytes_after": tensor_memory(device)[1], **chunk_metrics,
    }
    if is_openvino:
        metrics.update(session.runtime_metrics)
    if metrics["allocated_bytes_after"] != baseline:
        raise RuntimeError("XPU tensor allocations did not return to baseline.")
    with stage(profiler, "worker_result_serialization"):
        request_path.with_name("result.json").write_text(json.dumps({
            "raw": raw, "segments": [asdict(segment) for segment in segments], "metrics": metrics,
        }, ensure_ascii=False), encoding="utf-8")
    if profiler is not None:
        profiler.save()


if __name__ == "__main__":
    run_worker(sys.argv[1])
