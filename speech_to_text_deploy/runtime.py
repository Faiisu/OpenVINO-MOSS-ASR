"""Worker supervision, environment setup and shared model-memory cleanup."""
import gc
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic
import traceback


def tensor_memory(device):
    """Report PyTorch allocator bytes; other runtimes must report their own memory."""
    import torch

    device = torch.device(device)
    if device.type == "xpu":
        return torch.xpu.memory_allocated(device), torch.xpu.memory_reserved(device)
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    return 0, 0


def run_model_worker(command, *, env, cwd=None, timeout_seconds=600,
                     min_available_memory_bytes=1024**3, interrupt_grace_seconds=5):
    """Supervise a worker; None disables its deadline but keeps resource checks."""
    if ((timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0))
            or min_available_memory_bytes < 0 or interrupt_grace_seconds <= 0):
        raise ValueError("Worker timeout must be positive and memory reserve nonnegative.")
    process = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True)
    started = monotonic()
    try:
        while process.poll() is None:
            if timeout_seconds is not None and monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"Model worker exceeded {timeout_seconds}s; stopped before the next trial.")
            if min_available_memory_bytes and Path("/proc/meminfo").exists():
                values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available = int(values["MemAvailable"].split()[0]) * 1024
                if available < min_available_memory_bytes:
                    raise MemoryError("Model worker stopped: available system RAM fell below the configured reserve.")
            try:
                wait_seconds = (0.5 if timeout_seconds is None else
                                min(0.5, max(0.01, timeout_seconds - (monotonic() - started))))
                process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                pass
        return process.returncode
    finally:
        if process.poll() is None:
            # Give adapter finally blocks a chance to release tensors first.
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=interrupt_grace_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()


def openvino_environment():
    """Use the system OpenCL compiler instead of the optional Level Zero libraries."""
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[1] / ".tools" / "intel-iGPU"
    excluded = {str(root / "usr/lib/x86_64-linux-gnu"), str(root / "usr/local/lib")}
    paths = [p for p in environment.get("LD_LIBRARY_PATH", "").split(":") if p and p not in excluded]
    if paths:
        environment["LD_LIBRARY_PATH"] = ":".join(paths)
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return environment


def release_model_memory(device):
    """Release unreachable model tensors and cached device allocations on all exits."""
    import torch

    # Exceptions can otherwise retain model tensors in completed inference frames.
    active_traceback = sys.exc_info()[2]
    if active_traceback is not None:
        traceback.clear_frames(active_traceback)
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    elif device.type == "xpu" and torch.xpu.is_available():
        with torch.xpu.device(device):
            torch.xpu.synchronize(device)
            torch.xpu.empty_cache()
