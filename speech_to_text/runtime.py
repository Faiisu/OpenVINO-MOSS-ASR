"""Shared cleanup for model adapters used by scripts and notebooks."""
import gc
import sys
import traceback
import os
import signal
import subprocess
from contextlib import nullcontext
from time import monotonic
from pathlib import Path


def tensor_memory(device):
    """Report PyTorch allocator bytes; other runtimes must report their own memory."""
    import torch

    device = torch.device(device)
    if device.type == "xpu":
        return torch.xpu.memory_allocated(device), torch.xpu.memory_reserved(device)
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    return 0, 0


def inference_attention(device):
    """Avoid the oneDNN fused SDPA primitive failure on the local Intel iGPU."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if torch.device(device).type == "xpu":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def run_model_worker(command, *, env, cwd=None, timeout_seconds=600,
                     min_available_memory_bytes=1024**3, interrupt_grace_seconds=5):
    """Wait for one worker with a deadline and a Linux shared-RAM reserve."""
    if timeout_seconds <= 0 or min_available_memory_bytes < 0 or interrupt_grace_seconds <= 0:
        raise ValueError("Worker timeout must be positive and memory reserve nonnegative.")
    process = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True)
    started = monotonic()
    try:
        while process.poll() is None:
            if monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"Model worker exceeded {timeout_seconds}s; stopped before the next trial.")
            if min_available_memory_bytes and Path("/proc/meminfo").exists():
                values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available = int(values["MemAvailable"].split()[0]) * 1024
                if available < min_available_memory_bytes:
                    raise MemoryError("Model worker stopped: available system RAM fell below the configured reserve.")
            try:
                process.wait(timeout=min(0.5, max(0.01, timeout_seconds - (monotonic() - started))))
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


def igpu_environment():
    """Return child-process settings for the optional project-local Intel driver."""
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[1] / ".tools" / "intel-iGPU"
    libraries = [root / "usr/lib/x86_64-linux-gnu", root / "usr/local/lib"]
    if any((path / "libze_loader.so.1").exists() for path in libraries):
        paths = [str(path) for path in libraries if path.is_dir()]
        if environment.get("LD_LIBRARY_PATH"):
            paths.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = ":".join(paths)
    return environment


def release_model_memory(device, model=None):
    """Release unreachable model tensors and cached device allocations on all exits."""
    import torch

    if model is not None and getattr(model, "is_loaded_in_8bit", False):
        from bitsandbytes.nn import Linear8bitLt

        for module in model.modules():
            if isinstance(module, Linear8bitLt):
                # Outlier indices depend on the completed input, not the weights.
                module.state.idx = None
                module.state.subB = None

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
