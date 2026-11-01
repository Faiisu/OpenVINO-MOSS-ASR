"""Bound research inference memory by processing complete recordings in short windows."""
import math
import tempfile
from pathlib import Path
from contextlib import ExitStack

from .backend import Segment
from .audio import audio_properties, copy_audio_frames, open_audio_source, open_audio_target
from .profiling import stage
from .runtime import tensor_memory


def transcribe_chunked(audio_path, model_id, device, max_new_tokens, languages=None,
                       *, chunk_seconds=15, timeout_seconds=900,
                       min_available_memory_bytes=2 * 1024**3,
                       cpu_threads=None, interop_threads=1, batch_size=2, quantization="none",
                       profile_output_dir=None, use_cache=None, trace_batch=None,
                       backend="pytorch", ov_model_dir=None, ov_device="GPU"):
    """Keep weights in one worker while cycling audio windows through memory."""
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("Chunk duration must be finite and positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    from .isolated import transcribe_isolated

    return transcribe_isolated(
        audio_path, model_id, device, max_new_tokens, languages,
        chunk_seconds=chunk_seconds, timeout_seconds=timeout_seconds,
        min_available_memory_bytes=min_available_memory_bytes,
        cpu_threads=cpu_threads, interop_threads=interop_threads, batch_size=batch_size,
        quantization=quantization,
        profile_output_dir=profile_output_dir, use_cache=use_cache, trace_batch=trace_batch,
        backend=backend, ov_model_dir=ov_model_dir, ov_device=ov_device,
    )


def transcribe_chunks(audio_path, session, max_new_tokens, languages=None, *, chunk_seconds=15, batch_size=2):
    """Consume all audio windows using an already loaded model; keep only CPU results."""
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("Chunk duration must be finite and positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    baseline, _ = tensor_memory(session.device)
    profiler = getattr(session, "profiler", None)
    raw_parts, segments, chunks, batches = [], [], [], []
    next_speaker = 1
    with tempfile.TemporaryDirectory(prefix="moss-chunks-") as temporary:
        with ExitStack() as stack:
            with stage(profiler, "audio_source_open_and_decode"):
                source = stack.enter_context(open_audio_source(audio_path))
            properties = audio_properties(source)
            rate = source.samplerate
            chunk_frames = max(1, int(chunk_seconds * rate))
            total_frames = source.frames
            count = math.ceil(total_frames / chunk_frames)
            for first in range(0, count, batch_size):
                windows = []
                for index in range(first, min(first + batch_size, count)):
                    start = index * chunk_frames
                    frames = min(chunk_frames, total_frames - start)
                    offset, duration = start / rate, frames / rate
                    chunk_path = Path(temporary) / f"chunk-{index}.wav"
                    with stage(profiler, "chunk_write", chunk_index=index + 1):
                        with open_audio_target(chunk_path, source) as target:
                            copy_audio_frames(source, target, start, frames)
                    windows.append((index, offset, duration, chunk_path))
                batch_index = len(batches) + 1
                print(f"Batch {batch_index}: chunks {first + 1}-{first + len(windows)}/{count}", flush=True)
                predictions, elapsed = session.transcribe_batch(
                    [str(window[3]) for window in windows], max_new_tokens, languages)
                if len(predictions) != len(windows):
                    raise RuntimeError("Batch output count does not match submitted chunks.")
                allocated, reserved = tensor_memory(session.device)
                if allocated != baseline:
                    raise RuntimeError(f"Batch tensors did not return to the loaded-model baseline: {allocated} != {baseline}.")
                duration_sum = sum(window[2] for window in windows)
                batches.append({"index": batch_index, "chunk_indices": [w[0] + 1 for w in windows],
                                "audio_seconds": duration_sum, "elapsed_seconds": elapsed,
                                "rtf": elapsed / duration_sum,
                                "allocated_bytes_after_batch": allocated})
                print(f"Batch output: {elapsed:.3f}s; audio {duration_sum:.3f}s; RTF {elapsed / duration_sum:.3f}", flush=True)
                print(f"Batch PyTorch allocator: {allocated} bytes (baseline: {baseline})", flush=True)
                for (index, offset, duration, chunk_path), (raw, local_segments) in zip(windows, predictions):
                    chunk_path.unlink()
                    speaker_map = {}
                    for segment in local_segments:
                        if (not math.isfinite(segment.start) or not math.isfinite(segment.end)
                                or segment.start < 0 or segment.end <= segment.start
                                or segment.start >= duration):
                            raise ValueError(f"Invalid segment in chunk {index + 1}: {segment}")
                        if segment.speaker not in speaker_map:
                            speaker_map[segment.speaker] = f"S{next_speaker:02d}"
                            next_speaker += 1
                        segments.append(Segment(
                            offset + segment.start, offset + min(segment.end, duration),
                            speaker_map[segment.speaker], segment.text,
                        ))
                    raw_parts.append(f"Chunk {index + 1}, offset {offset:.2f}s (local timestamps):\n{raw}")
                    chunks.append({"index": index + 1, "offset_seconds": offset,
                                   "duration_seconds": duration, "speaker_map": speaker_map,
                                   "batch_index": batch_index,
                                   "elapsed_seconds": elapsed, "rtf": elapsed / duration,
                                   "model_baseline_allocated_bytes": baseline,
                                   "allocated_bytes_after_chunk": allocated,
                                   "reserved_bytes_after_chunk": reserved})
    if not chunks:
        raise ValueError("The recording contains no audio samples.")
    metrics = {
        "model_load_count": 1, "chunk_seconds": chunk_seconds,
        "source_audio": properties,
        "batch_size": batch_size, "batch_count": len(batches), "batches": batches,
        "chunk_count": len(chunks), "audio_seconds": total_frames / rate,
        "speaker_identity_scope": "Independent per chunk; no cross-chunk speaker matching.",
        "model_baseline_allocated_bytes": baseline, "chunks": chunks,
        "chunk_processing_seconds": sum(batch["elapsed_seconds"] for batch in batches),
        "chunk_timing_scope": "Batch submission through CPU transcript/segment outputs with completed device work; includes preprocessing, generation and parsing; excludes model loading, whole-recording decoding, window WAV writing, allocator cleanup and exports. Each chunk records its batch latency; aggregate time counts each batch once.",
    }
    metrics["chunk_processing_rtf"] = metrics["chunk_processing_seconds"] / metrics["audio_seconds"]
    return "\n\n".join(raw_parts), segments, metrics
