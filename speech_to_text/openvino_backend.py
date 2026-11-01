"""Run original MOSS weights with OpenVINO encoder and stateful greedy decoder."""
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .backend import Segment, transcription_prompt
from .batched import prepare_batch_inputs
from .profiling import stage
from .runtime import release_model_memory


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


def causal_mask(attention_mask, query_length):
    """Build the same causal and left-padding mask for prefill or one-token decode."""
    width = attention_mask.shape[1]
    query_positions = np.arange(width - query_length, width)
    allowed = ((np.arange(width)[None, :] <= query_positions[:, None])[None, None, :, :]
               & attention_mask[:, None, None, :].astype(bool))
    return np.where(allowed, np.float32(0), np.finfo(np.float32).min)


class OpenVinoSession:
    """Keep compiled graphs loaded, resetting KV state independently per batch."""

    def __init__(self, model_id, model_dir, *, ov_device="GPU", profiler=None, use_cache=True,
                 audio_window_seconds=None):
        import torch

        if use_cache is False:
            raise ValueError("The OpenVINO experiment requires its stateful KV cache.")
        if audio_window_seconds is not None and (
                not math.isfinite(audio_window_seconds) or audio_window_seconds <= 0):
            raise ValueError("Audio window must be positive and finite.")
        self.audio_window_seconds = audio_window_seconds
        self.model_id, self.model_dir = model_id, Path(model_dir)
        self.ov_device, self.profiler = ov_device, profiler
        self.device = torch.device("cpu")  # CPU preprocessing; OpenVINO owns GPU allocations.
        self.dtype = torch.float16 if ov_device.startswith("GPU") else torch.float32
        self.quantization, self.quantized_linear_count = "none", 0
        self.core = self.encoder = self.decoder = None
        self.encoder_request = self.decoder_request = self.processor = self.embeddings = None
        self.runtime_metrics = {"backend": "openvino", "ov_device": ov_device,
                                "torch_allocator_scope": "CPU preprocessing only; not OpenVINO GPU memory",
                                "audio_window_seconds": audio_window_seconds,
                                "openvino_batches": []}

    def memory_statistics(self):
        if self.core is None or not self.ov_device.startswith("GPU"):
            return None
        try:
            return {str(k): int(v) for k, v in self.core.get_property(self.ov_device, "GPU_MEMORY_STATISTICS").items()}
        except RuntimeError:
            return None

    def __enter__(self):
        import openvino as ov
        from transformers import AutoProcessor

        try:
            self.manifest = json.loads((self.model_dir / "manifest.json").read_text())
            if self.manifest["model_id"] != self.model_id or self.manifest["format_version"] != 1:
                raise ValueError("The OpenVINO export does not match the requested model.")
            self.core = ov.Core()
            self.runtime_metrics["ov_version"] = ov.__version__
            self.runtime_metrics["ov_device_name"] = self.core.get_property(self.ov_device, "FULL_DEVICE_NAME")
            self.runtime_metrics["ov_memory_before_compile"] = self.memory_statistics()
            precision = "f16" if self.ov_device.startswith("GPU") else "f32"
            config = {"INFERENCE_PRECISION_HINT": precision, "PERFORMANCE_HINT": "LATENCY"}
            with stage(self.profiler, "openvino_encoder_compile"):
                self.encoder = self.core.compile_model(str(self.model_dir / "encoder.xml"), self.ov_device, config)
                self.encoder_request = self.encoder.create_infer_request()
            with stage(self.profiler, "openvino_decoder_compile"):
                self.decoder = self.core.compile_model(str(self.model_dir / "decoder.xml"), self.ov_device, config)
                self.decoder_request = self.decoder.create_infer_request()
            self.runtime_metrics["encoder_execution_devices"] = self.encoder.get_property("EXECUTION_DEVICES")
            self.runtime_metrics["decoder_execution_devices"] = self.decoder.get_property("EXECUTION_DEVICES")
            with stage(self.profiler, "processor_and_embeddings_load"):
                self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
                self.embeddings = np.load(self.model_dir / "embeddings.npy", mmap_mode="r")
            if len(self.decoder_request.query_state()) != self.manifest["num_layers"] * 2:
                raise RuntimeError("The OpenVINO decoder has an unexpected number of KV states.")
            print("OpenVINO execution:", self.runtime_metrics, flush=True)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        with stage(self.profiler, "openvino_unload_and_cleanup", synchronize=False):
            self.runtime_metrics["ov_memory_before_unload"] = self.memory_statistics()
            self.encoder_request = self.decoder_request = None
            self.encoder = self.decoder = self.processor = self.embeddings = None
            release_model_memory(self.device)
            self.runtime_metrics["ov_memory_after_unload"] = self.memory_statistics()
            self.core = None

    def __exit__(self, *exc):
        self.close()

    def _empty_cache(self, batch_size):
        import openvino as ov

        shape = (batch_size, self.manifest["kv_heads"], 0, self.manifest["head_dim"])
        for state in self.decoder_request.query_state():
            state.state = ov.Tensor(np.zeros(shape, dtype=np.float32))

    def transcribe_batch(self, audio_paths, max_new_tokens, languages=None, *, audio_duration_seconds=None):
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

        if not audio_paths or max_new_tokens < 1:
            raise ValueError("Provide audio paths and a positive token limit.")
        inputs = features = projected = embeds = ids = mask = positions = logits = None
        started = perf_counter()
        batch_metrics = {"index": len(self.runtime_metrics["openvino_batches"]) + 1}
        duration = audio_duration_seconds if audio_duration_seconds is not None else self.audio_window_seconds
        batch_metrics["timestamp_limit_seconds"] = duration
        try:
            inputs = prepare_batch_inputs(self.processor, audio_paths,
                                          transcription_prompt(DEFAULT_PROMPT, languages,
                                                               audio_duration_seconds=duration),
                                          profiler=self.profiler)
            ids = inputs["input_ids"].numpy()
            mask = inputs["attention_mask"].numpy()
            features = inputs["input_features"].float().numpy()
            with stage(self.profiler, "openvino_audio_encoder"):
                self.encoder_request.infer({"features": features})
                projected = self.encoder_request.get_tensor("audio_embeds").data.copy()
            with stage(self.profiler, "audio_embedding_injection"):
                embeds = self.embeddings[ids].astype(np.float32)
                lengths = inputs["audio_feature_lengths"].tolist()
                mapping = inputs["audio_chunk_mapping"].tolist()
                for row in range(len(audio_paths)):
                    parts = [projected[i, :lengths[i]] for i, mapped in enumerate(mapping) if mapped == row]
                    audio = np.concatenate(parts)
                    audio_mask = ids[row] == self.manifest["audio_token_id"]
                    if audio_mask.sum() != len(audio):
                        raise ValueError("Audio token count does not match the exported features.")
                    embeds[row, audio_mask] = audio
            self._empty_cache(len(audio_paths))
            eos = self.manifest["eos_token_id"]
            eos = [eos] if isinstance(eos, int) else eos
            finished = np.zeros(len(audio_paths), dtype=bool)
            generated = [[] for _ in audio_paths]
            batch_metrics.update(prefill_seconds=0., decode_seconds=0., decode_calls=0)
            for step in range(max_new_tokens):
                positions = np.cumsum(mask, axis=1) - 1
                positions[mask == 0] = 1
                positions = positions[:, -embeds.shape[1]:].astype(np.int64)
                step_started = perf_counter()
                with stage(self.profiler, "openvino_prefill" if step == 0 else "openvino_decode"):
                    self.decoder_request.infer({"embeds": embeds, "mask": causal_mask(mask, embeds.shape[1]),
                                                "positions": positions})
                    logits = self.decoder_request.get_tensor("logits").data
                    next_ids = logits[:, -1, :].argmax(axis=-1).astype(np.int64)
                elapsed = perf_counter() - step_started
                batch_metrics["prefill_seconds" if step == 0 else "decode_seconds"] += elapsed
                batch_metrics["decode_calls"] += int(step > 0)
                for row, token in enumerate(next_ids.tolist()):
                    if not finished[row]:
                        generated[row].append(token)
                finished |= np.isin(next_ids, eos)
                if finished.all():
                    break
                next_ids[finished] = self.manifest["pad_token_id"]
                embeds = self.embeddings[next_ids[:, None]].astype(np.float32)
                mask = np.concatenate((mask, np.ones((len(audio_paths), 1), dtype=mask.dtype)), axis=1)
                if step == 0 or (step + 1) % 128 == 0:
                    print(f"OpenVINO generated {step + 1}/{max_new_tokens} steps", flush=True)
            if not finished.all():
                raise RuntimeError("OpenVINO generation reached the token limit; transcript may be incomplete.")
            with stage(self.profiler, "openvino_text_decode_and_parse"):
                results = []
                for tokens in generated:
                    raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    results.append((raw, [Segment(float(s.start), float(s.end), str(s.speaker), s.text)
                                          for s in parse_transcript(raw)]))
            batch_metrics["generated_tokens"] = [len(tokens) for tokens in generated]
            batch_metrics["kv_state_count"] = len(self.decoder_request.query_state())
            self.runtime_metrics["openvino_batches"].append(batch_metrics)
            return results, perf_counter() - started
        finally:
            inputs = features = projected = embeds = ids = mask = positions = logits = None
            if self.decoder_request is not None:
                self.decoder_request.reset_state()
            release_model_memory(self.device)

    def transcribe(self, audio_path, max_new_tokens, languages=None):
        duration = None
        if self.audio_window_seconds is not None:
            from .audio import open_audio_source

            with open_audio_source(audio_path) as source:
                duration = min(self.audio_window_seconds, source.frames / source.samplerate)
        results, _ = self.transcribe_batch([str(audio_path)], max_new_tokens, languages,
                                          audio_duration_seconds=duration)
        return results[0]
