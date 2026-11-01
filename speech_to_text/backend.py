"""Adapt the official MOSS inference API without duplicating its parser."""
from dataclasses import dataclass
import math
from time import perf_counter

from .runtime import inference_attention, release_model_memory
from .profiling import stage


def transcription_prompt(default_prompt, languages, *, audio_duration_seconds=None):
    """Extend the upstream prompt consistently for single and batched generation."""
    prompt = default_prompt
    if languages:
        prompt += (" The audio may contain the following languages: " + ", ".join(languages)
                   + ". Transcribe speech in its original language without translation."
                   " Preserve language switches, timestamps, and speaker labels.")
    if audio_duration_seconds is not None:
        if not math.isfinite(audio_duration_seconds) or audio_duration_seconds <= 0:
            raise ValueError("Audio duration must be positive and finite.")
        duration = f"{audio_duration_seconds:.9f}".rstrip("0").rstrip(".")
        prompt += (
            f" This audio is {duration} seconds long and spans 0 to {duration} seconds."
            " All timestamps are relative to the start of this input audio."
            f" Every segment must satisfy 0 <= start < end <= {duration} seconds."
            " Do not repeat or invent speech after the audio ends."
            " Preserve the required transcript format."
        )
    return prompt


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str


class MossSession:
    """Own one model while releasing audio and generation tensors after each call."""

    def __init__(self, model_id, device_name, *, quantization="none", profiler=None, use_cache=None):
        import torch
        from moss_transcribe_diarize.inference_utils import resolve_device

        self.model_id = model_id
        self.profiler = profiler
        self.use_cache = use_cache
        if quantization not in {"none", "int8"}:
            raise ValueError("Quantization must be 'int8' or 'none'.")
        self.quantization = quantization
        self.quantized_linear_count = 0
        self.device = resolve_device(device_name)
        self.dtype = torch.bfloat16 if self.device.type in {"cuda", "xpu"} else torch.float32
        self.model = self.processor = None

    def __enter__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self.device.type == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("Intel XPU unavailable. Configure the local Intel runtime before Python starts.")
        try:
            options = {}
            if self.quantization == "int8":
                from transformers import BitsAndBytesConfig

                options = {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": {"": str(self.device)},
                }
            print(f"Loading MOSS once on {self.device}; quantization={self.quantization}", flush=True)
            with stage(self.profiler, "model_weights_load"):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True, dtype=self.dtype, attn_implementation="sdpa",
                    **options,
                )
            if self.quantization == "none":
                with stage(self.profiler, "model_to_device"):
                    self.model.to(self.device)
            else:
                from bitsandbytes.nn import Linear8bitLt

                self.quantized_linear_count = sum(
                    isinstance(module, Linear8bitLt) and module.weight.dtype == torch.int8
                    for module in self.model.modules())
                if not self.quantized_linear_count:
                    raise RuntimeError("INT8 requested but no INT8 linear weights were loaded.")
                print(f"INT8 linear layers: {self.quantized_linear_count}", flush=True)
            self.model.eval()
            with stage(self.profiler, "processor_load"):
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        """Drop weights and processor references on success, failure or interruption."""
        with stage(getattr(self, "profiler", None), "model_unload_and_cleanup", synchronize=False):
            self.model = self.processor = None
            release_model_memory(self.device)

    def __exit__(self, *exc):
        self.close()

    def transcribe(self, audio_path, max_new_tokens, languages=None):
        import torch
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            DEFAULT_PROMPT, build_transcription_messages, generate_transcription,
        )

        if self.model is None or self.processor is None:
            raise RuntimeError("Open the MOSS session before transcribing.")
        if max_new_tokens < 1:
            raise ValueError("Token limit must be positive.")

        def report_tokens(count):
            if count == 1 or count % 128 == 0:
                print(f"Generated {count}/{max_new_tokens} tokens", flush=True)

        prompt = transcription_prompt(DEFAULT_PROMPT, languages)
        result = None
        try:
            with torch.inference_mode(), inference_attention(self.device):
                result = generate_transcription(
                    self.model, self.processor,
                    build_transcription_messages(str(audio_path), prompt=prompt),
                    max_new_tokens=max_new_tokens, do_sample=False,
                    device=self.device, dtype=self.dtype, token_callback=report_tokens,
                )
            print(f"Generation finished: {result['generated_tokens']} tokens", flush=True)
            if result["generated_tokens"] >= max_new_tokens:
                raise RuntimeError(
                    "Generation reached the token limit; transcript may be incomplete. "
                    "Use a shorter audio window or increase the token limit within available memory."
                )
            raw = result["text"]
            segments = [Segment(float(s.start), float(s.end), str(s.speaker), s.text)
                        for s in parse_transcript(raw)]
            return raw, segments
        finally:
            del result
            # Keep weights alive; clear only the completed generation's tensors/cache.
            if self.quantization == "int8":
                release_model_memory(self.device, self.model)
            else:
                release_model_memory(self.device)

    def transcribe_batch(self, audio_paths, max_new_tokens, languages=None):
        """Time batch submission through CPU outputs, excluding allocator cleanup."""
        import torch
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
        from .batched import generate_batch

        if self.model is None or self.processor is None:
            raise RuntimeError("Open the MOSS session before transcribing.")
        if not audio_paths or max_new_tokens < 1:
            raise ValueError("Provide audio paths and a positive token limit.")
        torch.xpu.synchronize(self.device)
        started = perf_counter()
        profiler = getattr(self, "profiler", None)
        try:
            with inference_attention(self.device):
                predictions = generate_batch(self.model, self.processor, audio_paths,
                                             transcription_prompt(DEFAULT_PROMPT, languages),
                                             max_new_tokens, self.device, profiler=profiler,
                                             use_cache=getattr(self, "use_cache", None))
            with stage(profiler, "parse_transcript"):
                results = [(item["text"], [Segment(float(s.start), float(s.end), str(s.speaker), s.text)
                                          for s in parse_transcript(item["text"])]) for item in predictions]
            torch.xpu.synchronize(self.device)
            elapsed = perf_counter() - started
            print("Batch generated tokens:", [item["generated_tokens"] for item in predictions], flush=True)
            return results, elapsed
        finally:
            with stage(profiler, "batch_allocator_cleanup"):
                if getattr(self, "quantization", "none") == "int8":
                    release_model_memory(self.device, self.model)
                else:
                    release_model_memory(self.device)


def transcribe(audio_path, model_id, device_name, max_new_tokens, languages=None):
    """Transcribe one input and unload the model on every exit."""
    with MossSession(model_id, device_name) as session:
        return session.transcribe(audio_path, max_new_tokens, languages)
