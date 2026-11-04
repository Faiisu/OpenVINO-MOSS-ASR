"""Export the original MOSS encoder/adaptor and stateful language decoder."""
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

from .profiling import StageProfiler
from .runtime import release_model_memory, run_model_worker


class AudioGraph(torch.nn.Module):
    """Project every padded Whisper window; trim valid audio tokens at runtime."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.whisper_encoder
        self.adaptor = model.model.vq_adaptor

    def forward(self, features):
        hidden = self.encoder(features, return_dict=True).last_hidden_state
        return self.adaptor(hidden.reshape(hidden.shape[0], hidden.shape[1] // 4, -1))


class DecoderGraph(torch.nn.Module):
    """Expose explicit KV ports for OpenVINO's stateful graph transformation."""

    def __init__(self, language_model, lm_head, config):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.config = config

    def forward(self, embeds, mask, positions, past):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.config)
        for layer, key, value in zip(cache.layers, past[::2], past[1::2]):
            layer.lazy_initialization(key, value)
            layer.keys, layer.values = key, value
        output = self.language_model(
            inputs_embeds=embeds, attention_mask={"full_attention": mask},
            position_ids=positions, past_key_values=cache, use_cache=True)
        logits = self.lm_head(output.last_hidden_state[:, -1:, :])
        return (logits,) + tuple(tensor for layer in cache.layers for tensor in (layer.keys, layer.values))


def stateful_decoder(graph, config):
    """Keep KV tensors in InferRequest state rather than copying them to Python."""
    from openvino import passes

    count = config.num_hidden_layers * 2
    inputs = ["embeds", "mask", "positions"] + [f"past_{i}" for i in range(count)]
    outputs = ["logits"] + [f"present_{i}" for i in range(count)]
    if len(graph.inputs) != len(inputs) or len(graph.outputs) != len(outputs):
        raise ValueError("Unexpected decoder graph ports; export cannot be validated.")
    for port, name in zip(graph.inputs, inputs):
        port.get_tensor().set_names({name})
    for port, name in zip(graph.outputs, outputs):
        port.get_tensor().set_names({name})
    graph.reshape({"embeds": [-1, -1, config.hidden_size], "mask": [-1, 1, -1, -1],
                   "positions": [-1, -1],
                   **{f"past_{i}": [-1, config.num_key_value_heads, -1, config.head_dim]
                      for i in range(count)}})
    manager = passes.Manager()
    manager.register_pass(passes.MakeStateful({f"past_{i}": f"present_{i}" for i in range(count)}))
    manager.run_passes(graph)
    return graph


def export_model(model_id, output_dir):
    """Convert in a disposable CPU worker and release source weights on all exits."""
    import numpy as np
    import openvino as ov
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils.hub import cached_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted overwrite must never leave an apparently complete cache.
    (output_dir / "manifest.json").unlink(missing_ok=True)
    profiler = StageProfiler(output_dir / "export_profile")
    model = processor = wrapper = graph = example = embedding = None
    started = perf_counter()
    try:
        with profiler.stage("source_weights_load"):
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager").eval()
            processor_class = AutoProcessor
            processor_reference = getattr(model.config, "auto_map", {}).get("AutoProcessor")
            if processor_reference:
                # Some fine-tunes omit processor_config.auto_map; the built-in
                # namesake uses different audio inputs than this source model.
                processor_class = get_class_from_dynamic_module(processor_reference, model_id)
                processor_class.register_for_auto_class("AutoProcessor")
            processor = processor_class.from_pretrained(model_id, trust_remote_code=True)
        config = model.config.text_config
        with profiler.stage("processor_and_embedding_save"):
            processor.save_pretrained(output_dir)
            embedding = model.model.language_model.embed_tokens.weight.detach().numpy().astype(np.float16)
            np.save(output_dir / "embeddings.npy", embedding)
            embedding = None
        with profiler.stage("encoder_convert_and_save"), torch.inference_mode():
            wrapper = AudioGraph(model).eval()
            example = torch.zeros(1, 80, 3000)
            graph = ov.convert_model(wrapper, example_input=example)
            graph.input(0).get_tensor().set_names({"features"})
            graph.output(0).get_tensor().set_names({"audio_embeds"})
            graph.reshape({"features": [-1, 80, 3000]})
            ov.save_model(graph, output_dir / "encoder.xml", compress_to_fp16=True)
            wrapper = graph = example = None
            model.model.whisper_encoder = model.model.vq_adaptor = None
            release_model_memory(torch.device("cpu"))
        with profiler.stage("stateful_decoder_convert_and_save"), torch.inference_mode():
            wrapper = DecoderGraph(model.model.language_model, model.lm_head, config).eval()
            example = (torch.zeros(1, 2, config.hidden_size), torch.zeros(1, 1, 2, 3),
                       torch.tensor([[1, 2]]),
                       tuple(torch.zeros(1, config.num_key_value_heads, 1, config.head_dim)
                             for _ in range(config.num_hidden_layers * 2)))
            graph = ov.convert_model(wrapper, example_input=example)
            graph = stateful_decoder(graph, config)
            ov.save_model(graph, output_dir / "decoder.xml", compress_to_fp16=True)
        manifest = {"format_version": 1, "model_id": model_id,
                    "source_snapshot": Path(cached_file(model_id, "config.json")).parent.name,
                    "openvino_version": ov.__version__, "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "weight_storage": "FP16", "inference_precision_hint": "FP16",
                    "kv_cache": "OpenVINO stateful; reset independently for every batch",
                    "num_layers": config.num_hidden_layers, "kv_heads": config.num_key_value_heads,
                    "head_dim": config.head_dim, "audio_token_id": model.config.audio_token_id,
                    "eos_token_id": model.generation_config.eos_token_id,
                    "pad_token_id": model.generation_config.pad_token_id,
                    "export_seconds": perf_counter() - started}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Export complete:", manifest, flush=True)
        return manifest
    finally:
        model = processor = wrapper = graph = example = embedding = None
        release_model_memory(torch.device("cpu"))
        profiler.save()


def export_isolated(model_id, output_dir, *, timeout_seconds=1800):
    """Run conversion outside notebook memory and retain a complete export manifest."""
    from .openvino_backend import openvino_environment

    output_dir = Path(output_dir).resolve()
    returncode = run_model_worker(
        [sys.executable, "-u", "-m", "speech_to_text.openvino_export", model_id, str(output_dir)],
        env=openvino_environment(), timeout_seconds=timeout_seconds,
        min_available_memory_bytes=2 * 1024**3)
    if returncode:
        raise RuntimeError(f"OpenVINO export worker exited {returncode}; inspect the traceback.")
    return json.loads((output_dir / "manifest.json").read_text())


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    export_model(sys.argv[1], sys.argv[2])
