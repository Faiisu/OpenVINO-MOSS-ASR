"""Generate independent audio transcripts together using one loaded MOSS model."""
import copy
from contextlib import nullcontext

from .profiling import stage


def prepare_batch_inputs(processor, audio_paths, prompt, *, profiler=None):
    """Prepare the same CPU audio features and left-padded prompts for both runtimes."""
    import torch
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info

    with stage(profiler, "prompt_and_chat_template"):
        messages = [build_transcription_messages(path, prompt=prompt) for path in audio_paths]
        texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                 for item in messages]
    with stage(profiler, "audio_read_resample_downmix"):
        audios = [audio for item in messages for audio in process_audio_info(
            item, sampling_rate=processor.feature_extractor.sampling_rate)]
    with stage(profiler, "processor_features_and_tokenization"):
        inputs = processor(text=texts, audio=audios, return_tensors="pt")
    # The remote processor always right-pads. Decoder generation needs left padding.
    with stage(profiler, "left_padding"):
        width = inputs["input_ids"].shape[1]
        lengths = inputs["attention_mask"].sum(dim=1)
        positions = (torch.arange(width)[None, :] - (width - lengths[:, None])) % width
        for key in ("input_ids", "attention_mask"):
            inputs[key] = inputs[key].gather(1, positions)
    return inputs


def generate_batch(model, processor, audio_paths, prompt, max_new_tokens, device, *, profiler=None, use_cache=None):
    """Return tensor-free texts and token counts; normalize prompts to left padding."""
    import torch

    inputs = outputs = row = generated = positions = lengths = None
    try:
        inputs = prepare_batch_inputs(processor, audio_paths, prompt, profiler=profiler)
        width = inputs["input_ids"].shape[1]
        with stage(profiler, "host_to_device"):
            inputs = inputs.to(device)
        config = copy.deepcopy(model.generation_config)
        config.max_new_tokens = max_new_tokens
        config.do_sample = False
        config.num_beams = 1
        config.num_return_sequences = 1
        if use_cache is not None:
            config.use_cache = use_cache
        with stage(profiler, "generation"), torch.inference_mode():
            with profiler.generation(model, inputs, config.use_cache) if profiler is not None else nullcontext():
                outputs = model.generate(**inputs, generation_config=config)
        with stage(profiler, "device_to_host_and_text_decode"):
            results = []
            eos = config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            for row in outputs:
                generated = row[width:].detach().cpu().tolist()
                stop = next((i + 1 for i, token in enumerate(generated) if token in eos_ids), len(generated))
                if stop >= max_new_tokens and (not generated or generated[stop - 1] not in eos_ids):
                    raise RuntimeError("Batch generation reached the token limit; use shorter chunks or increase the limit.")
                results.append({"text": processor.tokenizer.decode(generated[:stop], skip_special_tokens=True).strip(),
                                "generated_tokens": stop})
            if profiler is not None:
                profiler.batches[-1]["generated_tokens"] = [item["generated_tokens"] for item in results]
        if len(results) != len(audio_paths):
            raise RuntimeError("Model returned an unexpected number of batch outputs.")
        return results
    finally:
        with stage(profiler, "generation_tensor_release"):
            del inputs, outputs, row, generated, positions, lengths
