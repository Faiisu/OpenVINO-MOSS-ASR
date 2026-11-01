"""Typhoon Whisper ASR adapter; this model does not provide speaker diarization."""
from .runtime import release_model_memory


def _encoder_float_inputs(module, args, kwargs):
    """Use efficient FP32 kernels for the CPU encoder."""
    if args:
        args = (args[0].float(), *args[1:])
    if "input_features" in kwargs:
        kwargs = {**kwargs, "input_features": kwargs["input_features"].float()}
    return args, kwargs


def _encoder_decoder_output(module, args, output):
    """Match the GPU decoder precision after CPU encoding."""
    output.last_hidden_state = output.last_hidden_state.to(dtype=module._decoder_dtype)
    return output


def transcribe_typhoon(audio_path, model_id, device_name, language="thai", cpu_offload=False):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from moss_transcribe_diarize.inference_utils import load_audio_av

    device = torch.device(device_name)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError(f"Requested GPU is unavailable: {device}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = processor = recognizer = result = None
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=dtype, use_safetensors=True, attn_implementation="sdpa",
        )
        if cpu_offload:
            if device.type != "cuda":
                raise ValueError("CPU encoder offload requires a CUDA decoder device.")
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module
            # Keep real CPU weights rather than meta placeholders used by dynamic offload.
            model.model.encoder.float()
            model.model.encoder._decoder_dtype = dtype
            model.model.encoder.register_forward_pre_hook(_encoder_float_inputs, with_kwargs=True)
            model.model.encoder.register_forward_hook(_encoder_decoder_output)
            model.model.decoder.to(device)
            model.proj_out.to(device)
            add_hook_to_module(model, AlignDevicesHook(execution_device=device, io_same_device=True))
            add_hook_to_module(model.model.encoder, AlignDevicesHook(execution_device="cpu", io_same_device=True))
            model.hf_device_map = {"model.decoder": str(device), "model.encoder": "cpu", "proj_out": str(device)}
        else:
            model = model.to(device)
        model.eval()
        # A single total decoder-length limit includes Whisper prompt tokens.
        model.generation_config.max_new_tokens = None
        model.generation_config.max_length = model.config.max_target_positions
        processor = AutoProcessor.from_pretrained(model_id)
        recognizer = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            chunk_length_s=30, stride_length_s=5, batch_size=1, return_timestamps=True, **({} if cpu_offload else {"device": device}),
        )
        waveform = load_audio_av(str(audio_path), sampling_rate=16000).astype("float32")
        # The shared decoder returns normalized floating-point samples.
        generation = {"task": "transcribe", "do_sample": False, "num_beams": 1}
        if language:
            generation["language"] = language
        with torch.inference_mode():
            result = recognizer(
                {"array": waveform, "sampling_rate": 16000},
                generate_kwargs=generation,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"text": result["text"], "device": str(device), "speaker_diarization": False,
                "encoder_device": "cpu" if cpu_offload else str(device),
                "num_beams": 1, "chunk_length_s": 30, "stride_length_s": 5, "batch_size": 1,
                "return_timestamps": True, "max_length": model.config.max_target_positions,
                "max_new_tokens": None, "language": language, "chunks": result.get("chunks", []), "dtype": str(dtype),
                "encoder_dtype": "torch.float32" if cpu_offload else str(dtype)}
    finally:
        del recognizer, processor, model, result
        release_model_memory(device)
