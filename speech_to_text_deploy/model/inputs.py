"""Prepare CPU features and left-padded prompts with the upstream processor."""


def prepare_batch_inputs(processor, audio_paths, prompt):
    """Build CPU features and left-pad prompts for stateful decoder generation."""
    import torch
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info
    messages = [build_transcription_messages(path, prompt=prompt) for path in audio_paths]
    texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
             for item in messages]
    audios = [audio for item in messages for audio in process_audio_info(
        item, sampling_rate=processor.feature_extractor.sampling_rate)]
    inputs = processor(text=texts, audio=audios, return_tensors='pt')
    # The upstream processor right-pads; decoder generation needs left padding.
    width = inputs['input_ids'].shape[1]
    lengths = inputs['attention_mask'].sum(dim=1)
    positions = (torch.arange(width)[None, :] - (width - lengths[:, None])) % width
    for key in ('input_ids', 'attention_mask'):
        inputs[key] = inputs[key].gather(1, positions)
    return inputs
