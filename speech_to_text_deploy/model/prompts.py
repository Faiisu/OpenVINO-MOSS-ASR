"""Compose language, hotword and timestamp instructions for deployed MOSS."""
import math


def transcription_prompt(default_prompt, languages, *, hotwords=(), audio_duration_seconds=None):
    """Extend the upstream prompt consistently for single and batched generation."""
    prompt = default_prompt
    if hotwords:
        prompt += " Hotwords: " + ", ".join(hotwords) + "."
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
