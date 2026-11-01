"""Shared tensor-free model output schema."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str
