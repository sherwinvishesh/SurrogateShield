from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DetectedEntity:
    """Represents a single piece of detected PII."""
    text: str
    start: int
    end: int
    type: str
    score: float = 1.0
    source: str = "pattern"

    def overlaps(self, other: "DetectedEntity") -> bool:
        return not (self.end <= other.start or self.start >= other.end)


def mask_spans(text: str, entities: List[DetectedEntity], placeholder: str = "█") -> str:
    if not entities:
        return text
    chars = list(text)
    for ent in entities:
        for i in range(ent.start, min(ent.end, len(chars))):
            chars[i] = placeholder
    return "".join(chars)


def remove_span_overlap(candidate: DetectedEntity, existing: List[DetectedEntity]) -> bool:
    return any(candidate.overlaps(e) for e in existing)
