from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DetectedEntity:
    """Represents a single piece of detected PII."""
    text: str
    start: int
    end: int
    type: str
    score: float = 1.0
    source: str = "pattern"
    # Structured payload for entities that carry one (addresses carry a
    # ParsedAddress from the canonical address parser). Optional and unused
    # by all non-address code paths.
    parsed: Optional[object] = field(default=None, compare=False)

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


def apply_entity_surrogates(
    text: str,
    entities: List[DetectedEntity],
    mapping: Dict[str, str],
) -> str:
    """
    Replace each detected entity with its surrogate, span-safely.

    Two stages:
      1. Splice surrogates at the exact (start, end) offsets, right-to-left
         so earlier offsets stay valid.  This guarantees a surrogate for one
         entity can never rewrite text inside another entity's span (e.g. a
         standalone "Tempe" surrogate must not touch the "Tempe" inside a
         shift-mode address that is deliberately preserved).
      2. A word-boundary pass replaces any ADDITIONAL occurrences of each
         original outside the already-claimed spans, longest-original first
         (preserves recall for repeated values the cascade only saw once).
    """
    if not mapping:
        return text

    # Stage 1 — span splice (right-to-left, skip overlaps defensively)
    spliceable = sorted(
        (e for e in entities if e.text in mapping and 0 <= e.start < e.end <= len(text)),
        key=lambda e: e.start,
        reverse=True,
    )
    claimed_end = len(text) + 1
    result = text
    for ent in spliceable:
        if ent.end > claimed_end:
            continue  # overlaps an entity already spliced
        if result[ent.start:ent.end] != ent.text:
            continue  # offsets no longer line up with the text — skip
        result = result[:ent.start] + mapping[ent.text] + result[ent.end:]
        claimed_end = ent.start

    # Stage 2 — remaining occurrences of each original, outside surrogates.
    # Longest originals first so substrings never clobber longer values.
    for original in sorted(mapping, key=len, reverse=True):
        if original in result:
            pattern = re.compile(
                r"(?<![\w])" + re.escape(original) + r"(?![\w])"
            )
            surrogate = mapping[original]
            result = pattern.sub(lambda _m: surrogate, result)

    return result
