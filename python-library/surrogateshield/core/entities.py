from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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


def plan_substitutions(
    text: str,
    entities: List[DetectedEntity],
    mapping: Dict[str, str],
) -> List[Tuple[int, int, str, str]]:
    """
    Plan every replacement as ``(start, end, original, surrogate)`` in the
    coordinates of the ORIGINAL *text*. Edits never overlap.

    Two sources, both planned against the original text so a surrogate can
    never be rewritten by a later pass (E1):
      1. The detected entity spans whose text has a surrogate. Overlapping
         spans keep the longest one.
      2. Every ADDITIONAL whole-word occurrence of each mapped original that
         lies outside the claimed spans (repeated values the cascade only saw
         once), longest original first.
    """
    if not mapping:
        return []

    edits: List[Tuple[int, int, str, str]] = []

    def _free(s: int, e: int) -> bool:
        return all(e <= es or s >= ee for es, ee, _, _ in edits)

    for ent in sorted(
        (e for e in entities
         if e.text in mapping and 0 <= e.start < e.end <= len(text)
         and text[e.start:e.end] == e.text),
        key=lambda e: (-(e.end - e.start), e.start),
    ):
        if _free(ent.start, ent.end):
            edits.append((ent.start, ent.end, ent.text, mapping[ent.text]))

    for original in sorted(mapping, key=len, reverse=True):
        if not original or original not in text:
            continue
        pattern = re.compile(r"(?<![\w])" + re.escape(original) + r"(?![\w])")
        for m in pattern.finditer(text):
            if _free(m.start(), m.end()):
                edits.append((m.start(), m.end(), original, mapping[original]))

    edits.sort(key=lambda x: x[0])
    return edits


def splice(text: str, edits: List[Tuple[int, int, str, str]]) -> str:
    """Apply non-overlapping ``plan_substitutions`` edits, right to left."""
    result = text
    for start, end, _original, surrogate in sorted(edits, key=lambda x: x[0], reverse=True):
        result = result[:start] + surrogate + result[end:]
    return result


def apply_entity_surrogates(
    text: str,
    entities: List[DetectedEntity],
    mapping: Dict[str, str],
) -> str:
    """Replace each detected entity (and repeated occurrences of its text)
    with its surrogate. See ``plan_substitutions`` for the rules."""
    return splice(text, plan_substitutions(text, entities, mapping))
