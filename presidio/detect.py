"""
Run Presidio detection and return structured results.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PresidioEntity:
    entity_type: str   # Presidio's native label e.g. "PERSON", "US_SSN"
    text: str          # Extracted text from the original string
    start: int
    end: int
    score: float


def detect(text: str) -> list[PresidioEntity] | None:
    """
    Run Presidio on text with all entities enabled.

    Returns:
        List of PresidioEntity sorted by start position,
        or None if Presidio is unavailable.
        Returns empty list if Presidio ran but found nothing.
    """
    from presidio.engine import get_analyzer
    analyzer = get_analyzer()
    if analyzer is None:
        return None

    try:
        results = analyzer.analyze(text=text, language="en", entities=None)
    except Exception:
        return []

    entities: list[PresidioEntity] = []
    for r in results:
        # Guard against out-of-bounds offsets
        start = max(0, r.start)
        end = min(len(text), r.end)
        entity_text = text[start:end].strip()
        if not entity_text:
            continue
        entities.append(PresidioEntity(
            entity_type=r.entity_type,
            text=entity_text,
            start=start,
            end=end,
            score=round(float(r.score), 2),
        ))

    # Remove overlapping spans: keep highest-score entity when spans overlap
    entities.sort(key=lambda e: e.score, reverse=True)
    kept: list[PresidioEntity] = []
    occupied: list[tuple[int, int]] = []
    for ent in entities:
        overlaps = any(
            not (ent.end <= os or ent.start >= oe)
            for os, oe in occupied
        )
        if not overlaps:
            kept.append(ent)
            occupied.append((ent.start, ent.end))

    kept.sort(key=lambda e: e.start)
    return kept
