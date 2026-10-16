# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
Run Presidio detection and return structured results.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class PresidioEntity:
    entity_type: str   # Presidio's native label e.g. "PERSON", "US_SSN"
    text: str          # Extracted text from the original string
    start: int
    end: int
    score: float


def resolve_overlaps(entities: Iterable[PresidioEntity]) -> List[PresidioEntity]:
    """Keep one entity per overlapping region: the longest span, then the highest score.

    Presidio returns overlapping candidates (e.g. DATE_TIME '7911 123456' inside
    PHONE_NUMBER '+44 7911 123456') and leaves conflict resolution to the caller.
    Keeping the highest *score* drops the phone in favour of the spaCy fragment;
    keeping the longest span keeps the phone, as Presidio's anonymizer does.
    """
    ordered = sorted(entities, key=lambda e: (-(e.end - e.start), -e.score, e.start))
    kept: List[PresidioEntity] = []
    for ent in ordered:
        if all(ent.end <= k.start or ent.start >= k.end for k in kept):
            kept.append(ent)
    kept.sort(key=lambda e: e.start)
    return kept


def detect(text: str, score_threshold: Optional[float] = None) -> list[PresidioEntity] | None:
    """
    Run default-config Presidio on *text* with all of its entities enabled.

    Returns:
        List of PresidioEntity sorted by start position (overlaps resolved),
        or None if Presidio is not installed. An empty list means Presidio ran
        and found nothing. Analyzer exceptions propagate: a crash must never be
        scored as "found nothing".
    """
    from presidio.engine import SCORE_THRESHOLD, get_analyzer

    analyzer = get_analyzer()
    if analyzer is None:
        return None

    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold
    results = analyzer.analyze(
        text=text, language="en", entities=None, score_threshold=threshold
    )

    entities: list[PresidioEntity] = []
    for r in results:
        start = max(0, r.start)
        end = min(len(text), r.end)
        if not text[start:end].strip():
            continue
        entities.append(PresidioEntity(
            entity_type=r.entity_type,
            text=text[start:end],
            start=start,
            end=end,
            score=round(float(r.score), 2),
        ))

    return resolve_overlaps(entities)
