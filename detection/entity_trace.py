"""
detection/entity_trace.py — EntityTrace

spaCy-based Named Entity Recognition (NER) detection. This module contains
ONLY the NER logic. It does not run regex or call Ollama.

Loads en_core_web_lg and extracts PERSON, GPE, LOC, ORG, FAC entities.
Skips any span already covered by PatternScan results.

Returns:
    confirmed  — entities with spaCy score >= ENTITY_TRACE_HIGH_THRESHOLD
    borderline — entities with ENTITY_TRACE_LOW_THRESHOLD <= score < HIGH
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from config import ENTITY_TRACE_HIGH_THRESHOLD, ENTITY_TRACE_LOW_THRESHOLD, SPACY_MODEL
from util import DetectedEntity, get_logger, remove_span_overlap

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Lazy-load spaCy model (expensive; load once)
# ─────────────────────────────────────────────

_nlp = None


def _get_nlp():
    """
    Load and cache the spaCy pipeline.

    Returns:
        spaCy Language object, or None if the model is unavailable.
    """
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load(SPACY_MODEL)
        logger.info(f"[EntityTrace] Loaded spaCy model: {SPACY_MODEL}")
    except OSError:
        logger.error(
            f"[EntityTrace] spaCy model '{SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        )
        _nlp = None
    except Exception as exc:
        logger.error(f"[EntityTrace] Failed to load spaCy: {exc}")
        _nlp = None
    return _nlp


# ─────────────────────────────────────────────
# Entity types to capture
# ─────────────────────────────────────────────

_TARGET_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}


# ─────────────────────────────────────────────
# Main trace function
# ─────────────────────────────────────────────

def trace(
    text: str,
    existing_entities: Optional[List[DetectedEntity]] = None,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run spaCy NER on *text* and return confirmed and borderline entities.

    Skips any span that overlaps with an entity in *existing_entities*
    (i.e. spans already caught by PatternScan).

    Threshold rules:
        score >= ENTITY_TRACE_HIGH_THRESHOLD  → confirmed
        ENTITY_TRACE_LOW_THRESHOLD <= score < HIGH → borderline
        score < ENTITY_TRACE_LOW_THRESHOLD    → discarded

    Args:
        text:             Text to analyse (may be the partially-masked
                          remaining_text after PatternScan ran).
        existing_entities: Already-confirmed entities to avoid overlap.

    Returns:
        Tuple of (confirmed_entities, borderline_entities).
    """
    existing = existing_entities or []
    confirmed: List[DetectedEntity] = []
    borderline: List[DetectedEntity] = []

    nlp = _get_nlp()
    if nlp is None:
        logger.warning("[EntityTrace] spaCy unavailable — skipping NER stage")
        return confirmed, borderline

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error(f"[EntityTrace] spaCy processing failed: {exc}")
        return confirmed, borderline

    for ent in doc.ents:
        if ent.label_ not in _TARGET_LABELS:
            continue

        # spaCy en_core_web_lg does not always populate ent.kb_id / ent.sent.end_score
        # Use ent._.score if available (from component), otherwise default to 0.85
        # to represent reasonable NER confidence from the lg model.
        score: float = getattr(ent, "score_", None)
        if score is None:
            # en_core_web_lg doesn't expose per-entity scores natively;
            # use a sensible default that places it in the "confirmed" bucket.
            score = 0.85

        candidate = DetectedEntity(
            text=ent.text,
            start=ent.start_char,
            end=ent.end_char,
            type=ent.label_,
            score=score,
            source="ner",
        )

        # Skip spans already covered by PatternScan
        if remove_span_overlap(candidate, existing):
            logger.debug(
                f"[EntityTrace] Skipping '{ent.text}' — overlaps with existing entity"
            )
            continue

        if score >= ENTITY_TRACE_HIGH_THRESHOLD:
            confirmed.append(candidate)
            logger.debug(
                f"[EntityTrace] Confirmed: '{ent.text}' ({ent.label_}, {score:.2f})"
            )
        elif score >= ENTITY_TRACE_LOW_THRESHOLD:
            borderline.append(candidate)
            logger.debug(
                f"[EntityTrace] Borderline: '{ent.text}' ({ent.label_}, {score:.2f})"
            )
        else:
            logger.debug(
                f"[EntityTrace] Discarded (low score): '{ent.text}' ({score:.2f})"
            )

    logger.info(
        f"[EntityTrace] confirmed={len(confirmed)}, borderline={len(borderline)}"
    )
    return confirmed, borderline
