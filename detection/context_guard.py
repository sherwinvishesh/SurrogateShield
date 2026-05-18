"""
detection/context_guard.py — ContextGuard

NER-based detection of named entities using a local HuggingFace model
(dslim/distilbert-NER, ~250 MB). Replaces the previous Ollama phi3:mini
implementation — no external server required.

The model is downloaded from HuggingFace Hub on first use and cached
locally by the transformers library. Subsequent runs are fully offline.

ContextGuard is called ONLY with:
  1. Borderline entities from EntityTrace (to verify or upgrade)
  2. The remaining_text after PatternScan and EntityTrace have run

Graceful degradation: if transformers or torch are unavailable, logs a
warning and returns empty lists — the pipeline continues without crashing.
"""

from __future__ import annotations

from typing import List, Tuple

from config import (
    CONTEXT_GUARD_MODEL,
    CONTEXT_GUARD_ENABLED,
    CONTEXT_GUARD_CONFIDENCE_THRESHOLD,
)
from util import DetectedEntity, get_logger

logger = get_logger(__name__)

# Module-level cache — expensive to load, so we load it exactly once
# and reuse across all calls in a session.
_ner_pipeline = None


def _get_ner():
    """
    Lazy-load and cache the HuggingFace NER pipeline.

    Returns:
        The loaded pipeline callable, or None on failure.
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _ner_pipeline = hf_pipeline(
            "ner",
            model=CONTEXT_GUARD_MODEL,
            aggregation_strategy="simple",
            device=-1,   # CPU — change to 0 for GPU
        )
        logger.info(f"[ContextGuard] Loaded NER model: {CONTEXT_GUARD_MODEL}")
    except ImportError:
        logger.warning(
            "[ContextGuard] transformers not installed — skipping. "
            "Run: pip install transformers torch"
        )
        _ner_pipeline = None
    except Exception as exc:
        logger.warning(f"[ContextGuard] Failed to load NER model: {exc}")
        _ner_pipeline = None
    return _ner_pipeline


# ── Label mappings ────────────────────────────────────────────────────────────

# Map HuggingFace entity group labels to SurrogateShield types
_LABEL_MAP = {
    "PER":    "PERSON",
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "LOC":    "LOC",
    "GPE":    "GPE",
    "MISC":   "MISC",
}

# Entity types worth keeping (skip MISC — too noisy for privacy purposes)
_KEEP_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE"}


# ─────────────────────────────────────────────
# Main guard function
# ─────────────────────────────────────────────

def guard(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run NER on remaining_text and verify borderline_entities.

    Borderline entities from EntityTrace (score in [LOW, HIGH)) are
    re-evaluated using the confidence threshold: those that meet or
    exceed CONTEXT_GUARD_CONFIDENCE_THRESHOLD are confirmed, the rest
    remain uncertain.  The NER model then runs on any remaining text to
    catch entities missed by spaCy.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace
                             (may contain █ mask characters).
        borderline_entities: Entities EntityTrace was uncertain about.

    Returns:
        Tuple of (confirmed_entities, needs_user_confirmation_entities).
        Both lists contain DetectedEntity objects.
    """
    confirmed: List[DetectedEntity] = []
    uncertain: List[DetectedEntity] = []

    # ── Verify borderline entities from EntityTrace ───────────────────
    for ent in borderline_entities:
        if ent.score >= CONTEXT_GUARD_CONFIDENCE_THRESHOLD:
            confirmed.append(ent)
            logger.debug(
                f"[ContextGuard] Verified borderline: {ent.text!r} ({ent.type}, "
                f"score={ent.score:.2f})"
            )
        else:
            uncertain.append(ent)

    # ── Run NER on remaining text ─────────────────────────────────────
    # Replace mask placeholders with spaces so the model sees clean text
    clean = remaining_text.replace("█", " ").strip()
    if not clean:
        return confirmed, uncertain

    ner = _get_ner()
    if ner is None:
        return confirmed, uncertain

    try:
        results = ner(clean)
    except Exception as exc:
        logger.warning(f"[ContextGuard] NER inference failed: {exc}")
        return confirmed, uncertain

    for r in results:
        label = r.get("entity_group", r.get("entity", ""))
        if label not in _KEEP_LABELS:
            continue

        entity_type = _LABEL_MAP.get(label, label)
        score = float(r.get("score", 0.0))
        text = r.get("word", "").strip()
        if not text or len(text) < 2:
            continue

        # Remove HuggingFace subword tokenisation artefacts (## prefix)
        text = text.replace("##", "").strip()
        if not text:
            continue

        entity = DetectedEntity(
            text=text,
            start=r.get("start", 0),
            end=r.get("end", len(text)),
            type=entity_type,
            score=score,
            source="slm",
        )

        if score >= CONTEXT_GUARD_CONFIDENCE_THRESHOLD:
            confirmed.append(entity)
            logger.debug(
                f"[ContextGuard] Confirmed: {text!r} ({entity_type}, {score:.2f})"
            )
        else:
            uncertain.append(entity)
            logger.debug(
                f"[ContextGuard] Uncertain: {text!r} ({entity_type}, {score:.2f})"
            )

    logger.info(
        f"[ContextGuard] confirmed={len(confirmed)}, uncertain={len(uncertain)}"
    )
    return confirmed, uncertain