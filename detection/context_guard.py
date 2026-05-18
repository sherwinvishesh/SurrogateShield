"""
detection/context_guard.py — ContextGuard

NER-based detection of named entities using a local HuggingFace model
(dslim/distilbert-NER, ~250 MB). Replaces the previous Ollama phi3:mini
implementation — no external server required.

Geographic pass-through:
  The same GEO_PASS_THROUGH whitelist used by EntityTrace is applied here —
  US states, major countries, and major cities are never emitted as entities
  worth replacing, even when distilbert detects them with high confidence.

Tokenization artefact handling:
  distilbert word-piece tokenisation sometimes produces tokens starting with
  ". " or "##".  This module strips such artefacts and applies a blocklist of
  titles / short tokens (e.g. "Dr" alone, "DE", "Mr") that are never PII.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from config import (
    CONTEXT_GUARD_MODEL,
    CONTEXT_GUARD_ENABLED,
    CONTEXT_GUARD_CONFIDENCE_THRESHOLD,
)
from util import DetectedEntity, get_logger
from detection.geo_data import GEO_PASS_THROUGH

logger = get_logger(__name__)

_ner_pipeline = None


def _get_ner():
    """Lazy-load and cache the HuggingFace NER pipeline."""
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _ner_pipeline = hf_pipeline(
            "ner",
            model=CONTEXT_GUARD_MODEL,
            aggregation_strategy="simple",
            device=-1,
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

_LABEL_MAP = {
    "PER":    "PERSON",
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "LOC":    "LOC",
    "GPE":    "GPE",
    "MISC":   "MISC",
}

_KEEP_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE"}

# Titles and short tokens distilbert frequently fires on incorrectly
_CG_BLOCKLIST: frozenset = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "professor", "rev", "sr", "jr",
    "sir", "lord", "dame", "capt", "lt", "sgt", "col", "gen",
    "de", "le", "la", "el", "al", "van", "von",
})


def _clean_token(raw: str) -> str:
    """
    Strip HuggingFace word-piece artefacts and leading punctuation.

    Examples:
        "##wick"  → "wick"
        ". Sun"   → "Sun"    (leading period from "Dr. Sun" split)
        " Smith"  → "Smith"
    """
    text = raw.replace("##", "")
    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
    return text.strip()


# ─────────────────────────────────────────────
# Main guard function
# ─────────────────────────────────────────────

def guard(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run NER on remaining_text and verify borderline_entities.

    Geographic pass-through: GPE/LOC entities matching GEO_PASS_THROUGH
    (US states, major countries, major cities) are silently dropped — they
    are not PII and replacing them destroys answer utility.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace.
        borderline_entities: Entities EntityTrace was uncertain about.

    Returns:
        Tuple of (confirmed_entities, needs_user_confirmation_entities).
    """
    confirmed: List[DetectedEntity] = []
    uncertain: List[DetectedEntity] = []

    # ── Verify borderline entities from EntityTrace ───────────────────────────
    for ent in borderline_entities:
        # Apply geo whitelist to borderline entities too
        if ent.type in {"GPE", "LOC"} and ent.text.lower().strip() in GEO_PASS_THROUGH:
            logger.debug(
                f"[ContextGuard] Borderline entity is on geo whitelist, skipping: {ent.text!r}"
            )
            continue
        if ent.score >= CONTEXT_GUARD_CONFIDENCE_THRESHOLD:
            confirmed.append(ent)
            logger.debug(
                f"[ContextGuard] Verified borderline: {ent.text!r} ({ent.type}, "
                f"score={ent.score:.2f})"
            )
        else:
            uncertain.append(ent)

    # ── Run NER on remaining text ─────────────────────────────────────────────
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

        # Clean subword artefacts and leading punctuation
        raw_word = r.get("word", "")
        text = _clean_token(raw_word)

        # Require at least 3 characters after cleaning
        if len(text) < 3:
            logger.debug(f"[ContextGuard] Skipping too-short entity: {raw_word!r} → {text!r}")
            continue

        # Blocklist check — titles and common abbreviations
        if text.lower() in _CG_BLOCKLIST:
            logger.debug(f"[ContextGuard] Skipping blocklisted entity: {text!r}")
            continue

        # Geographic pass-through — US states, major countries, major cities
        if entity_type in {"GPE", "LOC"} and text.lower() in GEO_PASS_THROUGH:
            logger.debug(
                f"[ContextGuard] Skipping broad geo entity (whitelist): {text!r}"
            )
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
            logger.debug(f"[ContextGuard] Confirmed: {text!r} ({entity_type}, {score:.2f})")
        else:
            uncertain.append(entity)
            logger.debug(f"[ContextGuard] Uncertain: {text!r} ({entity_type}, {score:.2f})")

    logger.info(
        f"[ContextGuard] confirmed={len(confirmed)}, uncertain={len(uncertain)}"
    )
    return confirmed, uncertain