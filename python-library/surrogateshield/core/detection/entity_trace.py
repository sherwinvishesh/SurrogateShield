"""
detection/entity_trace.py — EntityTrace

spaCy-based Named Entity Recognition (NER) detection.

Loads the configured spaCy model and extracts PERSON, GPE, LOC, ORG, FAC
entities. Skips any span already covered by PatternScan results.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..entities import DetectedEntity, remove_span_overlap

logger = logging.getLogger(__name__)

# Cache loaded models by model name string
_nlp: Dict[str, object] = {}


def _get_nlp(model_name: str = "en_core_web_lg"):
    global _nlp
    if model_name in _nlp:
        return _nlp[model_name]
    try:
        import spacy
        try:
            model = spacy.load(model_name)
        except OSError:
            # Model not installed — download it on first use (~750 MB, cached after)
            logger.info(
                f"[EntityTrace] spaCy model '{model_name}' not found. "
                "Downloading — this happens once and is cached locally."
            )
            from spacy.cli import download as _spacy_download
            _spacy_download(model_name)
            model = spacy.load(model_name)
        _nlp[model_name] = model
        logger.info(f"[EntityTrace] Loaded spaCy model: {model_name}")
    except ImportError:
        logger.error(
            "[EntityTrace] spaCy is not installed. Run: pip install spacy"
        )
        _nlp[model_name] = None
    except Exception as exc:
        logger.error(
            f"[EntityTrace] Failed to load or download spaCy model '{model_name}': {exc}"
        )
        _nlp[model_name] = None
    return _nlp[model_name]


_TARGET_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}

_ENTITY_BLOCKLIST = {
    "ssn", "dob", "pin", "id", "uid", "email", "phone", "fax",
    "address", "zip", "postcode", "passport", "iban", "bic",
    "cvv", "cvc", "expiry",
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    "am", "pm", "gmt", "utc", "est", "pst",
}

_LOCATION_PREPS = {
    "in", "near",
    "live", "lives", "lived",
    "grew", "born", "raised",
    "moved", "relocate", "relocated",
    "residing", "reside",
    "hometown", "birthplace", "based",
}


def trace(
    text: str,
    existing_entities: Optional[List[DetectedEntity]] = None,
    spacy_model: str = "en_core_web_lg",
    high_threshold: float = 0.85,
    low_threshold: float = 0.60,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run spaCy NER on *text* and return confirmed and borderline entities.

    Args:
        text:              Text to analyse.
        existing_entities: Already-confirmed entities (avoid overlap).
        spacy_model:       Name of the spaCy model to load.
        high_threshold:    Score at or above which an entity is confirmed.
        low_threshold:     Score at or above which an entity is borderline.

    Returns:
        Tuple of (confirmed_entities, borderline_entities).
    """
    existing = existing_entities or []
    confirmed: List[DetectedEntity] = []
    borderline: List[DetectedEntity] = []

    nlp = _get_nlp(spacy_model)
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

        if ent.text.lower().strip() in _ENTITY_BLOCKLIST:
            logger.debug(f"[EntityTrace] Skipping blocklisted token: {ent.text!r}")
            continue

        _TYPE_DEFAULTS = {
            "PERSON": 0.88,
            "GPE":    0.85,
            "ORG":    0.85,
            "LOC":    0.74,
            "FAC":    0.70,
        }
        score: float = getattr(ent, "score_", None)
        if score is None:
            score = _TYPE_DEFAULTS.get(ent.label_, 0.80)

        effective_label = ent.label_
        if ent.label_ == "ORG":
            context_before = text[max(0, ent.start_char - 50): ent.start_char].lower()
            if _LOCATION_PREPS & set(context_before.split()):
                effective_label = "GPE"
                score = _TYPE_DEFAULTS["GPE"]
                logger.debug(f"[EntityTrace] Reclassified ORG→GPE: {ent.text!r}")

        candidate = DetectedEntity(
            text=ent.text,
            start=ent.start_char,
            end=ent.end_char,
            type=effective_label,
            score=score,
            source="ner",
        )

        if remove_span_overlap(candidate, existing):
            logger.debug(f"[EntityTrace] Skipping '{ent.text}' — overlaps existing")
            continue

        if score >= high_threshold:
            confirmed.append(candidate)
            logger.debug(f"[EntityTrace] Confirmed: '{ent.text}' ({effective_label}, {score:.2f})")
        elif score >= low_threshold:
            borderline.append(candidate)
            logger.debug(f"[EntityTrace] Borderline: '{ent.text}' ({effective_label}, {score:.2f})")
        else:
            logger.debug(f"[EntityTrace] Discarded (low score): '{ent.text}' ({score:.2f})")

    logger.info(f"[EntityTrace] confirmed={len(confirmed)}, borderline={len(borderline)}")
    return confirmed, borderline
