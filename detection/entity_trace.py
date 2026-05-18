"""
detection/entity_trace.py — EntityTrace

spaCy-based Named Entity Recognition (NER) detection.

Loads en_core_web_lg and extracts PERSON, GPE, LOC, ORG, FAC entities.
Skips any span already covered by PatternScan results.

Geographic pass-through:
  US states, major countries, and major world cities (population > ~500k)
  are NEVER detected as PII here.  They are too broad to be identifying
  under any k-anonymity threshold and replacing them destroys answer utility
  without any privacy gain.  See detection/geo_data.py for the full list.

Returns:
    confirmed  — entities with spaCy score >= ENTITY_TRACE_HIGH_THRESHOLD
    borderline — entities with ENTITY_TRACE_LOW_THRESHOLD <= score < HIGH
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from config import ENTITY_TRACE_HIGH_THRESHOLD, ENTITY_TRACE_LOW_THRESHOLD, SPACY_MODEL
from util import DetectedEntity, get_logger, remove_span_overlap
from detection.geo_data import GEO_PASS_THROUGH

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Lazy-load spaCy model (expensive; load once)
# ─────────────────────────────────────────────

_nlp = None


def _get_nlp():
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

# Tokens that spaCy frequently mis-classifies as named entities
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

# ─────────────────────────────────────────────────────────────────────────────
# Location prepositions for ORG→GPE reclassification
# ─────────────────────────────────────────────────────────────────────────────
# "from" and "visit/visited" were removed — they caused false positives on
# company names ("offer from Google", "site visit to Amazon").
_LOCATION_PREPS = {
    "in", "near",
    "live", "lives", "lived",
    "grew", "born", "raised",
    "moved", "relocate", "relocated",
    "residing", "reside",
    "hometown", "birthplace", "based",
}


# ─────────────────────────────────────────────
# Main trace function
# ─────────────────────────────────────────────

def trace(
    text: str,
    existing_entities: Optional[List[DetectedEntity]] = None,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run spaCy NER on *text* and return confirmed and borderline entities.

    Geographic pass-through: GPE/LOC entities whose text (case-insensitive)
    appears in GEO_PASS_THROUGH are silently skipped — US states, major
    countries, and major cities are never treated as PII.

    Args:
        text:              Text to analyse.
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

        # Common PII label words mis-classified as entities
        if ent.text.lower().strip() in _ENTITY_BLOCKLIST:
            logger.debug(f"[EntityTrace] Skipping blocklisted token: {ent.text!r}")
            continue

        # ── Geographic pass-through ─────────────────────────────────────────
        # US states, major countries, and major cities are NEVER PII.
        # Replacing "Wyoming" or "Phoenix" or "Germany" provides no privacy
        # benefit and destroys answer utility.
        if ent.label_ in {"GPE", "LOC"} and ent.text.lower().strip() in GEO_PASS_THROUGH:
            logger.debug(
                f"[EntityTrace] Skipping broad geo entity (whitelist): {ent.text!r}"
            )
            continue

        # Type-specific default confidence scores
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

        # ── Context-aware ORG→GPE reclassification ──────────────────────────
        # Catches informal city abbreviations spaCy labels as ORG
        # (e.g. "Phili", "Cali", "Frisco") near residential prepositions.
        effective_label = ent.label_
        if ent.label_ == "ORG":
            context_before = text[max(0, ent.start_char - 50): ent.start_char].lower()
            context_words  = set(context_before.split())
            if context_words & _LOCATION_PREPS:
                effective_label = "GPE"
                score = _TYPE_DEFAULTS["GPE"]
                logger.debug(
                    f"[EntityTrace] Reclassified ORG→GPE: {ent.text!r}"
                )
                # After reclassification, check the geo whitelist again
                if ent.text.lower().strip() in GEO_PASS_THROUGH:
                    logger.debug(
                        f"[EntityTrace] Reclassified entity is on whitelist, skipping: {ent.text!r}"
                    )
                    continue

        candidate = DetectedEntity(
            text=ent.text,
            start=ent.start_char,
            end=ent.end_char,
            type=effective_label,
            score=score,
            source="ner",
        )

        if remove_span_overlap(candidate, existing):
            logger.debug(
                f"[EntityTrace] Skipping '{ent.text}' — overlaps with existing entity"
            )
            continue

        if score >= ENTITY_TRACE_HIGH_THRESHOLD:
            confirmed.append(candidate)
            logger.debug(
                f"[EntityTrace] Confirmed: '{ent.text}' ({effective_label}, {score:.2f})"
            )
        elif score >= ENTITY_TRACE_LOW_THRESHOLD:
            borderline.append(candidate)
            logger.debug(
                f"[EntityTrace] Borderline: '{ent.text}' ({effective_label}, {score:.2f})"
            )
        else:
            logger.debug(
                f"[EntityTrace] Discarded (low score): '{ent.text}' ({score:.2f})"
            )

    logger.info(
        f"[EntityTrace] confirmed={len(confirmed)}, borderline={len(borderline)}"
    )
    return confirmed, borderline