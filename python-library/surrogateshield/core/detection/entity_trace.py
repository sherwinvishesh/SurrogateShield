"""
detection/entity_trace.py — EntityTrace

spaCy-based Named Entity Recognition (NER) detection.

Loads the configured spaCy model and extracts PERSON, GPE, LOC, ORG, FAC
entities. Skips any span already covered by PatternScan results.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

from ..consistency import strip_clitic
from ..entities import DetectedEntity, remove_span_overlap
from ..errors import DetectorUnavailable

logger = logging.getLogger(__name__)

# Cache loaded models (or the reason loading failed) by model name
_nlp: Dict[str, object] = {}
_nlp_lock = threading.Lock()


def _get_nlp(model_name: str = "en_core_web_lg"):
    """Return the loaded spaCy model; raise DetectorUnavailable if it cannot
    be loaded (audit I17: never skip the NER stage silently)."""
    cached = _nlp.get(model_name)
    if cached is None:
        with _nlp_lock:
            cached = _nlp.get(model_name)
            if cached is None:
                cached = _nlp[model_name] = _load_nlp(model_name)
    if isinstance(cached, DetectorUnavailable):
        raise cached
    return cached


def _load_nlp(model_name: str):
    try:
        import spacy
    except ImportError:
        return DetectorUnavailable(
            "EntityTrace needs spaCy, which is not installed. Run: pip install spacy"
        )
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Never download from library code (audit I10): tell the user how.
        return DetectorUnavailable(
            f"EntityTrace needs the spaCy model {model_name!r}, which is not "
            f"installed. Run: python -m spacy download {model_name}"
        )
    logger.info(f"[EntityTrace] Loaded spaCy model: {model_name}")
    return nlp


_TARGET_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}

_ENTITY_BLOCKLIST = {
    "ssn", "dob", "pin", "id", "uid", "email", "phone", "fax",
    "address", "zip", "postcode", "passport", "iban", "bic",
    "cvv", "cvc", "expiry",
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr",
    "mon", "tue", "wed", "thu", "fri", "sat",
    # "sun" intentionally NOT here — it's a real surname ("Dr. Sun")
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    "am", "pm", "gmt", "utc", "est", "pst",
    # generic tech/domain acronyms spaCy mislabels as ORG/GPE.
    # (ssh/ux/sql are intentionally NOT here — they occur as real ORG names.)
    "api", "ip", "url", "http", "https", "sti", "std", "faq",
    "crm", "pdf", "dns", "vpn", "cpu", "gpu", "html", "css",
    "bearer", "bear", "token", "auth",
    # contact-channel labels ("Mobile +91…", "Cell: …") — never entities
    "mobile", "cell", "tel",
    # document-field labels NER mistakes for entities ("VIN 1HGB…")
    "vin", "mrn", "ktn", "er", "agi",
}

_LOCATION_PREPS = {
    "in", "near",
    "live", "lives", "lived",
    "grew", "born", "raised",
    "moved", "relocate", "relocated",
    "residing", "reside",
    "hometown", "birthplace", "based",
}


# Clause punctuation never belongs to a name; an English model run on CJK
# text otherwise returns spans such as "…收件人：李娜，邮箱：" (audit I4).
_EDGE_PUNCT = set(" \t,;:!?\"“”«»()[]{}，。：；！？、（）「」『』《》【】")


def _clean_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Trim edge punctuation and a possessive clitic ("Mia Lopez's" →
    "Mia Lopez", "Jennings'" → "Jennings") from an NER span (audit I4)."""
    while start < end and text[start] in _EDGE_PUNCT:
        start += 1
    while end > start and text[end - 1] in _EDGE_PUNCT:
        end -= 1
    end = start + len(strip_clitic(text[start:end]))
    return start, end


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
    try:
        doc = nlp(text)
    except Exception as exc:
        # fail closed: an input spaCy cannot process is not "no entities"
        raise DetectorUnavailable(f"EntityTrace failed on this input: {exc!r}") from exc

    for ent in doc.ents:
        if ent.label_ not in _TARGET_LABELS:
            continue

        if ent.text.lower().strip() in _ENTITY_BLOCKLIST:
            logger.debug(f"[EntityTrace] Skipping blocklisted token: {ent.text!r}")
            continue

        # A real name never spans a line break — entities that do are
        # artifacts of form-style text ("DESHAWN M\nADDRESS").
        if "\n" in ent.text:
            logger.debug(f"[EntityTrace] Skipping newline-spanning: {ent.text!r}")
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

        start, end = _clean_span(text, ent.start_char, ent.end_char)
        if start >= end:
            continue
        candidate = DetectedEntity(
            text=text[start:end],
            start=start,
            end=end,
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
