# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
detection/context_guard.py — ContextGuard

NER-based detection of named entities using a local HuggingFace model
(dslim/distilbert-NER, ~250 MB).

This module detects named entities in the text that PatternScan and
EntityTrace missed.  It does NOT decide whether a geographic entity is
PII — that decision is made in detection/logic.py by analysing the
entity type co-occurrence across the full set of detected entities.

Tokenization artefact handling:
  distilbert word-piece tokenisation sometimes produces tokens like ". Sun"
  (a period attached to the next word when "Dr. Sun" is split) or "##wick"
  (a subword continuation prefix).  Both are stripped before emitting entities.
  Titles and very short tokens are also filtered to avoid false positives
  like "Dr" alone or "DE" being detected as PERSON/ORG.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from config import (
    CONTEXT_GUARD_MODEL,
    CONTEXT_GUARD_DEVICE,
    CONTEXT_GUARD_ENABLED,
    CONTEXT_GUARD_CONFIDENCE_THRESHOLD,
)
from util import DetectedEntity, get_logger

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
            device=CONTEXT_GUARD_DEVICE,
        )
        logger.info(
            f"[ContextGuard] Loaded NER model: {CONTEXT_GUARD_MODEL} "
            f"(device={CONTEXT_GUARD_DEVICE})"
        )
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


_LABEL_MAP = {
    "PER":    "PERSON",
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "LOC":    "LOC",
    "GPE":    "GPE",
    "MISC":   "MISC",
}

_KEEP_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE"}

# Titles and short tokens distilbert commonly fires on incorrectly.
# These are NEVER meaningful PII on their own.
_CG_BLOCKLIST: frozenset = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "professor", "rev", "sr", "jr",
    "sir", "lord", "dame", "capt", "lt", "sgt", "col", "gen",
    "de", "le", "la", "el", "al", "van", "von",
    # generic tech/domain acronyms the NER model mistakes for ORGs.
    # (ssh/ux/sql intentionally excluded — they occur as real ORG names.)
    "api", "ip", "sti", "std", "url", "http", "https",
    "faq", "crm", "pdf", "dns", "vpn", "bear", "bearer",
    # contact-channel labels ("Mobile +91…", "Cell: …") — never entities
    "mobile", "cell", "tel", "fax",
    # document-field labels NER mistakes for entities ("VIN 1HGB…")
    "vin", "mrn", "ktn", "er", "agi",
    # role nouns fragment-expansion can surface ("Client", "Batch", "Median")
    "client", "customer", "vendor", "supplier", "contact", "batch", "median",
    "user", "patient", "member",
    # form-field labels ("Name - X, Email - Y" intake rows)
    "email", "phone", "name", "dob", "ssn", "address",
})


def _clean_token(raw: str) -> str:
    """
    Strip HuggingFace word-piece artefacts and leading punctuation.

    Examples:
        "##wick"  → "wick"       (subword continuation prefix)
        ". Sun"   → "Sun"        (leading period from "Dr. Sun" split)
        " Smith"  → "Smith"      (leading whitespace)
    """
    text = raw.replace("##", "")
    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
    return text.strip()


def guard(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run NER on remaining_text and verify borderline_entities.

    Returns entities as detected.  The decision of whether a geographic
    entity is PII in context is deferred to detection/logic.py which
    analyses the full entity set across the whole message.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace.
        borderline_entities: Entities EntityTrace was uncertain about.

    Returns:
        Tuple of (confirmed_entities, needs_user_confirmation_entities).
    """
    confirmed: List[DetectedEntity] = []
    uncertain: List[DetectedEntity] = []

    # Verify borderline entities from EntityTrace against the threshold
    for ent in borderline_entities:
        if ent.score >= CONTEXT_GUARD_CONFIDENCE_THRESHOLD:
            confirmed.append(ent)
            logger.debug(
                f"[ContextGuard] Verified borderline: {ent.text!r} "
                f"({ent.type}, score={ent.score:.2f})"
            )
        else:
            uncertain.append(ent)

    # Run NER on remaining text.  █ placeholders become spaces; the text is
    # deliberately NOT stripped so model offsets stay aligned with
    # remaining_text (stripping used to shift every span left).
    clean = remaining_text.replace("█", " ")
    if not clean.strip():
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

        # Strip word-piece artefacts and leading punctuation
        raw_word = r.get("word", "")
        text = _clean_token(raw_word)

        # Minimum 3 characters after cleaning (blocks "Dr", "DE", "Mr", etc.)
        if len(text) < 3:
            logger.debug(
                f"[ContextGuard] Skipping too-short token: {raw_word!r} → {text!r}"
            )
            continue

        # Title / abbreviation blocklist
        if text.lower() in _CG_BLOCKLIST:
            logger.debug(f"[ContextGuard] Skipping blocklisted token: {text!r}")
            continue

        # A real name never spans a line break
        if "\n" in text:
            logger.debug(f"[ContextGuard] Skipping newline-spanning: {text!r}")
            continue

        # Word-boundary alignment: word-piece aggregation sometimes emits
        # fragments of longer words ("ri Nkosi" from "Zuberi Nkosi",
        # "lient" from "Client").  A fragment is real signal with wrong
        # boundaries — EXPAND it to the enclosing words, then re-screen.
        start = int(r.get("start", 0))
        end   = int(r.get("end", len(text)))
        idx = clean.find(text, max(0, start - 2), end + 2)
        if idx != -1:
            start, end = idx, idx + len(text)
            grew = False
            while start > 0 and clean[start - 1].isalnum() and (idx - start) < 12:
                start -= 1
                grew = True
            while (end < len(clean) and clean[end].isalnum()
                   and (end - (idx + len(text))) < 12):
                end += 1
                grew = True
            if grew:
                text = clean[start:end].strip()
                logger.debug(
                    f"[ContextGuard] Expanded fragment {raw_word!r} → {text!r}"
                )
                if len(text) < 3 or text.lower() in _CG_BLOCKLIST:
                    continue

        entity = DetectedEntity(
            text=text,
            start=start,
            end=end,
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