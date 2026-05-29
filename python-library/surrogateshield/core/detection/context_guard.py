"""
detection/context_guard.py — ContextGuard

NER-based detection of named entities using a local HuggingFace model
(dslim/distilbert-NER by default, ~250 MB).

This module detects named entities in the text that PatternScan and
EntityTrace missed.  It does NOT decide whether a geographic entity is
PII — that decision is made in detection/pipeline.py.

Tokenization artefact handling:
  distilbert word-piece tokenisation sometimes produces tokens like ". Sun"
  or "##wick".  Both are stripped before emitting entities.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from ..entities import DetectedEntity

logger = logging.getLogger(__name__)

# Cache pipelines keyed by model name
_ner_pipelines: Dict[str, object] = {}


def _get_ner(model_name: str = "dslim/distilbert-NER"):
    """Lazy-load and cache the HuggingFace NER pipeline by model name."""
    global _ner_pipelines
    if model_name in _ner_pipelines:
        return _ner_pipelines[model_name]
    try:
        from transformers import pipeline as hf_pipeline
        pipeline = hf_pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=-1,
        )
        _ner_pipelines[model_name] = pipeline
        logger.info(f"[ContextGuard] Loaded NER model: {model_name}")
    except ImportError:
        logger.warning(
            "[ContextGuard] transformers not installed — skipping. "
            "Run: pip install transformers torch"
        )
        _ner_pipelines[model_name] = None
    except Exception as exc:
        logger.warning(f"[ContextGuard] Failed to load NER model: {exc}")
        _ner_pipelines[model_name] = None
    return _ner_pipelines[model_name]


_LABEL_MAP = {
    "PER":    "PERSON",
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "LOC":    "LOC",
    "GPE":    "GPE",
    "MISC":   "MISC",
}

_KEEP_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE"}

_CG_BLOCKLIST: frozenset = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "professor", "rev", "sr", "jr",
    "sir", "lord", "dame", "capt", "lt", "sgt", "col", "gen",
    "de", "le", "la", "el", "al", "van", "von",
})


def _clean_token(raw: str) -> str:
    """Strip HuggingFace word-piece artefacts and leading punctuation."""
    text = raw.replace("##", "")
    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
    return text.strip()


def guard(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
    model_name: str = "dslim/distilbert-NER",
    enabled: bool = True,
    confidence_threshold: float = 0.70,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run NER on remaining_text and verify borderline_entities.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace.
        borderline_entities: Entities EntityTrace was uncertain about.
        model_name:          HuggingFace model to use for NER inference.
        enabled:             If False, skip NER inference entirely.
        confidence_threshold: Minimum score to promote a borderline entity.

    Returns:
        Tuple of (confirmed_entities, uncertain_entities).
    """
    confirmed: List[DetectedEntity] = []
    uncertain: List[DetectedEntity] = []

    # Verify borderline entities from EntityTrace against the threshold
    for ent in borderline_entities:
        if ent.score >= confidence_threshold:
            confirmed.append(ent)
            logger.debug(
                f"[ContextGuard] Verified borderline: {ent.text!r} "
                f"({ent.type}, score={ent.score:.2f})"
            )
        else:
            uncertain.append(ent)

    if not enabled:
        return confirmed, uncertain

    # Run NER on remaining text
    clean = remaining_text.replace("█", " ").strip()
    if not clean:
        return confirmed, uncertain

    ner = _get_ner(model_name)
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

        raw_word = r.get("word", "")
        text = _clean_token(raw_word)

        if len(text) < 3:
            logger.debug(
                f"[ContextGuard] Skipping too-short token: {raw_word!r} → {text!r}"
            )
            continue

        if text.lower() in _CG_BLOCKLIST:
            logger.debug(f"[ContextGuard] Skipping blocklisted token: {text!r}")
            continue

        entity = DetectedEntity(
            text=text,
            start=r.get("start", 0),
            end=r.get("end", len(text)),
            type=entity_type,
            score=score,
            source="slm",
        )

        if score >= confidence_threshold:
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
