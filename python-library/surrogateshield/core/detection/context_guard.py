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

import os
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

from ..entities import DetectedEntity
from ..errors import DetectorUnavailable

logger = logging.getLogger(__name__)

# Cache pipelines (or the reason loading failed) keyed by (model, device, revision)
_ner_pipelines: Dict[object, object] = {}
_ner_lock = threading.Lock()


def _get_ner(model_name: str = "dslim/distilbert-NER", device: int = -1,
             revision: Optional[str] = None):
    """Lazy-load and cache the HuggingFace NER pipeline by (model, device,
    revision).

    Raises DetectorUnavailable if it cannot be loaded (audit I17): an
    enabled stage is never skipped silently.
    """
    cache_key = (model_name, device, revision)
    cached = _ner_pipelines.get(cache_key)
    if cached is None:
        with _ner_lock:
            cached = _ner_pipelines.get(cache_key)
            if cached is None:
                cached = _ner_pipelines[cache_key] = _load_ner(model_name, device, revision)
    if isinstance(cached, DetectorUnavailable):
        raise cached
    return cached


def _local_model(model_name: str, revision: Optional[str] = None) -> str:
    """The cached copy of *model_name* (at *revision*, a commit or branch;
    None is the default branch), so a load makes no network request (audit
    N7). Only a model that is not cached yet is fetched, once."""
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name
    try:
        return snapshot_download(model_name, revision=revision, local_files_only=True)
    except OSError:         # not cached: LocalEntryNotFoundError
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
            return model_name   # the pipeline raises; _load_ner reports it
        return snapshot_download(model_name, revision=revision)


def _load_ner(model_name: str, device: int, revision: Optional[str] = None):
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return DetectorUnavailable(
            "ContextGuard needs transformers and torch, which are not installed. "
            "Run: pip install transformers torch — or switch ContextGuard off."
        )
    try:
        pipeline = hf_pipeline(
            "ner",
            model=_local_model(model_name, revision),
            aggregation_strategy="simple",
            device=device,
        )
    except (OSError, ValueError) as exc:
        return DetectorUnavailable(
            f"ContextGuard could not load {model_name!r} ({exc}). Download it once "
            "with network access (HF_HUB_OFFLINE unset), or switch ContextGuard off."
        )
    logger.info(f"[ContextGuard] Loaded NER model: {model_name} (device={device})")
    return pipeline


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
    device: int = -1,
    revision: Optional[str] = None,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run NER on remaining_text and verify borderline_entities.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace.
        borderline_entities: Entities EntityTrace was uncertain about.
        model_name:          HuggingFace model to use for NER inference.
        enabled:             If False, skip NER inference entirely.
        confidence_threshold: Minimum score to promote a borderline entity.
        revision:            Model revision (commit) to load; None is the
                             default branch.

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

    # Run NER on remaining text.  █ placeholders become spaces; the text is
    # deliberately NOT stripped so model offsets stay aligned with
    # remaining_text (stripping used to shift every span left).
    clean = remaining_text.replace("█", " ")
    if not clean.strip():
        return confirmed, uncertain

    ner = _get_ner(model_name, device, revision)
    try:
        results = ner(clean)
    except Exception as exc:
        raise DetectorUnavailable(f"ContextGuard failed on this input: {exc!r}") from exc

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
        else:
            # The word is not in the text near its offsets (an [UNK] piece, a
            # normalised character): the offsets are the authority, so the
            # entity's text is what an edit at those offsets would replace.
            # A span across a line break ("Bartholomew Ekwueme\nCell") keeps
            # the line with most letters: an entity never spans a break.
            if not 0 <= start < end <= len(clean):
                continue
            lines = [(sum(c.isalpha() for c in m.group()), start + m.start(), start + m.end())
                     for m in re.finditer(r"[^\n]+", clean[start:end])]
            if not lines:
                continue
            _, start, end = max(lines, key=lambda x: x[0])
            span = clean[start:end]
            start += len(span) - len(span.lstrip())
            end = start + len(span.strip())
            text = clean[start:end]
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
