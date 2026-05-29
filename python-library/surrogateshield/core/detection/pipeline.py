"""
detection/pipeline.py — SentinelLayer

Cascade: PatternScan → EntityTrace → ContextGuard, followed by four
post-processing passes.

Post-processing passes:
  Pass A — Structural ORG detection
  Pass B — Email-username → PERSON reclassification
  Pass C — PERSON component deduplication
  Pass D — Topical geo-entity filter
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from dataclasses import replace as _dc_replace
from ..entities import DetectedEntity, mask_spans
from . import pattern_scan, entity_trace, context_guard
from .quasi_identifier import score as qi_score

logger = logging.getLogger(__name__)


class _TaggedList(list):
    """list subclass that allows attribute assignment (used for _qi_matches)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# pii_off alias resolution
# ─────────────────────────────────────────────────────────────────────────────

_PII_OFF_ALIASES: Dict[str, Set[str]] = {
    "phone":       {"phone_us", "phone_uk", "phone_intl"},
    "postal_code": {"zip_us", "postcode_uk"},
    "zip":         {"zip_us"},
    "postcode":    {"postcode_uk"},
    "name":        {"PERSON"},
    "names":       {"PERSON"},
    "location":    {"GPE", "LOC"},
    "org":         {"ORG"},
    "facility":    {"FAC"},
    "crypto":      {"crypto"},
    "bank":        {"us_bank_number"},
    "license":     {"us_driver_license"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Pass A — Structural ORG detection
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURAL_ORG_PATTERN = re.compile(
    r'\b(?:the|a|an)\s+'
    r'([A-Za-z][A-Za-z]*(?:\s+[A-Za-z]+){0,3}?)'
    r'\s+(?:corporation|company|corp|inc|ltd|llc|group|firm|enterprise'
    r'|organization|organisation|associates|holdings|ventures|solutions)\b',
    re.IGNORECASE,
)


def _detect_structural_orgs(
    text: str,
    existing_entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    occupied = {(e.start, e.end) for e in existing_entities}
    new_ents: List[DetectedEntity] = []

    for m in _STRUCTURAL_ORG_PATTERN.finditer(text):
        name_text  = m.group(1).strip()
        name_start = m.start(1)
        name_end   = m.end(1)

        if any(not (name_end <= os or name_start >= oe) for os, oe in occupied):
            continue

        ent = DetectedEntity(
            text=name_text,
            start=name_start,
            end=name_end,
            type="ORG",
            score=0.90,
            source="pattern",
        )
        new_ents.append(ent)
        occupied.add((name_start, name_end))
        logger.debug(f"[SentinelLayer] Pass A structural ORG: {name_text!r}")

    return new_ents


# ─────────────────────────────────────────────────────────────────────────────
# Pass B — Email-username → PERSON reclassification
# ─────────────────────────────────────────────────────────────────────────────

def _reclassify_email_username_orgs(
    entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    email_usernames: Set[str] = set()
    for ent in entities:
        if ent.type == "email" and "@" in ent.text:
            email_usernames.add(ent.text.split("@")[0].lower())

    if not email_usernames:
        return entities

    result = []
    for ent in entities:
        if ent.type == "ORG" and len(ent.text) >= 3:
            ent_lower = ent.text.lower()
            for username in email_usernames:
                if username.startswith(ent_lower):
                    ent = _dc_replace(ent, type="PERSON", score=0.88)
                    logger.debug(
                        f"[SentinelLayer] Pass B ORG→PERSON (email prefix): "
                        f"{ent.text!r}"
                    )
                    break
        result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass C — PERSON component deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_person_components(
    entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    persons = [e for e in entities if e.type == "PERSON"]
    others  = [e for e in entities if e.type != "PERSON"]

    to_remove: Set[str] = set()
    for short in persons:
        short_words = set(short.text.split())
        for long_ent in persons:
            if short.text != long_ent.text:
                long_words = set(long_ent.text.split())
                if short_words <= long_words:
                    to_remove.add(short.text)
                    logger.debug(
                        f"[SentinelLayer] Pass C removing component PERSON "
                        f"{short.text!r} (subset of {long_ent.text!r})"
                    )
                    break

    return others + [e for e in persons if e.text not in to_remove]


# ─────────────────────────────────────────────────────────────────────────────
# Pass D — Topical geo-entity filter
# ─────────────────────────────────────────────────────────────────────────────

_CLAUSE_SPLIT = re.compile(
    r'[.!?]+\s+'
    r'|\s+(?:and|or|but|however|yet|because|since|although|while|when'
    r'|therefore|whereas|unless|though|despite|nevertheless|so)\s+'
    r'|,\s+',
    re.IGNORECASE,
)

_QUERY_FRAME = re.compile(
    r'^\s*'
    r'(?:please\s+|could\s+you\s+(?:please\s+)?|can\s+you\s+(?:please\s+)?'
    r'|would\s+you\s+(?:please\s+)?)?'
    r'(?:'
    r'give\s+me|tell\s+me|show\s+me|find\s+me|help\s+me|get\s+me'
    r'|look\s+up|search\s+for|look\s+for|explain|list|describe'
    r'|summarize|summarise|compare|recommend|suggest|advise'
    r'|what\s+(?:is|are|was|were|would|can|do|does|did)'
    r'|how\s+(?:do|does|did|can|much|many|to|would)'
    r'|which\s+(?:is|are|was|were|would)'
    r'|where\s+(?:is|are|can|do|should|would)'
    r'|when\s+(?:is|are|was|were|do|does|did|can)'
    r'|who\s+(?:is|are|was|were|can|would|do|does)'
    r'|why\s+(?:is|are|was|were|do|does|did|would|should)'
    r'|is\s+there|are\s+there'
    r'|i\s+(?:want|need|would\s+like|\'d\s+like)\s+(?:to\s+know|to\s+find|'
    r'to\s+learn|to\s+understand|information|details|advice|help)'
    r')',
    re.IGNORECASE,
)

_GEO_FILTERABLE = {"GPE", "LOC"}


def _all_sub_clauses(text: str) -> List[str]:
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _contains_entity(entity_text: str, clause: str) -> bool:
    return entity_text.lower() in clause.lower()


def _is_proper_capitalized(entity_text: str, text: str) -> bool:
    if entity_text[0].isupper():
        return True

    idx = text.find(entity_text)
    if idx == -1:
        idx = text.lower().find(entity_text.lower())
    if idx == -1:
        return True

    prefix = text[:idx].rstrip()
    if not prefix or prefix[-1] in ".!?;":
        return True

    return False


def _filter_topical_geo_entities(
    entities: List[DetectedEntity],
    text: str,
) -> tuple:
    geo_ents   = [e for e in entities if e.type in _GEO_FILTERABLE]
    other_ents = [e for e in entities if e.type not in _GEO_FILTERABLE]

    if not geo_ents:
        return entities, []

    all_clauses     = _all_sub_clauses(text)
    clause_is_query = [bool(_QUERY_FRAME.match(c)) for c in all_clauses]

    skipped: List[DetectedEntity] = []
    result = list(other_ents)

    for geo_ent in geo_ents:
        if not _is_proper_capitalized(geo_ent.text, text):
            logger.debug(
                f"[SentinelLayer] Pass D: lowercase geo skipped (not proper noun): "
                f"{geo_ent.text!r}"
            )
            skipped.append(geo_ent)
            continue

        in_query    = False
        in_personal = False

        for i, clause in enumerate(all_clauses):
            if _contains_entity(geo_ent.text, clause):
                if clause_is_query[i]:
                    in_query = True
                else:
                    in_personal = True

        if in_query and not in_personal:
            logger.debug(
                f"[SentinelLayer] Pass D: topical geo (query-only): {geo_ent.text!r}"
            )
            skipped.append(geo_ent)
            continue

        logger.debug(
            f"[SentinelLayer] Pass D: geo kept (non-query context): {geo_ent.text!r}"
        )
        result.append(geo_ent)

    return result, skipped


# ─────────────────────────────────────────────────────────────────────────────
# ORG→GPE reclassification
# ─────────────────────────────────────────────────────────────────────────────

_LOCATION_PREPS = {
    "in", "near", "live", "lives", "lived", "grew", "born", "raised",
    "moved", "relocate", "relocated", "residing", "reside",
    "hometown", "birthplace", "based",
}

_GEO_TYPES = {"GPE", "LOC", "FAC"}


def _reclassify_location_orgs(
    entities: List[DetectedEntity],
    original_text: str,
) -> List[DetectedEntity]:
    result = []
    for ent in entities:
        if ent.type == "ORG":
            ctx = original_text[max(0, ent.start - 50): ent.start].lower()
            if _LOCATION_PREPS & set(ctx.split()):
                ent = _dc_replace(ent, type="GPE", score=0.85)
                logger.debug(f"[SentinelLayer] Reclassified ORG→GPE: {ent.text!r}")
        result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main cascade
# ─────────────────────────────────────────────────────────────────────────────

def run_cascade(
    text: str,
    skip_values: Optional[Set[str]] = None,
    skip_location_entities: bool = False,
    pii_off=None,
    spacy_model: str = "en_core_web_lg",
    context_guard_enabled: bool = True,
    entity_trace_high_threshold: float = 0.85,
    entity_trace_low_threshold: float = 0.60,
    context_guard_threshold: float = 0.70,
    entity_trace_fallback_threshold: float = 0.65,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Execute the full SentinelLayer cascade then apply post-processing passes.

    Args:
        text:                         Raw user message.
        skip_values:                  Surrogate strings to skip in PatternScan.
        skip_location_entities:       Suppress ALL geo entities (service-query mode).
        pii_off:                      List of PII type names/aliases to exclude.
        spacy_model:                  spaCy model name for EntityTrace.
        context_guard_enabled:        Whether to run ContextGuard NER inference.
        entity_trace_high_threshold:  Score threshold to auto-confirm NER entities.
        entity_trace_low_threshold:   Score threshold for borderline NER entities.
        context_guard_threshold:      Score threshold for ContextGuard confirmation.
        entity_trace_fallback_threshold: Promotion threshold when ContextGuard disabled.
    """
    confirmed: List[DetectedEntity] = []
    needs_confirmation: List[DetectedEntity] = []
    all_skipped: List[DetectedEntity] = []

    # ── Stage 1: PatternScan ──────────────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 1: PatternScan")
    pattern_results = pattern_scan.scan(text, skip_values=skip_values)
    confirmed.extend(pattern_results)
    remaining_text = mask_spans(text, pattern_results)

    # ── Stage 2: EntityTrace ──────────────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 2: EntityTrace")
    ner_confirmed, ner_borderline = entity_trace.trace(
        remaining_text,
        existing_entities=confirmed,
        spacy_model=spacy_model,
        high_threshold=entity_trace_high_threshold,
        low_threshold=entity_trace_low_threshold,
    )
    ner_confirmed  = _reclassify_location_orgs(ner_confirmed,  text)
    ner_borderline = _reclassify_location_orgs(ner_borderline, text)

    if skip_location_entities:
        ner_confirmed_filtered  = [e for e in ner_confirmed  if e.type in _GEO_TYPES]
        ner_borderline_filtered = [e for e in ner_borderline if e.type in _GEO_TYPES]
        ner_confirmed  = [e for e in ner_confirmed  if e.type not in _GEO_TYPES]
        ner_borderline = [e for e in ner_borderline if e.type not in _GEO_TYPES]
        all_skipped.extend(ner_confirmed_filtered)
        all_skipped.extend(ner_borderline_filtered)

    confirmed.extend(ner_confirmed)
    remaining_text = mask_spans(remaining_text, ner_confirmed)

    # ── Stage 3: ContextGuard ─────────────────────────────────────────────────
    if context_guard_enabled:
        logger.info("[SentinelLayer] Stage 3: ContextGuard")
        slm_confirmed, slm_uncertain = context_guard.guard(
            remaining_text=remaining_text,
            borderline_entities=ner_borderline,
            model_name="dslim/distilbert-NER",
            enabled=True,
            confidence_threshold=context_guard_threshold,
        )
        if skip_location_entities:
            slm_confirmed = [e for e in slm_confirmed if e.type not in _GEO_TYPES]
            slm_uncertain = [e for e in slm_uncertain if e.type not in _GEO_TYPES]
        confirmed.extend(slm_confirmed)
        needs_confirmation.extend(slm_uncertain)
    else:
        promoted = [
            e for e in ner_borderline
            if e.score >= entity_trace_fallback_threshold
        ]
        if promoted:
            confirmed.extend(promoted)

    # ── Pass A: Structural ORG detection ─────────────────────────────────────
    structural_orgs = _detect_structural_orgs(text, confirmed)
    if structural_orgs:
        logger.info(
            f"[SentinelLayer] Pass A: +{len(structural_orgs)} structural ORG(s)"
        )
        confirmed.extend(structural_orgs)

    # ── Pass B: Email-username → PERSON reclassification ─────────────────────
    confirmed          = _reclassify_email_username_orgs(confirmed)
    needs_confirmation = _reclassify_email_username_orgs(needs_confirmation)

    # ── Pass C: PERSON component deduplication ────────────────────────────────
    confirmed          = _deduplicate_person_components(confirmed)
    needs_confirmation = _deduplicate_person_components(needs_confirmation)

    # ── Pass D: Topical geo-entity filter ─────────────────────────────────────
    if not skip_location_entities:
        confirmed,          skipped_confirmed = _filter_topical_geo_entities(confirmed,          text)
        needs_confirmation, skipped_nc        = _filter_topical_geo_entities(needs_confirmation, text)
        all_skipped = skipped_confirmed + skipped_nc

    # ── Quasi-identifier combination scoring ──────────────────────────────────
    confirmed = _TaggedList(confirmed)
    qi_matches = qi_score(confirmed)
    if qi_matches:
        for match in qi_matches:
            logger.info(
                f"[SentinelLayer] Quasi-ID risk: {match.combination_name} "
                f"(fields: {match.matched_fields}, risk: {match.risk_level})"
            )
    confirmed._qi_matches = qi_matches
    confirmed._skipped_entities = all_skipped

    # ── pii_off filtering ─────────────────────────────────────────────────────
    if pii_off:
        exclude_types: Set[str] = set()
        for item in pii_off:
            item_lower = item.lower()
            if item_lower in _PII_OFF_ALIASES:
                exclude_types.update(_PII_OFF_ALIASES[item_lower])
            else:
                exclude_types.add(item)

        old_qi      = confirmed._qi_matches
        old_skipped = confirmed._skipped_entities
        confirmed = _TaggedList([e for e in confirmed if e.type not in exclude_types])
        confirmed._qi_matches       = old_qi
        confirmed._skipped_entities = old_skipped

    logger.info(
        f"[SentinelLayer] Final → "
        f"confirmed={len(confirmed)}, "
        f"needs_confirmation={len(needs_confirmation)}"
    )
    return confirmed, needs_confirmation


def deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """Remove duplicate entities by text, keeping the highest-scored one."""
    seen: dict = {}
    for ent in entities:
        key = ent.text.strip()
        if key not in seen or ent.score > seen[key].score:
            seen[key] = ent
    result = _TaggedList(seen.values())
    result.sort(key=lambda e: e.start)
    if hasattr(entities, "_qi_matches"):
        result._qi_matches = entities._qi_matches
    if hasattr(entities, "_skipped_entities"):
        result._skipped_entities = entities._skipped_entities
    return result
