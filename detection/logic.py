"""
detection/logic.py — SentinelLayer

Cascade orchestration pipeline: PatternScan → EntityTrace → ContextGuard.

Geographic pass-through:
  The GEO_PASS_THROUGH whitelist (US states, major countries, major cities)
  is applied as a final filter on the confirmed entity list.  Even if a
  broad geographic entity slips through EntityTrace or ContextGuard, it is
  removed here before surrogates are generated.  This ensures "wyoming",
  "phoenix", "germany", etc. are never replaced regardless of query type.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from dataclasses import replace as _dc_replace
from util import DetectedEntity, get_logger, mask_spans
from detection import pattern_scan, entity_trace, context_guard
from detection.geo_data import GEO_PASS_THROUGH

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Location prepositions for ORG→GPE reclassification
# ─────────────────────────────────────────────────────────────────────────────
# "from" and "visit/visited" removed — caused false positives on company names.
_LOCATION_PREPS = {
    "in", "near",
    "live", "lives", "lived",
    "grew", "born", "raised",
    "moved", "relocate", "relocated",
    "residing", "reside",
    "hometown", "birthplace", "based",
}

# Entity types that count as geographic (filtered in service-query mode or by whitelist)
_GEO_TYPES = {"GPE", "LOC", "FAC"}


def _reclassify_location_orgs(
    entities: List[DetectedEntity],
    original_text: str,
) -> List[DetectedEntity]:
    """
    Reclassify ORG entities near residential prepositions as GPE.
    Post-reclassification, applies the geo whitelist so broad place names
    are not accidentally promoted into replaceable entities.
    """
    result = []
    for ent in entities:
        if ent.type == "ORG":
            ctx = original_text[max(0, ent.start - 50): ent.start].lower()
            if _LOCATION_PREPS & set(ctx.split()):
                ent = _dc_replace(ent, type="GPE", score=0.85)
                logger.debug(
                    f"[SentinelLayer] Reclassified ORG→GPE: {ent.text!r}"
                )
                # After reclassification, whitelist check
                if ent.text.lower().strip() in GEO_PASS_THROUGH:
                    logger.debug(
                        f"[SentinelLayer] Reclassified GPE on whitelist, dropping: {ent.text!r}"
                    )
                    continue
        result.append(ent)
    return result


def _apply_geo_whitelist(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """
    Remove any GPE/LOC entity whose text is in the geographic pass-through whitelist.

    This is the final safety net applied to the entire confirmed list before
    surrogate generation.  US states, major countries, and major cities are
    NEVER replaced regardless of how they entered the confirmed list.

    Args:
        entities: List of confirmed DetectedEntity objects.

    Returns:
        Filtered list with broad geographic entities removed.
    """
    filtered = []
    for ent in entities:
        if ent.type in {"GPE", "LOC"} and ent.text.lower().strip() in GEO_PASS_THROUGH:
            logger.debug(
                f"[SentinelLayer] Geo whitelist: dropping {ent.text!r} ({ent.type})"
            )
            continue
        filtered.append(ent)
    return filtered


def run_cascade(
    text: str,
    skip_values: Optional[Set[str]] = None,
    skip_location_entities: bool = False,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Execute the full three-stage SentinelLayer detection cascade.

    Geographic pass-through is applied as a final step on the confirmed list
    before returning.  This ensures US states, major countries, and major
    cities are never in the confirmed set, regardless of whether
    skip_location_entities is True or False.

    Args:
        text:                   Raw user message to analyse.
        skip_values:            Set of surrogate strings to ignore in PatternScan.
        skip_location_entities: When True, ALL geo entities (GPE/LOC/FAC) are
                                suppressed.  Used in service-query mode so even
                                small towns and specific addresses are not replaced
                                (the house number is fuzzed instead).

    Returns:
        Tuple of (confirmed_entities, needs_user_confirmation).
    """
    confirmed: List[DetectedEntity] = []
    needs_confirmation: List[DetectedEntity] = []

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
    )
    ner_confirmed  = _reclassify_location_orgs(ner_confirmed,  text)
    ner_borderline = _reclassify_location_orgs(ner_borderline, text)

    if skip_location_entities:
        ner_confirmed  = [e for e in ner_confirmed  if e.type not in _GEO_TYPES]
        ner_borderline = [e for e in ner_borderline if e.type not in _GEO_TYPES]
        logger.debug("[SentinelLayer] skip_location_entities=True — geo suppressed")

    confirmed.extend(ner_confirmed)
    remaining_text = mask_spans(remaining_text, ner_confirmed)

    # ── Stage 3: ContextGuard ─────────────────────────────────────────────────
    from config import CONTEXT_GUARD_ENABLED, ENTITY_TRACE_FALLBACK_THRESHOLD
    if CONTEXT_GUARD_ENABLED:
        logger.info("[SentinelLayer] Stage 3: ContextGuard")
        slm_confirmed, slm_uncertain = context_guard.guard(
            remaining_text=remaining_text,
            borderline_entities=ner_borderline,
        )
        if skip_location_entities:
            slm_confirmed = [e for e in slm_confirmed if e.type not in _GEO_TYPES]
            slm_uncertain = [e for e in slm_uncertain if e.type not in _GEO_TYPES]

        confirmed.extend(slm_confirmed)
        needs_confirmation.extend(slm_uncertain)
    else:
        promoted = [
            e for e in ner_borderline
            if e.score >= ENTITY_TRACE_FALLBACK_THRESHOLD
        ]
        if promoted:
            confirmed.extend(promoted)
            logger.debug(
                f"[SentinelLayer] Stage 3: disabled — promoted {len(promoted)} via fallback"
            )
        else:
            logger.debug("[SentinelLayer] Stage 3: ContextGuard disabled")

    # ── Final: geographic pass-through filter ─────────────────────────────────
    # Belt-and-suspenders: remove any broad geo entity that slipped through.
    # This catches cases where ContextGuard detected e.g. "Wyoming" as LOC.
    confirmed = _apply_geo_whitelist(confirmed)
    needs_confirmation = _apply_geo_whitelist(needs_confirmation)

    logger.info(
        f"[SentinelLayer] Final → "
        f"confirmed={len(confirmed)}, "
        f"needs_confirmation={len(needs_confirmation)}"
    )
    return confirmed, needs_confirmation


def deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """
    Remove duplicate entities by text value, keeping the highest-scored one.
    """
    seen: dict = {}
    for ent in entities:
        key = ent.text.strip()
        if key not in seen or ent.score > seen[key].score:
            seen[key] = ent
    result = list(seen.values())
    result.sort(key=lambda e: e.start)
    return result