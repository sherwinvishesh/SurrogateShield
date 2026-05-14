"""
detection/logic.py — SentinelLayer

Cascade orchestration pipeline that runs PatternScan → EntityTrace →
ContextGuard in order and returns a unified set of confirmed entities
plus any entities that need user confirmation.

This module contains ONLY the cascade logic. All regex, NER, and SLM
code lives in the three sibling modules it imports from.

Cascade flow:
    1. PatternScan runs on full message text → score=1.0 → confirmed
    2. PatternScan spans are masked from remaining_text
    3. EntityTrace runs on remaining_text
       - score >= HIGH_THRESHOLD → confirmed
       - LOW_THRESHOLD <= score < HIGH → borderline
       - score < LOW → discarded
    4. EntityTrace confirmed spans masked from remaining_text
    5. ContextGuard runs on:
       - borderline_entities (verify or upgrade to confirmed)
       - remaining_text after stages 1 and 2
    6. ContextGuard confirmed → add to confirmed_entities
       ContextGuard uncertain → needs_user_confirmation

Returns:
    confirmed_entities       — auto-replace these
    needs_user_confirmation  — ask the user before replacing
"""

from __future__ import annotations

from typing import List, Tuple

from dataclasses import replace as _dc_replace
from util import DetectedEntity, get_logger, mask_spans
from detection import pattern_scan, entity_trace, context_guard

# Location prepositions used for ORG→GPE reclassification
_LOCATION_PREPS = {
    "from", "in", "near", "live", "lives", "lived",
    "grew", "born", "moved", "relocate", "relocated",
    "residing", "reside", "hometown", "birthplace",
}


def _reclassify_location_orgs(
    entities: "List[DetectedEntity]",
    original_text: str,
) -> "List[DetectedEntity]":
    """
    Reclassify ORG entities that appear after location prepositions as GPE.

    spaCy often labels informal/abbreviated place names (e.g. "Phili",
    "Cali", "Frisco") as ORG. This causes MimicGen to generate a company
    surrogate instead of a city name. Checking the 50 characters before
    the entity for location prepositions catches these cases reliably.

    Args:
        entities:      List of DetectedEntity objects to check.
        original_text: The full original (unmasked) user message.

    Returns:
        New list with eligible ORG entities reclassified as GPE.
    """
    result = []
    for ent in entities:
        if ent.type == "ORG":
            ctx = original_text[max(0, ent.start - 50): ent.start].lower()
            if _LOCATION_PREPS & set(ctx.split()):
                ent = _dc_replace(ent, type="GPE", score=0.85)
                logger.debug(
                    f"[SentinelLayer] Reclassified ORG→GPE: {ent.text!r} "
                    f"(location preposition in context)"
                )
        result.append(ent)
    return result

logger = get_logger(__name__)


def run_cascade(text: str) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Execute the full three-stage SentinelLayer detection cascade.

    Args:
        text: Raw user message to analyse.

    Returns:
        Tuple of:
            confirmed_entities        — high-confidence, ready to replace
            needs_user_confirmation   — uncertain, present to user
    """
    confirmed: List[DetectedEntity] = []
    needs_confirmation: List[DetectedEntity] = []

    # ── Stage 1: PatternScan ─────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 1: PatternScan")
    pattern_results = pattern_scan.scan(text)
    confirmed.extend(pattern_results)

    # Mask pattern-matched spans so downstream stages don't double-detect
    remaining_text = mask_spans(text, pattern_results)

    # ── Stage 2: EntityTrace ─────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 2: EntityTrace")
    ner_confirmed, ner_borderline = entity_trace.trace(
        remaining_text,
        existing_entities=confirmed,
    )
    # Reclassify ORG entities that follow location prepositions as GPE.
    # Uses the ORIGINAL text (not remaining_text) so masking doesn't hide
    # the prepositions. This is the authoritative fix — independent of
    # entity_trace.py's own reclassification logic.
    ner_confirmed  = _reclassify_location_orgs(ner_confirmed,  text)
    ner_borderline = _reclassify_location_orgs(ner_borderline, text)
    confirmed.extend(ner_confirmed)

    # Mask NER-confirmed spans from remaining text
    remaining_text = mask_spans(remaining_text, ner_confirmed)

    # ── Stage 3: ContextGuard ────────────────────────────────────────
    from config import CONTEXT_GUARD_ENABLED, ENTITY_TRACE_FALLBACK_THRESHOLD
    if CONTEXT_GUARD_ENABLED:
        logger.info("[SentinelLayer] Stage 3: ContextGuard")
        slm_confirmed, slm_uncertain = context_guard.guard(
            remaining_text=remaining_text,
            borderline_entities=ner_borderline,
        )
        confirmed.extend(slm_confirmed)
        needs_confirmation.extend(slm_uncertain)
    else:
        # ContextGuard is disabled — promote borderline entities that scored
        # above the fallback threshold rather than silently discarding them.
        # Without this, LOC (0.74) and FAC (0.70) are always dropped in the
        # default configuration, leaving location names unprotected.
        promoted = [
            e for e in ner_borderline
            if e.score >= ENTITY_TRACE_FALLBACK_THRESHOLD
        ]
        if promoted:
            confirmed.extend(promoted)
            logger.debug(
                f"[SentinelLayer] Stage 3: ContextGuard disabled — "
                f"promoted {len(promoted)} borderline entities via fallback threshold"
            )
        else:
            logger.debug("[SentinelLayer] Stage 3: ContextGuard disabled")

    logger.info(
        f"[SentinelLayer] Final → "
        f"confirmed={len(confirmed)}, "
        f"needs_confirmation={len(needs_confirmation)}"
    )
    return confirmed, needs_confirmation


def deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """
    Remove duplicate entities by text value, keeping the highest-scored one.

    Called after user approvals are merged into confirmed_entities to
    ensure no text value appears twice in the replacement map.

    Args:
        entities: List of potentially duplicate DetectedEntity objects.

    Returns:
        Deduplicated list sorted by start position.
    """
    seen: dict = {}
    for ent in entities:
        key = ent.text.strip()
        if key not in seen or ent.score > seen[key].score:
            seen[key] = ent
    result = list(seen.values())
    result.sort(key=lambda e: e.start)
    return result