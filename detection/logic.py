"""
detection/logic.py — SentinelLayer

Cascade: PatternScan → EntityTrace → ContextGuard, followed by four
model-output-driven post-processing passes.

Post-processing passes (all use model outputs + linguistic structure,
no keyword lists):

  Pass A — Structural ORG detection
      Regex: "[the/a/an] <name> [corporation|company|corp|inc|ltd|llc…]"
      emits <name> as ORG.  The organisational suffix is the signal that
      the preceding word is a company name, regardless of capitalisation.
      This is structural detection (like PatternScan's address pattern),
      not a list of company names.

  Pass B — Email-username → PERSON reclassification
      If an ORG entity's text is a prefix of a detected email username
      (e.g. "Sherwin" is prefix of "sherwinvishesh"), the NER mis-labelling
      is corrected to PERSON.  Decision is made from pattern-detection output.

  Pass C — PERSON component deduplication
      When entity A ("Mitchell") is a word-component of entity B
      ("Sarah Mitchell"), both PERSON, A is removed.  ResolvePass
      component matching handles standalone surname occurrences from the
      full-name surrogate, giving consistent replacement.

  Pass D — Topical geo-entity filter (revised)
      A GPE/LOC is dropped ONLY if it appears exclusively in query
      sub-clauses.  Appearing in any non-query sub-clause → always kept,
      regardless of whether a PERSON is present.

      Additionally: mid-sentence lowercase geo entities (common-noun
      usages like "phoenix bird" or "springfield field team") are skipped
      via a capitalisation check — proper place names are capitalised in
      English.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from dataclasses import replace as _dc_replace
from util import DetectedEntity, get_logger, mask_spans
from detection import pattern_scan, entity_trace, context_guard
from detection.quasi_identifier import score as qi_score

logger = get_logger(__name__)


class _TaggedList(list):
    """list subclass that allows attribute assignment (used for _qi_matches)."""
    pass


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
    """
    Emit ORG entities for company names identified by organisational suffixes.

    The structural suffix (corporation, company, etc.) signals that the
    preceding word is used as a company name regardless of capitalisation.
    This catches "target corporation", "the phoenix group", etc. without
    any list of known company names.

    The detected span covers ONLY the name (e.g. "target"), not the suffix,
    so the surrogate replaces only "target" and "corporation" is preserved,
    giving "The RetailHoldings corporation announced…" as expected.
    """
    occupied = {(e.start, e.end) for e in existing_entities}
    new_ents: List[DetectedEntity] = []

    for m in _STRUCTURAL_ORG_PATTERN.finditer(text):
        name_text  = m.group(1).strip()
        name_start = m.start(1)
        name_end   = m.end(1)

        # Skip if overlaps with an already-detected entity
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
        logger.debug(
            f"[SentinelLayer] Pass A structural ORG: {name_text!r}"
        )

    return new_ents


# ─────────────────────────────────────────────────────────────────────────────
# Pass B — Email-username → PERSON reclassification
# ─────────────────────────────────────────────────────────────────────────────

def _reclassify_email_username_orgs(
    entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    """
    Reclassify ORG entities whose text is a name-prefix of a detected email.

    spaCy sees masked text after PatternScan — the email is hidden — so it
    sometimes labels the standalone first name as ORG.  This pass uses the
    email detection output (from PatternScan) to correct that labelling.

    Example: "sherwinvishesh@gmail.com" detected → username "sherwinvishesh"
             "Sherwin" detected as ORG → "sherwin" is prefix of username
             → reclassify "Sherwin" as PERSON
    """
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
    """
    Remove PERSON entities that are word-components of longer PERSON entities.

    When "Mitchell" and "Sarah Mitchell" are both detected:
      • Keep "Sarah Mitchell"
      • Remove "Mitchell" (ResolvePass component matching will handle
        standalone "Mitchell" occurrences using the full-name surrogate,
        giving consistent output like "Clark" from "Jessica Clark")

    Decision is purely structural (word-set containment), no name lists.
    """
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
    """
    Return True if this entity should be treated as a proper noun.

    English proper nouns are capitalised.  A geo entity whose surface form
    starts with a lowercase letter and appears mid-sentence is almost
    certainly a common-noun usage ("phoenix bird", "springfield field team"),
    not the name of a place.

    This is a linguistic structure rule — it uses no geographic keyword lists.
    """
    if entity_text[0].isupper():
        return True  # capitalised → proper noun ✓

    # Lowercase entity — check if it's at the start of a sentence/clause
    idx = text.find(entity_text)
    if idx == -1:
        idx = text.lower().find(entity_text.lower())
    if idx == -1:
        return True  # not found → conservative: keep

    prefix = text[:idx].rstrip()
    if not prefix or prefix[-1] in ".!?;":
        return True  # sentence-start → might be proper noun despite lowercase

    return False  # lowercase, mid-sentence → common noun usage → skip


def _filter_topical_geo_entities(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """
    Remove GPE/LOC entities that are query topics rather than personal refs.

    Revised rule (no PERSON co-occurrence requirement):
      • Geo entity appears ONLY in query sub-clauses → topical → DROPPED
      • Geo entity appears in any non-query sub-clause → personal/narrative
        → KEPT regardless of whether a PERSON entity is present

    This means:
      "give me tax benefits of Wyoming"         → query only → dropped
      "from the ashes of Phoenix"               → non-query  → kept ✓
      "The city of Springfield is beautiful"    → non-query  → kept ✓
      "What restaurants are near London?"       → query only → dropped ✓
      "Revanth lives in Wyoming"                → non-query  → kept ✓

    The capitalisation check (_is_proper_capitalized) additionally filters
    mid-sentence lowercase usages ("phoenix bird", "springfield field team").
    """
    geo_ents   = [e for e in entities if e.type in _GEO_FILTERABLE]
    other_ents = [e for e in entities if e.type not in _GEO_FILTERABLE]

    if not geo_ents:
        return entities

    all_clauses     = _all_sub_clauses(text)
    clause_is_query = [bool(_QUERY_FRAME.match(c)) for c in all_clauses]

    result = list(other_ents)

    for geo_ent in geo_ents:
        # Capitalisation check: lowercase mid-sentence = common noun → skip
        if not _is_proper_capitalized(geo_ent.text, text):
            logger.debug(
                f"[SentinelLayer] Pass D: lowercase geo skipped (not proper noun): "
                f"{geo_ent.text!r}"
            )
            continue

        # Query-clause filter
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
            continue

        logger.debug(
            f"[SentinelLayer] Pass D: geo kept (non-query context): {geo_ent.text!r}"
        )
        result.append(geo_ent)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ORG→GPE reclassification (narrow preps only)
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
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Execute the full SentinelLayer cascade then apply post-processing passes.

    Args:
        text:                   Raw user message.
        skip_values:            Surrogate strings to skip in PatternScan.
        skip_location_entities: Suppress ALL geo entities (service-query mode).
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
        remaining_text, existing_entities=confirmed,
    )
    ner_confirmed  = _reclassify_location_orgs(ner_confirmed,  text)
    ner_borderline = _reclassify_location_orgs(ner_borderline, text)

    if skip_location_entities:
        ner_confirmed  = [e for e in ner_confirmed  if e.type not in _GEO_TYPES]
        ner_borderline = [e for e in ner_borderline if e.type not in _GEO_TYPES]

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
        confirmed          = _filter_topical_geo_entities(confirmed,          text)
        needs_confirmation = _filter_topical_geo_entities(needs_confirmation, text)

    # ── Quasi-identifier combination scoring ──────────────────────────────────
    confirmed = _TaggedList(confirmed)  # wrap to allow attribute assignment
    qi_matches = qi_score(confirmed)
    if qi_matches:
        for match in qi_matches:
            logger.info(
                f"[SentinelLayer] Quasi-ID risk: {match.combination_name} "
                f"(fields: {match.matched_fields}, risk: {match.risk_level})"
            )
    confirmed._qi_matches = qi_matches

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
    return result