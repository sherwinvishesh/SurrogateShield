"""
detection/logic.py — SentinelLayer

Cascade: PatternScan → EntityTrace → ContextGuard, followed by a
model-driven geographic entity filter.

The geographic filter — purely model-output driven
───────────────────────────────────────────────────
No hardcoded lists of states, countries, or city names.
The decision is based entirely on the entity TYPES that spaCy and
distilbert returned and on whether each sub-clause is a personal
statement or a knowledge/service request.

Algorithm
─────────
1. Split the text into sub-clauses at ALL clause boundaries:
       sentence endings  .  !  ?
       conjunctions      and  or  but  because  while  when  since  …
       commas / semicolons

2. Classify each sub-clause as "query" or "personal":
     • Query  — starts with a request/interrogative frame
                  "give me …", "tell me …", "what is …", "how do …",
                  "find me …", "please …", "can you …", etc.
     • Personal — everything else (personal disclosures, descriptions)

3. For each GPE or LOC entity:
     a. If it appears in a QUERY sub-clause  → it is the TOPIC of the
        request, not the user's personal data → DROP (do not replace).
     b. Else (it is only in PERSONAL sub-clauses):
        • If any PERSONAL sub-clause in the same SENTENCE also contains a
          PERSON entity → the geo entity is personal location context → KEEP.
        • Otherwise → no person anywhere in the sentence → DROP.

Why this handles all the key cases (entity-type decisions, no keyword lists)
─────────────────────────────────────────────────────────────────────────────
"give me the tax benefits of wyoming"
    sub-clause: "give me the tax benefits of wyoming" → QUERY
    Wyoming in QUERY → dropped ✓

"my name is Revanth and give me tax benefits of wyoming"
    sub-clauses: "my name is Revanth" (personal),
                 "give me tax benefits of wyoming" (QUERY)
    Wyoming in QUERY → dropped ✓  |  Revanth kept ✓

"Revanth lives in Wyoming"
    one personal sub-clause; sentence has PERSON (Revanth) → Wyoming kept ✓

"I am Emma and I live in Seattle"
    sub-clauses: "I am Emma" (personal), "I live in Seattle" (personal)
    Seattle in personal sub-clause; sentence has PERSON (Emma) → kept ✓

"I am Emma but give me California tax rates"
    sub-clauses: "I am Emma" (personal), "give me California tax rates" (QUERY)
    California in QUERY → dropped ✓

"What is the weather in Arizona?"
    sub-clause: "What is the weather in Arizona?" → QUERY
    Arizona dropped ✓
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from dataclasses import replace as _dc_replace
from util import DetectedEntity, get_logger, mask_spans
from detection import pattern_scan, entity_trace, context_guard

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-clause splitting
# ─────────────────────────────────────────────────────────────────────────────

# Split at ALL grammatical clause boundaries, including coordinating "and/or".
# We split at "and" here so that "give me X and tell me Y" cleanly separates
# the two requests.  Personal coordinated clauses ("I am Emma and I live in
# Seattle") still work because neither sub-clause matches the QUERY frame.
_CLAUSE_SPLIT = re.compile(
    r'[.!?]+\s+'                                      # sentence endings
    r'|\s+(?:and|or|but|however|yet|because|since|although|while|when'
    r'|therefore|whereas|unless|though|despite|nevertheless|so)\s+'
    r'|,\s+',                                         # comma
    re.IGNORECASE,
)

# Sentence-only split (used to scope "does this sentence have a PERSON?")
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ─────────────────────────────────────────────────────────────────────────────
# Query-frame classifier
# ─────────────────────────────────────────────────────────────────────────────

# A sub-clause is a "query clause" if it starts with (optionally preceded by
# polite markers like "please" / "could you") an imperative or interrogative
# opener that signals a request for information rather than a personal disclosure.
#
# Why a regex here and not a model?
#   • The regex matches STRUCTURAL sentence openers (imperative / wh-word),
#     not topic keywords.  "give me the tax benefits of [anything]" is a query
#     regardless of what [anything] is — the request structure matters, not the
#     geographic content.
#   • This is different from a geo whitelist: the regex has no knowledge of
#     states, countries, or cities.
_QUERY_FRAME = re.compile(
    r'^\s*'
    # Optional polite prefix
    r'(?:please\s+|could\s+you\s+(?:please\s+)?|can\s+you\s+(?:please\s+)?'
    r'|would\s+you\s+(?:please\s+)?)?'
    r'(?:'
    # Imperatives
    r'give\s+me|tell\s+me|show\s+me|find\s+me|help\s+me|get\s+me'
    r'|look\s+up|search\s+for|look\s+for|explain\s+to\s+me|explain'
    r'|list\s+(?:the|all|some)?|describe|summarize|summarise|compare'
    r'|recommend|suggest|advise|help'
    r'|'
    # Wh-questions and auxiliaries
    r'what\s+(?:is|are|was|were|would|can|do|does|did|are\s+the|is\s+the)'
    r'|how\s+(?:do|does|did|can|much|many|to|would|come)'
    r'|which\s+(?:is|are|was|were|would|one|ones)'
    r'|where\s+(?:is|are|can|do|should|would|to\s+find)'
    r'|when\s+(?:is|are|was|were|do|does|did|can)'
    r'|who\s+(?:is|are|was|were|can|would|do|does)'
    r'|why\s+(?:is|are|was|were|do|does|did|would|should)'
    r'|is\s+there|are\s+there|was\s+there|were\s+there'
    r'|do\s+you\s+know|can\s+you\s+tell\s+me'
    r'|'
    # "I want/need to know" patterns
    r'i\s+(?:want|need|would\s+like|\'d\s+like)\s+(?:to\s+know|to\s+find\s+out'
    r'|to\s+learn|to\s+understand|information|details|advice|help)'
    r')',
    re.IGNORECASE,
)


def _all_sub_clauses(text: str) -> List[str]:
    """Split text into all sub-clauses at every grammatical boundary."""
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sentences(text: str) -> List[str]:
    """Split text into sentences at [.!?] boundaries."""
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _contains_entity(entity_text: str, clause: str) -> bool:
    """Return True if entity_text appears in clause (case-insensitive)."""
    return entity_text.lower() in clause.lower()


# ─────────────────────────────────────────────────────────────────────────────
# The model-driven geographic entity filter
# ─────────────────────────────────────────────────────────────────────────────

_GEO_FILTERABLE = {"GPE", "LOC"}


def _filter_topical_geo_entities(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """
    Remove GPE/LOC entities that are query topics rather than personal data.

    Decision is made from entity types the NER models detected + whether
    each sub-clause is a personal statement or a knowledge request.
    No geographic keyword lists are used.

    See module docstring for the full algorithm and worked examples.
    """
    geo_ents  = [e for e in entities if e.type in _GEO_FILTERABLE]
    other_ents = [e for e in entities if e.type not in _GEO_FILTERABLE]

    if not geo_ents:
        return entities

    # ── Build sub-clause metadata ─────────────────────────────────────────────
    all_clauses = _all_sub_clauses(text)
    all_sents   = _sentences(text) or [text]

    # For each sub-clause: is it a query clause?
    clause_is_query = [bool(_QUERY_FRAME.match(c)) for c in all_clauses]

    # For each sentence: which sub-clauses (indices) belong to it?
    # We map by text presence since we don't have character offsets after splitting.
    def sent_for_clause(clause: str) -> int:
        for i, s in enumerate(all_sents):
            if clause.lower() in s.lower():
                return i
        return 0  # fallback: first sentence

    sent_of_clause: List[int] = [sent_for_clause(c) for c in all_clauses]

    # For each sentence: does it have a PERSON entity in a PERSONAL sub-clause?
    sent_has_person_in_personal: Dict[int, bool] = defaultdict(bool)
    for i, clause in enumerate(all_clauses):
        if clause_is_query[i]:
            continue  # skip query clauses
        sent_idx = sent_of_clause[i]
        for ent in other_ents:
            if ent.type == "PERSON" and _contains_entity(ent.text, clause):
                sent_has_person_in_personal[sent_idx] = True

    # ── Decide each geo entity ────────────────────────────────────────────────
    result = list(other_ents)

    for geo_ent in geo_ents:
        in_query  = False
        in_personal = False

        for i, clause in enumerate(all_clauses):
            if not _contains_entity(geo_ent.text, clause):
                continue
            if clause_is_query[i]:
                in_query = True
                logger.debug(
                    f"[SentinelLayer] {geo_ent.text!r} found in query sub-clause: "
                    f"{clause[:60]!r}"
                )
            else:
                in_personal = True

        if in_query:
            # Entity appears in at least one QUERY clause → it is the topic,
            # not personal data → do NOT replace
            logger.debug(
                f"[SentinelLayer] Topical geo (query sub-clause): {geo_ent.text!r} → skip"
            )
            continue

        if in_personal:
            # Entity is only in personal clauses — keep it only if a PERSON
            # entity co-occurs in the same sentence's personal clauses.
            any_sent = next(
                (sent_of_clause[i] for i, c in enumerate(all_clauses)
                 if _contains_entity(geo_ent.text, c) and not clause_is_query[i]),
                0,
            )
            if sent_has_person_in_personal[any_sent]:
                logger.debug(
                    f"[SentinelLayer] Personal geo (PERSON co-occurs in sentence): "
                    f"{geo_ent.text!r} → keep"
                )
                result.append(geo_ent)
            else:
                logger.debug(
                    f"[SentinelLayer] Topical geo (no PERSON in sentence): "
                    f"{geo_ent.text!r} → skip"
                )
            continue

        # Entity not found in any clause (e.g. ContextGuard entity on masked text)
        # → conservative: keep it
        logger.debug(
            f"[SentinelLayer] Geo entity not found in any clause → keep (conservative): "
            f"{geo_ent.text!r}"
        )
        result.append(geo_ent)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ORG→GPE reclassification (narrow preps only)
# ─────────────────────────────────────────────────────────────────────────────

_LOCATION_PREPS = {
    "in", "near",
    "live", "lives", "lived",
    "grew", "born", "raised",
    "moved", "relocate", "relocated",
    "residing", "reside",
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
    Execute the full SentinelLayer cascade then apply the model-driven
    geographic entity filter.

    Args:
        text:                   Raw user message.
        skip_values:            Surrogate strings to skip in PatternScan.
        skip_location_entities: Suppress ALL geo entities (service-query mode).
    """
    confirmed: List[DetectedEntity] = []
    needs_confirmation: List[DetectedEntity] = []

    logger.info("[SentinelLayer] Stage 1: PatternScan")
    pattern_results = pattern_scan.scan(text, skip_values=skip_values)
    confirmed.extend(pattern_results)
    remaining_text = mask_spans(text, pattern_results)

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
                f"[SentinelLayer] ContextGuard disabled — promoted {len(promoted)}"
            )

    # Model-driven geo filter (Stage 4)
    if not skip_location_entities:
        confirmed          = _filter_topical_geo_entities(confirmed,          text)
        needs_confirmation = _filter_topical_geo_entities(needs_confirmation, text)

    logger.info(
        f"[SentinelLayer] Final → "
        f"confirmed={len(confirmed)}, "
        f"needs_confirmation={len(needs_confirmation)}"
    )
    return confirmed, needs_confirmation


def deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    seen: dict = {}
    for ent in entities:
        key = ent.text.strip()
        if key not in seen or ent.score > seen[key].score:
            seen[key] = ent
    result = list(seen.values())
    result.sort(key=lambda e: e.start)
    return result