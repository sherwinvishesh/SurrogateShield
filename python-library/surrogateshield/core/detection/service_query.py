"""
detection/service_query.py — ServiceQueryDetector

Detects whether a user message is a service or knowledge query where
full PII replacement would break answer utility, and whether it touches
a sensitive topic that must force full anonymization regardless.

v2: address parsing/fuzzing moved to the canonical address parser
(detection/address_parser.py) and the shift generator
(generation.shift_house_number). All patterns are precompiled at import.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Tuple

import re

logger = logging.getLogger(__name__)


# ─── Service query patterns (precompiled) ────────────────────────────────────

# I6: a service query needs an explicit request ("find", "is there", a
# question opener…) AND a proximity phrase whose object is a location
# ("near 12 Elm St", "around Phoenix", "close to me", "near the stadium").
# A bare "in <Place>" is weaker and also needs a venue word ("pizza in
# Tempe"). The v1 keyword soup flagged prose such as "I'm a nurse at Mercy
# General Hospital in Tempe" or "I'm nowhere near finished" and then kept
# every location in it verbatim.
_VENUE_WORDS = (
    r"restaurants?|caf[eé]s?|coffee(?:\s+shops?)?|bars?|pubs?|hotels?|motels?|clinics?"
    r"|hospitals?|pharmac(?:y|ies)|chemists?|dentists?|doctors?|urgent\s+care|emergency\s+rooms?"
    r"|gyms?|stores?|shops?|supermarkets?|grocer(?:y|ies)|banks?|atms?|gas\s+stations?"
    r"|petrol\s+stations?|charging\s+stations?|parking|parks?|schools?|daycares?|vets?|salons?"
    r"|barbers?|mechanics?|librar(?:y|ies)|museums?|malls?|markets?|breakfast|brunch|lunch|dinner"
    r"|food|pizza|sushi|tacos?|bakery|bakeries|places?\s+to\s+(?:eat|stay|go|visit)|eat"
    r"|things?\s+to\s+do|weather|forecast|temperature|directions?|airports?|stadiums?"
    r"|shelters?|centers?|centres?|lawyers?|attorneys?|therapists?|counsell?ors?"
    r"|testing\s+sites?|food\s+banks?|laundromats?|hikes?|trails?|beach(?:es)?"
)
_VENUE = re.compile(r"\b(?:" + _VENUE_WORDS + r")\b", re.IGNORECASE)
_INTENT = re.compile(
    r"\b(?:find|finding|locate|recommend|suggest|looking\s+for|look\s+for|search(?:ing)?\s+for"
    r"|is\s+there|are\s+there|where(?:'s|\s+(?:can|could|do|does|is|are|should|would))"
    r"|any\s+good|open\s+now|directions?|how\s+(?:do|can)\s+i\s+get|what'?s\s+the\s+weather"
    r"|weather|forecast|check\s+(?:if|whether))\b"
    # a ranking word only counts right before a venue ("best pizza", not
    # "my nearest relative")
    r"|\b(?:best|top|good|great|cheap(?:est)?|nearest|closest|popular)\s+(?:\w+\s+){0,2}?(?:"
    + _VENUE_WORDS + r")\b"
    # a question opener at the start of a sentence ("What cafes near …")
    r"|(?:^|[.?!]\s+)\W*(?:what|which|where|any|how\s+far|how\s+long|can\s+you|could\s+you)\b",
    re.IGNORECASE,
)
# A bare search fragment ("rehab centers around Phoenix") has no verb: a
# short message that opens with a venue counts as the intent.
_FRAGMENT = re.compile(r"^\W*(?:[\w'-]+\s+){0,3}?(?:" + _VENUE_WORDS + r")\b", re.IGNORECASE)
_FRAGMENT_MAX_WORDS = 15
# …unless it is a first-person statement ("My doctor in Mesa said…",
# "Hospital in Tempe is where I work"). "near my house" is still a query.
_STATEMENT = re.compile(
    r"\b(?:I|I'm|I've|I'd|I'll|we|we're|our)\b|(?<!near\s)(?<!around\s)(?<!to\s)(?<!from\s)\b[Mm]y\b"
)


def _fragment_intent(text: str) -> bool:
    return (bool(_FRAGMENT.search(text)) and len(text.split()) <= _FRAGMENT_MAX_WORDS
            and not _STATEMENT.search(text))


# The object of the proximity word must be a location: an address or ZIP, a
# capitalised name, me/here, "my home/area/…", "the <noun>", downtown/campus.
_LOC_OBJECT = (
    r"(?=\d|[A-Z]|(?i:me\b|here\b|downtown\b|campus\b|town\b|the\s+\w"
    r"|my\s+(?:house|home|place|area|location|apartment|flat|office|work|job|hotel|campus"
    r"|school|neighbou?rhood|zip|address)\b))"
)
_PLACE = re.compile(
    r"(?i:\b(?<!nowhere\s)(?:near(?:by)?|nearest\s+to|close(?:st)?\s+to|around(?!\s+the\s+corner)"
    r"|within\s+\d+\s*(?:miles?|mi|km|minutes?|mins?|blocks?)\s+(?:of|from)"
    r"|walking\s+distance\s+(?:of|from|to)|directions?\s+(?:to|from)|get\s+(?:to|from)))\s+"
    + _LOC_OBJECT +
    r"|(?i:\b(?:nearby|close\s+by|around\s+here|in\s+my\s+area|near\s+me)\b)"
)
_PLACE_WEAK = re.compile(r"\b(?i:in)\s+(?=[A-Z0-9])")     # "in Tempe", "in 85281"

_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")


def _located(text: str) -> bool:
    return bool(_PLACE.search(text) or (_PLACE_WEAK.search(text) and _VENUE.search(text)))


def _is_request(sentence: str) -> bool:
    """Request and location in the SAME sentence: "I live near Tempe. What's
    a good 401k?" is not a service query."""
    return bool(_INTENT.search(sentence)) and _located(sentence)


# Kept for the precompiled-pattern guard and for callers that introspect it.
_SERVICE_PATTERNS = [_INTENT, _FRAGMENT, _PLACE, _PLACE_WEAK, _VENUE]

# Sensitive topics that override service classification → full anonymization
_SENSITIVE_OVERRIDES = [re.compile(p, re.IGNORECASE) for p in [
    r"(hiv|aids|std|sti|abortion|rehab|rehabil|addiction|mental health|psychiatr|"
    r"therapy|therapist|counsel|domestic violence|shelter|homeless|immigration|undocumented|"
    r"substance abuse|overdose|suicide|self.harm|eating disorder|detox)",
]]


# ─── Public API ───────────────────────────────────────────────────────────────

def is_sensitive_topic(text: str) -> bool:
    """Return True if the message touches a topic that must force full
    anonymization even inside a service query (medical, legal, immigration…)."""
    return any(p.search(text) for p in _SENSITIVE_OVERRIDES)


def classify(text: str) -> str:
    """``"none"`` | ``"service"`` | ``"service_coarse"`` (audit I6).

    A service query keeps real city/state names so the answer is useful.
    A *sensitive* service query ("rehab clinic near 316 Citrus Blvd,
    Orlando") is still a service query — inventing a fictional city makes
    the answer useless exactly when location matters most — but its street
    line is coarsened to "my area" instead of shifted.
    """
    if not any(_is_request(sentence) for sentence in _SENTENCE.split(text)) and not (
        _fragment_intent(text) and _located(text)
    ):
        return "none"
    if is_sensitive_topic(text):
        logger.debug("[ServiceQuery] sensitive service query — coarse location")
        return "service_coarse"
    logger.debug("[ServiceQuery] service query — locations kept, addresses shifted")
    return "service"


def is_service_query(text: str) -> bool:
    """True for a (possibly sensitive) service query: city/state names are
    kept. Use :func:`classify` to tell the sensitive case apart."""
    return classify(text) != "none"


def resolve(text: str, mode: str, enabled: bool = True) -> Tuple[bool, str]:
    """``(is_service_query, effective_address_mode)`` for one message — the
    single entry point every caller uses (CLI, pipeline, library, evaluator)."""
    kind = classify(text) if enabled else "none"
    return kind != "none", resolve_address_mode(mode, kind != "none", kind == "service_coarse")


def resolve_address_mode(mode: str, service_query: bool, sensitive: bool = False) -> str:
    """Effective address mode for one message.

    ``"auto"`` shifts inside a service query, coarsens the street line to
    "my area" inside a sensitive service query and replaces everywhere else;
    explicit ``"shift"``/``"replace"`` pass through.
    """
    if mode == "auto":
        if not service_query:
            return "replace"
        return "coarse" if sensitive else "shift"
    return mode


# ─── Deprecated compatibility wrapper ─────────────────────────────────────────

def fuzz_addresses(
    text: str,
    verify: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """
    DEPRECATED — kept for backward compatibility only.

    Address handling now flows through the canonical parser
    (detection/address_parser.py) and the shift generator
    (generation.shift_house_number). Use those directly, or simply call
    mask() with address_mode="shift" (or "auto" for service queries).

    *verify* is ignored: the Nominatim existence check sent each street
    to a third party and its result never changed the output (audit F3).

    Returns:
        Tuple of (fuzzed_text, {original_address: fuzzed_address}).
    """
    warnings.warn(
        "fuzz_addresses() is deprecated; address shifting is handled by the "
        "detection/generation pipeline (address_mode='shift').",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import address_parser
    from ..generation.mimic import shift_house_number

    mappings: Dict[str, str] = {}
    parsed_list = address_parser.find_addresses(text)

    forbidden = {p.full_text for p in parsed_list}
    for parsed in parsed_list:
        if parsed.full_text in mappings:
            continue
        fuzzed = shift_house_number(parsed, forbidden=frozenset(forbidden))
        if fuzzed is None:
            continue
        mappings[parsed.full_text] = fuzzed
        forbidden.add(fuzzed)
        logger.debug(f"[ServiceQuery] {parsed.full_text!r} → {fuzzed!r}")

    from ..entities import splice

    # Span-based: each parsed address is replaced at its own offsets (E1).
    result = splice(text, [
        (p.start, p.end, p.full_text, mappings[p.full_text])
        for p in parsed_list if p.full_text in mappings
    ])

    if mappings:
        logger.info(f"[ServiceQuery] Fuzzed {len(mappings)} address(es)")

    return result, mappings
