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

# Every keyword group is \b-anchored: without anchors, substrings inside
# ordinary words classify prose as service queries ("sTOPped … MilTOn"
# used to satisfy "(top).{0,40}(to)") and silently suppress geo entities.
_SERVICE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # Food / dining
    r"\b(find|locate|show|get|recommend|suggest|any|good|best)\b.{0,60}"
    r"\b(restaurant|cafe|coffee|breakfast|lunch|dinner|food|brunch|spot|place|eatery|bistro|diner)s?\b"
    r".{0,40}\b(near|in|around|close|by)\b",

    # Generic "what/where X near Y"
    r"\b(what|which|where|any)\b.{0,50}\b(near|close to|around|in the area)\b",

    # Nearest / closest / open now
    r"\b(nearest|closest|best|top|good|popular|open)\b.{0,40}\b(near|close|around|by|to)\b",

    # "Is there a / are there any / find me"
    r"\b(is there a?|are there any|find (a|some|me|the))\b.{0,60}"
    r"\b(near|in|around|close|by)\b",

    # Directions
    r"\bdirections?\b.{0,25}\b(to|from)\b",
    r"\b(how (do i|to|can i) get|navigate|route)\b.{0,25}\b(to|from)\b",

    # Weather
    r"\b(weather|temperature|forecast|rain|snow|humidity)\b.{0,25}\b(in|at|near|for)\b",

    # Hours / availability
    r"\b(what.{0,15}(open|closed|hours|close)|is.{0,5}(open|closed))\b.{0,40}\b(near|in)\b",

    # Activities / places
    r"\b(places?|spots?|areas?|things? to do|activities?)\b.{0,25}\b(in|near|around)\b",

    # Specific service types
    r"\b(charging station|parking|atm|gas station|petrol|fuel)\b.{0,40}\b(near|close|around)\b",
    r"\b(pharmacy|chemist|hospital|clinic|doctor|urgent care)\b.{0,40}\b(near|in|around|close)\b",
    r"\b(grocery|supermarket|store|shop|mall|market)\b.{0,40}\b(near|in|around|close)\b",

    # "check if ... near"
    r"\bcheck (if|whether)\b.{0,60}\b(near|in|around|close)\b",
]]

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


def is_service_query(text: str) -> bool:
    """
    Return True if the message is a service or knowledge query.

    Sensitive topics always override and force full anonymization.
    """
    if is_sensitive_topic(text):
        logger.debug("[ServiceQuery] Sensitive topic — full anonymization")
        return False

    for pattern in _SERVICE_PATTERNS:
        if pattern.search(text):
            logger.debug("[ServiceQuery] Service query detected — minimal fuzzing")
            return True

    return False


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
    mask() with address_mode="shift" (the default).

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
        if verify:
            address_parser.verify_address_exists(parsed)
        fuzzed = shift_house_number(parsed, forbidden=frozenset(forbidden))
        if fuzzed is None:
            continue
        mappings[parsed.full_text] = fuzzed
        forbidden.add(fuzzed)
        logger.debug(f"[ServiceQuery] {parsed.full_text!r} → {fuzzed!r}")

    result = text
    for original in sorted(mappings, key=len, reverse=True):
        result = result.replace(original, mappings[original])

    if mappings:
        logger.info(f"[ServiceQuery] Fuzzed {len(mappings)} address(es)")

    return result, mappings
