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

_SERVICE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # Food / dining
    r"(find|locate|show|get|recommend|suggest|any|good|best).{0,60}"
    r"(restaurant|cafe|coffee|breakfast|lunch|dinner|food|brunch|spot|place|eatery|bistro|diner)"
    r".{0,40}(near|in|around|close|by)",

    # Generic "what/where X near Y"
    r"(what|which|where|any).{0,50}(near|close to|around|in the area)",

    # Nearest / closest / open now
    r"(nearest|closest|best|top|good|popular|open).{0,40}(near|close|around|by|to)",

    # "Is there a / are there any / find me"
    r"(is there a?|are there any|find (a|some|me|the)).{0,60}"
    r"(near|in|around|close|by)",

    # Directions
    r"directions?.{0,25}(to|from)",
    r"(how (do i|to|can i) get|navigate|route).{0,25}(to|from)",

    # Weather
    r"(weather|temperature|forecast|rain|snow|humidity).{0,25}(in|at|near|for)",

    # Hours / availability
    r"(what.{0,15}(open|closed|hours|close)|is.{0,5}(open|closed)).{0,40}(near|in)",

    # Activities / places
    r"(places?|spots?|areas?|things? to do|activities?).{0,25}(in|near|around)",

    # Specific service types
    r"(charging station|parking|atm|gas station|petrol|fuel).{0,40}(near|close|around)",
    r"(pharmacy|chemist|hospital|clinic|doctor|urgent care).{0,40}(near|in|around|close)",
    r"(grocery|supermarket|store|shop|mall|market).{0,40}(near|in|around|close)",

    # "check if ... near"
    r"check (if|whether).{0,60}(near|in|around|close)",
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
