"""
detection/service_query.py — ServiceQueryDetector

Detects whether a user message is a service or knowledge query where
full PII replacement would break answer utility.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# ─── Service query patterns ───────────────────────────────────────────────────

_SERVICE_PATTERNS = [
    # Food / dining
    r"(?i)(find|locate|show|get|recommend|suggest|any|good|best).{0,60}"
    r"(restaurant|cafe|coffee|breakfast|lunch|dinner|food|brunch|spot|place|eatery|bistro|diner)"
    r".{0,40}(near|in|around|close|by)",

    # Generic "what/where X near Y"
    r"(?i)(what|which|where|any).{0,50}(near|close to|around|in the area)",

    # Nearest / closest / open now
    r"(?i)(nearest|closest|best|top|good|popular|open).{0,40}(near|close|around|by|to)",

    # "Is there a / are there any / find me"
    r"(?i)(is there a?|are there any|find (a|some|me|the)).{0,60}"
    r"(near|in|around|close|by)",

    # Directions
    r"(?i)directions?.{0,25}(to|from)",
    r"(?i)(how (do i|to|can i) get|navigate|route).{0,25}(to|from)",

    # Weather
    r"(?i)(weather|temperature|forecast|rain|snow|humidity).{0,25}(in|at|near|for)",

    # Hours / availability
    r"(?i)(what.{0,15}(open|closed|hours|close)|is.{0,5}(open|closed)).{0,40}(near|in)",

    # Activities / places
    r"(?i)(places?|spots?|areas?|things? to do|activities?).{0,25}(in|near|around)",

    # Specific service types
    r"(?i)(charging station|parking|atm|gas station|petrol|fuel).{0,40}(near|close|around)",
    r"(?i)(pharmacy|chemist|hospital|clinic|doctor|urgent care).{0,40}(near|in|around|close)",
    r"(?i)(grocery|supermarket|store|shop|mall|market).{0,40}(near|in|around|close)",

    # "check if ... near"
    r"(?i)check (if|whether).{0,60}(near|in|around|close)",
]

# Sensitive topics that override service classification → full anonymization
_SENSITIVE_OVERRIDES = [
    r"(?i)(hiv|aids|std|sti|abortion|rehab|rehabil|addiction|mental health|psychiatr|"
    r"therapy|therapist|counsel|domestic violence|shelter|homeless|immigration|undocumented|"
    r"substance abuse|overdose|suicide|self.harm|eating disorder|detox)",
]


# ─── Address pattern ──────────────────────────────────────────────────────────

_ADDRESS_PATTERN = re.compile(
    r"\b(\d+)\s+"
    r"([A-Za-z0-9 ]+?"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|"
    r"Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|"
    r"Row|Mews|Close|Crescent|Cres|"
    r"Parkway|Pkwy|Highway|Hwy|Freeway|Fwy|"
    r"Terrace|Terr|Circle|Cir|Loop|Trail|Trl|"
    r"Plaza|Pass|Square|Sq)\.?)"
    r"(?:\s*,\s*([A-Za-z][A-Za-z\s]{1,30}?))?"
    r"(?:\s*,\s*([A-Z]{2}))?"
    r"\b",
    re.IGNORECASE,
)


# ─── Public API ───────────────────────────────────────────────────────────────

def is_service_query(text: str) -> bool:
    """
    Return True if the message is a service or knowledge query.

    Sensitive topics always override and force full anonymization.
    """
    for pattern in _SENSITIVE_OVERRIDES:
        if re.search(pattern, text):
            logger.debug("[ServiceQuery] Sensitive topic — full anonymization")
            return False

    for pattern in _SERVICE_PATTERNS:
        if re.search(pattern, text):
            logger.debug("[ServiceQuery] Service query detected — minimal fuzzing")
            return True

    return False


def fuzz_addresses(
    text: str,
    verify: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
    Find street addresses in text and apply minimal house-number fuzzing.

    The house number is shifted by exactly ±1 (randomly chosen).  All other
    address components are preserved verbatim.

    Returns:
        Tuple of (fuzzed_text, {original_address: fuzzed_address}).
    """
    mappings: Dict[str, str] = {}
    result = text

    for match in _ADDRESS_PATTERN.finditer(text):
        original = match.group(0).strip()
        if original in mappings:
            continue

        try:
            house_number = int(match.group(1))
        except (ValueError, TypeError):
            continue

        street = match.group(2).strip() if match.group(2) else ""
        city   = match.group(3).strip() if match.group(3) else ""
        state  = match.group(4).strip() if match.group(4) else ""

        delta      = random.choice([-1, 1])
        new_number = max(1, house_number + delta)

        parts = [f"{new_number} {street}"]
        if city:
            parts.append(city)
        if state:
            parts.append(state)
        fuzzed = ", ".join(parts)

        if verify and city:
            fuzzed = _verify_or_fallback(new_number, street, city, state, fuzzed)

        mappings[original] = fuzzed
        logger.debug(f"[ServiceQuery] {original!r} → {fuzzed!r}")

    for original in sorted(mappings, key=len, reverse=True):
        result = result.replace(original, mappings[original])

    if mappings:
        logger.info(f"[ServiceQuery] Fuzzed {len(mappings)} address(es)")

    return result, mappings


def _verify_or_fallback(
    number: int,
    street: str,
    city: str,
    state: str,
    fallback: str,
    timeout: float = 2.0,
) -> str:
    """Verify fuzzed address via OpenStreetMap Nominatim. Always returns the fuzzed address."""
    try:
        import requests
        query = f"{number} {street}, {city}"
        if state:
            query += f", {state}"
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "SurrogateShield-Research/1.0"},
            timeout=timeout,
        )
        data = r.json()
        if data:
            logger.debug(f"[ServiceQuery] Nominatim verified: {query!r}")
        else:
            logger.debug(f"[ServiceQuery] Nominatim: no result for {query!r} — using anyway")
    except Exception as exc:
        logger.debug(f"[ServiceQuery] Nominatim failed: {exc} — using fallback")
    return fallback
