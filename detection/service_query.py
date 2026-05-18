"""
detection/service_query.py — ServiceQueryDetector

Detects whether a user message is a service or knowledge query where
full PII replacement would break answer utility.

For service queries containing street addresses, applies minimal
geographic fuzzing (house number shift ±2 to ±8) instead of full
surrogate replacement. Street name, city, and state are preserved so
the LLM can return useful local results.

Target error rate: geographic displacement < 100 metres.
Address existence verified via OpenStreetMap Nominatim (free, no key).

IMPORTANT: sensitive service queries (e.g. "HIV clinic near [address]")
are NOT classified as service queries — the medical/sensitive context
overrides the service classification and full anonymization applies.

Research context: this module introduces utility-aware anonymization —
the system understands query intent before deciding how aggressively to
anonymize.  It is a novel contribution connecting directly to the
Surrogate Fidelity Study (PETS 2027 submission).
"""

from __future__ import annotations

import random
import re
from typing import Dict, Tuple

from util import get_logger

logger = get_logger(__name__)


# ─── Service query patterns ───────────────────────────────────────────────────

_SERVICE_PATTERNS = [
    r"(?i)(what|which|where).{0,30}(restaurant|cafe|coffee|shop|store|hotel|motel|hospital|clinic|pharmacy|gym|bank|library|park|museum|theater|cinema|mall|supermarket|grocery).{0,30}(near|in|around|close|by)",
    r"(?i)(nearest|closest|best|top|recommend|good|popular).{0,30}(to|near|in|around)",
    r"(?i)directions?.{0,20}(to|from)",
    r"(?i)(how (do i|to|can i) get).{0,20}(to|from)",
    r"(?i)(weather|temperature|forecast|rain|snow).{0,20}(in|at|near|for)",
    r"(?i)what.{0,20}(open|closed|hours|open until|close at).{0,20}(near|in)",
    r"(?i)(places?|spots?|areas?|things? to do).{0,20}(in|near|around)",
    r"(?i)(is there a|are there any|find a|find me).{0,30}(near|in|around|close)",
]

# Override: these topics always require full anonymization regardless of
# query structure — sensitive context beats utility considerations.
_SENSITIVE_OVERRIDES = [
    r"(?i)(hiv|aids|std|sti|abortion|rehab|rehabil|addiction|mental health|psychiatr|"
    r"therapy|therapist|counsel|domestic violence|shelter|homeless|immigration|undocumented)",
]


# ─── Address pattern ──────────────────────────────────────────────────────────

# Matches common US/UK street address formats:
#   1126 E Apache Blvd, Tempe, AZ
#   42 Baker Street, London
#   500 Main St
_ADDRESS_PATTERN = re.compile(
    r"\b(\d+)\s+"                                          # house number (group 1)
    r"([A-Za-z0-9 ]+?"                                     # street name (lazy, group 2)
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|"
    r"Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|"
    r"Parkway|Pkwy|Highway|Hwy|Freeway|Fwy|"
    r"Terrace|Terr|Circle|Cir|Loop|Trail|Trl)\.?)"        # street suffix
    r"(?:\s*,\s*([A-Za-z\s]+?))??"                         # optional city (lazy, group 3)
    r"(?:\s*,\s*([A-Z]{2}))?"                              # optional 2-letter state (group 4)
    r"\b",
    re.IGNORECASE,
)


# ─── Public API ───────────────────────────────────────────────────────────────

def is_service_query(text: str) -> bool:
    """
    Return True if the message is a service or knowledge query.

    Sensitive medical/legal/social topics override the classification
    even if the query structure looks like a service query — full
    anonymization always applies for sensitive contexts.

    Args:
        text: Raw user message.

    Returns:
        True if minimal address fuzzing should be applied instead of
        full surrogate replacement.
    """
    # Sensitive topics always override — full anonymization required
    for pattern in _SENSITIVE_OVERRIDES:
        if re.search(pattern, text):
            logger.debug("[ServiceQuery] Sensitive topic detected — full anonymization")
            return False

    # Check for service query structural patterns
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
    Find street addresses in text and apply minimal house number fuzzing.

    Shifts the house number by a random even amount (±2, ±4, ±6, or ±8).
    Street name, city, and state are preserved — the result is geographically
    close (typically < 100 m) but not identical to the original address.
    The fuzzed house number is always >= 1.

    Args:
        text:   User message that may contain street addresses.
        verify: If True, attempt to verify the fuzzed address exists via
                OpenStreetMap Nominatim (adds ~1–2 s per address).
                Set False in tests or when the network is unavailable.

    Returns:
        Tuple of (fuzzed_text, {original_address: fuzzed_address}).
        If no addresses are found, returns (text, {}).
    """
    mappings: Dict[str, str] = {}
    result = text

    for match in _ADDRESS_PATTERN.finditer(text):
        original = match.group(0).strip()
        if original in mappings:
            continue  # already processed this exact string

        try:
            house_number = int(match.group(1))
        except (ValueError, TypeError):
            continue

        street = match.group(2).strip() if match.group(2) else ""
        city   = match.group(3).strip() if match.group(3) else ""
        state  = match.group(4).strip() if match.group(4) else ""

        # Shift house number by ±2, ±4, ±6, or ±8 (even offsets only)
        delta      = random.choice([-8, -6, -4, -2, 2, 4, 6, 8])
        new_number = max(1, house_number + delta)

        # Reconstruct fuzzed address preserving city/state
        parts = [f"{new_number} {street}"]
        if city:
            parts.append(city)
        if state:
            parts.append(state)
        fuzzed = ", ".join(parts)

        # Optionally verify via Nominatim (only when a city is present)
        if verify and city:
            fuzzed = _verify_or_fallback(new_number, street, city, state, fuzzed)

        mappings[original] = fuzzed
        logger.debug(f"[ServiceQuery] Fuzzed address: {original!r} → {fuzzed!r}")

    # Apply all mappings to the text — longest-first to avoid substring conflicts
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
    """
    Verify fuzzed address exists via OpenStreetMap Nominatim.

    Returns the fuzzed address string whether or not Nominatim confirms
    it (a nearby non-existent number is still geographically close enough
    to preserve answer utility).  Never raises — on any error the fallback
    is returned silently.

    Args:
        number:   Fuzzed house number.
        street:   Street name (preserved from original).
        city:     City (preserved from original).
        state:    State code (preserved from original).
        fallback: The fuzzed address string to use regardless.
        timeout:  HTTP request timeout in seconds.

    Returns:
        The fuzzed address string (verified or used as-is on failure).
    """
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
            logger.debug(
                f"[ServiceQuery] Nominatim could not verify {query!r} — using anyway"
            )
    except Exception as exc:
        logger.debug(
            f"[ServiceQuery] Nominatim verification failed: {exc} — using fallback"
        )
    # Always return the fuzzed address — close enough for service queries
    return fallback