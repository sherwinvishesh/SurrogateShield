"""
detection/pattern_scan.py — PatternScan

Regex-based PII detection. This module contains ONLY the regex logic.
It does not call NER models or SLMs.

Detects: SSN, email, phone (US/UK), credit card (Luhn-validated),
date of birth, IPv4, API keys/secrets, UK postcodes, US ZIP codes.

Every match returns a DetectedEntity with score = 1.0 (deterministic).
"""

from __future__ import annotations

import re
from typing import List

from util import DetectedEntity, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Luhn algorithm for credit card validation
# ─────────────────────────────────────────────

def _luhn_valid(number: str) -> bool:
    """
    Validate a credit card number string using the Luhn algorithm.

    Args:
        number: Digit-only string (no spaces or dashes).

    Returns:
        True if the number passes Luhn validation.
    """
    digits = [int(d) for d in number]
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ─────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────

# Each entry: (entity_type, compiled_regex, post_validator_or_None)
_PATTERNS: list = [
    (
        "ssn",
        re.compile(r"\b(\d{3}-\d{2}-\d{4}|\d{9})\b"),
        None,
    ),
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "phone_us",
        re.compile(
            r"(?<!\d)"
            r"(\+1[\s\-.]?)?"
            r"\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
            r"(?!\d)"
        ),
        None,
    ),
    (
        "phone_uk",
        re.compile(
            r"(?<!\d)"
            r"(\+44\s?|0)"
            r"(\d{4}[\s\-]?\d{6}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{2}[\s\-]?\d{4}[\s\-]?\d{4})"
            r"(?!\d)"
        ),
        None,
    ),
    (
        "credit_card",
        re.compile(
            r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
        ),
        lambda m: _luhn_valid(re.sub(r"[\s\-]", "", m.group())),
    ),
    (
        "dob",
        re.compile(
            r"\b(?:"
            # MM/DD/YYYY or DD/MM/YYYY
            r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}"
            r"|"
            # Written: January 14, 1990 / 14 January 1990 / Jan 14 1990
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{1,2}[\s,]+\d{4}"
            r"|"
            r"\d{1,2}[\s,]+"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        None,
    ),
    (
        "api_key",
        re.compile(
            r"(?:"
            r"sk-[A-Za-z0-9\-_]{16,}"           # OpenAI / Anthropic style
            r"|Bearer\s+[A-Za-z0-9\-_.]{16,}"   # Bearer token
            r"|ghp_[A-Za-z0-9]{20,}"             # GitHub personal access token
            r"|gho_[A-Za-z0-9]{20,}"             # GitHub OAuth token
            r"|AKIA[0-9A-Z]{16}"                # AWS access key ID
            r"|[A-Za-z0-9+/]{40}={0,2}"         # Generic base64 secret (>=40 chars)
            r")"
        ),
        None,
    ),
    (
        "postcode_uk",
        re.compile(
            r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "zip_us",
        re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        None,
    ),
]


# ─────────────────────────────────────────────
# Main scan function
# ─────────────────────────────────────────────

def scan(text: str) -> List[DetectedEntity]:
    """
    Run all regex patterns against *text* and return detected entities.

    Overlapping matches for the same span are deduplicated (first match
    wins). All returned entities carry score = 1.0.

    Args:
        text: The raw user message to scan.

    Returns:
        List of DetectedEntity objects, sorted by start position.
    """
    results: List[DetectedEntity] = []
    occupied_spans: List[tuple] = []  # (start, end) already claimed

    def _span_free(start: int, end: int) -> bool:
        """Check whether a span overlaps with any already-occupied span."""
        for s, e in occupied_spans:
            if not (end <= s or start >= e):
                return False
        return True

    for entity_type, pattern, validator in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if span already claimed by an earlier (higher-priority) pattern
            if not _span_free(start, end):
                continue

            # Run optional post-validator (e.g. Luhn for credit cards)
            if validator is not None and not validator(match):
                continue

            entity = DetectedEntity(
                text=match.group().strip(),
                start=start,
                end=end,
                type=entity_type,
                score=1.0,
                source="pattern",
            )
            results.append(entity)
            occupied_spans.append((start, end))
            logger.debug(
                f"[PatternScan] {entity_type}: {entity.text!r} "
                f"at [{start}:{end}]"
            )

    results.sort(key=lambda e: e.start)
    logger.info(f"[PatternScan] Found {len(results)} entities")
    return results