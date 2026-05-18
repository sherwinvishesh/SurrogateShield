"""
detection/pattern_scan.py — PatternScan

Regex-based PII detection. This module contains ONLY the regex logic.
It does not call NER models or SLMs.

Detects: SSN, email, phone (US/UK/international), credit card (Luhn-validated),
date of birth, IPv4, API keys/secrets, UK postcodes, US ZIP codes.

Every match returns a DetectedEntity with score = 1.0 (deterministic).

Pattern order matters: patterns are evaluated in declaration order and earlier
patterns claim spans that later patterns cannot overlap.  phone_intl MUST come
before zip_us so international numbers like "+91 98765 43210" are not dismembered
into US ZIP codes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

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

    # ── SSN ────────────────────────────────────────────────────────────────────
    # Accepts dash-separated (123-45-6789) and space-separated (123 45 6789).
    # The bare 9-digit format (\d{9}) has too many false positives and is omitted.
    (
        "ssn",
        re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b"),
        None,
    ),

    # ── Email ──────────────────────────────────────────────────────────────────
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── US phone ───────────────────────────────────────────────────────────────
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

    # ── UK phone ───────────────────────────────────────────────────────────────
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

    # ── International phone (non-US, non-UK) ───────────────────────────────────
    # MUST appear BEFORE zip_us so "+91 98765 43210" is claimed as one entity
    # and its digit-groups are never subsequently split into US ZIP codes.
    # Matches: +CC NNNNN NNNNNN  /  +CC-NNN-NNNNN  etc.
    # Excludes +1 (US) and +44 (UK) — handled above.
    (
        "phone_intl",
        re.compile(
            r"(?<!\d)"
            r"\+(?!1[ \-.]|44[ \-.])"   # not US (+1) or UK (+44)
            r"[1-9]\d{0,2}"              # country code 1–3 digits
            r"[ \-.]"                    # separator
            r"\d{3,6}"                   # subscriber block 1
            r"[ \-.]"                    # separator
            r"\d{3,8}"                   # subscriber block 2
            r"(?:[ \-.]\d{2,6})?"        # optional trailing block
            r"(?!\d)"
        ),
        None,
    ),

    # ── Credit card (Luhn-validated) ───────────────────────────────────────────
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
        lambda m: _luhn_valid(re.sub(r"[\s\-]", "", m.group())),
    ),

    # ── Date of birth / dates ──────────────────────────────────────────────────
    # Four sub-formats, ordered to maximise correct matching:
    #   1. ISO 8601:        YYYY-MM-DD (e.g. 1990-08-30)
    #   2. Numeric:         MM/DD/YYYY  DD.MM.YYYY  DD-MM-YYYY
    #   3. Day-first:       14 January 1990 / 14-Jan-1990 / 14,Jan,1990
    #   4. Month-first:     January 14, 1990 / Jan 14 1990
    (
        "dob",
        re.compile(
            r"\b(?:"
            # Format 1 — ISO 8601 (year-first, strict ranges)
            r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            r"|"
            # Format 2 — numeric with / - . separators
            r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}"
            r"|"
            # Format 3 — day first  (14 Jan 1990, 14-January-1990, 14,Dec,1985)
            r"\d{1,2}[\s,\-]+"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,\-]+\d{4}"
            r"|"
            # Format 4 — month first  (January 14, 1990 / Jan 14 1990)
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{1,2}[\s,]+\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── IPv4 address ───────────────────────────────────────────────────────────
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        None,
    ),

    # ── API keys / secrets ─────────────────────────────────────────────────────
    # Coverage:
    #   • OpenAI/Anthropic sk- / sk_   (sk-proj-abc, sk_abc)
    #   • Anthropic ant-api-           (ant-api-ik92jbs...)
    #   • Bearer token
    #   • GitHub ghp_ / gho_
    #   • AWS AKIA
    #   • Google AIzaSy
    #   • Env-var assignment: VAR_NAME=<known-prefix>...
    #     e.g. CLAUDE_API_KEY=ant-api-...  OPENAI_KEY=sk-...
    (
        "api_key",
        re.compile(
            r"(?:"
            r"sk[-_][A-Za-z0-9\-_]{16,}"
            r"|ant-api-[A-Za-z0-9\-_]{16,}"
            r"|Bearer\s+[A-Za-z0-9\-_.]{16,}"
            r"|ghp_[A-Za-z0-9]{20,}"
            r"|gho_[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|AIzaSy[A-Za-z0-9\-_]{26,}"
            r"|[A-Z][A-Z0-9_]*=(?:sk[-_]|ant-api-|AIzaSy|ghp_|gho_|AKIA)"
            r"[A-Za-z0-9\-_.]{12,}"
            r")"
        ),
        None,
    ),

    # ── UK postcode ────────────────────────────────────────────────────────────
    (
        "postcode_uk",
        re.compile(
            r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── US ZIP code ────────────────────────────────────────────────────────────
    # MUST come AFTER phone_intl — see note above.
    (
        "zip_us",
        re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        None,
    ),
]


# ─────────────────────────────────────────────
# Main scan function
# ─────────────────────────────────────────────

def scan(text: str, skip_values: Optional[Set[str]] = None) -> List[DetectedEntity]:
    """
    Run all regex patterns against *text* and return detected entities.

    Overlapping matches for the same span are deduplicated (first match wins).
    All returned entities carry score = 1.0.

    The *skip_values* parameter prevents re-detection of surrogates that were
    generated in a previous turn.  Pass the current ShadowMap keys here to
    prevent double-wrapping of surrogate values quoted back by the user.

    Args:
        text:        The raw user message to scan.
        skip_values: Set of strings to skip even if they match a pattern.

    Returns:
        List of DetectedEntity objects, sorted by start position.
    """
    _skip: Set[str] = skip_values or set()

    results: List[DetectedEntity] = []
    occupied_spans: List[tuple] = []  # (start, end) already claimed

    def _span_free(start: int, end: int) -> bool:
        for s, e in occupied_spans:
            if not (end <= s or start >= e):
                return False
        return True

    for entity_type, pattern, validator in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            if not _span_free(start, end):
                continue

            matched_text = match.group().strip()

            if matched_text in _skip:
                logger.debug(
                    f"[PatternScan] Skipping known surrogate: {matched_text!r}"
                )
                continue

            if validator is not None and not validator(match):
                continue

            entity = DetectedEntity(
                text=matched_text,
                start=start,
                end=end,
                type=entity_type,
                score=1.0,
                source="pattern",
            )
            results.append(entity)
            occupied_spans.append((start, end))
            logger.debug(
                f"[PatternScan] {entity_type}: {entity.text!r} at [{start}:{end}]"
            )

    results.sort(key=lambda e: e.start)
    logger.info(f"[PatternScan] Found {len(results)} entities")
    return results