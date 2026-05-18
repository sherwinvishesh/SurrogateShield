"""
detection/pattern_scan.py — PatternScan

Regex-based PII detection. This module contains ONLY the regex logic.

Detects: street addresses, SSN, email, phone (US/UK/international),
credit card (Luhn-validated), date of birth, IPv4, API keys/secrets,
UK postcodes, US ZIP codes.

Key design decisions
────────────────────
• street address is detected HERE (PatternScan, structural regex) — not by
  downstream NER.  Detecting addresses in PatternScan means they are masked
  before EntityTrace and ContextGuard run, so the NER models never see
  address components and the geo-entity filter never mis-applies to them.
  This is how "99 Cathedral Close" is protected even without a person name
  in the same sentence.

• phone_intl comes before zip_us in the pattern list so that digit groups
  inside international numbers (e.g. "+91 98765 43210") are claimed as a
  phone entity before zip_us can split them into spurious ZIP codes.

• skip_values: exact match PLUS substring check (len >= 6).  The substring
  check is needed for service-query compatibility: after address fuzzing the
  fuzzed full address (e.g. "790 Crescent Row, Tempe, AZ") lives in
  skip_values, but PatternScan's address pattern would match only the street
  portion ("790 Crescent Row").  The substring check catches this.

Pattern order matters — patterns claim character spans; later patterns cannot
overlap earlier ones.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from util import DetectedEntity, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Luhn algorithm
# ─────────────────────────────────────────────

def _luhn_valid(number: str) -> bool:
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

_PATTERNS: list = [

    # ── Street address ─────────────────────────────────────────────────────────
    # Structural pattern: house-number + optional street-name words + type suffix.
    # Catches "99 Cathedral Close", "456 Innovation Plaza", "1126 E Apache Blvd",
    # "789 Crescent Row", "12 Close Mews" etc.
    # Being caught here means they are masked BEFORE NER runs, so they never
    # enter the topical-geo filter and are always protected.
    (
        "address",
        re.compile(
            r"\b\d+\s+"
            r"(?:[A-Za-z]+\s+){0,4}"
            r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
            r"Lane|Ln|Way|Court|Ct|Place|Pl|Row|Mews|Close|Crescent|Cres|"
            r"Parkway|Pkwy|Highway|Hwy|Freeway|Fwy|Terrace|Terr|"
            r"Circle|Cir|Loop|Trail|Trl|Plaza|Pass|Square|Sq|"
            r"Grove|Green|Park|Gardens?|View|Walk|Rise|Mount|Hill|"
            r"Gate|Alley|Chase|Heath|Meadow|Ridge|Vale|Glen)"
            r"\.?\b",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── SSN ────────────────────────────────────────────────────────────────────
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
            r"(?<!\d)(\+1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"
        ),
        None,
    ),

    # ── UK phone ───────────────────────────────────────────────────────────────
    (
        "phone_uk",
        re.compile(
            r"(?<!\d)(\+44\s?|0)"
            r"(\d{4}[\s\-]?\d{6}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{2}[\s\-]?\d{4}[\s\-]?\d{4})"
            r"(?!\d)"
        ),
        None,
    ),

    # ── International phone (non-US, non-UK) ───────────────────────────────────
    # MUST appear before zip_us.
    (
        "phone_intl",
        re.compile(
            r"(?<!\d)"
            r"\+(?!1[ \-.]|44[ \-.])"
            r"[1-9]\d{0,2}"
            r"[ \-.]"
            r"\d{3,6}"
            r"[ \-.]"
            r"\d{3,8}"
            r"(?:[ \-.]\d{2,6})?"
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
    (
        "dob",
        re.compile(
            r"\b(?:"
            r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            r"|"
            r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}"
            r"|"
            r"\d{1,2}[\s,\-]+"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,\-]+\d{4}"
            r"|"
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
    # - AIzaSy threshold lowered to {10,} (was {26,}) — catches shorter test keys
    # - KEY=value char class is [A-Za-z0-9\-_] (no dot) to prevent capturing
    #   trailing sentence-ending periods like "ant-api-abc123."
    (
        "api_key",
        re.compile(
            r"(?:"
            r"sk[-_][A-Za-z0-9\-_]{16,}"
            r"|ant-api-[A-Za-z0-9\-_]{16,}"
            r"|Bearer\s+[A-Za-z0-9\-_]{16,}"
            r"|ghp_[A-Za-z0-9]{20,}"
            r"|gho_[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|AIzaSy[A-Za-z0-9\-_]{10,}"
            r"|[A-Z][A-Z0-9_]*=(?:sk[-_]|ant-api-|AIzaSy|ghp_|gho_|AKIA)"
            r"[A-Za-z0-9\-_]{12,}"
            r")"
        ),
        None,
    ),

    # ── UK postcode ────────────────────────────────────────────────────────────
    (
        "postcode_uk",
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE),
        None,
    ),

    # ── US ZIP code — MUST come after phone_intl ───────────────────────────────
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

    Args:
        text:        Raw user message.
        skip_values: Surrogate strings to skip even if they match a pattern.
                     Checked by exact match AND substring (len >= 6) for
                     service-query address compatibility.

    Returns:
        List of DetectedEntity objects, sorted by start position.
    """
    _skip: Set[str] = skip_values or set()

    results: List[DetectedEntity] = []
    occupied_spans: List[tuple] = []

    def _span_free(s: int, e: int) -> bool:
        for os, oe in occupied_spans:
            if not (e <= os or s >= oe):
                return False
        return True

    def _should_skip(matched: str) -> bool:
        # Exact match (fast path)
        if matched in _skip:
            return True
        # Substring check for longer matches — handles fuzzed-address sub-parts
        # e.g. "790 Crescent Row" (14 chars) is a substring of skip value
        # "790 Crescent Row, Tempe, AZ"
        if len(matched) >= 6:
            for sv in _skip:
                if len(sv) > len(matched) and matched in sv:
                    return True
        return False

    for entity_type, pattern, validator in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            if not _span_free(start, end):
                continue

            matched_text = match.group().strip()

            if _should_skip(matched_text):
                logger.debug(f"[PatternScan] Skipping (skip_values): {matched_text!r}")
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
            logger.debug(f"[PatternScan] {entity_type}: {entity.text!r} at [{start}:{end}]")

    results.sort(key=lambda e: e.start)
    logger.info(f"[PatternScan] Found {len(results)} entities")
    return results