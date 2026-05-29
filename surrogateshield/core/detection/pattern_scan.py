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

import logging
import re
from typing import List, Optional, Set

from ..entities import DetectedEntity

logger = logging.getLogger(__name__)


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
# ABA routing number checksum
# ─────────────────────────────────────────────

def _aba_routing_valid(number: str) -> bool:
    """
    Validate a 9-digit ABA routing number using the standard checksum.
    Formula: (3*d0 + 7*d1 + d2 + 3*d3 + 7*d4 + d5 + 3*d6 + 7*d7 + d8) % 10 == 0
    """
    if len(number) != 9 or not number.isdigit():
        return False
    digits = [int(c) for c in number]
    checksum = (
        3 * digits[0] + 7 * digits[1] + digits[2] +
        3 * digits[3] + 7 * digits[4] + digits[5] +
        3 * digits[6] + 7 * digits[7] + digits[8]
    )
    return checksum % 10 == 0


# ─────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────

_PATTERNS: list = [

    # ── Street address ─────────────────────────────────────────────────────────
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
        re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b|\b\d{9}\b"),
        lambda m: (
            True if re.search(r"[ -]", m.group())
            else not _aba_routing_valid(m.group())
        ),
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

    # ── Gender indicator ───────────────────────────────────────────────────────
    (
        "gender_indicator",
        re.compile(
            r'\b(?:'
            r'(?:gender|sex)\s*[:=]\s*(?:male|female|m|f|man|woman|boy|girl|non-binary|nb)'
            r'|(?:i\s+am\s+a|i\'m\s+a)\s+(?:male|female|man|woman|boy|girl)'
            r'|(?:he/him|she/her|they/them)'
            r')\b',
            re.IGNORECASE,
        ),
        None,
    ),

    # ── UK postcode ────────────────────────────────────────────────────────────
    (
        "postcode_uk",
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE),
        None,
    ),

    # ── Cryptocurrency wallet address ────────────────────────────────────────
    (
        "crypto",
        re.compile(
            r"(?:"
            r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,36}\b"
            r"|\bbc1[ac-hj-np-z02-9]{6,87}\b"
            r"|\b0x[0-9a-fA-F]{40}\b"
            r")"
        ),
        None,
    ),

    # ── US ABA routing number ────────────────────────────────────────────────
    (
        "us_bank_number",
        re.compile(r"(?<!\d)\d{9}(?!\d)"),
        lambda m: _aba_routing_valid(m.group().strip()),
    ),

    # ── US Driver's License ──────────────────────────────────────────────────
    (
        "us_driver_license",
        re.compile(
            r"(?:driver'?s?\s+licen[sc]e(?:\s+(?:number|no|num|#))?"
            r"|licen[sc]e\s*(?:number|no|#|num)"
            r"|\bDL\b|\bD\.L\.\b)"
            r"[\s:\-#]*(?:is\s+|was\s+)?"
            r"(?-i:([A-Z0-9]{5,20}))\b",
            re.IGNORECASE,
        ),
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
        if matched in _skip:
            return True
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

            if entity_type == "us_driver_license":
                matched_text = (match.group(1) or "").strip()
                if not matched_text:
                    continue
                start = match.start(1)
                end   = match.end(1)
            else:
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
