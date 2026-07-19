# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
detection/pattern_scan.py — PatternScan

Regex-based PII detection. This module contains ONLY the regex logic.

Detects: street addresses, SSN, email, phone (US/UK/international),
payment cards (Luhn-validated, 13-19 digits incl. Amex/Diners), date of
birth, IPv4 + IPv6, MAC addresses, API keys/secrets, IBAN (mod-97
validated), VIN (check-digit validated), crypto wallets, US routing
numbers (ABA checksum), driver's licenses, passports, license plates,
generic labelled ID numbers (MRN, insurance/member IDs, account numbers,
KTN/PASSID, USCIS, Aadhaar…), UK postcodes, US ZIP codes.

Key design decisions
────────────────────
• street address is detected HERE (PatternScan, structural regex) — not by
  downstream NER.  Detecting addresses in PatternScan means they are masked
  before EntityTrace and ContextGuard run, so the NER models never see
  address components and the geo-entity filter never mis-applies to them.
  This is how "99 Cathedral Close" is protected even without a person name
  in the same sentence.

• checksum > keyword: patterns with a mathematical validator (Luhn, ABA,
  IBAN mod-97, VIN check digit) fire unconditionally — the checksum IS the
  evidence.  Bare numeric patterns with no checksum (standalone ZIP, bare
  9-digit SSN, routing numbers) require a nearby *positive* context word,
  and are suppressed by *negative* context ("order #", "invoice", "SKU",
  "port", "build") so counters and identifiers in business prose never
  become false positives.

• phones carry alphanumeric boundary guards so digit runs inside longer
  tokens (ETH addresses, artifact hashes, "PO-4805550123") can never be
  claimed as a phone number.

• phone_intl comes before phone_us so "+7 495 374 8120" is claimed whole,
  and all phones come after crypto/MAC/IP so hex-adjacent digit runs are
  claimed by the more specific pattern first.

Pattern order matters — patterns claim character spans; later patterns cannot
overlap earlier ones.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from detection import address_parser
from util import DetectedEntity, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Checksum validators
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


def _iban_valid(candidate: str) -> bool:
    """ISO 13616 mod-97 check. The checksum makes context words unnecessary."""
    s = re.sub(r"\s+", "", candidate).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


_VIN_VALUES = {c: v for c, v in zip(
    "0123456789ABCDEFGHJKLMNPRSTUVWXYZ",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
     1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9],
)}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def _vin_valid(vin: str) -> bool:
    """ISO 3779 check digit (position 9) — vanishingly rare in random text."""
    vin = vin.upper()
    if len(vin) != 17 or any(c not in _VIN_VALUES for c in vin):
        return False
    total = sum(_VIN_VALUES[c] * w for c, w in zip(vin, _VIN_WEIGHTS))
    check = total % 11
    expected = "X" if check == 10 else str(check)
    return vin[8] == expected


# ─────────────────────────────────────────────
# Context probes (positive / negative evidence for bare numeric patterns)
# ─────────────────────────────────────────────

# Business-prose labels that mean "this number is a counter, not PII".
_NEG_NUM_CONTEXT = re.compile(
    r"(?:order|invoice|sku|part|batch|build|budget|port|ticket|tracking"
    r"|item|model|serial|imei|ref|reference|case|confirmation|receipt"
    r"|txn|transaction|version|score|error|hash|artifact|quantity|qty"
    r"|p\.?o\.?)"
    r"\s*(?:number|no\.?|num|id)?\s*[:#\-]*\s*$",
    re.IGNORECASE,
)

_ZIP_POS_CONTEXT = re.compile(
    r"(?:zip|zipcode|postal|postcode)\s*(?:code)?\s*(?:is|was|[:=#])?\s*$",
    re.IGNORECASE,
)

# Geographic framing on either side ("within the 60611 area", "deliver to
# 85281") — weaker than an explicit ZIP label but still zip-shaped usage.
_ZIP_GEO_CONTEXT = re.compile(
    r"\b(?:zip|zipcode|postal|postcode|area|region|neighborhood|neighbourhood"
    r"|city|county|district|address|located|location|deliver(?:y|ed)?"
    r"|residents?|resid\w+|coverage|zones?|municipal|borough"
    r"|boundar(?:y|ies)|destinations?|living|lives?)\b",
    re.IGNORECASE,
)

_SSN_PRE_CONTEXT = re.compile(
    r"(?:\bssn\b|social\s+security|soc\.?\s*sec|taxpayer|tax\s*id|\bitin\b|\btin\b)"
    r"[^.\n]{0,50}$",
    re.IGNORECASE,
)
_SSN_POST_CONTEXT = re.compile(
    r"^[^.\n]{0,30}?(?:\bssn\b|social\s+security)",
    re.IGNORECASE,
)

_BANK_CONTEXT = re.compile(
    r"(?:routing|\baba\b|\brtn\b|bank|account|acct|wire|ach|deposit)"
    r"[^.\n]{0,50}$",
    re.IGNORECASE,
)


def _before(m: "re.Match", n: int) -> str:
    return m.string[max(0, m.start() - n):m.start()]


def _after(m: "re.Match", n: int) -> str:
    return m.string[m.end():m.end() + n]


def _ssn_validator(m: "re.Match") -> bool:
    s = m.group()
    if re.search(r"[ -]", s):
        return True  # formatted SSN is distinctive on its own
    if _aba_routing_valid(s):
        return False  # leave ABA-valid 9-digit numbers to us_bank_number
    # Bare 9 digits: needs SSN-ish context, else it's an invoice/serial.
    return bool(
        _SSN_PRE_CONTEXT.search(_before(m, 60))
        or _SSN_POST_CONTEXT.match(_after(m, 40))
    )


def _bank_validator(m: "re.Match") -> bool:
    if not _aba_routing_valid(m.group().strip()):
        return False
    # 10% of random 9-digit numbers pass the ABA checksum — require a
    # banking context word so invoice numbers can't slip through.
    return bool(_BANK_CONTEXT.search(_before(m, 60)))


def _zip_validator(m: "re.Match") -> bool:
    pre = _before(m, 30)
    if _NEG_NUM_CONTEXT.search(pre):
        return False
    if _ZIP_POS_CONTEXT.search(pre):
        return True
    # geographic framing nearby ("within the 60611 area", "deliver to 85281")
    if (_ZIP_GEO_CONTEXT.search(_before(m, 50))
            or _ZIP_GEO_CONTEXT.search(_after(m, 50))):
        return True
    # a bare uppercase state abbreviation right before ("IL 60611")
    if re.search(r"\b[A-Z]{2}\s*,?\s*$", pre):
        return True
    # ZIP+4 shape is specific enough on its own; bare 5 digits are not
    # (ZIPs inside addresses are already claimed by the address parser).
    return "-" in m.group()


def _phone_validator(m: "re.Match") -> bool:
    return not _NEG_NUM_CONTEXT.search(_before(m, 30))


def _card_validator(m: "re.Match") -> bool:
    digits = re.sub(r"[\s\-]", "", m.group())
    if not (13 <= len(digits) <= 19):
        return False
    if not _luhn_valid(digits):
        return False
    return not _NEG_NUM_CONTEXT.search(_before(m, 30))


_CARD_CONTEXT = re.compile(
    r"(?:credit|debit|card|visa|mastercard|amex|american\s+express"
    r"|discover|payment)\b[^.\n]{0,30}$",
    re.IGNORECASE,
)


def _card_ctx_validator(m: "re.Match") -> bool:
    """Luhn-INVALID card-length numbers still get masked when the text says
    they are a card ("payment card is Visa 5412751234123412") — typos and
    synthetic numbers are card-intent PII."""
    digits = re.sub(r"[\s\-]", "", m.group())
    if not (13 <= len(digits) <= 19):
        return False
    if _NEG_NUM_CONTEXT.search(_before(m, 30)):
        return False
    return bool(_CARD_CONTEXT.search(_before(m, 40)))


def _has_digit(m: "re.Match") -> bool:
    return any(c.isdigit() for c in (m.group(1) or ""))


def _plate_validator(m: "re.Match") -> bool:
    v = m.group(1) or ""
    # real plates virtually always mix letters and digits; requiring both
    # stops "price tag is 45000" from becoming a plate
    return any(c.isdigit() for c in v) and any(c.isalpha() for c in v)


def _dob_validator(m: "re.Match") -> bool:
    s = m.group()
    if _before(m, 1) in ("v", "V"):
        return False  # "v2.5.10" is a version, not a date
    # short-year form (mm/dd/yy) is the weakest shape — reject it in
    # counter/version contexts ("release 2.5.24") and impossible dates
    if re.fullmatch(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2}", s):
        if _NEG_NUM_CONTEXT.search(_before(m, 30)):
            return False
        a, b, _ = re.split(r"[/\-.]", s)
        a, b = int(a), int(b)
        if not (1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12)):
            return False
    return True


# ─────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────

# Patterns whose PII value is capture group 1 (keyword-gated patterns).
_GROUP1_TYPES = frozenset({
    "us_driver_license", "passport", "id_number", "license_plate",
})

_PATTERNS: list = [
    # NOTE: street addresses are detected by the canonical structured parser
    # (address_parser.find_addresses) in scan() BEFORE this pattern list runs,
    # so the full span — street + unit + city + state + ZIP — is claimed as
    # ONE entity and later patterns (zip_us, ssn, phone) can never split
    # components out of an address. Being caught first also means they are
    # masked BEFORE NER runs, so they never enter the topical-geo filter.

    # ── Email ──────────────────────────────────────────────────────────────────
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
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
            r"|eyJ[A-Za-z0-9\-_]{8,}(?:\.[A-Za-z0-9\-_]+){0,2}"
            r"|[A-Z][A-Z0-9_]*=(?:sk[-_]|ant-api-|AIzaSy|ghp_|gho_|AKIA)"
            r"[A-Za-z0-9\-_]{12,}"
            # snake_case identifier that names itself a secret AND embeds a
            # long digit run ("secret_auth_token_9988776655_admin")
            r"|\b(?=[A-Za-z0-9_]*(?:secret|token|auth|key))"
            r"(?=[A-Za-z0-9_]*\d{6,})"
            r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b"
            r")"
        ),
        None,
    ),

    # ── Cryptocurrency wallet address ────────────────────────────────────────
    # MUST come before the phone patterns: ETH addresses contain 10-digit
    # runs bounded by hex letters that a phone pattern would otherwise claim.
    (
        "crypto",
        re.compile(
            r"(?:"
            r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,36}\b"   # BTC legacy P2PKH/P2SH
            r"|\bbc1[ac-hj-np-z02-9]{6,87}\b"          # BTC Bech32
            r"|\b0x[0-9a-fA-F]{40}\b"                   # Ethereum
            r")"
        ),
        None,
    ),

    # ── MAC address — before IPv6 (both are colon-separated hex) ─────────────
    (
        "mac_address",
        re.compile(
            r"(?<![0-9A-Fa-f:\-])"
            r"(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}"
            r"(?![0-9A-Fa-f:\-])"
        ),
        None,
    ),

    # ── IPv6 address ───────────────────────────────────────────────────────────
    # Full form needs all 8 groups; every compressed alternative structurally
    # requires "::", so clock times ("12:30:45") can never match.
    (
        "ip_address",
        re.compile(
            r"(?<![0-9A-Za-z:.])"
            r"(?:"
            r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,7}:"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}"
            r"|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}"
            r"|:(?::[0-9A-Fa-f]{1,4}){1,7}"
            r")"
            r"(?![0-9A-Za-z:])"
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

    # ── VIN (check-digit validated — no keyword needed) ──────────────────────
    (
        "vin",
        re.compile(r"\b(?-i:[A-HJ-NPR-Z0-9]{17})\b"),
        lambda m: _vin_valid(m.group()),
    ),

    # ── IBAN (mod-97 validated — no keyword needed) ──────────────────────────
    # Accepts compact ("DE8937040044…") and 4-char-grouped spaced form
    # ("GB82 WEST 1234 5698 7654 32"); the checksum kills false matches.
    (
        "iban",
        re.compile(
            r"\b(?-i:[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?)\b"
        ),
        lambda m: _iban_valid(m.group()),
    ),

    # ── International phone (non-US, non-UK) ───────────────────────────────────
    # MUST appear before phone_us (so "+7 495 374 8120" is claimed whole and
    # phone_us cannot grab just the "495 374 8120" tail) and before zip_us.
    (
        "phone_intl",
        re.compile(
            r"(?<![0-9A-Za-z_])"
            r"\+(?!1[ \-.]|44[ \-.])"
            r"[1-9]\d{0,2}"
            r"(?:[ \-.]\d{1,9}){1,6}"
            r"(?![0-9A-Za-z_])"
        ),
        lambda m: 9 <= len(re.sub(r"\D", "", m.group())) <= 15,
    ),

    # ── US phone (alnum-guarded, optional extension) ───────────────────────────
    (
        "phone_us",
        re.compile(
            r"(?<![A-Za-z0-9_.\-])(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
            r"(?:\s*(?:ext|extension|x)\.?\s*\d{1,6})?"
            r"(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
        _phone_validator,
    ),

    # ── UK phone ───────────────────────────────────────────────────────────────
    (
        "phone_uk",
        re.compile(
            r"(?<![A-Za-z0-9_.\-])(\+44\s?|0)"
            r"(\d{4}[\s\-]?\d{6}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{2}[\s\-]?\d{4}[\s\-]?\d{4})"
            r"(?![A-Za-z0-9_])"
        ),
        _phone_validator,
    ),

    # ── Payment card (Luhn-validated, 13-19 digits) ────────────────────────────
    # Covers Visa/MC 16, Amex 15 (4-6-5 or bare), Diners 14, Discover 16-19.
    (
        "credit_card",
        re.compile(
            r"(?<![0-9A-Za-z_.\-])"
            r"(?:\d[ \-]?){12,18}\d"
            r"(?![0-9A-Za-z_])"
        ),
        _card_validator,
    ),

    # ── US Driver's License (keyword-gated, value = group 1) ─────────────────
    # Value must contain a digit ("license AGREEMENT" can never match).
    # Dashed formats (Florida "G645-201-88-123-0") are supported.
    (
        "us_driver_license",
        re.compile(
            r"(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e"
            r"|licen[sc]e\s*(?:number|no\.?|#|num)"
            r"|\bDL\b|\bD\.L\.\b"
            # bare "license" gates ONLY the unmistakable dashed DL shape
            r"|licen[sc]e(?=[\s:\-#]*(?:is\s+|was\s+)?[A-Z]?\d{3}-\d{3}-\d{2}-\d{3}-\d)"
            r")"
            r"[\s:\-#]*(?:is\s+|was\s+)?"
            r"(?-i:([A-Z]?\d{3}-\d{3}-\d{2}-\d{3}-\d|[A-Z0-9]{5,20}))\b",
            re.IGNORECASE,
        ),
        _has_digit,
    ),

    # ── Passport (keyword-gated, value = group 1) ────────────────────────────
    # Must come before ssn: US passports are 9 digits.
    (
        "passport",
        re.compile(
            r"passport(?:\s+(?:number|no\.?|num|card|#))?"
            r"\s*[:\-#]*\s*(?:is\s+|was\s+)?"
            r"(?-i:([A-Z0-9]{6,9}))\b",
            re.IGNORECASE,
        ),
        _has_digit,
    ),

    # ── Labelled ID numbers (keyword-gated, value = group 1) ─────────────────
    # MRN, insurance/member/subscriber IDs, account numbers, patient IDs,
    # employee/badge IDs, KTN/PASSID, USCIS/A-numbers, Aadhaar.
    (
        "id_number",
        re.compile(
            r"(?:\bmrn\b|medical\s+record(?:\s+(?:number|no\.?))?"
            r"|insurance\s*(?:id|number|no\.?|#)"
            r"|medicare\s*(?:id|number|no\.?|#)|medicaid\s*(?:id|number|no\.?|#)"
            r"|policy\s*(?:id|number|no\.?|#)"
            r"|member(?:ship)?\s*(?:id|number|no\.?|#)"
            r"|subscriber\s*(?:id|number|no\.?)"
            r"|acc(?:oun)?t\s*(?:number|no\.?|num|#)"
            r"|patient\s*(?:id|identifier|number|no\.?)"
            r"|employee\s*(?:id|number|no\.?)"
            r"|badge\s*(?:id|number|no\.?)"
            r"|\bktn\b|known\s+traveler(?:\s+number)?"
            r"|\bpassid\b|global\s+entry(?:\s+passid)?"
            r"|uscis(?:\s*(?:number|no\.?|#))?|a-?number"
            r"|alien\s+(?:registration\s+)?number"
            r"|aadha{1,2}r(?:\s+(?:number|no\.?|card))?"
            r"|national\s+id(?:\s+(?:number|no\.?))?"
            r")"
            r"\s*[:\-#]*\s*(?:is\s+|was\s+)?"
            r"(\d{4}\s\d{4}\s\d{4}|(?-i:[A-Z0-9][A-Z0-9\-]{3,19}))\b",
            re.IGNORECASE,
        ),
        _has_digit,
    ),

    # ── License plate (keyword-gated, value = group 1) ───────────────────────
    (
        "license_plate",
        re.compile(
            r"(?:licen[sc]e\s+plate|number\s+plate|\bplate\b|\btag\b)"
            r"\s*(?:number|no\.?|#)?\s*[:\-#]*\s*(?:is\s+|was\s+)?"
            r"(?-i:([A-Z0-9]{2,3}[\- ]?[A-Z0-9]{2,5}))\b",
            re.IGNORECASE,
        ),
        _plate_validator,
    ),

    # ── Payment card, context-gated fallback (Luhn-invalid but card-framed) ──
    (
        "credit_card",
        re.compile(
            r"(?<![0-9A-Za-z_.\-])"
            r"(?:\d[ \-]?){12,18}\d"
            r"(?![0-9A-Za-z_])"
        ),
        _card_ctx_validator,
    ),

    # ── SSN ────────────────────────────────────────────────────────────────────
    # Formatted (123-45-6789 / 123 45 6789) fires unconditionally; bare
    # 9-digit needs SSN-ish context and must not be ABA-valid.
    (
        "ssn",
        re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b|\b\d{9}\b"),
        _ssn_validator,
    ),

    # ── US ABA routing number (checksum + banking context) ───────────────────
    (
        "us_bank_number",
        re.compile(r"(?<!\d)\d{9}(?!\d)"),
        _bank_validator,
    ),

    # ── Date of birth / dates ──────────────────────────────────────────────────
    # ISO, mm/dd/yyyy, mm/dd/yy, ordinal day forms ("March 3rd, 1946",
    # "3rd of March 1946"), month-name forms.
    (
        "dob",
        re.compile(
            r"\b(?:"
            r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            r"|"
            r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}"
            r"|"
            r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2}(?![\d/\-.])"
            r"|"
            r"\d{1,2}(?:st|nd|rd|th)?[\s,\-]+(?:of\s+)?"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,\-]+\d{4}"
            r"|"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
        _dob_validator,
    ),

    # ── Gender indicator ───────────────────────────────────────────────────────
    (
        "gender_indicator",
        re.compile(
            r'\b(?:'
            r'(?:gender|sex)\s*[:=]\s*(?:male|female|m|f|man|woman|boy|girl|non-binary|nb)'
            r'|(?:i\s+am\s+a|i\'m\s+a)\s+(?:male|female|man|woman|boy|girl)'
            r'|identif(?:y|ies)\s+as\s+(?:male|female|a\s+man|a\s+woman|non-?binary|nb)'
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

    # ── US ZIP code (context-gated) ────────────────────────────────────────────
    # ZIPs inside addresses are claimed by the address parser; a standalone
    # 5-digit number is only a ZIP when the text says so.
    (
        "zip_us",
        re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        _zip_validator,
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

    # ── Street addresses first: the canonical parser claims the FULL span
    # (street + unit + city + state + ZIP) as one entity.
    for parsed in address_parser.find_addresses(text):
        if not _span_free(parsed.start, parsed.end):
            continue
        if _should_skip(parsed.full_text):
            logger.debug(f"[PatternScan] Skipping (skip_values): {parsed.full_text!r}")
            continue
        entity = DetectedEntity(
            text=parsed.full_text,
            start=parsed.start,
            end=parsed.end,
            type="address",
            score=1.0,
            source="pattern",
            parsed=parsed,
        )
        results.append(entity)
        occupied_spans.append((parsed.start, parsed.end))
        logger.debug(
            f"[PatternScan] address: {entity.text!r} at [{parsed.start}:{parsed.end}]"
        )

    for entity_type, pattern, validator in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            if not _span_free(start, end):
                continue

            if entity_type in _GROUP1_TYPES:
                matched_text = (match.group(1) or "").strip()
                if not matched_text:
                    continue
                start = match.start(1)
                end   = match.end(1)
                if not _span_free(start, end):
                    continue
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
