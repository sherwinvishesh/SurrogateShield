"""
detection/pattern_scan.py — PatternScan

Regex-based PII detection. This module contains ONLY the regex logic.

Detects: street addresses, SSN, email, phone (US/UK/international),
payment cards (Luhn-validated, 13-19 digits incl. Amex/Diners), date of
birth, IPv4 + IPv6, MAC addresses, API keys/secrets, IBAN (mod-97
validated), VIN (check-digit validated), crypto wallets, US routing
numbers (ABA checksum), driver's licenses, passports, license plates,
generic labelled ID numbers (MRN, insurance/member IDs, account numbers,
KTN/PASSID, USCIS, Aadhaar, EIN/ITIN, customer/student/client numbers,
CPF/DNI/NIF…), UK postcodes, US ZIP codes, ages ("34 years old", "my mom
is 72"), years of birth, social/chat/game handles, personal URLs, and
credentials (passwords, one-time and backup codes, tokens) — audit I14.

• URLs are found first (audit I1). A personal URL (a profile on a known
  host, a person path such as /in/ or /team/, a personal domain introduced
  as "my site" or named after someone in the message) is one "url" entity. Any other URL is opaque: nothing inside
  it is matched except secret query parameters (?t=, token=) and e-mail
  addresses, and opaque_spans() lets the pipeline hide it from the NER
  stages.

• dates are a DOB only with a birth cue in the 40 characters before them,
  or when the year is at least five years in the past; ISO timestamps,
  version strings and future dates (deadlines, lease ends) are kept.

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

import datetime
import logging
import re
import unicodedata
from typing import List, Optional, Set

from ..entities import DetectedEntity
from . import address_assembly, address_parser
from .canonical import number_value

logger = logging.getLogger(__name__)


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
    r"|item|model|serial|ref|reference|case|confirmation|receipt"
    r"|txn|transaction|version|score|error|hash|artifact|quantity|qty"
    r"|p\.?o\.?)"
    r"\s*(?:number|no\.?|num|id|code)?\s*[:#\-]*\s*$",     # "error code 4631872866"
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
    if _NEG_NUM_CONTEXT.search(_before(m, 30)):
        return False  # "item 219-44-8812 is listed twice"
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
    if re.search(r"[A-Za-z0-9][\-#/]$", pre):
        return False                      # tail of a code ("Order #A-77219")
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
    digits = re.sub(r"\D", "", m.group())      # the digits, whatever separates them
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
    return (any(c.isdigit() for c in v) and any(c.isalpha() for c in v)
            and sum(c.isalnum() for c in v) <= 8)


_BIRTH_CUE = re.compile(
    r"(?:born|birth|\bdob\b|d\.o\.b|b-?day|nacimiento|naissance|geburt|nascimento"
    r"|nacido|n[ée]e?\s+le|geboren)",
    re.IGNORECASE,
)

# A labelled date that is not a birth date
_NOT_BIRTH_DATE = re.compile(
    r"(?i)(?:\bdate\s+of\s+(?:entry|arrival|departure|issue|expiry|expiration|purchase|service"
    r"|incident|loss|travel|hire|admission|discharge|marriage|death)|\b(?:entry|arrival|departure"
    r"|issue|expiry|expiration|purchase|hire|start|end|due|ship|delivery)\s+date"
    r"|\bday\s+of\s+the\s+week\s+(?:was|is|will\s+be|fell\s+on)?)\W{0,4}$")

# A date with no birth cue is treated as a date of birth only when its year
# is at least this many years in the past (audit I8). Ship dates, deadlines,
# lease ends and log timestamps are recent or future and stay in the text;
# an old date in personal prose is still masked (fail closed).
_DOB_MIN_AGE_WITHOUT_CUE = 5
_THIS_YEAR = datetime.date.today().year


def _date_year(s: str) -> Optional[int]:
    """The year of a date string, or None when it has no recognisable year."""
    m = re.search(r"(?:19|20)\d{2}", s)
    if m:
        return int(m.group())
    m = re.search(r"(?:'|[/\-.])(\d{2})$", s.strip())
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy <= _THIS_YEAR % 100 else 1900 + yy
    return None


def _dob_validator(m: "re.Match") -> bool:
    s = m.group()
    if _before(m, 1) in ("v", "V"):
        return False  # "v2.5.10" is a version, not a date
    if _after(m, 1) == "T":
        return False  # ISO-8601 timestamp ("2026-10-03T14:21:55Z")
    # short-year form (mm/dd/yy) is the weakest shape — reject it in
    # counter/version contexts ("release 2.5.24") and impossible dates
    if re.fullmatch(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2}", s):
        if _NEG_NUM_CONTEXT.search(_before(m, 30)):
            return False
        a, b, _ = re.split(r"[/\-.]", s)
        a, b = int(a), int(b)
        if not (1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12)):
            return False
    if _BIRTH_CUE.search(_before(m, 40)):
        return True
    if _NOT_BIRTH_DATE.search(_before(m, 40)):
        return False              # "Date of entry: 04/12/2019", "what day of the week was …"
    year = _date_year(s)
    return year is not None and year <= _THIS_YEAR - _DOB_MIN_AGE_WITHOUT_CUE


def _ipv4_validator(m: "re.Match") -> bool:
    # loopback, "any" and netmasks identify nobody (audit I8)
    s = m.group()
    return not (s.startswith(("127.", "0.", "255.")) or s == "0.0.0.0")


# ── age ──────────────────────────────────────────────────────────────────────

# The letters of a name in the age rules: ASCII and the Latin letters beyond
# it (Latin-1 Supplement, Extended-A and -B: "Özge", "Ștefan", "Łukasz",
# "İlkay"), capitals and small letters kept apart.
_LATIN = [chr(c) for c in range(0xC0, 0x250)]
_UP = "A-Z" + "".join(c for c in _LATIN if c.isupper())
_LO = "a-z" + "".join(c for c in _LATIN if c.islower())

_KIN = (
    r"(?:mom|mum|mother|dad|father|son|daughter|kid|child|baby|toddler|brother"
    r"|sister|grandma|grandmother|granny|grandpa|grandfather|grandson|granddaughter"
    r"|wife|husband|partner|spouse|boyfriend|girlfriend|fianc[ée]e?|aunt|uncle"
    r"|niece|nephew|cousin|friend|roommate|boss|patient|client|neighbou?r|stepson"
    r"|stepdaughter|twins?|grandad|granddad|gran|nana|nanna|nani|nonna|nonno"
    r"|abuel[ao]|oma|opa)"
)
# Words that turn "I'm 30" into a measurement or a count, not an age.
_NOT_AGE_AFTER = re.compile(
    r"\s*(?:%|percent|['\"′″]|[.:,]\d|/|[-–]\d|x\b|k\b|am\b|pm\b|st\b|nd\b|rd\b|th\b"
    r"|(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|seconds?|secs?|lbs?|kg|kilos?"
    r"|pounds|cm|mm|inch(?:es)?|in(?=\s*[.,;)\n]|\s*$|\s+(?:tall|long|wide|high|deep))|feet|ft|foot|miles?|mi\b|km|meters?|metres?"
    r"|dollars?|bucks|euros?|times|of\b|out\b|minutes|points?|pts|degrees?"
    r"|steps?|items?|pages?|people|units?|cents?)\b)",
    re.IGNORECASE,
)


def _value_group(m: "re.Match"):
    """The value group of a match: ``v``, or ``w`` for a pattern whose
    alternatives each name their value (only one of them participates)."""
    if m.group("v") is not None:
        return "v"
    if "w" in m.re.groupindex and m.group("w") is not None:
        return "w"
    return None


def _age_validator(m: "re.Match") -> bool:
    digits = re.findall(r"\d+", m.group(_value_group(m)))
    if not digits or not (0 < int(digits[0]) <= 120):
        return False
    return not _NOT_AGE_AFTER.match(_after(m, 12))


# The shape of an age a model reads (V4 §3.3: the tagger's AGE candidates
# keep only these): a number from 1 to 120, in digits or in words, with at
# most an age cue around it ("34", "34yo", "34 years old", "aged 34",
# "thirty-four"), and no unit or count after it, as for the patterns.
_AGE_SHAPE = re.compile(
    r"(?i)(?:aged?[ \t]*[:=]?[ \t]*)?(?P<v>\d{1,3}|[a-z]+(?:[ \t]*-[ \t]*|[ \t]+)?[a-z]*)"
    r"(?:[ \t]*-?[ \t]*(?:years?|yrs?\.?|y/o|yo)(?:[ \t]*-?[ \t]*old)?|[mf])?"
)


def age_shape(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]``, read as an age by a model, has an age's shape."""
    m = _AGE_SHAPE.fullmatch(text, start, end)
    if m is None:
        return False
    v = m.group("v")
    n = int(v) if v.isdigit() else number_value(v)
    return n is not None and 0 < n <= 120 and not _NOT_AGE_AFTER.match(text, end)


# ── handles, credentials ─────────────────────────────────────────────────────

# "@token" forms that are code or chat syntax, not a person's handle.
_NOT_HANDLE = frozenset({
    "media", "import", "keyframes", "font-face", "charset", "supports", "page",
    "layer", "container", "tailwind", "apply", "use", "include", "mixin",
    "extend", "return", "if", "else", "each", "function", "property",
    "dataclass", "staticmethod", "classmethod", "override", "here", "channel",
    "everyone", "all", "param", "returns", "throws", "author", "since",
    "deprecated", "see", "type", "example", "todo", "component", "test",
})

# default system accounts identify nobody
_SYSTEM_ACCOUNTS = frozenset({
    "root", "admin", "administrator", "guest", "postgres", "ubuntu", "pi", "user",
    "test", "default", "anonymous", "nobody", "sa", "oracle", "mysql", "ftp",
    "www-data", "nginx", "apache", "git", "jenkins", "deploy", "ec2-user", "centos",
    "debian", "support", "operator", "daemon", "bin", "sys", "backup", "service",
})

_HANDLE_STOP = frozenset({
    "is", "was", "the", "a", "an", "and", "or", "my", "your", "for", "to",
    "not", "name", "names", "here", "there", "below", "above", "same",
    "different", "wrong", "correct", "incorrect", "invalid", "taken",
    "available", "required", "missing", "blank", "empty", "changed", "field",
    "password", "username", "account", "login", "email", "please", "help",
    "gamertag", "handle", "id", "tag", "on", "in", "at", "with",
    "one", "ones", "too", "also", "got", "has", "had", "and", "but", "it",
}) | _SYSTEM_ACCOUNTS


def _at_handle_validator(m: "re.Match") -> bool:
    v = m.group("v")[1:]
    if v.lower() in _NOT_HANDLE:
        return False
    if _after(m, 1) == "(":
        return False                      # "@app.route(" — a decorator call
    line_start = m.string.rfind("\n", 0, m.start()) + 1
    line_end = m.string.find("\n", m.end())
    line = m.string[line_start:line_end if line_end >= 0 else None].strip()
    return line != m.group("v")           # "@dataclass" alone on its line


def _tag_handle_validator(m: "re.Match") -> bool:
    name = m.group("v").split("#")[0]
    return (any(c.islower() for c in name)
            and name.lower() not in {"issue", "pr", "ticket", "order", "item",
                                     "room", "no", "case", "bug", "build", "ref",
                                     "apt", "unit", "suite", "page", "step",
                                     "line", "rule", "task", "store", "table"})


def _keyword_handle_validator(m: "re.Match") -> bool:
    v = m.group("v")
    if v.lower() in _HANDLE_STOP:
        return False
    # a plain lowercase word is a handle only behind an explicit username label
    distinctive = (any(c.isdigit() or c in "._-" for c in v)
                   or (v[:1].islower() and any(c.isupper() for c in v[1:]))
                   or (any(c.isupper() for c in v) and any(c.islower() for c in v)
                       and "_" in v))
    strong = m.groupdict().get("strong")
    if not strong and not distinctive and m.re.groupindex.get("strong"):
        gap = m.string[m.start():m.start("v")]
        tail = m.string[m.end():m.end() + 2]
        return (bool(re.search(r"[:=]\s*@?$", gap)) and v.islower()
                and (tail == "" or tail[:1] in "\n,;)" or tail in (". ", ".\n", ".")))
    if strong and strong.lower() == "handle" and not distinctive:
        # the verb: "which roses handle August heat" — a bare word is a
        # handle only as "my handle is X" / "handle: X"
        gap = m.string[m.end("strong"):m.start("v")]
        before = m.string[max(0, m.start() - 12):m.start()].lower()
        if not re.search(r"[=:@]|\b(?:is|was)\b", gap) and not re.search(
                r"\b(?:my|his|her|their|your|our|the)\s+$", before):
            return False
    return distinctive or bool(strong)


_CODE_REFERENCE = re.compile(
    r"^(?:[$%{<]|[A-Za-z_][\w.]*[\[(]|[A-Z][A-Z0-9_]*$|(?:os|process|env|config"
    r"|settings|self|this|request|req)\.)"
)


def _credential_validator(m: "re.Match") -> bool:
    v = m.group("v").rstrip(".,;:)]}")
    if _CODE_REFERENCE.match(v):
        return False          # a variable, env lookup or template, not a secret
    has_digit = any(c.isdigit() for c in v)
    has_symbol = any(not c.isalnum() for c in v)
    mixed = any(c.isupper() for c in v) and any(c.islower() for c in v)
    return has_digit or (has_symbol and len(v) >= 6) or (mixed and len(v) >= 8)


_INSURANCE_CUE = re.compile(r"(?i)\b(?:member|policy|insurance|insurer|plan|eob|payer"
                            r"|subscriber|rx|bin|pcn|copay|deductible|claim)\b")


def _id_value_validator(m: "re.Match") -> bool:
    v = m.group("v") or ""
    digits = sum(c.isdigit() for c in v)
    if re.match(r"(?i)group\b", m.group()):
        # "group 44190" is a health-plan group number only among plan details
        return digits >= 4 and bool(_INSURANCE_CUE.search(_before(m, 120)))
    # a letter-and-digit serial ("serial PF3K8L2M") carries fewer digits
    mixed = len(v) >= 6 and digits >= 2 and sum(c.isalpha() for c in v) >= 2
    return (digits >= 4 or mixed) and not re.fullmatch(r"(?:19|20)\d\d", v)


_SSN_SHAPE = re.compile(r"\d{3}-\d{2}-\d{4}")


def _phone_ctx_validator(m: "re.Match") -> bool:
    # The label may be far back ("Contact alice@corp.com, SSN 123-45-6789"):
    # the gap never crosses an e-mail, and the US SSN shape is left to "ssn".
    n = len(re.sub(r"\D", "", m.group("v")))
    return (8 <= n <= 13 and not _SSN_SHAPE.fullmatch(m.group("v"))
            and not _NEG_NUM_CONTEXT.search(_before(m, 30)))


# ── URLs ─────────────────────────────────────────────────────────────────────

_URL_TLDS = (
    r"com|org|net|edu|gov|io|dev|me|app|co|ai|gg|tv|xyz|info|biz|site|page"
    r"|blog|online|tech|us|uk|ca|de|fr|es|it|nl|br|au|jp|cn|ru|ch|se|no|fi"
    r"|dk|pl|pt|ie|nz|in|ly|to|sh|so|link|social|art|design|studio|email"
    r"|ng|za|ke|ar|mx|cl|pe|co\.uk|photo|photography|photos|name|portfolio|bio|cv|family|shop|store"
)
_URL_RE = re.compile(
    r"(?:https?://|\bwww\.)[^\s<>()\[\]{}\"'`|]+[^\s<>()\[\]{}\"'`|.,;:!?]"
    # bare host.tld/path or host.tld (lowercase TLD; never after "@" or ".")
    r"|(?<![\w@.%+\-/])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"
    r"(?-i:" + _URL_TLDS + r")\b(?![@\-])"
    r"(?:/[^\s<>()\[\]{}\"'`|]*[^\s<>()\[\]{}\"'`|.,;:!?])?",
)

# Hosts whose first path segment (or the one after /in/, /u/, …) is a person.
_PROFILE_HOSTS = frozenset({
    "linkedin.com", "github.com", "gitlab.com", "bitbucket.org", "twitter.com",
    "x.com", "instagram.com", "facebook.com", "fb.com", "tiktok.com",
    "youtube.com", "medium.com", "reddit.com", "behance.net", "dribbble.com",
    "threads.net", "bsky.app", "t.me", "wa.me", "venmo.com", "paypal.me",
    "cash.app", "linktr.ee", "about.me", "keybase.io", "pinterest.com",
    "snapchat.com", "twitch.tv", "soundcloud.com", "patreon.com",
    "calendly.com", "stackoverflow.com", "kaggle.com", "huggingface.co",
    "mastodon.social", "orcid.org", "scholar.google.com", "vimeo.com",
    "ko-fi.com", "buymeacoffee.com", "depop.com", "gumroad.com", "onlyfans.com",
})
_PERSONAL_HOST_SUFFIXES = (
    ".github.io", ".gitlab.io", ".substack.com", ".wordpress.com",
    ".blogspot.com", ".medium.com", ".carrd.co", ".netlify.app",
    ".vercel.app", ".bsky.social", ".tumblr.com", ".firebaseapp.com",
    ".web.app", ".herokuapp.com", ".pages.dev", ".wixsite.com", ".weebly.com",
    ".neocities.org", ".onrender.com", ".fly.dev", ".glitch.me",
    ".squarespace.com", ".gumroad.com", ".myshopify.com", ".bigcartel.com", ".itch.io",
    ".square.site", ".etsy.com",
)
# Marketplace hosts and the path segment before a seller ("etsy.com/shop/…").
_SELLER_PATHS = {
    "etsy.com": ("shop", "people"), "ebay.com": ("usr", "str"), "ebay.co.uk": ("usr", "str"),
    "poshmark.com": ("closet",), "vinted.com": ("member",), "amazon.com": ("shops",),
    "redbubble.com": ("people",), "society6.com": ("",), "mercari.com": ("u",),
}
# Top-level domains that are mostly personal sites ("analuisakr.photo").
_PERSONAL_TLDS = ("me", "name", "photo", "photography", "photos", "art", "design", "studio",
                  "portfolio", "blog", "bio", "cv", "family")
# "okeke-family.ng", "thejonesfamily.com"
_FAMILY_HOST = re.compile(r"(?:^|[.\-])(?:the)?[a-z]+-?family\.|(?:^|\.)family-")
# File-sharing hosts: a link carrying a document token grants access to
# someone's file ("docs.google.com/document/d/1xQ7vB…/edit?usp=sharing").
_SHARE_HOSTS = frozenset({
    "docs.google.com", "drive.google.com", "dropbox.com", "dl.dropboxusercontent.com",
    "onedrive.live.com", "1drv.ms", "app.box.com", "wetransfer.com", "we.tl",
    "icloud.com", "photos.app.goo.gl", "photos.google.com", "sharepoint.com",
})
_SHARE_TOKEN = re.compile(r"/(?=[\w\-]*\d)(?=[\w\-]*[A-Za-z])[\w\-]{12,}(?:/|$|\?)")
# "Personal site: https://…", "Portfolio – …", "Profile: …"
_PERSONAL_URL_LABEL = re.compile(
    r"(?:^|[\n.;|•·]|\s{2})\s*(?:personal\s+(?:site|website|page|blog|homepage|domain)"
    r"|portfolio|profile|homepage|my\s+(?:site|website|blog))\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)
# Platform pages that are not a person ("github.com/features").
_NON_PROFILE_SEGMENTS = frozenset({
    "", "about", "help", "settings", "login", "signup", "search", "explore",
    "topics", "pricing", "docs", "blog", "features", "jobs", "company",
    "school", "home", "watch", "results", "hashtag", "i", "intent", "share",
    "sharer", "p", "reel", "groups", "events", "pages", "marketplace",
    "orgs", "organizations", "enterprise", "marketplace", "apps", "r",
    "questions", "tags", "c", "playlist", "embed", "status", "policies",
    "legal", "privacy", "terms", "security", "sponsors", "collections",
    "trending", "notifications", "messages", "feed", "news",
})
_PERSON_PATH_MARKERS = re.compile(
    r"/(?:in|pub|u|user|users|profile|profiles|people|person|team|staff"
    # "/u/0/" is a Google account index, not a person
    r"|members?|author|authors|employees?|faculty|directory)/(?!\d+(?:[/?#]|$))[^/?#]+"
    r"|/~[^/?#]+|/@[^/?#]+",
    re.IGNORECASE,
)
_PERSONAL_URL_CONTEXT = re.compile(
    r"\b(?:my|our)\s+(?:own|personal|self[\s\-]?hosted|home|family)\s+[\w\- ]{1,25}?"
    r"\s*(?:at|is|on|:|-|–)?\s*$"           # "my personal nextcloud at …"
    r"|\b(?:my|our|his|her|their)\s+(?:own\s+|personal\s+)?(?:site|website"
    r"|portfolio|blog|homepage|home\s+page|page|profile|r[ée]sum[ée]|cv|link"
    r"|linkedin|github|twitter|instagram|domain|channel)\b[^.\n]{0,25}$",
    re.IGNORECASE,
)
# Query parameters whose value is a secret ("reset?t=8f3a91c2e7").
_SECRET_PARAM = re.compile(
    r"[?&#](?:t|token|access_token|refresh_token|id_token|key|api_key|apikey"
    r"|code|auth|sig|signature|session|sid|otp|reset|secret|password|pwd"
    r"|pass|ticket)=(?P<v>[^&#\s]{4,})",
    re.IGNORECASE,
)


def _split_url(url: str):
    """(host, path) of a URL with or without a scheme; host is lowercased
    without "www."."""
    rest = re.sub(r"(?i)^https?://", "", url)
    host, _, path = rest.partition("/")
    host = host.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host, "/" + path if path else ""


def _is_personal_url(url: str, before: str) -> bool:
    host, path = _split_url(url)
    if host in _PROFILE_HOSTS:
        first = path.strip("/").split("/")[0].split("?")[0].lower()
        if host == "linkedin.com":
            return bool(re.match(r"/(?:in|pub)/[^/?#]+", path))
        return first not in _NON_PROFILE_SEGMENTS
    if host.endswith(_PERSONAL_HOST_SUFFIXES):
        return True
    if (host in _SHARE_HOSTS or host.endswith(".sharepoint.com")) and (
            _SHARE_TOKEN.search(path) or path.strip("/").count("/") >= 2 and re.search(
                r"/(?=[\w\-]*\d)(?=[\w\-]*[A-Za-z])[\w\-]{6,}(?:/|$|\?)|/[^/?#]+\.\w{2,4}(?:\?|$)", path)):
        return True                         # a document link grants access
    segs = path.strip("/").split("/")
    if host in _SELLER_PATHS and len(segs) >= 2 and segs[0].lower() in _SELLER_PATHS[host] and segs[1]:
        return True                         # "etsy.com/shop/KnotsByNaledi"
    if host.rsplit(".", 1)[-1] in _PERSONAL_TLDS or _FAMILY_HOST.search(host):
        return True                         # "analuisakr.photo", "okeke-family.ng"
    if _PERSONAL_URL_LABEL.search(before):
        return True
    if _PERSON_PATH_MARKERS.search(path):
        return True
    return bool(_PERSONAL_URL_CONTEXT.search(before))


def find_urls(text: str) -> List[tuple]:
    """``(start, end, personal)`` for every URL or bare domain in *text*.

    Personal URLs (a profile on a social or code host, a personal domain
    introduced as "my site" or spelling a name from the message, such as
    "anakovac.dev" beside "Ana Kovač", a page under /team/ or /people/) are
    PII and get a surrogate. Every other URL is opaque: no pattern or NER
    stage may match inside it, so "api.example.com/v1?key=…" never yields a phone number and
    "github.com/Microsoft" never yields an ORG (audit I1, I8)."""
    found = [m for m in _URL_RE.finditer(text) if "." in m.group()]
    names = None
    out = []
    for m in found:
        url = m.group()
        personal = _is_personal_url(url, text[max(0, m.start() - 60):m.start()])
        if not personal:
            label = _host_label(_split_url(url)[0])
            if len(label) >= 6:
                if names is None:
                    names = _name_spellings(text, [x.span() for x in found])
                personal = label in names
        out.append((m.start(), m.end(), personal))
    return out


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


# second-level labels under which a country registers names ("co.uk")
_SLD = frozenset({"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "go"})


def _host_label(host: str) -> str:
    """The registered name of *host*, letters only: "anakovac" for
    "www.ana-kovac.co.uk"."""
    labels = host.split(".")
    if len(labels) < 2:
        return ""
    i = -3 if len(labels) >= 3 and labels[-2] in _SLD else -2
    return re.sub(r"[^a-z]", "", _fold(labels[i]))


def _name_spellings(text: str, url_spans: List[tuple]) -> Set[str]:
    """What a site named after someone in *text* would be called: two to
    four adjacent words run together, in order or surname first ("Ó
    Cainnigh, Niall" → "niallocainnigh"), with or without their initials,
    and the letters of each e-mail's local part. The URLs themselves are
    left out, so "jane-doe.com" alone names no one."""
    chars = list(text)
    for s, e in url_spans:
        chars[s:e] = " " * (e - s)
    folded = _fold("".join(chars))
    out = {re.sub(r"[^a-z]", "", m.group(1)) for m in re.finditer(r"([\w.+\-]+)@", folded)}
    toks = re.findall(r"[a-z]+", folded)
    for i in range(len(toks)):
        for k in (2, 3, 4):
            w = toks[i:i + k]
            if len(w) < k:
                break
            for v in (w, w[-1:] + w[:-1]):
                out.add("".join(v))
                long = [t for t in v if len(t) > 1]
                if len(long) >= 2:
                    out.add("".join(long))
    return out


def opaque_spans(text: str) -> List[tuple]:
    """Spans the NER stages must not see: every URL in *text*."""
    return [(s, e) for s, e, _ in find_urls(text)]


# ── labelled ID numbers ──────────────────────────────────────────────────────

_NUM = r"(?:number|no\.?|num|nr\.?|#|id)"
_ID_KEYWORDS = (
    r"\bmrn\b|medical[\s_]+record(?:[\s_]+(?:number|no\.?))?"
    rf"|insurance[\s_]*{_NUM}|medicare[\s_]*{_NUM}|medicaid[\s_]*{_NUM}"
    rf"|policy(?:[\s_]*{_NUM})?|member(?:ship)?[\s_]*{_NUM}|subscriber[\s_]*{_NUM}"
    rf"|acc(?:oun)?t(?:[\s_]*{_NUM})?|patient[\s_]*(?:identifier|{_NUM})"
    rf"|employee[\s_]*{_NUM}|badge[\s_]*{_NUM}|customer[\s_]*{_NUM}"
    rf"|client[\s_]*{_NUM}|student[\s_]*{_NUM}|loyalty[\s_]*{_NUM}"
    r"|\bktn\b|known\s+traveler(?:\s+number)?"
    r"|\bpassid\b|global\s+entry(?:\s+passid)?"
    r"|uscis(?:\s*(?:number|no\.?|#))?|\ba-number|alien\s+(?:registration(?:\s+number)?|number)"
    r"|aadha{1,2}r(?:\s+(?:number|no\.?|card))?"
    r"|national[\s_]+(?:id|identity|insurance)(?:\s+(?:number|no\.?|card))?"
    r"|tax[\s_]*(?:id|identification)(?:[\s_]*(?:number|no\.?))?"
    r"|\bf?ein\b|\bitin\b|\btin\b|\bvat[\s_]*(?:id|number|no\.?)"
    r"|\bnhs[\s_]*(?:number|no\.?)|social\s+insurance\s+number|\bpan\s+card"
    r"|card\s+(?:ending|ends)(?:\s+(?:in|with))?"
    r"|(?:device[\s_]+)?serial(?:[\s_]*(?:number|no\.?|#|num))?|\bs/n\b"
    r"|\bimei\d?\b|\bmeid\b|\bgroup\b"
    r"|num[ée]ro\s+(?:de\s+)?client|n[úu]mero\s+de\s+(?:cliente|cuenta|socio)"
    r"|kunden(?:nummer|-?nr\.?)|mitglieds?(?:nummer|-?nr\.?)|versicherungsnummer"
    r"|c[óo]digo\s+(?:de\s+)?cliente|(?-i:\bSIN\b)|\bcpf\b|\bcnpj\b|\bdni\b|\bnie\b|\bnif\b|\bpesel\b"
    r"|\bbsn\b|steuer-?id|personalausweis(?:nummer)?"
    # national / tax / academic identifiers named in other languages
    r"|codice\s+fiscale|s[ée]curit[ée]\s+sociale|\bnir\b|\bird\b|\borcid\b"
    # national, tax and health numbers by their usual names (general knowledge)
    r"|(?-i:\b(?:TFN|ABN|BSB|OHIP|PPSN?|CURP|RFC|RUT|CUI[TL]|NRIC|HKID|BVN|NIN|NINO"
    r"|EPIC|UAN|GSTIN|NIP|REGON|OIB|EGN|JMBG|CPR|NSS|AHV|CNH|RG|PAN|NPI|IRD)\b)"
    r"|emirates\s+id|\biqama\b|mykad|personnummer|f[øo]dselsnummer|henkil[öo]tunnus"
    r"|burgerservicenummer|steuer(?:nummer|-?identifikationsnummer)|sozialversicherungsnummer"
    r"|tessera\s+sanitaria|num[ée]ro\s+fiscal|t\.?c\.?\s+kimlik|kimlik\s+no"
    r"|health\s+(?:card|insurance)(?:\s+(?:number|no\.?|#))?|\bni\s+(?:number|no\.?)"
    r"|license\s*#|licence\s*#"
    r"|matr[ií]cul[ae]|身份证号?码?|sort\s*code"
    # "ID number", "steam id", court / benefit case numbers
    r"|\b(?:id|identity|identification)\s+(?:card\s+)?(?:number|no\.?|num|#)"
    r"|\b(?:steam|player|voter|personal)\s*id\b|\bcase\s+(?:number|no\.?)"
)
_ID_VALUE = (
    r"\d{3}\.\d{3}\.\d{3}-\d{2}"                 # CPF
    r"|\d{1,3}(?:\.\d{3}){2,3}-?[\dkKxX]{0,2}"     # RUT, RG, DNI with dots
    r"|(?-i:[A-Z]{2} ?\d{2} ?\d{2} ?\d{2} ?[A-D])"  # UK NI number
    r"|\d{1,6}(?:[ \-]\d{1,8}){1,5}"             # grouped digits
    r"|(?-i:[A-Z0-9][A-Z0-9\-]{3,19})"
)
_ID_PATTERN = re.compile(
    # "Global Entry number is …", "employee ID (QM-20417)", "身份证号是…"
    rf"(?:{_ID_KEYWORDS})(?:[\s_]*(?:number|no\.?|num|nr\.?|#))?[\"'\s:=\-#(：]*"
    r"(?:(?:is|was|est|es|ist|é|è|era|c'est)\s+|(?:是|为)\s*)?[\"']?"
    rf"(?P<v>{_ID_VALUE})(?![\w\-])",
    re.IGNORECASE,
)
# A second value of the same shape after a list separator
# ("Passport: A09382716 / A11746620").
_LIST_CONTINUATION = re.compile(r"\s*(?:/|,|&|\band\b|\bor\b|\by\b|\bund\b|\bet\b)\s*")


_ID_KEYWORD_RE = re.compile(_ID_KEYWORDS, re.IGNORECASE)
_BARE_ID = re.compile(
    r"(?:(?-i:\bID\b)|\bidentifier\b|\blicen[sc]e\b|\binsurance\b|保险单?号|保单号)"
    r"(?:[\s:=#(\-：]|\bis\b|\bwas\b)*"
    r"(?P<v>(?-i:(?:[A-Z]{1,4}[ \-]?)?\d{6,14}))(?![\w\-])",
    re.IGNORECASE,
)
# The ID of a thing, not a person ("process ID 4412345", "order ID …")
_NOT_PERSONAL_ID = re.compile(
    r"\b(?:process|thread|job|task|build|commit|request|run|event|object|product|item|error"
    r"|issue|bug|sku|model|version|app|channel|guild|server|chat|message|video|post|order"
    r"|transaction|tracking|session|node|row|column|table|field|file)\s*$",
    re.IGNORECASE,
)
_MONTH_NAMES = (r"(?i:January|February|March|April|May|June|July|August|September"
                r"|October|November|December)")


def _nearby_id_validator(m: "re.Match") -> bool:
    """A long digit group a few words after an ID label ("numéro de sécurité
    sociale … C'est 2 84 07 75 112 058 43") — same line, nothing numeric in
    between, not a counter."""
    if sum(c.isdigit() for c in m.group()) < 8:
        return False
    line_start = m.string.rfind("\n", 0, m.start()) + 1
    before = m.string[max(line_start, m.start() - 80):m.start()]
    hits = list(_ID_KEYWORD_RE.finditer(before))
    if not hits or _NEG_NUM_CONTEXT.search(before[-30:]):
        return False
    return not re.search(r"\d{3}", before[hits[-1].end():])


_CODE_VERB = re.compile(r"(?i)(?:get|set|is|has|on|to|from|make|create|update|delete|handle"
                        r"|fetch|load|init|use|test|do|run|add|remove|parse|build|with)_")
_PHONE_EXT = re.compile(r"(?i),?[ \t]*(?:ext\.?|extension|x)[ \t]*\d{1,5}\b")
_AGE_UNIT = re.compile(r"(?i)[ \t]+(?:years?|yrs?\.?|y/o|yo)(?:[ \t]+old)?\b")


def _shape(s: str) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", s))


# ── age and year of birth ────────────────────────────────────────────────────

_AGE_HEADER = frozenset({"age", "ages", "age (years)", "age (yrs)", "edad", "alter",
                         "âge", "age_years", "idade", "età", "leeftijd", "ålder"})


def _table_age_cells(text: str):
    """(start, end) of each number under an "Age" column of a pasted CSV,
    TSV or pipe table ("Name,Age,Allergy\nKai Nakamura,8,peanuts")."""
    lines = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+", text)]
    i = 0
    while i < len(lines):
        _, head = lines[i]
        i += 1
        for d in (",", "\t", ";", "|"):
            cells = [c.strip().lower() for c in head.split(d)]
            if len(cells) >= 3 and any(c in _AGE_HEADER for c in cells):
                col = next(k for k, c in enumerate(cells) if c in _AGE_HEADER)
                while i < len(lines):
                    off, row = lines[i]
                    parts = row.split(d)
                    if len(parts) != len(cells):
                        break
                    i += 1
                    s = off + sum(len(x) + 1 for x in parts[:col])
                    cell = parts[col]
                    v = cell.strip()
                    if re.fullmatch(r"\d{1,3}", v) and 0 < int(v) <= 120:
                        s += len(cell) - len(cell.lstrip())
                        yield s, s + len(v)
                break

_AGE_PATTERNS = [
    # "34 years old", "a 41 year old", "34yo", "34 y/o"
    re.compile(
        r"\b(?P<v>\d{1,3}[\s\-]*(?:(?:years?|yrs?)[\s\-]*old|yo|y/o|y\.o\.)(?![\w/]))",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)(?P<v>\d{1,3}\s*(?:años|ans|jahre\s+alt|anos|岁|歳|साल))",
               re.IGNORECASE),
    # a JSON or YAML key: '"age": "34"', "'edad': 34"
    re.compile(
        r"(?i)[\"'](?:age|age_years|edad|alter|âge|idade|età|leeftijd|ålder)[\"']\s*[:=]\s*[\"']?"
        r"(?P<v>\d{1,3})[\"']?(?=\s*(?:[,}\]\n]|$))"
    ),
    # "turned 7", "aged 34", "age: 34", "turning 40"
    re.compile(
        r"\b(?P<v>(?:age[ds]?|turn(?:ed|s|ing)?)(?:\s*(?:is|was|are|:|=|-|–|—))?\s+\d{1,3})\b",
        re.IGNORECASE,
    ),
    # "I'm 29", "they're 5", "my grandma Evelyn is 89"
    re.compile(
        r"(?:\b(?:i'?m|i\s+am|she'?s|he'?s|they'?re|they\s+are|she\s+is|he\s+is"
        r"|who'?s|who\s+is|(?:are|were)\s+both)"
        rf"|\b(?:my|our|his|her|their)\s+{_KIN}s?(?:\s+[{_UP}][\w'\-]+"
        rf"(?:\s+(?:and|&)\s+[{_UP}][\w'\-]+)?)?\s+(?:is|was|are|were|turns|just\s+turned))"
        r"\s+(?:(?:only|just|almost|nearly|about)\s+)?(?P<v>\d{1,3})\b",
        re.IGNORECASE,
    ),
    # "Esther is 13 and has dyslexia", "my youngest, Talia, is 4": a
    # capitalised name, then the number ends the clause
    re.compile(
        r"(?<![\w'])(?!(?:It|This|That|There|Here|What|Which|Who|Where|When|How|Version"
        r"|Price|Total|Score|Rate|Chapter|Page|Step|Level|Size|Count|Number|Python|Java"
        r"|Node|Room|Floor|Gate|Platform|Section|Part|Item|Order|Answer|Result|Value"
        r"|Mine|Ours|Yours|Today|Tomorrow|Yesterday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b)"
        rf"[{_UP}][{_LO}]{{2,}}(?:\s+and\s+[{_UP}][{_LO}]{{2,}})?,?\s+(?:is|are|was|turns|just\s+turned)"
        r"\s+(?:(?:only|just|almost|nearly)\s+)?(?P<v>\d{1,2})"
        r"(?=\s+(?:and|but|now|today|this|next|so|too|already)\b|\s*[.,;!?)]|\s*$)",
    ),
    # Reddit style: "I (29F)", "my bf (31 M)", "[F/24]", "34M here"
    re.compile(
        r"[(\[]\s*(?P<v>[1-9]\d)\s*/?\s*(?:[MF]|NB|nb|m|f)\s*[)\]]"
        r"|[(\[]\s*(?:[MF]|NB|m|f)\s*/?\s*(?P<w>[1-9]\d)\s*[)\]]"
    ),
    re.compile(r"(?:^|(?<=[\s,]))(?P<v>\d{2})\s?[MF](?=\s+here\b|\s*[,:)]\s*(?:i|my)\b)"),
    # "(gloria, 63)", "name 'Dmitri, 39'"
    re.compile(r"[(\"'‘“]\s*(?!(?:fig|figure|table|eq|page|pp|vol|ch|chapter|item|step|line"
               r"|row|col|box|note|part|section|sec|ref|art|para|no|nr|level|round|week|day)\b)"
               rf"[{_UP}{_LO}][{_LO}]{{2,}},\s*(?P<v>\d{{1,2}})\s*[)\"'’”]", re.IGNORECASE),
    # "Nana's 80th", "my 40th birthday"
    re.compile(
        rf"\b(?:{_KIN}|[{_UP}][{_LO}]{{2,}})['’]s\s+(?P<v>\d{{1,3}}(?:st|nd|rd|th))"
        r"(?=\s+birthday|\s*[.,!?;)\n]|\s*$)"
        r"|\b(?:my|his|her|their|your)\s+(?P<w>\d{1,3}(?:st|nd|rd|th))(?=\s+birthday)",
        re.IGNORECASE,
    ),
    # "My daughter Ava Lindqvist (14)"
    re.compile(
        rf"\b{_KIN}(?:\s+[{_UP}][\w'\-]+){{0,3}}\s*\((?P<v>\d{{1,2}})\)",
        re.IGNORECASE,
    ),
    # "at the age of 55", "dad had a heart attack at 55", "mum died at 71"
    re.compile(
        r"\bat\s+the\s+age\s+of\s+(?P<v>\d{1,3})\b"
        rf"|\b{_KIN}\s+(?:had|died|passed(?:\s+away)?|was\s+diagnosed|got\s+diagnosed"
        r"|retired|married|got\s+married)\b[^.\n;:]{0,30}?\bat\s+(?P<w>\d{1,2})\b",
        re.IGNORECASE,
    ),
]

# "Ana-Maria Popescu, 29, MSc …", "Our intern Folake Mensah (45) starts …",
# a sign-off "— Özge Yıldız, 34" at a line's end: a capitalised two- or
# three-part name and a bare number. The words must not be a heading, a unit
# or a place ("Chapter Two (12)", "Room 4, 12,", "Windows Server, 12").
_NAME_PARTS = (rf"(?<![\w'\-])(?P<n>[{_UP}][{_LO}'\-]+"
               rf"(?:[ \-](?:[{_UP}][{_LO}'\-]+|D'[{_UP}][{_LO}]+)){{1,2}})")
_NAMED_AGE = re.compile(_NAME_PARTS + r"(?:,\s+(?P<v>\d{1,2}),|\s*\((?P<w>\d{1,2})\))")
# ... or two digits closing the line, a full stop or bracket at most after
# them (a single digit there is more often a rank, a sequel or a score)
_NAMED_AGE_END = re.compile(_NAME_PARTS + r",[ \t]+(?P<v>[1-9]\d)[.;:!?)]?(?=[ \t]*(?:\n|$))")
_NOT_NAME_WORD = frozenset({
    "chapter", "section", "part", "step", "page", "table", "figure", "room",
    "level", "season", "episode", "volume", "version", "phase", "stage",
    "grade", "year", "class", "unit", "lesson", "week", "day", "item", "round",
    "game", "match", "team", "group", "box", "line", "row", "column", "floor",
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "for", "to", "with",
    "my", "our", "your", "his", "her", "their", "this", "that", "these",
    "mr", "mrs", "ms", "dr", "street", "road", "avenue", "lane", "city",
    "county", "state", "north", "south", "east", "west", "new", "san", "st",
    "windows", "server", "office", "python", "java", "node", "iphone", "galaxy",
    "pixel", "android", "edition", "release", "model", "series", "pro", "max",
    "mini", "plus", "ultra", "air", "studio", "core", "gen", "mark",
})


def _named_age_validator(m: "re.Match") -> bool:
    words = re.split(r"[ \-]", m.group("n"))
    if any(w.lower() in _NOT_NAME_WORD for w in words):
        return False
    n = int(m.group("v") or m.group("w"))
    # at a line's end there is no unit after the number: the next line's
    # first word ("People Ops lead") says nothing about it
    return 1 <= n <= 99 and (m.re is _NAMED_AGE_END or not _NOT_AGE_AFTER.match(_after(m, 12)))


# "aged 7 and 10", "ages 3, 6 and 9" — the numbers after the first one
_AGE_LIST_CONT = re.compile(r"(?:\s*,\s*|\s*,?\s*(?:and|&)\s+)(\d{1,2})\b(?![.,:]\d)")


# ─────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────

# Patterns whose PII value is capture group 1 (keyword-gated patterns).
_GROUP1_TYPES = frozenset({
    "us_driver_license", "passport", "id_number", "license_plate",
})

# Types whose value may be followed by more values of the same shape
# ("Passport: A09382716 / A11746620").
_LIST_TYPES = frozenset({"us_driver_license", "passport", "id_number", "credential"})

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
            # published provider token prefixes (Slack, GitHub, GitLab, npm,
            # PyPI, Hugging Face, SendGrid, Shopify, DigitalOcean, Google
            # OAuth, AWS session keys, Telegram bots)
            r"|\bxox[abposre]-[0-9A-Za-z\-]{10,}"
            r"|\bxapp-\d-[0-9A-Za-z\-]{10,}"
            r"|\b(?:ghs|ghu|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"
            r"|\bglpat-[A-Za-z0-9\-_]{20}"
            r"|\bnpm_[A-Za-z0-9]{36}"
            r"|\bpypi-[A-Za-z0-9\-_]{40,}"
            r"|\bhf_[A-Za-z0-9]{30,}"
            r"|\bSG\.[A-Za-z0-9\-_]{16,}\.[A-Za-z0-9\-_]{16,}"
            r"|\bshp(?:at|ca|pa|ss)_[a-fA-F0-9]{32}"
            r"|\bdop_v1_[a-f0-9]{64}"
            r"|\bya29\.[A-Za-z0-9\-_]{20,}"
            r"|\bASIA[0-9A-Z]{16}"
            r"|\b\d{8,10}:AA[A-Za-z0-9\-_]{33}\b"
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

    # ── Credentials (keyword-gated, value = group "v") ───────────────────────
    # Passwords, PINs, one-time / backup / verification codes, tokens. The
    # value must look like a secret (a digit, a symbol, or mixed case), so
    # "reset my password for Gmail" never matches (audit I1, I14).
    (
        "credential",
        re.compile(
            r"(?:pas+w(?:or|ro)?d|pass(?:phrase|code)|\bpwd\b|\bpw\b|\bpword\b"
            r"|\bpass(?=\s*[:=])|\bpin(?:\s*(?:code|number))?\b"
            r"|(?:2fa|mfa|otp|backup|recovery|verification|verify|security"
            r"|one[\s-]?time|login|auth(?:entication)?|access|sms|reset|confirmation"
            r"|al(?:ar|ra)m|door|gate|garage|lock|keypad|entry|safe|unlock)\s+codes?"
            r"|\botp\b|(?:api|secret|access|auth|refresh|session)[\s_\-]*(?:key|token)"
            r"|\btoken\b|\bsecret\b|contrase[ñn]a|mot\s+de\s+passe|passwort|senha"
            r"|pass\s?key|(?:wi-?fi|wlan|wpa2?|network|router)\s+(?:key|pass\b)"
            r"|(?:private|secret|license|licence|product|activation)\s+key"
            r"|wachtwoord|parola\s+d'ordine|has[łl]o|kennwort|codice\s+pin|c[óo]digo\s+pin)"
            # "the reset code they sent was 482913"
            r"(?:\s+(?:they|you|we|he|she|it|i)\s+(?:just\s+)?(?:sent|gave|texted"
            r"|emailed)(?:\s+(?:me|us))?)?"
            r"\s*(?:(?:is|was|are|=|:|-|of|es|est|ist|é)\s*)?"
            r"(?:(?:now|still|set\s+to|changed\s+to)\s+)?[\"'`]?"
            r"(?P<v>\d{3,4}[ \-]\d{3,4}(?!\d)|[^\s\"'`]{4,128})",     # "OTP 559 104"
            re.IGNORECASE,
        ),
        _credential_validator,
    ),
    # A config / env entry that names itself a secret ("DB_PASS=…",
    # "stripe_key: rk_live_…"); the value must look generated, so
    # "primary_key: id" and "foreign_key: users.id" stay.
    (
        "credential",
        re.compile(
            r"\b[A-Za-z][A-Za-z0-9]*_(?:key|token|secret|pass(?:word|wd)?|pwd)[\"']?"
            r"\s*[=:]\s*[\"'`]?(?P<v>[^\s\"'`,;]{6,128})",
            re.IGNORECASE,
        ),
        lambda m: (_credential_validator(m) and any(c.isdigit() for c in m.group("v"))
                   and not _CODE_REFERENCE.match(m.group("v"))),
    ),
    # "login is admin / HARBOR!LIGHT29", "creds: root / t0ps3cret"
    (
        "credential",
        re.compile(
            r"\b(?:login|log-in|credentials?|creds)\s*(?:is|are|=|:)?\s*[^\s/]{2,40}"
            r"\s*/\s*(?P<v>[^\s\"'`]{4,128})",
            re.IGNORECASE,
        ),
        _credential_validator,
    ),

    # "your Kloverbank code is 482 913": six digits after "code is" are a
    # one-time code unless the code is named as something else
    (
        "credential",
        re.compile(r"\bcode\s+(?:is|was|:)\s*(?P<v>\d{3}[ \-]?\d{3})(?![\d\-.,]\d)",
                   re.IGNORECASE),
        lambda m: not re.search(r"(?:zip|postal|post|area|country|dial|promo|discount|coupon"
                                r"|error|status|exit|http|product|sku|item|tracking|order"
                                r"|reference|ref|tax|class|course|program+e?)\s*$",
                                _before(m, 30), re.IGNORECASE),
    ),
    # SMS one-time codes as pasted: "G-482913", "482913 is your Steam code"
    (
        "credential",
        re.compile(r"\b(?P<v>G-\d{6})\b"
                   r"|(?<![\w\-.])(?P<w>\d{4,8}|\d{3}[ \-]\d{3})\s+is\s+your\s+"
                   r"(?:[\w\-]+\s+){0,4}(?:code|pin|password|otp|passcode)\b",
                   re.IGNORECASE),
        None,
    ),
    # A password in quotes after its keyword is a password, whatever it
    # looks like ("my password is 'sunshine'")
    (
        "credential",
        re.compile(r"(?:pas+w(?:or)?d|passcode|passphrase|\bpin\b|\bpw\b)\s*(?:is|was|:|=)?\s*"
                   r"[\"'‘“`](?P<v>[^\s\"'’”`]{4,64})[\"'’”`]", re.IGNORECASE),
        lambda m: not _CODE_REFERENCE.match(m.group("v")),
    ),
    # A quoted key: '"password": "Thistle%summit637"', "'pin': '4821'"
    (
        "credential",
        re.compile(r"[\"'](?:pas+w(?:or)?d|passcode|passphrase|pin|pwd|pw|secret|token)[\"']"
                   r"\s*[:=]\s*[\"'](?P<v>[^\s\"']{4,128})[\"']", re.IGNORECASE),
        _credential_validator,
    ),
    # The secret named for something: "my password for the sim session is
    # …", "password for registry: …". "is" or a colon is required, so
    # "reset my password for Gmail" stays.
    (
        "credential",
        re.compile(r"(?:pas+w(?:or)?d|passcode|passphrase|\bpin\b|\bkey\b|\btoken\b)\s+(?:for|to|on|at)\s+"
                   r"(?:[\w\-.]+\s+){0,3}?[\w\-.]+\s*(?:\bis\b|\bwas\b|:|=)\s*[\"'`]?"
                   r"(?P<v>[^\s\"'`]{4,128})", re.IGNORECASE),
        _credential_validator,
    ),
    # "Use Nectar*summit835 as the access password"
    (
        "credential",
        re.compile(r"\buse\s+[\"'`]?(?P<v>[^\s\"'`]{4,128}?)[\"'`]?\s+as\s+"
                   r"(?:the\s+|my\s+|your\s+|our\s+|an?\s+)?(?:[\w\-]+\s+){0,2}?"
                   r"(?:password|passcode|passphrase|pin)\b", re.IGNORECASE),
        _credential_validator,
    ),
    # A bare "key: …" whose value looks generated: letters and digits
    # alternating ("key: 5xyjpbtq-1t4bh4zw-sqa79f5e", "Key 4f8a9b2c7d1e3f6a"),
    # so "key: value", "cache key: session12345" and "key of C" stay.
    (
        "credential",
        re.compile(r"(?<![\w\-])key\s*(?:\bis\b|:|=)?\s*[\"'`]?"
                   r"(?P<v>[A-Za-z0-9][A-Za-z0-9\-]{11,127})(?![\w\-])", re.IGNORECASE),
        lambda m: len(re.findall(r"[A-Za-z](?=\d)|\d(?=[A-Za-z])", m.group("v"))) >= 4,
    ),
    # A seed / recovery phrase: 12–24 lower-case words
    (
        "credential",
        re.compile(r"(?:seed|recovery|backup|secret|mnemonic)\s+(?:phrase|words)\s*"
                   r"(?:is|are|was|were|:|=|-)?\s*[\"'`]?"
                   r"(?P<v>[a-z]{3,8}(?:[ ,]+[a-z]{3,8}){11,23})", re.IGNORECASE),
        None,
    ),
    # A base32 TOTP setup key written in groups of four
    # ("MZXW 6YTB OI3K 5QPL R2DN 4HVE"); most groups carry a digit, which
    # all-caps prose never does.
    (
        "credential",
        re.compile(r"(?<![A-Za-z0-9])(?P<v>[A-Z2-7]{4}(?: [A-Z2-7]{4}){3,7})(?![A-Za-z0-9])"),
        lambda m: sum(any(c.isdigit() for c in g) for g in m.group("v").split()) >= 3,
    ),

    # ── Social / chat / game handles ─────────────────────────────────────────
    (
        "handle",
        re.compile(
            r"(?<![\w@./])(?P<v>@[A-Za-z0-9_](?:[A-Za-z0-9_.\-]{0,28}[A-Za-z0-9_])?)"
            r"(?![\w@])"
        ),
        _at_handle_validator,
    ),
    (
        "handle",
        re.compile(r"(?<![\w.#])(?P<v>[A-Za-z0-9_.]{2,32}#\d{4})\b"),
        _tag_handle_validator,
    ),
    (
        "handle",
        re.compile(
            r"(?:(?P<strong>(?:(?:xbox|psn|steam|discord|epic|switch|ps\d)\s+)?"
            r"(?:gamer\s*tag|user\s*name|screen\s*name|login\s+name|user\s*id)"
            # "user=", JSON "user":, env DB_USER=
            r"|\bpsn(?:\s+id)?|\bhandle|(?:\b|(?<=_))user(?=[\"']?\s*[=:]))"
            r"|\b(?:discord|insta(?:gram)?|\big|twitter|tiktok|snap(?:chat)?|telegram"
            r"|reddit|twitch|steam|nick(?:name)?|alias|login)"
            r"(?:\s+(?:name|handle|tag|account))?)"
            # "Login: username X" — let the stronger label claim the value
            r"(?!\s*(?:[=:\-]\s*)?(?:user\s*name|gamer\s*tag|screen\s*name|handle)\b)"
            r"[\"']?\s*(?:(?:is|was|=|:|-)\s*)?[\"']?@?"
            r"(?P<v>[A-Za-z0-9][A-Za-z0-9_.\-]{1,30}[A-Za-z0-9])(?![\w@])",
            re.IGNORECASE,
        ),
        _keyword_handle_validator,
    ),
    # "minecraft name is PixelMoth_77", "stream on twitch as lunarkoi_tv",
    # "a seller called GREENLEAF_GEAR", "with the account r.mwangi.dev"
    (
        "handle",
        re.compile(
            r"(?:\b(?i:minecraft|roblox|fortnite|xbox|psn|steam|twitch|youtube|kick|valorant"
            r"|riot|league|epic|battle\.?net|github|gitlab|tiktok|insta(?:gram)?|snap(?:chat)?"
            r"|reddit|telegram|discord|twitter|threads|bluesky|mastodon|spotify|soundcloud"
            r"|chess\.com|lichess|osu|genshin)\s+(?i:ign|name|user\s*name|id|handle|tag)"
            r"\s*(?i:is|was|[:=\-])\s*@?"
            r"|\b(?i:stream(?:s|ed|ing)?|post(?:s|ed|ing)?|play(?:s|ed|ing)?|go(?:es)?\s+by)"
            r"\s+(?i:on\s+[a-z]+\s+)?(?i:as|under)\s+@?"
            r"|\b(?i:seller|buyer|vendor|shop|store|user|player|streamer|channel|account)"
            r"\s+(?i:called|named)\s+@?)"
            r"(?P<v>[A-Za-z0-9][A-Za-z0-9_.\-]{1,30}[A-Za-z0-9])(?![\w@])"
        ),
        _keyword_handle_validator,
    ),
    (
        "handle",
        re.compile(r"\b(?i:the|my|his|her|their|your|with)\s+account\s+@?"
                   r"(?P<v>[a-z][a-z0-9]*(?:[._][a-z0-9]+)+)(?![\w@(/])"),
        lambda m: not re.search(r"\.(?:json|ya?ml|py|js|ts|txt|csv|xml|env|cfg|ini|conf|log)$",
                                m.group("v")),
    ),
    # a game tag shaped "xX_VoidRunner_Xx": mixed case around underscores
    (
        "handle",
        re.compile(r"(?<![\w.@/`$\-])(?P<v>(?=[A-Za-z0-9_]*[a-z])(?=[A-Za-z0-9_]*[A-Z])"
                   r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)(?![\w@(=.`\-])"),
        lambda m: (m.string.count("`", 0, m.start()) % 2 == 0
                   and "```" not in m.string[:m.start()]
                   and not _CODE_VERB.match(m.group("v"))
                   and bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+_|_[A-Z][a-z]", m.group("v")))),
    ),
    # combined / common web-log user ("1.2.3.4 - mbanda_r [12/Mar/2026:…")
    (
        "handle",
        re.compile(r"(?m)^[ \t]*(?:\d{1,3}\.){3}\d{1,3}[ \t]+\S+[ \t]+"
                   r"(?P<v>[A-Za-z0-9_][A-Za-z0-9_.@\-]{0,63})[ \t]+\[\d{1,2}/[A-Za-z]{3}/\d{4}:"),
        lambda m: m.group("v") != "-" and m.group("v").lower() not in _SYSTEM_ACCOUNTS,
    ),
    # Reddit user ("u/quietlark_88"); r/subreddits are public
    (
        "handle",
        re.compile(r"(?<![\w/])(?P<v>u/[A-Za-z0-9_\-]{3,20})(?![\w/])"),
        None,
    ),
    # auth-log user names ("Failed password for jmorales from …")
    (
        "handle",
        re.compile(
            r"(?:failed|accepted|invalid)\s+(?:password|publickey|keyboard-interactive)"
            r"\s+for\s+(?:invalid\s+user\s+)?(?P<v>[a-z_][a-z0-9_.\-]{1,31})\b",
            re.IGNORECASE,
        ),
        lambda m: m.group("v").lower() not in _SYSTEM_ACCOUNTS,   # "invalid user admin"
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
        # not four groups of a dotted phone ("+33.7.25.61.98.87")
        re.compile(
            r"(?<![+.\d])\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b(?!\.\d)"
        ),
        _ipv4_validator,
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
    # Labelled IBAN / VIN with a failing checksum: a typo or a made-up
    # number is still the user's account or car ("IBAN DE27 1007 …",
    # "is VIN 2T1BURHE6KC218845 valid?") — fail closed.
    (
        "iban",
        re.compile(
            r"\biban\b[\s:#\-]*(?:(?:is|was|:)\s*)?"
            r"(?P<v>(?-i:[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?))\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "vin",
        re.compile(
            r"\b(?:vin|vehicle\s+identification\s+number|chassis\s+(?:number|no\.?))\b"
            r"[\s:#\-]*(?:(?:is|was)\s+)?(?P<v>(?-i:[A-HJ-NPR-Z0-9]{17}))\b",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── Labelled ID numbers (keyword-gated, value = group 1) ─────────────────
    # Runs before every phone pattern: a labelled number ("customer no.
    # 55-019283", "Account 8774 10 223 1186654") is an ID, not a phone.
    (
        "id_number",
        _ID_PATTERN,
        _id_value_validator,
    ),
    # A bare label: "ID: U043480", "(ID S7701234)", "licence Q0262707",
    # "insurance HMO-2098444", "保险单号 INS-9495746". The value is mostly
    # digits (six or more, after at most four capitals), so "MIT license
    # 2024" and "insurance costs" never match.
    (
        "id_number",
        _BARE_ID,
        lambda m: not _NOT_PERSONAL_ID.search(_before(m, 30)),
    ),

    # ── International phone (non-US, non-UK) ───────────────────────────────────
    # MUST appear before phone_us (so "+7 495 374 8120" is claimed whole and
    # phone_us cannot grab just the "495 374 8120" tail) and before zip_us.
    (
        "phone_intl",
        re.compile(
            r"(?<![0-9A-Za-z_])"
            # "+56 9 7741 2208", "(+56) 9 7741 2208"
            r"(?:\+(?!1[ \-.]|44[ \-.])[1-9]\d{0,2}|\(\+[1-9]\d{0,2}\))"
            r"(?:[ \-.]\d{1,9}){1,6}"
            r"(?![0-9A-Za-z_])"
        ),
        lambda m: 9 <= len(re.sub(r"\D", "", m.group())) <= 15,
    ),

    # ── Chinese mobile (11 digits, 1[3-9]x) ────────────────────────────────────
    (
        "phone_intl",
        re.compile(
            r"(?<![0-9A-Za-z_.\-])(?:\+?86[\s\-]?)?1[3-9]\d(?:[\s\-]?\d{4}){2}(?![0-9A-Za-z])"
        ),
        _phone_validator,
    ),

    # ── US phone (alnum-guarded, optional extension) ───────────────────────────
    # A markdown "_" wrapper ("_312-849-2031_") is not part of a token; an
    # underscore glued to a word ("PO_4805550123") still is.
    (
        "phone_us",
        re.compile(
            r"(?<![A-Za-z0-9.\-])(?<![A-Za-z0-9]_)"
            # not the national part of "+44 116 296 4471" (phone_uk / _intl)
            r"(?<!\+\d )(?<!\+\d\d )(?<!\+\d\d\d )(?<!\+\d\d)(?<!\+\d\d\d)"
            # area-code brackets come in pairs: "(875.228.2393)" is a bracketed number
            r"(\+?1[\s\-.]?)?(?:\(\d{3}\)|\d{3})[\s\-.]?\d{3}[\s\-.]?\d{4}"
            r"(?:\s*(?:ext|extension|x)\.?\s*\d{1,6})?"
            r"(?![A-Za-z0-9])(?!_[A-Za-z0-9])",
            re.IGNORECASE,
        ),
        _phone_validator,
    ),

    # ── UK phone ───────────────────────────────────────────────────────────────
    # phone_intl leaves every "+44" followed by a space, "-" or "." to this
    # rule, so the +44 form takes those separators too ("+44.7700.900123").
    (
        "phone_uk",
        re.compile(
            r"(?<![A-Za-z0-9_.\-])"
            r"(?:\+44[\s.\-]?(?:\d{4}[\s.\-]?\d{6}|\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}|\d{2}[\s.\-]?\d{4}[\s.\-]?\d{4})"
            r"|0(?:\d{4}[\s\-]?\d{6}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4}|\d{2}[\s\-]?\d{4}[\s\-]?\d{4}))"
            r"(?![A-Za-z0-9_])"
        ),
        _phone_validator,
    ),

    # ── National-format phone with a trunk 0 ───────────────────────────────────
    # "07412 680 335" (UK mobile), "0412 773 519" (AU), "06 47 21 93 58" (FR),
    # "0176 4482 1903" (DE), "06-41982277" (NL): a leading 0 and 10–12 digits
    # in two or more groups. Labelled IDs ("Account 0123 …") are claimed
    # earlier; a bare 0-led digit run stays an ID.
    (
        "phone_intl",
        re.compile(
            r"(?<![0-9A-Za-z_.\-/:#])(?<!\d[ \-])0\d{1,4}(?:[ \-]\d{2,8}){1,4}(?![0-9A-Za-z_]|[.\-/]\d)"
        ),
        lambda m: (10 <= len(re.sub(r"\D", "", m.group())) <= 12
                   and _phone_validator(m)),
    ),
    # ... and with dots: "06.45.68.14.80" (FR), "0176.20476296" (DE),
    # "06.45681480" (NL); four groups of up to three digits are an IPv4 shape.
    (
        "phone_intl",
        re.compile(
            r"(?<![0-9A-Za-z_.\-/:#])0\d{1,4}(?:\.\d{2,8}){1,4}(?![0-9A-Za-z_]|[.\-/]\d)"
        ),
        lambda m: (10 <= len(re.sub(r"\D", "", m.group())) <= 12
                   and not re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", m.group())
                   and _phone_validator(m)),
    ),

    # ── National-format phone behind a phone word ("teléfono 612 345 678",
    # "फ़ोन नंबर 98765 43210") — no country code, so it needs the label.
    (
        "phone_intl",
        re.compile(
            r"(?:phone|\bph\b|\btel\b|tel[ée]fono|telefon|mobile|\bmob\b|cell(?:ular)?"
            r"|celular|m[óo]vil|handy|whats\s?app|call|text|contact|portable|ported"
            r"|\b(?:my|his|her|their|our)\s+number(?!\s+of\b)"
            r"|फ़ोन|फोन|电话|手机|携帯|전화)"
            r"[^\n\d+@]{0,25}?(?P<v>\+?\d{2,5}(?:[\s.\-]\d{2,5}){1,4}|\d{8,13})"
            r"(?![0-9A-Za-z])",
            re.IGNORECASE,
        ),
        _phone_ctx_validator,
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
            r"(?-i:([A-Z]?\d{3}-\d{3}-\d{2}-\d{3}-\d|[A-Z0-9]{1,6}(?:-[A-Z0-9]{2,8}){1,4}|[A-Z0-9]{5,20}))\b",
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

    # ── License plate (keyword-gated, value = group 1) ───────────────────────
    (
        "license_plate",
        re.compile(
            r"(?:licen[sc]e\s+plate|number\s+plate|\bplate\b|\btag\b|\brego\b|\breg\b"
            r"|registration(?:\s+(?:plate|mark))?|kenteken|kennzeichen|immatriculation"
            r"|\btarga\b|\bplaca\b)"
            r"\s*(?:number|no\.?|#)?\s*[:\-#]*\s*(?:(?:is|was|its|it's|es|ist|is\s+now)\s+)?"
            r"(?-i:([A-Z0-9]{1,4}(?:[\- ]?[A-Z0-9]{1,4}){1,2}))\b",
            re.IGNORECASE,
        ),
        _plate_validator,
    ),
    # typed in lower case in chat ("my plate is yk19 rzt"); only behind
    # "plate", since a lower-case "tag" is usually a git or HTML tag
    (
        "license_plate",
        re.compile(
            r"(?:licen[sc]e\s+plate|number\s+plate|\bplate\b|\brego\b)"
            r"\s*(?:number|no\.?|#)?\s*[:\-#]*\s*(?:(?:is|was|its|it's)\s+)?"
            r"(?-i:([a-z0-9]{2,4}[\- ]?[a-z0-9]{2,4}))\b",
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
            r"[\s,\-]+(?:\d{4}|'\d{2})"
            r"|"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+(?:\d{4}|'\d{2})"
            r"|(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])"
            r"|\d{1,2}(?:\.|º|er)?\s+(?:de\s+)?"
            r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre"
            r"|octubre|noviembre|diciembre|janvier|f[ée]vrier|mars|avril|mai|juin|juillet"
            r"|ao[ûu]t|septembre|octobre|novembre|d[ée]cembre|januar|februar|m[äa]rz|april"
            r"|juni|juli|august|oktober|dezember|gennaio|febbraio|aprile|maggio|giugno"
            r"|luglio|settembre|ottobre|dicembre|janeiro|fevereiro|mar[çc]o|maio|junho"
            r"|julho|setembro|outubro|novembro|dezembro|januari|februari|maart|mei|augustus)"
            r"\s+(?:de\s+)?\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
        _dob_validator,
    ),

    # a birthday without its year after a birth cue ("TURNED 18 ON 9 SEPTEMBER",
    # "her birthday is March 3rd")
    (
        "dob",
        re.compile(
            r"(?i:\bturn(?:ed|s|ing)\s+\d{1,3}\s+on|\bbirthday\s+(?:is|was|falls\s+on|on)"
            r"|\bborn\s+on)\s+(?i:the\s+)?"
            r"(?P<v>\d{1,2}(?i:st|nd|rd|th)?\s+(?i:of\s+)?" + _MONTH_NAMES + r"\b"
            r"|" + _MONTH_NAMES + r"\s+\d{1,2}(?i:st|nd|rd|th)?\b)(?![\s,]*(?:\d{4}|'\d\d))"
        ),
        None,
    ),

    # ── Year of birth ("born in 1990", "b. 1990") ────────────────────────────
    (
        "dob",
        re.compile(
            r"(?:\b(?:born|b\.|year\s+of\s+birth|yob|birth\s+year|n[ée]e?|nat[oa]|nacid[oa]"
            r"|geboren|nascid[oa])\s*(?::|-|–)?\s*(?:in\s+|on\s+|en\s+|nel\s+|em\s+|im\s+)?"
            r")(?P<v>(?:(?:January|February|March|April|May|June|July|August|September"
            r"|October|November|December)\s+)?(?:19|20)\d{2})\b(?![/\-.]\d)",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── Age (audit I14) ──────────────────────────────────────────────────────
    *[("age", _p, _age_validator) for _p in _AGE_PATTERNS],
    ("age", _NAMED_AGE, _named_age_validator),
    ("age", _NAMED_AGE_END, _named_age_validator),

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

    # ── ID label a few words back (last resort for long digit groups) ────────
    (
        "id_number",
        re.compile(r"(?<![\w\-.+])(?P<v>\d{1,6}(?:[ \-]\d{1,8}){1,6}|\d{8,18})(?![\w\-])"),
        _nearby_id_validator,
    ),

    # ── labelled postcode, any country: "zip": "X5000", "PLZ 79098",
    # "CAP 20121", "código postal 1100-148". "PIN code" is left to the
    # credential rule above: a card PIN and an Indian PIN code both get masked.
    (
        "zip_us",
        re.compile(r"(?i)\b(?:zip(?:[\s_]*code)?|post(?:al)?[\s_]*code|postcode|plz|(?-i:CAP)"
                   r"|c[oó]digo[\s_]+postal|(?-i:CP)|c\.p\.)\b"
                   r"[\"']?\s*[:=#\-–]?\s*[\"']?(?-i:(?P<v>[A-Z]{0,2}-?\d{3,6}(?:[ \-]?[A-Z]{2,3}(?![\w])|-\d{3,4})?"
                   r"|[A-Z]\d[A-Z][ ]?\d[A-Z]\d))(?![\w])"),
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

    def _claim(entity_type: str, s: int, e: int) -> None:
        results.append(DetectedEntity(text=text[s:e], start=s, end=e,
                                      type=entity_type, score=1.0, source="pattern"))
        occupied_spans.append((s, e))
        logger.debug(f"[PatternScan] {entity_type}: {text[s:e]!r} at [{s}:{e}]")

    # ── URLs first: a personal URL is one entity; any other URL is opaque
    # (nothing inside it is matched) except secret query parameters and
    # e-mail addresses.
    for us, ue, personal in find_urls(text):
        url = text[us:ue]
        if (not personal and "/" not in url and url.count(".") >= 2
                and re.search(r"(?i)\b(?:the|my|his|her|their|your|with)\s+account\s+$",
                              text[max(0, us - 30):us])):
            if not _should_skip(url):     # "the account r.mwangi.dev": a login
                _claim("handle", us, ue)
            continue
        if personal:
            if not _should_skip(url):
                _claim("url", us, ue)
            continue
        for m in _SECRET_PARAM.finditer(url):
            if _credential_validator(m) and not _should_skip(m.group("v")):
                _claim("credential", us + m.start("v"), us + m.end("v"))
        for m in _PATTERNS[0][1].finditer(url):          # e-mail
            if _span_free(us + m.start(), us + m.end()):
                _claim("email", us + m.start(), us + m.end())
        occupied_spans.append((us, ue))

    # ── Street addresses next: the canonical parser claims the FULL span
    # (street + unit + city + state + ZIP) as one entity; the assembler adds
    # the international layouts.  Where the two read the same words the
    # longer span wins, the parser on a tie.
    found = [(p, 0) for p in address_parser.find_addresses(text)]
    found += [(p, 1) for p in address_assembly.find(text)]
    picked: List[address_parser.ParsedAddress] = []
    for parsed, _rank in sorted(found, key=lambda f: (-(f[0].end - f[0].start), f[1], f[0].start)):
        if all(parsed.end <= q.start or parsed.start >= q.end for q in picked):
            picked.append(parsed)
    for parsed in sorted(picked, key=lambda p: p.start):
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

    def _claim_list_continuation(entity_type: str, pos: int, first: str) -> None:
        shape = _shape(first)
        while True:
            sep = _LIST_CONTINUATION.match(text, pos)
            if not sep:
                return
            s, e = sep.end(), sep.end() + len(first)
            cand = text[s:e]
            if (_shape(cand) != shape or (e < len(text) and text[e].isalnum())
                    or not _span_free(s, e) or _should_skip(cand)):
                return
            _claim(entity_type, s, e)
            pos = e

    for entity_type, pattern, validator in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            if "v" not in pattern.groupindex and not _span_free(start, end):
                continue

            if "v" in pattern.groupindex:
                g = _value_group(match)
                if g is None:
                    continue
                start, end = match.span(g)
            elif entity_type in _GROUP1_TYPES:
                if not match.group(1):
                    continue
                start, end = match.span(1)
            # trim whitespace (and sentence punctuation after a credential)
            # so the span and the text always agree
            while start < end and text[start].isspace():
                start += 1
            while start < end and (text[end - 1].isspace() or (
                    entity_type == "credential" and text[end - 1] in ".,;:)]}")):
                end -= 1
            if start >= end or not _span_free(start, end):
                continue
            matched_text = text[start:end]

            if _should_skip(matched_text):
                logger.debug(f"[PatternScan] Skipping (skip_values): {matched_text!r}")
                continue

            if validator is not None and not validator(match):
                continue

            tail = (_PHONE_EXT if entity_type.startswith("phone")
                    else _AGE_UNIT if entity_type == "age" else None)
            if tail and (t := tail.match(text, end)) and _span_free(end, t.end()):
                end = t.end()                 # "0161 496 0732 extension 214", "74 years"

            _claim(entity_type, start, end)

            if entity_type in _LIST_TYPES:
                _claim_list_continuation(entity_type, end, matched_text)
            elif entity_type == "age":
                pos = end
                while (lm := _AGE_LIST_CONT.match(text, pos)) and _span_free(*lm.span(1)) \
                        and not _NOT_AGE_AFTER.match(text, lm.end()):
                    _claim("age", *lm.span(1))
                    pos = lm.end()

    for s, e in _table_age_cells(text):
        if _span_free(s, e) and not _should_skip(text[s:e]):
            _claim("age", s, e)

    results.sort(key=lambda e: e.start)
    logger.info(f"[PatternScan] Found {len(results)} entities")
    return results
