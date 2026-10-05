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
  as "my site") is one "url" entity. Any other URL is opaque: nothing inside
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
from typing import List, Optional, Set

from ..entities import DetectedEntity
from . import address_parser

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


_BIRTH_CUE = re.compile(
    r"(?:born|birth|\bdob\b|d\.o\.b|b-?day|nacimiento|naissance|geburt|nascimento"
    r"|nacido|n[ée]e?\s+le|geboren)",
    re.IGNORECASE,
)

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
    year = _date_year(s)
    return year is not None and year <= _THIS_YEAR - _DOB_MIN_AGE_WITHOUT_CUE


def _ipv4_validator(m: "re.Match") -> bool:
    # loopback, "any" and netmasks identify nobody (audit I8)
    s = m.group()
    return not (s.startswith(("127.", "0.", "255.")) or s == "0.0.0.0")


# ── age ──────────────────────────────────────────────────────────────────────

_KIN = (
    r"(?:mom|mum|mother|dad|father|son|daughter|kid|child|baby|toddler|brother"
    r"|sister|grandma|grandmother|granny|grandpa|grandfather|grandson|granddaughter"
    r"|wife|husband|partner|spouse|boyfriend|girlfriend|fianc[ée]e?|aunt|uncle"
    r"|niece|nephew|cousin|friend|roommate|boss|patient|client|neighbou?r|stepson"
    r"|stepdaughter)"
)
# Words that turn "I'm 30" into a measurement or a count, not an age.
_NOT_AGE_AFTER = re.compile(
    r"\s*(?:%|percent|['\"′″]|[.:,]\d|/|[-–]\d|x\b|k\b|am\b|pm\b|st\b|nd\b|rd\b|th\b"
    r"|(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|seconds?|secs?|lbs?|kg|kilos?"
    r"|pounds|cm|mm|inch(?:es)?|in\b|feet|ft|foot|miles?|mi\b|km|meters?|metres?"
    r"|dollars?|bucks|euros?|times|of\b|out\b|minutes|points?|pts|degrees?"
    r"|steps?|items?|pages?|people|units?|cents?)\b)",
    re.IGNORECASE,
)


def _age_validator(m: "re.Match") -> bool:
    digits = re.findall(r"\d+", m.group("v"))
    if not digits or not (0 < int(digits[0]) <= 120):
        return False
    return not _NOT_AGE_AFTER.match(_after(m, 12))


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

_HANDLE_STOP = frozenset({
    "is", "was", "the", "a", "an", "and", "or", "my", "your", "for", "to",
    "not", "name", "names", "here", "there", "below", "above", "same",
    "different", "wrong", "correct", "incorrect", "invalid", "taken",
    "available", "required", "missing", "blank", "empty", "changed", "field",
    "password", "username", "account", "login", "email", "please", "help",
    "gamertag", "handle", "id", "tag", "on", "in", "at", "with",
})


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
    return distinctive or bool(m.group("strong"))


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


def _id_value_validator(m: "re.Match") -> bool:
    return sum(c.isdigit() for c in (m.group("v") or "")) >= 4


def _phone_ctx_validator(m: "re.Match") -> bool:
    n = len(re.sub(r"\D", "", m.group("v")))
    return 8 <= n <= 13 and not _NEG_NUM_CONTEXT.search(_before(m, 30))


# ── URLs ─────────────────────────────────────────────────────────────────────

_URL_TLDS = (
    r"com|org|net|edu|gov|io|dev|me|app|co|ai|gg|tv|xyz|info|biz|site|page"
    r"|blog|online|tech|us|uk|ca|de|fr|es|it|nl|br|au|jp|cn|ru|ch|se|no|fi"
    r"|dk|pl|pt|ie|nz|in|ly|to|sh|so|link|social|art|design|studio|email"
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
})
_PERSONAL_HOST_SUFFIXES = (
    ".github.io", ".gitlab.io", ".substack.com", ".wordpress.com",
    ".blogspot.com", ".medium.com", ".carrd.co", ".netlify.app",
    ".vercel.app", ".bsky.social", ".tumblr.com",
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
    r"|members?|author|authors|employees?|faculty|directory)/[^/?#]+"
    r"|/~[^/?#]+|/@[^/?#]+",
    re.IGNORECASE,
)
_PERSONAL_URL_CONTEXT = re.compile(
    r"\b(?:my|our|his|her|their)\s+(?:own\s+|personal\s+)?(?:site|website"
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
    if _PERSON_PATH_MARKERS.search(path):
        return True
    return bool(_PERSONAL_URL_CONTEXT.search(before))


def find_urls(text: str) -> List[tuple]:
    """``(start, end, personal)`` for every URL or bare domain in *text*.

    Personal URLs (a profile on a social or code host, a personal domain
    introduced as "my site", a page under /team/ or /people/) are PII and get
    a surrogate. Every other URL is opaque: no pattern or NER stage may match
    inside it, so "api.example.com/v1?key=…" never yields a phone number and
    "github.com/Microsoft" never yields an ORG (audit I1, I8)."""
    out = []
    for m in _URL_RE.finditer(text):
        url = m.group()
        if "." not in url:
            continue
        out.append((m.start(), m.end(), _is_personal_url(url, text[max(0, m.start() - 60):m.start()])))
    return out


def opaque_spans(text: str) -> List[tuple]:
    """Spans the NER stages must not see: every URL in *text*."""
    return [(s, e) for s, e, _ in find_urls(text)]


# ── labelled ID numbers ──────────────────────────────────────────────────────

_NUM = r"(?:number|no\.?|num|nr\.?|#|id)"
_ID_KEYWORDS = (
    r"\bmrn\b|medical[\s_]+record(?:[\s_]+(?:number|no\.?))?"
    rf"|insurance[\s_]*{_NUM}|medicare[\s_]*{_NUM}|medicaid[\s_]*{_NUM}"
    rf"|policy[\s_]*{_NUM}|member(?:ship)?[\s_]*{_NUM}|subscriber[\s_]*{_NUM}"
    rf"|acc(?:oun)?t(?:[\s_]*{_NUM})?|patient[\s_]*(?:identifier|{_NUM})"
    rf"|employee[\s_]*{_NUM}|badge[\s_]*{_NUM}|customer[\s_]*{_NUM}"
    rf"|client[\s_]*{_NUM}|student[\s_]*{_NUM}|loyalty[\s_]*{_NUM}"
    r"|\bktn\b|known\s+traveler(?:\s+number)?"
    r"|\bpassid\b|global\s+entry(?:\s+passid)?"
    r"|uscis(?:\s*(?:number|no\.?|#))?|\ba-number|alien\s+(?:registration\s+)?number"
    r"|aadha{1,2}r(?:\s+(?:number|no\.?|card))?"
    r"|national[\s_]+(?:id|identity|insurance)(?:\s+(?:number|no\.?|card))?"
    r"|tax[\s_]*(?:id|identification)(?:[\s_]*(?:number|no\.?))?"
    r"|\bf?ein\b|\bitin\b|\btin\b|\bvat[\s_]*(?:id|number|no\.?)"
    r"|\bnhs[\s_]*(?:number|no\.?)|social\s+insurance\s+number|\bpan\s+card"
    r"|card\s+(?:ending|ends)(?:\s+(?:in|with))?"
    r"|(?:device[\s_]+)?serial(?:[\s_]*(?:number|no\.?|#|num))?|\bs/n\b"
    r"|num[ée]ro\s+(?:de\s+)?client|n[úu]mero\s+de\s+(?:cliente|cuenta|socio)"
    r"|kunden(?:nummer|-?nr\.?)|\bcpf\b|\bcnpj\b|\bdni\b|\bnie\b|\bnif\b|\bpesel\b"
    r"|\bbsn\b|steuer-?id|personalausweis(?:nummer)?"
)
_ID_VALUE = (
    r"\d{3}\.\d{3}\.\d{3}-\d{2}"                 # CPF
    r"|\d{1,6}(?:[ \-]\d{1,8}){1,5}"             # grouped digits
    r"|(?-i:[A-Z0-9][A-Z0-9\-]{3,19})"
)
_ID_PATTERN = re.compile(
    rf"(?:{_ID_KEYWORDS})[\"'\s:=\-#]*"
    r"(?:(?:is|was|est|es|ist|é)\s+)?[\"']?"
    rf"(?P<v>{_ID_VALUE})(?![\w\-])",
    re.IGNORECASE,
)
# A second value of the same shape after a list separator
# ("Passport: A09382716 / A11746620").
_LIST_CONTINUATION = re.compile(r"\s*(?:/|,|&|\band\b|\bor\b|\by\b|\bund\b|\bet\b)\s*")


def _shape(s: str) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", s))


# ── age and year of birth ────────────────────────────────────────────────────

_AGE_PATTERNS = [
    # "34 years old", "a 41 year old", "34yo", "34 y/o"
    re.compile(
        r"\b(?P<v>\d{1,3}[\s\-]*(?:(?:years?|yrs?)[\s\-]*old|yo|y/o|y\.o\.)(?![\w/]))",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)(?P<v>\d{1,3}\s*(?:años|ans|jahre\s+alt|anos|岁|歳|साल))",
               re.IGNORECASE),
    # "turned 7", "aged 34", "age: 34", "turning 40"
    re.compile(
        r"\b(?P<v>(?:aged?|turn(?:ed|s|ing)?)(?:\s*(?:is|was|:|=|-))?\s+\d{1,3})\b",
        re.IGNORECASE,
    ),
    # "I'm 29", "they're 5", "my grandma Evelyn is 89"
    re.compile(
        r"(?:\b(?:i'?m|i\s+am|she'?s|he'?s|they'?re|they\s+are|she\s+is|he\s+is"
        r"|who'?s|who\s+is)"
        rf"|\b(?:my|our|his|her|their)\s+{_KIN}(?:\s+[A-Z][\w'\-]+)?\s+"
        r"(?:is|was|turns|just\s+turned))"
        r"\s+(?:(?:only|just|almost|nearly|about)\s+)?(?P<v>\d{1,3})\b",
        re.IGNORECASE,
    ),
    # "My daughter Ava Lindqvist (14)"
    re.compile(
        rf"\b{_KIN}(?:\s+[A-Z][\w'\-]+){{0,3}}\s*\((?P<v>\d{{1,2}})\)",
        re.IGNORECASE,
    ),
]


# ─────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────

# Patterns whose PII value is capture group 1 (keyword-gated patterns).
_GROUP1_TYPES = frozenset({
    "us_driver_license", "passport", "id_number", "license_plate",
})

# Types whose value may be followed by more values of the same shape
# ("Passport: A09382716 / A11746620").
_LIST_TYPES = frozenset({"us_driver_license", "passport", "id_number"})

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

    # ── Credentials (keyword-gated, value = group "v") ───────────────────────
    # Passwords, PINs, one-time / backup / verification codes, tokens. The
    # value must look like a secret (a digit, a symbol, or mixed case), so
    # "reset my password for Gmail" never matches (audit I1, I14).
    (
        "credential",
        re.compile(
            r"(?:pass(?:word|wd|phrase|code)|\bpwd\b|\bpin(?:\s*(?:code|number))?\b"
            r"|(?:2fa|mfa|otp|backup|recovery|verification|verify|security"
            r"|one[\s-]?time|login|auth(?:entication)?|access|sms)\s+codes?"
            r"|\botp\b|(?:api|secret|access|auth|refresh|session)[\s_]*(?:key|token)"
            r"|\btoken\b|\bsecret\b|contrase[ñn]a|mot\s+de\s+passe|passwort|senha)"
            r"\s*(?:(?:is|was|are|=|:|-|of|es|est|ist|é)\s*)?"
            r"(?:(?:now|still|set\s+to|changed\s+to)\s+)?[\"'`]?"
            r"(?P<v>[^\s\"'`]{4,128})",
            re.IGNORECASE,
        ),
        _credential_validator,
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
            r"|\bpsn(?:\s+id)?|\bhandle|\buser(?=\s*[=:]))"
            r"|\b(?:discord|insta(?:gram)?|\big|twitter|tiktok|snap(?:chat)?|telegram"
            r"|reddit|twitch|steam|nick(?:name)?|alias|login))"
            r"\s*(?:(?:is|was|=|:|-)\s*)?@?"
            r"(?P<v>[A-Za-z0-9][A-Za-z0-9_.\-]{1,30}[A-Za-z0-9])(?![\w@])",
            re.IGNORECASE,
        ),
        _keyword_handle_validator,
    ),
    # auth-log user names ("Failed password for jmorales from …")
    (
        "handle",
        re.compile(
            r"(?:failed|accepted|invalid)\s+(?:password|publickey|keyboard-interactive)"
            r"\s+for\s+(?:invalid\s+user\s+)?(?P<v>[a-z_][a-z0-9_.\-]{1,31})\b",
            re.IGNORECASE,
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

    # ── Labelled ID numbers (keyword-gated, value = group 1) ─────────────────
    # Runs before every phone pattern: a labelled number ("customer no.
    # 55-019283", "Account 8774 10 223 1186654") is an ID, not a phone.
    (
        "id_number",
        _ID_PATTERN,
        _id_value_validator,
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
            r"(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
            r"(?:\s*(?:ext|extension|x)\.?\s*\d{1,6})?"
            r"(?![A-Za-z0-9])(?!_[A-Za-z0-9])",
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

    # ── National-format phone behind a phone word ("teléfono 612 345 678",
    # "फ़ोन नंबर 98765 43210") — no country code, so it needs the label.
    (
        "phone_intl",
        re.compile(
            r"(?:phone|\bph\b|\btel\b|tel[ée]fono|telefon|mobile|\bmob\b|cell(?:ular)?"
            r"|celular|m[óo]vil|handy|whats\s?app|call|text|contact|portable"
            r"|फ़ोन|फोन|电话|手机|携帯|전화)"
            r"[^\n\d+]{0,25}?(?P<v>\+?\d{2,5}(?:[\s.\-]\d{2,5}){1,4}|\d{8,13})"
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
            r"[\s,\-]+(?:\d{4}|'\d{2})"
            r"|"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+(?:\d{4}|'\d{2})"
            r")\b",
            re.IGNORECASE,
        ),
        _dob_validator,
    ),

    # ── Year of birth ("born in 1990", "b. 1990") ────────────────────────────
    (
        "dob",
        re.compile(
            r"\b(?:born|b\.)\s+(?:in\s+|on\s+)?(?P<v>(?:19|20)\d{2})\b(?![/\-.]\d)",
            re.IGNORECASE,
        ),
        None,
    ),

    # ── Age (audit I14) ──────────────────────────────────────────────────────
    *[("age", _p, _age_validator) for _p in _AGE_PATTERNS],

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
                if match.group("v") is None:
                    continue
                start, end = match.span("v")
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

            _claim(entity_type, start, end)

            if entity_type in _LIST_TYPES:
                _claim_list_continuation(entity_type, end, matched_text)

    results.sort(key=lambda e: e.start)
    logger.info(f"[PatternScan] Found {len(results)} entities")
    return results
