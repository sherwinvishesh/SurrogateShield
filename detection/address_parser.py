"""
detection/address_parser.py — Canonical structured address parser.

Single source of truth for "what is an address" across detection,
surrogate generation, and reconstruction.  Pure stdlib, zero dependencies.

Captures the FULL address span — house number, directionals, street name,
suffix, unit, city, state, ZIP — as one unit so that downstream stages never
surrogate the components independently (the v1 "garbled address" bug).

Handles: numbered/ordinal streets ("123 5th Ave"), pre/post directionals
("742 W 3rd St NW"), apartment/suite/unit designators, PO boxes, ZIP+4,
highway/route forms ("1500 Highway 50"), multi-line addresses, full state
names and uppercase 2-letter abbreviations, comma-free formats.

Rejects: prices ("$20"), versions ("2.4.1"), counts ("5 stars", "3 point
turn"), and other numeric noise via a two-tier suffix system plus a
code-level plausibility validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection.geo_data import MAJOR_CITIES, US_STATES, US_STATE_ABBREVS
from util import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParsedAddress:
    """A structured address with absolute offsets into the source text."""
    full_text: str                                    # exact span, formatting preserved
    start: int
    end: int
    house_number: Optional[str] = None                # raw token: "123", "0123", "123A"
    house_number_span: Optional[Tuple[int, int]] = None  # ABSOLUTE offsets in source
    pre_directional: Optional[str] = None             # "E", "W.", "North"
    street_name: Optional[str] = None                 # "Crescent", "5th", "Martin Luther King"
    suffix: Optional[str] = None                      # "Row", "Blvd."
    post_directional: Optional[str] = None            # "NW"
    unit: Optional[str] = None                        # "Apt 4B", "Suite 210", "# 12"
    city: Optional[str] = None
    state: Optional[str] = None                       # "AZ" or "Arizona" (as written)
    zip_code: Optional[str] = None                    # "85281" or "85281-1234"
    po_box: Optional[str] = None                      # "PO Box 1234"
    is_po_box: bool = False

    @property
    def house_number_relative_span(self) -> Optional[Tuple[int, int]]:
        """House-number offsets relative to full_text (for surrogate splicing)."""
        if self.house_number_span is None:
            return None
        return (self.house_number_span[0] - self.start,
                self.house_number_span[1] - self.start)


# ─────────────────────────────────────────────────────────────────────────────
# Suffix vocabulary — USPS Publication 28 street suffixes, two tiers
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1: tokens that essentially only occur as street suffixes.
_UNAMBIGUOUS_SUFFIXES = frozenset({
    "street", "st", "avenue", "ave", "av", "boulevard", "blvd", "road", "rd",
    "drive", "dr", "lane", "ln", "court", "ct", "place", "pl", "plaza", "plz",
    "circle", "cir", "terrace", "ter", "terr", "parkway", "pkwy", "pky",
    "highway", "hwy", "freeway", "fwy", "expressway", "expy",
    "turnpike", "tpke", "causeway", "cswy", "crescent", "cres",
    "cove", "cv", "crossing", "xing", "trace", "trce", "trail", "trl",
    "landing", "lndg", "manor", "mnr", "gardens", "gdns",
    "heights", "hts", "square", "sq", "alley", "aly",
    "row", "mews", "byway", "pike", "esplanade", "boardwalk",
})

# Tier 2: common English nouns that are also street suffixes — these require
# extra evidence (capitalized street name, or a city/state/ZIP tail).
_AMBIGUOUS_SUFFIXES = frozenset({
    "way", "point", "pt", "run", "park", "green", "hill", "pass", "walk",
    "view", "vw", "rise", "mount", "mt", "gate", "path", "bend", "loop",
    "close", "chase", "heath", "meadow", "meadows", "mdws", "ridge", "rdg",
    "vale", "glen", "grove", "grv", "mill", "shore", "shores", "summit",
    "village", "vista", "pines", "spring", "spg", "springs", "spgs",
    "knoll", "hollow", "holw", "cliff", "creek", "crk", "dale", "dell",
    "fork", "glade", "harbor", "hbr", "haven", "hvn", "isle", "key",
    "lake", "oval", "orchard", "plains", "pointe", "port", "ranch",
    "station", "stream", "wells",
})

_ALL_SUFFIXES = _UNAMBIGUOUS_SUFFIXES | _AMBIGUOUS_SUFFIXES

# Function/quantity words that never appear inside a real street name.
# Checked lowercase-as-written, so "3-5 business days on Oak St" is rejected
# while a legitimately capitalized street ("123 Days Inn Dr") still passes.
_STREET_STOPWORDS = frozenset({
    "on", "in", "at", "by", "to", "of", "the", "a", "an", "and", "or",
    "near", "from", "for", "with", "was", "is", "are", "be", "off",
    "minutes", "mins", "min", "hours", "hrs", "days", "weeks", "months",
    "years", "miles", "blocks", "stars", "percent", "people", "times",
    "business", "more", "less", "about", "around", "over", "under",
})

# Words that must never be accepted as a "city" (query/filler words that can
# appear capitalized at sentence starts).
_CITY_BLOCKLIST = frozenset({
    "and", "or", "the", "near", "in", "at", "on", "by", "to", "from",
    "please", "thanks", "also", "but", "so", "then", "where", "which",
    "what", "when", "how", "can", "could", "would", "should", "call",
    "email", "phone", "contact", "attn", "attention", "usa",
})

_STATE_NAMES_LOWER = US_STATES  # already lowercase in geo_data


# ─────────────────────────────────────────────────────────────────────────────
# Grammar (compiled once at import)
# ─────────────────────────────────────────────────────────────────────────────

def _alt(words) -> str:
    """Longest-first alternation of literal words."""
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_SUFFIX_ALT = _alt(_ALL_SUFFIXES)
_STATE_NAME_ALT = _alt(_STATE_NAMES_LOWER)
_STATE_ABBREV_ALT = "|".join(sorted(US_STATE_ABBREVS))

# House number: 1-6 digits, optional NYC-style hyphenated part, optional
# letter ("123A").  Lookbehind rejects prices, decimals, ranges, fragments
# of longer numbers ("$20", "2.4.1", "3-5", "#42", "9999123 …").
_HOUSE = r"(?<![\d.,$#/-])(?P<house>\d{1,6}(?:-\d{1,4})?[A-Za-z]?)"

_DIRECTIONAL = (
    r"(?:N\.?E\.?|N\.?W\.?|S\.?E\.?|S\.?W\.?|[NSEW]\.?|"
    r"North|South|East|West|Northeast|Northwest|Southeast|Southwest)"
)

# A street-name word: ordinal ("5th", "42nd") or alphabetic word with
# apostrophes/hyphens ("James's", "Winston-Salem").
_STREET_WORD = r"(?:\d{1,3}(?:st|nd|rd|th)|[A-Za-z][A-Za-z'’\-]*)"

# Highway/route street forms where the number FOLLOWS the road word:
# "Highway 50", "Route 66", "County Road 12", "US-19", "I-90", "FM 1325".
# Short abbreviations (US, CR, FM, I) are case-sensitive to avoid matching
# prose ("give me 5 us 19" must not become an address).
_HWY_STREET = (
    r"(?:"
    r"(?:(?:State|County|Old)\s+)?(?:Highway|Hwy|Route|Rte|Rt)\.?\s*-?\s*\d{1,4}"
    r"|County\s+Road\s+\d{1,4}"
    r"|Interstate\s+\d{1,4}"
    r"|(?-i:US|CR|FM)\s*-?\s*\d{1,4}"
    r"|(?-i:I)-\d{1,4}"
    r")"
)

# Unit id must contain a digit or be a single letter — "Suite 210", "Apt 4B",
# "Unit B" pass; "Building permits" does not.
_UNIT_ID = r"(?:\d[A-Za-z0-9\-]{0,7}|[A-Za-z]\d[A-Za-z0-9\-]{0,6}|[A-Za-z](?![A-Za-z0-9]))"
_UNIT = (
    r"(?:Apartment|Apt|Suite|Ste|Unit|Building|Bldg|Floor|Room|Rm|Lot|Trailer|Space|Spc)"
    rf"\.?\s*#?\s*{_UNIT_ID}"
    rf"|#\s?{_UNIT_ID}"
)

# City: 1-4 capitalized words (case enforced even under IGNORECASE via (?-i:)).
_CITY = r"(?-i:[A-Z][A-Za-z'’.\-]{1,15}(?:[ ][A-Z][A-Za-z'’.\-]{1,15}){0,3})"

# State: full name (any case) or UPPERCASE-only 2-letter USPS abbreviation.
_STATE = rf"(?:{_STATE_NAME_ALT}|(?-i:(?:{_STATE_ABBREV_ALT}))\.?)"

_ZIP = r"\d{5}(?:-\d{4})?(?!\d)"

# Separators: comma (optionally with whitespace/newlines) or bare whitespace
# (which includes newlines, covering multi-line addresses).
_SEP = r"\s*,\s*|\s+"

# Shared tail: unit, city, state, ZIP — every part optional; acceptance rules
# are enforced by the code-level validator, which may trim the span back.
_TAIL = (
    rf"(?:(?P<unitsep>{_SEP})(?P<unit>{_UNIT}))?"
    rf"(?:(?P<citysep>{_SEP})(?P<city>{_CITY}))?"
    rf"(?:(?P<statesep>{_SEP})(?P<state>{_STATE})\b)?"
    rf"(?:(?P<zipsep>{_SEP})(?P<zip>{_ZIP}))?"
)

_STREET_ADDRESS_RE = re.compile(
    rf"{_HOUSE}"
    r"\s+"
    rf"(?:(?P<predir>{_DIRECTIONAL})\s+)?"
    r"(?:"
    rf"(?P<hwystreet>{_HWY_STREET})"
    r"|"
    rf"(?P<street>(?:{_STREET_WORD}\s+){{0,4}})"
    rf"(?P<suffix>(?:{_SUFFIX_ALT})\.?)(?=[\s,.;:!?)\"']|$)"
    rf"(?:\s+(?P<postdir>{_DIRECTIONAL})(?=[\s,.;:!?)\"']|$))?"
    r")"
    rf"{_TAIL}",
    re.IGNORECASE,
)

_PO_BOX_RE = re.compile(
    rf"(?P<pobox>(?:P\.?\s*O\.?\s*|Post\s+Office\s+)Box|POB)"
    rf"\s+#?(?P<house>\d{{1,8}})"
    rf"{_TAIL}",
    re.IGNORECASE,
)

_ORDINAL_RE = re.compile(r"\d{1,3}(?:st|nd|rd|th)$", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_suffix(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    return token.rstrip(".").lower()


def _is_state_token(token: str) -> bool:
    stripped = token.rstrip(".")
    return stripped in US_STATE_ABBREVS or stripped.lower() in _STATE_NAMES_LOWER


def _is_sep_delimited(sep: Optional[str]) -> bool:
    """True when the separator contains a comma or newline (explicit delimiter)."""
    return sep is not None and ("," in sep or "\n" in sep)


def _street_is_capitalized(street: Optional[str]) -> bool:
    """True when every word of the street name is capitalized or an ordinal."""
    if not street:
        return False
    for word in street.split():
        if _ORDINAL_RE.match(word):
            continue
        if not word[0].isupper():
            return False
    return True


def _city_word_ok(city: str) -> bool:
    """Reject blocklisted / query words masquerading as a city."""
    return city.split()[0].lower() not in _CITY_BLOCKLIST


def _followed_by_boundary(text: str, pos: int) -> bool:
    """True when *pos* is at end of text or followed by punctuation/newline
    or another capitalized construct — i.e. a plausible address ending."""
    rest = text[pos:]
    if not rest:
        return True
    return bool(re.match(r"[\s]*[.,;:!?)\"'\n]|[\s]*$", rest))


# ─────────────────────────────────────────────────────────────────────────────
# Match → ParsedAddress assembly (with span trimming)
# ─────────────────────────────────────────────────────────────────────────────

def _assemble(match: "re.Match", text: str, is_po_box: bool) -> Optional[ParsedAddress]:
    g = match.groupdict()
    start = match.start()

    house = g.get("house")
    predir = (g.get("predir") or "").strip() or None
    street = (g.get("street") or "").strip() or None
    suffix = g.get("suffix")
    hwystreet = g.get("hwystreet")
    postdir = g.get("postdir")

    if is_po_box:
        core_end = match.end("house")
    elif hwystreet:
        street = hwystreet.strip()
        suffix = None
        core_end = match.end("hwystreet")
    else:
        # Plausibility gate: an address needs a street name or a directional
        # ("42 St" alone is not an address).
        if not street and not predir:
            return None

        # Prose gate: real street names never contain lowercase function or
        # quantity words ("3-5 business days on Oak St").
        if street and any(
            w in _STREET_STOPWORDS for w in street.split()
        ):
            return None

        norm = _norm_suffix(suffix)
        has_tail = bool(g.get("city") or g.get("state") or g.get("zip"))
        if norm in _AMBIGUOUS_SUFFIXES:
            # Ambiguous suffixes need extra evidence: a capitalized street
            # name, or an explicit city/state/ZIP tail.
            if not (_street_is_capitalized(street) or has_tail):
                return None
        core_end = match.end("postdir") if postdir else match.end("suffix")

    # ── Tail acceptance (prefix-monotone: rejecting a part drops later parts) ──
    unit = city = state = zip_code = None
    end = core_end

    if g.get("unit"):
        unit = g["unit"]
        end = match.end("unit")

    city_txt = g.get("city")
    state_txt = g.get("state")
    zip_txt = g.get("zip")

    # Fixup 1: the city group can swallow an uppercase state abbreviation
    # ("Tempe AZ" as a two-word city, or "AZ" alone as the city).
    if city_txt and not state_txt:
        words = city_txt.split()
        if _is_state_token(words[-1]) and words[-1].isupper():
            if len(words) == 1:
                state_txt, city_txt = words[0], None
            else:
                state_txt = words[-1]
                city_txt = " ".join(words[:-1])

    # The city group's char class allows "." (for "St. Louis"), which can
    # absorb sentence punctuation — trim it off the component value.
    if city_txt and city_txt.endswith("."):
        city_txt = city_txt.rstrip(".")

    city_accepted = False
    if city_txt:
        delimited = _is_sep_delimited(g.get("citysep"))
        has_following = bool(state_txt or zip_txt)
        if _city_word_ok(city_txt):
            if has_following:
                # "…, Tempe, AZ" / "… Tempe AZ 85281" — safe.
                city_accepted = True
            elif delimited:
                # Bare city ("123 Main St, Tempe"): require a sentence-like
                # boundary after it, or a well-known city name, so we don't
                # absorb "…, John was there".
                if city_txt.lower() in MAJOR_CITIES or _followed_by_boundary(
                    text, match.end("city")
                ):
                    city_accepted = True

    if city_accepted:
        city = city_txt
        end = match.end("city")

    # Track whether the state token came from the regex state group or was
    # recovered out of the city group (Fixup 1) — the span end differs.
    state_from_city_group = state_txt is not None and g.get("state") is None

    state_accepted = False
    if state_txt and (city_accepted or city_txt is None or state_from_city_group):
        # A state is accepted when comma/newline-delimited, or preceded by an
        # accepted city, or followed by a ZIP ("123 Main St OR take a left"
        # has none of these).
        if (
            _is_sep_delimited(g.get("statesep"))
            or city_accepted
            or (state_from_city_group and _is_sep_delimited(g.get("citysep")))
            or zip_txt
        ):
            state_accepted = True

    if state_accepted:
        state = state_txt
        end = match.end("city") if state_from_city_group else match.end("state")

    zip_accepted = bool(zip_txt) and (state_accepted or city_accepted)
    if zip_accepted:
        zip_code = zip_txt
        end = match.end("zip")

    full_text = text[start:end].rstrip()
    end = start + len(full_text)

    # A span-final "." followed by end-of-text or whitespace is sentence
    # punctuation, not part of the address ("I live at 7 Main St.") — leave
    # it outside the span. Mid-address periods ("123 Main St., Tempe") and
    # runs of dots ("42 Main St..") are untouched.
    if full_text.endswith(".") and not full_text.endswith(".."):
        nxt = text[end:end + 1]
        if nxt == "" or nxt.isspace():
            full_text = full_text[:-1]
            end -= 1

    if not full_text:
        return None

    return ParsedAddress(
        full_text=full_text,
        start=start,
        end=end,
        house_number=house,
        house_number_span=(match.start("house"), match.end("house")) if house else None,
        pre_directional=predir,
        street_name=street,
        suffix=suffix,
        post_directional=postdir,
        unit=unit,
        city=city,
        state=state,
        zip_code=zip_code,
        po_box=text[match.start():match.end("house")] if is_po_box else None,
        is_po_box=is_po_box,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_addresses(text: str) -> List[ParsedAddress]:
    """
    Find every street address / PO box in *text*.

    Returns non-overlapping ParsedAddress objects sorted by start offset.
    """
    if not text:
        return []

    found: List[ParsedAddress] = []

    def _claimed(s: int, e: int) -> bool:
        return any(not (e <= a.start or s >= a.end) for a in found)

    # PO boxes first (more specific — no street suffix to anchor on).
    for match in _PO_BOX_RE.finditer(text):
        parsed = _assemble(match, text, is_po_box=True)
        if parsed and not _claimed(parsed.start, parsed.end):
            found.append(parsed)

    for match in _STREET_ADDRESS_RE.finditer(text):
        parsed = _assemble(match, text, is_po_box=False)
        if parsed and not _claimed(parsed.start, parsed.end):
            found.append(parsed)

    found.sort(key=lambda a: a.start)
    if found:
        logger.debug(f"[AddressParser] Found {len(found)} address(es)")
    return found


def parse(fragment: str) -> Optional[ParsedAddress]:
    """Parse a standalone address string. Returns None if no address found."""
    results = find_addresses(fragment)
    return results[0] if results else None


def verify_address_exists(parsed: ParsedAddress, timeout: float = 2.0) -> bool:
    """
    OPT-IN network check against OpenStreetMap Nominatim.

    Never called on the hot path unless the user enables address
    verification in config. Returns True when Nominatim knows the address;
    False on no-result or any network failure.
    """
    try:
        import requests
        query_parts = [p for p in (
            f"{parsed.house_number} {parsed.street_name or ''} {parsed.suffix or ''}".strip(),
            parsed.city, parsed.state,
        ) if p]
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": ", ".join(query_parts), "format": "json", "limit": 1},
            headers={"User-Agent": "SurrogateShield-Research/2.0"},
            timeout=timeout,
        )
        return bool(r.json())
    except Exception as exc:
        logger.debug(f"[AddressParser] Nominatim verify failed: {exc}")
        return False
