"""
generation/mimic.py — MimicGen

Surrogates that keep what an answer may depend on and nothing that
identifies (audit D1–D4, I7):

* people and e-mails are linked — one surrogate person per real person,
  same gender, locale and form (:mod:`.identity`);
* a phone keeps its country code, trunk digit and grouping; a date of birth
  moves by at most two years in the same format; a town becomes a real town
  of the same country (:mod:`.places`); an organisation keeps its legal and
  industry words; card, SSN, routing and IBAN numbers stay valid;
* one seeded RNG per generator: the same seed gives the same surrogates.

No surrogate equals or contains an original of the conversation (J4).
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Dict, FrozenSet, List, Optional, Set

import datetime
from faker import Faker

from ..detection import address_parser
from ..detection.geo_data import MAJOR_COUNTRIES, US_STATE_ABBREVS
from ..consistency import is_low_entropy, occurs
from ..entities import DetectedEntity
from . import places
from .identity import People

logger = logging.getLogger(__name__)

# Types whose surrogates mirror the original's exact character shape
# (an ID number swaps to a same-shape ID number).
_SHAPE_TYPES = frozenset({
    "mac_address", "vin", "passport", "id_number", "license_plate", "credential",
})

# Types whose surrogate is built from the original's format (audit I14): a
# URL keeps its host and path layout, a handle its @/#1234 frame, an age its
# words, a date its separators and month-name style.
_FORMAT_TYPES = frozenset({"url", "handle", "age", "dob"})

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_MONTH_RE = re.compile(
    r"(?i)\b(?:" + "|".join(m if len(m) <= 4 else f"{m[:3]}(?:{m[3:]})?" for m in _MONTHS)
    + r"|Sept)\b")


def _ordinal(m: "re.Match") -> str:
    n, suffix = int(m.group(1)), m.group(2)
    right = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return m.group(1) + (right.upper() if suffix.isupper() else right)


def _aba_check(number: str) -> bool:
    digits = [int(c) for c in number]
    return (
        3 * digits[0] + 7 * digits[1] + digits[2] +
        3 * digits[3] + 7 * digits[4] + digits[5] +
        3 * digits[6] + 7 * digits[7] + digits[8]
    ) % 10 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Address surrogate helpers (module-level, unit-testable without MimicGen)
# ─────────────────────────────────────────────────────────────────────────────

def shift_house_number(
    parsed: "address_parser.ParsedAddress",
    shift_range: int = 1,
    rng: Optional[random.Random] = None,
    forbidden: FrozenSet[str] = frozenset(),
) -> Optional[str]:
    """
    Produce a surrogate address by shifting ONLY the house number.

    The new number token is spliced into the ORIGINAL matched span, so
    commas, newlines, casing, and abbreviations survive byte-for-byte.

    Rules:
      • candidate deltas ±1…±shift_range, random order;
      • a shifted number below 1 is rejected (house number 1 always goes UP —
        fixes the v1 max(1, 0) identity-surrogate bug);
      • leading zeros are preserved ("0123" → "0124");
      • alpha suffixes ride along unchanged ("123A" → "124A");
      • candidates found in *forbidden* (other real addresses in the text,
        already-issued surrogates, shadow-map values) are skipped; the range
        widens up to shift_range+8 before giving up.

    Returns the full surrogate address string, or None when no valid
    candidate exists (caller should fall back to replace mode).
    """
    if parsed is None or parsed.house_number is None:
        return None
    rel = parsed.house_number_relative_span
    if rel is None:
        return None

    match = re.match(r"\d+", parsed.house_number)
    if not match:
        return None
    numeric = match.group()
    n = int(numeric)
    _rng = rng if rng is not None else random

    deltas: List[int] = []
    for d in range(1, max(1, shift_range) + 1):
        deltas.extend((d, -d))
    _rng.shuffle(deltas)
    # Widening tail (deterministic order) for collision escape.
    for d in range(max(1, shift_range) + 1, max(1, shift_range) + 9):
        deltas.extend((d, -d))

    for delta in deltas:
        new_n = n + delta
        if new_n < 1:
            continue
        new_numeric = str(new_n).zfill(len(numeric)) if numeric[0] == "0" else str(new_n)
        new_token = new_numeric + parsed.house_number[len(numeric):]
        s, e = rel
        candidate = parsed.full_text[:s] + new_token + parsed.full_text[e:]
        if candidate != parsed.full_text and candidate not in forbidden:
            return candidate
    return None


# Gender terms grouped by grammatical form, so a surrogate keeps the form of
# the original ("gender: male" → "gender: female", "she/her" → "he/him").
_GENDER_FAMILIES = (
    ("he/him", "she/her", "they/them"),
    ("male", "female", "non-binary"),
    ("man", "woman", "non-binary person"),
)
_GENDER_TERM = re.compile(
    r"(?<![\w/-])(he/him|she/her|they/them|non-?binary(?: person)?|female|male|woman|man)(?![\w/-])",
    re.IGNORECASE,
)


def _swap_gender(text: str, rng) -> str:
    """Replace the last gender term in *text* with a different term of the same form."""
    matches = list(_GENDER_TERM.finditer(text))
    if not matches:
        return rng.choice(("male", "female", "non-binary"))
    m = matches[-1]
    term = m.group(1).lower().replace("nonbinary", "non-binary")
    family = next((f for f in _GENDER_FAMILIES if term in f), _GENDER_FAMILIES[1])
    choice = rng.choice([t for t in family if t != term])
    if m.group(1)[:1].isupper():
        choice = choice[:1].upper() + choice[1:]
    return text[: m.start()] + choice + text[m.end():]



_MAX_ORIGINAL_RETRIES = 20


_EURO_COMPOUND = (r"stra(?:ß|ss)e|weg|platz|allee|gasse|ring|damm|straat|laan|gracht|plein|kade"
                  r"|singel|dijk|steeg|dreef|gata|gatan|gaten|vägen|veien|vegen|vej|gade|stræde"
                  r"|katu|tie")
_EURO_STREET_WORD = re.compile(
    r"(?i)\b(?:rue|avenue|av|avda|boulevard|bd|bv|bvd|blvd|bulevar|chemin|all[ée]e|impasse|place"
    r"|quai|route|cours|rua|avenida|travessa|trav|alameda|estrada|largo|pra[çc]a|rodovia|calle"
    r"|paseo|plaza|carrer|camino|carrera|via|viale|piazza|corso|vicolo|strada|ul|ulica|aleja"
    r"|apto|apartamento|apt|piso|planta|depto|dpto|puerta|pta|esq|dto|dta|dir|izq|izqda|dcha"
    r"|frente|appartement|appt|[ée]tage|wohnung|whg|og|eg|dg|ug|stock|links|rechts|mitte"
    r"|interno|sala|bloco|bloque|escalera|esc|portal|unit|flat|suite|ste|top|stiege|bus|bo[iî]te"
    r"|h|no|house|plot|door|\w+?(?:" + _EURO_COMPOUND + r"))\b"
)
# a stem that suits the compound's language: "Kerkstraat" -> "Molenstraat"
_EURO_STEMS = (
    (re.compile(r"(?i)straat|laan|gracht|plein|kade|singel|dijk|steeg|dreef"),
     ("Molen", "Kerk", "Linden", "Dorps", "Bloemen", "Beuken")),
    (re.compile(r"(?i)gata|gatan|gaten|vägen|veien|vegen|vej|gade|stræde"),
     ("Stor", "Kirke", "Skole", "Strand", "Bjørke", "Møller")),
    (re.compile(r"(?i)katu|tie"), ("Koulu", "Kirkko", "Puisto", "Ranta")),
    (re.compile(r".*"), ("Linden", "Berg", "Garten", "Schiller", "Mühlen", "Kirch", "Wald")),
)
_ZH_ROAD_NAMES = ("人民", "中山", "解放", "建设", "和平", "文化", "新华", "长江", "光明", "朝阳")
_EURO_JOINERS = frozenset("de des du la le da do dos das del della di von der bis ter".split())
_EURO_STREET_NAMES = ("Flores", "Liberdade", "Garibaldi", "Mozart", "Pasteur", "Castelo",
                      "Oliveira", "Roma", "Goethe", "Mayor", "Vitória", "Tilleuls",
                      "Sol", "Lumière", "Lindenhof")


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_RE = re.compile(r"(?i)\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|nesday|rsday|urday)?\b")

# ITU-T E.164 country codes: one digit (1, 7) and the two-digit ones; the
# rest are three digits.
_CC2 = frozenset("""20 27 30 31 32 33 34 36 39 40 41 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58
    60 61 62 63 64 65 66 81 82 84 86 90 91 92 93 94 95 98""".split())
_TOLL_FREE = frozenset({"800", "833", "844", "855", "866", "877", "888"})

_KEY_PREFIXES = ("sk-ant-api03-", "sk-ant-", "sk-proj-", "sk_live_", "sk_test_", "pk_live_",
                 "pk_test_", "rk_live_", "github_pat_", "ghp_", "gho_", "ghs_", "ghu_", "glpat-",
                 "xoxb-", "xoxp-", "xoxa-", "xapp-", "AKIA", "ASIA", "AIza", "SG.", "hf_", "sk-")
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

# Words of an organisation or facility name that say what it is, not who:
# kept as written ("Mitchell Family Dental LLC" → "Garrison Family Dental LLC").
_LEGAL = re.compile(r"(?i)^(?:inc|llc|l\.l\.c|ltd|limited|gmbh|plc|corp|corporation|co|company"
                    r"|llp|lp|s\.a|sa|ag|pty|pvt|bv|nv|srl|spa|sarl|oy|ab|kk|pc|pllc)\.?$")
_GENERIC = frozenset("""
    the of and for at in on de la le del des du di da y und
    hospital medical center centre clinic health healthcare care dental dentistry pharmacy
    family pediatrics pediatric orthopedics surgery surgical urgent emergency rehab rehabilitation
    therapy wellness vision eye animal veterinary vet labs lab laboratory diagnostics imaging
    bank credit union savings trust financial capital investments investment insurance mutual
    wealth advisors advisory accounting tax law legal attorneys lawyers firm partners associates
    group holdings enterprises industries international global national regional community
    general memorial united american first city county state federal department office agency
    school schools academy university college institute elementary middle high primary
    secondary preschool kindergarten daycare montessori charter learning education
    church temple mosque synagogue ministries chapel cathedral parish
    consulting solutions systems technologies technology tech software digital data networks
    media marketing design creative studio studios productions entertainment publishing press
    construction builders building contractors roofing plumbing electric electrical hvac
    landscaping cleaning services service logistics transport transportation trucking freight
    shipping motors auto automotive airlines airways energy power utilities water gas oil
    foods food restaurant restaurants cafe coffee bakery grill kitchen bar pizza deli market
    supermarket grocery store stores shop shops boutique salon spa fitness gym yoga
    realty real estate properties property homes housing apartments management
    library museum gallery theater theatre hotel inn resort motel airport station terminal
    stadium arena mall plaza tower towers hall park center court courthouse bridge tunnel
    building complex campus warehouse factory plant mill farm farms ranch vineyard winery
    brewery distillery foundation society association council club league network alliance
    corp inc llc ltd co company corporation
    dermatology cardiology oncology neurology radiology urology gynecology obstetrics psychiatry
    psychology optometry chiropractic physiotherapy podiatry orthodontics
    """.split())

_FUNCTION_WORDS = frozenset("the of and for at in on de la le del des du di da y und".split())
_STREET_WORDS = frozenset("""ave avenue st street rd road blvd boulevard dr drive ln lane way pl
    place ct court pkwy parkway hwy highway sq square ter terrace cir circle""".split())
_GENERIC = _GENERIC | _STREET_WORDS
# the word that says what a facility is: kept when a name has no name word
_KIND_WORDS = _STREET_WORDS | frozenset("""hospital clinic center centre school university
    college academy institute bank church temple mosque synagogue library museum gallery hotel
    airport station stadium arena mall plaza tower hall park building office pharmacy""".split())


_SAINTS = ("Mary", "Luke", "Francis", "Vincent", "Anthony", "Jude", "Elizabeth", "Catherine",
           "Michael", "Thomas", "Patrick", "Anne", "Peter", "Paul", "John", "Mark", "Clare",
           "Agnes", "Andrew", "Joseph", "Bernard", "Teresa")


def _luhn_digit(payload: str) -> str:
    total = 0
    for i, c in enumerate(reversed(payload)):
        d = int(c)
        if i % 2 == 0:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return str((10 - total % 10) % 10)


def _splice_digits(original: str, digits: str) -> str:
    it = iter(digits)
    return "".join(next(it) if c.isdigit() else c for c in original)


def _case_like(model: str, value: str) -> str:
    if len(model) > 1 and model.isupper():
        return value.upper()
    if model.islower():
        return value.lower()
    return value


class MimicGen:
    """Generates realistic, collision-resistant surrogates for detected PII.

    Args:
        seed: Seed for this generator's RNG. The same seed gives the same
              surrogates for the same inputs; None draws one from the OS.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.used_surrogates: Set[str] = set()
        self._rng = random.Random(seed)
        self._fake = Faker()
        self._fake.seed_instance(self._rng.getrandbits(64))
        self.people = People(self._rng)
        self._context = ""                  # the message being masked
        self._avoid: Set[str] = set()       # casefold originals of the conversation
        self._fresh = False                 # retrying a rejected person surrogate
        self._used_cf: Set[str] = set()     # casefold used_surrogates (see _taken)
        self._used_cf_n = 0

    def _taken(self, candidate: str) -> bool:
        """*candidate* was issued before, in any case: restoration matches
        case-insensitively, so "Shreveport" and "shreveport" are one value."""
        if self._used_cf_n != len(self.used_surrogates):      # the set only grows
            self._used_cf = {u.casefold() for u in self.used_surrogates}
            self._used_cf_n = len(self.used_surrogates)
        return candidate.casefold() in self._used_cf

    def _unique(self, generator_fn, max_attempts: int = 50) -> str:
        for _ in range(max_attempts):
            candidate = str(generator_fn())
            # Low-entropy values (an age, a gender term) are restored only in
            # their own turn, so they need not be unique per conversation —
            # there are only a handful of near ages (audit I5).
            if is_low_entropy(candidate):
                return candidate
            if not self._taken(candidate):
                self.used_surrogates.add(candidate)
                return candidate
        fallback = str(generator_fn()) + "_" + "".join(
            self._rng.choices(string.ascii_lowercase, k=4)
        )
        self.used_surrogates.add(fallback)
        return fallback

    def _digits(self, n: int) -> str:
        return "".join(str(self._rng.randint(0, 9)) for _ in range(n))

    # ── Per-type generators ────────────────────────────────────────────────────

    def _nxx(self, area: bool) -> str:
        """A valid NANP area code or exchange: N in 2–9, not N11; an area
        code is not N9X or toll-free, an exchange is not 555."""
        while True:
            v = str(self._rng.randint(2, 9)) + self._digits(2)
            if v[1:] == "11" or (area and (v[1] == "9" or v in _TOLL_FREE)) \
                    or (not area and v == "555"):
                continue
            return v

    def _gen_phone_like(self, original: str) -> str:
        """Same country code, trunk prefix, first national digit and every
        separator; a NANP number keeps a valid area code and exchange (and a
        toll-free prefix). An extension keeps its label and length."""
        ext = re.search(r"(?i)(,?[ \t]*(?:ext\.?|extension|x)[ \t]*)(\d{1,5})$", original)
        if ext and ext.start() > 0:
            return (self._gen_phone_like(original[:ext.start()]) + ext.group(1)
                    + self._digits(len(ext.group(2))))
        ds = "".join(c for c in original if c.isdigit())
        lead = original.lstrip()
        keep, nanp = 0, False
        if lead.startswith(("+", "00")):
            off = 2 if lead.startswith("00") else 0
            cc = 1 if ds[off:off + 1] in ("1", "7") else 2 if ds[off:off + 2] in _CC2 else 3
            keep, nanp = off + cc, ds[off:off + 1] == "1"
        elif len(ds) == 11 and ds[0] == "1":
            keep, nanp = 1, True
        elif len(ds) == 10 and ds[0] in "23456789":
            nanp = True
        nat = ds[keep:]
        for _ in range(50):
            if nanp and len(nat) == 10:
                area = nat[:3] if nat[:3] in _TOLL_FREE else self._nxx(area=True)
                new = area + self._nxx(area=False) + self._digits(4)
            else:
                k = min(2 if nat[:1] == "0" else 1, max(0, len(nat) - 2))
                new = nat[:k] + self._digits(len(nat) - k)
            if ds[:keep] + new != ds:
                break
        return _splice_digits(original, ds[:keep] + new)

    def _gen_ssn_like(self, original: str) -> str:
        """A valid SSN (area 001–899 except 666, non-zero group and serial)
        in the original's layout."""
        ds = "".join(c for c in original if c.isdigit())
        if len(ds) != 9:
            return self._shape_like(original)
        while True:
            area = self._rng.randint(1, 899)
            if area == 666:
                continue
            new = f"{area:03d}{self._rng.randint(1, 99):02d}{self._rng.randint(1, 9999):04d}"
            if new != ds:
                return _splice_digits(original, new)

    def _gen_card_like(self, original: str) -> str:
        """Same length, separators and network prefix (two digits); valid
        Luhn check digit."""
        ds = "".join(c for c in original if c.isdigit())
        if len(ds) < 12:
            return self._shape_like(original)
        while True:
            payload = ds[:2] + self._digits(len(ds) - 3)
            new = payload + _luhn_digit(payload)
            if new != ds:
                return _splice_digits(original, new)

    def _gen_zip_like(self, original: str) -> str:
        return re.sub(r"\d+", lambda m: self._fake_number_like(m.group()), original)

    def _gen_ip_like(self, original: str) -> str:
        """IPv4: a private address keeps its private prefix, a public one
        stays public; a CIDR suffix or port is kept. IPv6: same shape."""
        m = re.fullmatch(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(.*)", original)
        if not m or any(int(g) > 255 for g in m.groups()[:4]):
            return self._shape_like(original)
        o = [int(g) for g in m.groups()[:4]]
        if o[0] in (10, 127):
            keep = 1
        elif (o[0], o[1]) in ((192, 168), (169, 254)) or (o[0] == 172 and 16 <= o[1] <= 31) \
                or (o[0] == 100 and 64 <= o[1] <= 127):
            keep = 2
        else:
            keep = 0
        while True:
            new = o[:keep]
            if keep == 0:
                first = self._rng.randint(1, 223)
                if first in (10, 100, 127, 169, 172, 192):
                    continue
                new = [first]
            new += [self._rng.randint(0, 255) for _ in range(3 - len(new))] + \
                   [self._rng.randint(1, 254)]
            if new != o:
                return ".".join(map(str, new)) + m.group(5)

    def _token_like(self, value: str) -> str:
        """A random string of the same shape; hexadecimal stays hexadecimal."""
        if len(value) >= 8 and re.fullmatch(r"[0-9a-f]+", value) and re.search(r"[a-f]", value):
            return "".join(self._rng.choice("0123456789abcdef") for _ in value)
        if len(value) >= 8 and re.fullmatch(r"[0-9A-F]+", value) and re.search(r"[A-F]", value):
            return "".join(self._rng.choice("0123456789ABCDEF") for _ in value)
        return self._shape_like(value)

    def _gen_api_key_like(self, original: str) -> str:
        """The provider prefix ("sk-", "AKIA", "ghp_") and the shape kept."""
        pre = next((p for p in _KEY_PREFIXES if original.startswith(p)), "")
        rest = original[len(pre):]
        return pre + self._token_like(rest) if rest else self._token_like(original)

    def _gen_crypto_like(self, original: str) -> str:
        """Same chain prefix, length and alphabet (base58, bech32, hex)."""
        rng = self._rng
        if original[:2].lower() == "0x":
            body = original[2:]
            hexd = "0123456789ABCDEF" if body.isupper() else "0123456789abcdef"
            return original[:2] + "".join(rng.choice(hexd) for _ in body)
        m = re.match(r"(?i)(bc1|tb1|ltc1)", original)
        if m:
            body = "".join(rng.choice(_BECH32) for _ in original[m.end():])
            return m.group() + (body.upper() if original.isupper() else body)
        return original[:1] + "".join(rng.choice(_B58) for _ in original[1:])

    def _gen_bank_like(self, original: str) -> str:
        """A routing number stays ABA-valid; an account number keeps its shape."""
        ds = "".join(c for c in original if c.isdigit())
        if len(ds) != 9 or not _aba_check(ds):
            return self._shape_like(original)
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        while True:
            digits = [self._rng.randint(0, 9) for _ in range(8)]
            digits[0] = self._rng.choice([0, 1, 2, 3])
            check = (10 - sum(w * d for w, d in zip(weights, digits)) % 10) % 10
            new = "".join(map(str, digits)) + str(check)
            if new != ds and _aba_check(new):
                return _splice_digits(original, new)

    def _place_taken(self, candidate: str) -> bool:
        return (candidate.casefold() in self._avoid or self._taken(candidate)
                or bool(self._context and occurs(self._context, candidate)))

    def _gen_place(self, original: str) -> str:
        """A real place of the same kind (audit I7): a town of the same
        country, a state for a state, a lake for a lake."""
        cf = original.strip().casefold()
        if cf in MAJOR_COUNTRIES:
            pool = sorted(c.title() for c in MAJOR_COUNTRIES if len(c) > 3 and c != cf
                          and not self._place_taken(c.title()))
            if pool:
                return _case_like(original, self._rng.choice(pool))
        parts = re.split(r"(,\s*)", original)
        out = []
        for part in parts:
            if not part.strip() or part.lstrip().startswith(","):
                out.append(part)
                continue
            place = places.real_place(part, self._context, self._rng.choice, self._place_taken)
            out.append(place if place is not None else self._fake.city())
        return "".join(out)

    def _gen_institution(self, original: str) -> str:
        """An organisation or facility of the same kind: legal suffix and
        industry words kept, name words replaced (one surrogate surname per
        word per conversation, shared with people), "of <place>" → another
        real place. Adds no comma."""
        # "of New Mexico", "at Fort Collins": a known place after a preposition
        place_at = {}
        for m in re.finditer(r"\b(?:of|at|in)\s+((?:[A-Z][\w.'’-]*)(?:\s+[A-Z][\w.'’-]*){0,2})",
                             original):
            words = m.group(1).split()
            for n in range(len(words), 0, -1):
                name = " ".join(words[:n])
                if places.is_known_place(name):
                    place_at[m.start(1)] = m.start(1) + len(name)
                    break
        out, pos, prev, replaced = [], 0, "", False
        skip_to = -1
        # in a lowercase name ("mill ave") the lowercase words are the name
        titled = any(c.isupper() for c in original)
        for m in re.finditer(r"[^\W_][\w'’.\-]*", original):
            if m.start() < skip_to:
                continue
            if m.start() in place_at:
                end = place_at[m.start()]
                rep = self._gen_place(original[m.start():end])
                out += [original[pos:m.start()], rep]
                pos = skip_to = end
                replaced = True
                continue
            w = m.group()
            bare = w.rstrip(".")
            poss = re.search(r"['’]s$", bare)
            stem = bare[:poss.start()] if poss else bare
            tail = w[len(stem):]
            low = stem.lower()
            if (_LEGAL.match(bare) or low in _GENERIC or low in ("st", "saint", "mt")
                    or titled and not stem[:1].isupper() and not stem[:1].isdigit()):
                rep = w
            elif prev.lower().rstrip(".") in ("st", "saint", "san", "santa", "sainte"):
                rep = _case_like(stem, self._rng.choice([n for n in _SAINTS if n.lower() != low])
                                 ) + tail
            elif stem.isdigit():
                rep = self._fake_number_like(stem) + tail
            elif stem.isupper() and len(stem) <= 5:
                rep = "".join(self._rng.choice(string.ascii_uppercase) for _ in stem) + tail
            else:
                rep = _case_like(stem, self.people.surname(stem)) + tail
            replaced |= rep != w
            out += [original[pos:m.start()], rep]
            pos, prev = m.end(), bare
        out.append(original[pos:])
        result = "".join(out)
        if not replaced:
            # only generic words ("Medical Center", "mill ave"): the first word
            # that is not the facility's kind becomes a name ("Hampton Center").
            # A surname in front would leave the original whole in the surrogate.
            words = list(re.finditer(r"[^\W_][\w'’\-]*", original))
            plain = [m for m in words if m.group().lower() not in _FUNCTION_WORDS]
            pick = next((m for m in plain if m.group().lower() not in _KIND_WORDS
                         and not _LEGAL.match(m.group())), plain[0] if plain else None)
            if pick is None:
                return self.people.surname(original) + " " + original
            name = _case_like(pick.group(), self.people.surname(pick.group()))
            result = original[:pick.start()] + name + original[pick.end():]
        return result
    # ── Address surrogates (mode-aware) ────────────────────────────────────────

    def _gen_address(
        self,
        entity: Optional[DetectedEntity] = None,
        mode: str = "shift",
        shift_range: int = 1,
        forbidden: FrozenSet[str] = frozenset(),
    ) -> str:
        """
        Address surrogate honoring the configured address mode.

          shift   → house number ±shift_range spliced into the original span
                    (street/city/state/ZIP and all formatting preserved);
          replace → structure-preserving fake: every component replaced by a
                    fake of the same shape, in the same slot.

        Falls back: shift → replace (on collision exhaustion),
        replace without a parse → plain Faker address (last resort).
        """
        parsed = getattr(entity, "parsed", None) if entity is not None else None
        if parsed is None and entity is not None:
            parsed = address_parser.parse(entity.text)

        blocked = frozenset(forbidden | self.used_surrogates)

        if mode == "coarse":
            # I6: sensitive service query — drop the street line, keep the
            # (already public-scale) city/state/ZIP-less tail as written.
            tail = ""
            if parsed is not None:
                anchor = parsed.city or parsed.state
                if anchor and anchor in parsed.full_text:
                    tail = parsed.full_text[parsed.full_text.index(anchor):]
                    if parsed.zip_code:
                        tail = tail.replace(parsed.zip_code, "").rstrip(" ,")
            coarse = f"my area, {tail}" if tail else "my area"
            if coarse not in blocked:
                self.used_surrogates.add(coarse)
                return coarse
            # a second address in the same coarse message: a structure-
            # preserving fake keeps the mapping one-to-one (exact restore)

        if mode == "shift" and parsed is not None:
            shifted = shift_house_number(
                parsed, shift_range=shift_range, rng=self._rng, forbidden=blocked
            )
            if shifted is not None:
                self.used_surrogates.add(shifted)
                return shifted
            logger.warning(
                "[MimicGen] shift collision exhausted for %r — falling back to replace",
                entity.text if entity else "?",
            )

        if parsed is not None:
            for _ in range(20):
                candidate = self._replace_address(parsed)
                if candidate not in blocked and not self._taken(candidate):
                    self.used_surrogates.add(candidate)
                    return candidate

        # A street or unit the US parser cannot read ("14 rue des Lilas",
        # "Rua Augusta 1508", "apto 52", "3º B", "H.No. 3-118"): keep its
        # words and shape, swap the numbers and the name words (I13). A full
        # Faker address in place of "2. OG links" would read as nonsense.
        if entity is not None and (_EURO_STREET_WORD.search(entity.text)
                                   or re.search(r"\d", entity.text)):
            for _ in range(20):
                candidate = self._euro_street_like(entity.text)
                if candidate not in blocked and candidate != entity.text \
                        and not self._taken(candidate):
                    self.used_surrogates.add(candidate)
                    return candidate

        # Last resort: no structure available.
        return self._unique(lambda: self._fake.address().replace("\n", ", "))

    def _euro_street_like(self, text: str) -> str:
        def digits(m):
            d = m.group(0)
            out = str(self._rng.randint(1, 9)) + "".join(
                str(self._rng.randint(0, 9)) for _ in d[1:])
            return out
        out = re.sub(r"\d+", digits, text)

        def word(m):
            w = m.group(0)
            de = re.fullmatch(r"(?i)(\w+?)(" + _EURO_COMPOUND + r")", w)
            if de:   # a compound (de/nl/nordic/fi): the prefix IS the street name
                stems = next(st for rx, st in _EURO_STEMS if rx.fullmatch(de.group(2)))
                stem = self._rng.choice([n for n in stems if n.lower() != de.group(1).lower()])
                return stem + de.group(2)
            if _EURO_STREET_WORD.fullmatch(w) or w.lower() in _EURO_JOINERS:
                return w
            if w.isupper() and w.isalpha() and len(w) <= 3:
                return letters(w)    # postcode letters "8861 AJ"
            choice = self._rng.choice([n for n in _EURO_STREET_NAMES if n != w])
            return choice.upper() if w.isupper() else choice

        def letters(w):
            return "".join(self._rng.choice([c for c in "ABCDEFGHJKLMNPRSTVWXZ" if c != x])
                           for x in w)
        out = re.sub(r"[A-ZÀ-Ý][\w'’\-]+", word, out)
        # a Chinese road name: "桃园路88号" -> "建设路31号"
        out = re.sub(r"[\u4e00-\u9fff]{1,4}(?=路|街|大道|巷|弄|胡同)", lambda m: self._rng.choice(
            [n for n in _ZH_ROAD_NAMES if n != m.group()]), out)
        # a door or stair letter: "3º B", "Dpto B" (not the "H" of "H.No.")
        return re.sub(r"(?<![\w.])[A-Z](?![\w.])", lambda m: letters(m.group()), out)

    def _replace_address(self, parsed: "address_parser.ParsedAddress") -> str:
        """Build a structure-preserving fake: same components, same separators,
        same shapes — ONE unit, never concatenated fragments."""
        text = parsed.full_text
        replacements = []  # (start, end, replacement) relative to full_text
        cursor = 0

        def locate(value: str) -> Optional[tuple]:
            nonlocal cursor
            idx = text.find(value, cursor)
            if idx < 0:
                return None
            cursor = idx + len(value)
            return (idx, idx + len(value))

        # House / box number: same digit count, never identical.
        if parsed.house_number:
            span = parsed.house_number_relative_span
            cursor = span[1]
            numeric = re.match(r"\d+", parsed.house_number).group()
            new_numeric = self._fake_number_like(numeric)
            replacements.append(
                (span[0], span[0] + len(numeric), new_numeric)
            )

        # Street name (skip for highway forms, whose digits are handled below).
        if parsed.street_name:
            span = locate(parsed.street_name)
            if span:
                if re.search(r"\d", parsed.street_name):
                    # "Highway 50" → fake the route number, keep the road word.
                    route = re.search(r"\d+", parsed.street_name)
                    new_route = self._fake_number_like(route.group())
                    replacements.append(
                        (span[0] + route.start(), span[0] + route.end(), new_route)
                    )
                else:
                    replacements.append((span[0], span[1], self._fake.last_name()))

        # Unit: keep the designator, fake the digits of the id.
        if parsed.unit:
            span = locate(parsed.unit)
            if span:
                for digits in re.finditer(r"\d+", parsed.unit):
                    replacements.append((
                        span[0] + digits.start(),
                        span[0] + digits.end(),
                        self._fake_number_like(digits.group()),
                    ))

        # City: fake city of similar shape.
        if parsed.city:
            span = locate(parsed.city)
            if span:
                town = places.real_place(parsed.city, self._context, self._rng.choice,
                                         self._place_taken) or self._fake.city()
                replacements.append((span[0], span[1], town))

        # State: different state, same form (2-letter stays 2-letter).
        if parsed.state:
            span = locate(parsed.state)
            if span:
                token = parsed.state.rstrip(".")
                if token.isupper() and len(token) == 2:
                    choices = sorted(US_STATE_ABBREVS - {token})
                    new_state = self._rng.choice(choices)
                else:
                    new_state = self._fake.state()
                replacements.append((span[0], span[0] + len(token), new_state))

        # ZIP: same shape (ZIP+4 stays ZIP+4).
        if parsed.zip_code:
            span = locate(parsed.zip_code)
            if span:
                for digits in re.finditer(r"\d+", parsed.zip_code):
                    replacements.append((
                        span[0] + digits.start(),
                        span[0] + digits.end(),
                        self._fake_number_like(digits.group()),
                    ))

        result = text
        for start, end, value in sorted(replacements, reverse=True):
            result = result[:start] + value + result[end:]
        return result

    def _fake_number_like(self, numeric: str) -> str:
        """Random number with the same digit count, different value,
        leading-zero style preserved."""
        for _ in range(20):
            candidate = "".join(
                str(self._rng.randint(0, 9)) for _ in range(len(numeric))
            )
            if candidate != numeric and (len(numeric) == 1 or candidate[0] != "0"):
                if numeric[0] == "0":
                    candidate = "0" + candidate[1:]
                return candidate
        return str(int(numeric) + 1).zfill(len(numeric))

    def _shape_like(self, original: str) -> str:
        """
        Random string with the same character-class shape as *original*:
        digits→digits, uppercase→uppercase, lowercase→lowercase, separators
        preserved.  Hex-looking strings (IPv6, MAC) stay within hex digits
        so the surrogate remains format-valid.
        """
        stripped = original.replace(":", "").replace("-", "").replace(".", "")
        is_hexish = (
            (":" in original or "." in original)
            and bool(re.fullmatch(r"[0-9A-Fa-f]*", stripped))
        )
        for _ in range(20):
            out = []
            for c in original:
                if c.isdigit():
                    out.append(str(self._rng.randint(0, 9)))
                elif is_hexish and c.isalpha():
                    pool = "abcdef" if c.islower() else "ABCDEF"
                    out.append(self._rng.choice(pool))
                elif c.isupper():
                    out.append(self._rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
                elif c.islower():
                    out.append(self._rng.choice("abcdefghijkmnpqrstuvwxyz"))
                else:
                    out.append(c)
            candidate = "".join(out)
            if candidate != original:
                return candidate
        return original[::-1]

    def _gen_iban_like(self, original: str) -> str:
        """
        Same-country fake IBAN with VALID mod-97 check digits and the
        original grouping/spacing preserved.
        """
        compact = re.sub(r"\s+", "", original)
        country = compact[:2]
        body = "".join(
            str(self._rng.randint(0, 9)) if c.isdigit()
            else self._rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
            for c in compact[4:]
        )
        rearranged = body + country + "00"
        numeric = "".join(str(int(c, 36)) for c in rearranged)
        check = 98 - (int(numeric) % 97)
        candidate = f"{country}{check:02d}{body}"
        out, i = [], 0
        for c in original:
            if c.isspace():
                out.append(c)
            else:
                out.append(candidate[i])
                i += 1
        return "".join(out)

    # ── Format-preserving generators (url, handle, age, dob) ──────────────────

    def _slug(self, like: str = "") -> str:
        """A user name in the style of *like*, linked to the surrogate of the
        person it names ("sarah.mitchell" → "mia.lopez")."""
        return self.people.local(like) if like else self.people.local("user")

    def _gen_handle_like(self, original: str) -> str:
        m = re.fullmatch(r"(@?)(.+?)(#\d{4})?", original)
        prefix, body, tag = m.group(1), m.group(2), m.group(3)
        if tag:
            tag = "#" + "".join(str(self._rng.randint(0, 9)) for _ in range(4))
        slug = self._slug(body)
        if slug.casefold() == body.casefold():
            # a role name ("admin", "support") is kept as an e-mail local
            # part, but a handle that keeps it is the original
            slug = self.people._slug_like(body)
        return f"{prefix}{slug}{tag or ''}"

    def _gen_url_like(self, original: str) -> str:
        """Same scheme and host style; the identifying part (profile path
        segment, personal sub-domain or domain label) becomes a fake slug."""
        from ..detection import pattern_scan as ps
        m = re.match(r"(?i)((?:https?://)?(?:www\.)?)([^/?#\s]+)(.*)", original)
        lead, host, path = m.groups()
        hl = host.lower()
        marker = ps._PERSON_PATH_MARKERS.search(path)
        if hl in ps._PROFILE_HOSTS or marker:
            segs = path.split("/")
            # the segment after a person marker, or the first path segment
            idx = 1
            if marker:
                idx = path[:marker.start()].count("/") + 2
                if marker.group().startswith(("/~", "/@")):
                    idx -= 1
            if idx < len(segs) and segs[idx]:
                seg = segs[idx]
                pre = seg[:1] if seg[:1] in "~@" else ""
                segs[idx] = pre + self._slug(seg.lstrip("~@"))
                return lead + host + "/".join(segs)
        labels = host.split(".")
        if hl.endswith(ps._PERSONAL_HOST_SUFFIXES):
            i = 0                                   # name.github.io → sub-domain
        else:
            i = max(0, len(labels) - 2)             # sarahmitchell.dev → label
        labels[i] = self._slug(labels[i]).replace(".", "-").replace("_", "-").lower()
        return lead + ".".join(labels) + path

    def _gen_hostname_like(self, original: str) -> str:
        """"Galaxy-S23-Ritika" -> "Galaxy-S23-Priya": the owner's name part
        becomes the surrogate of that name; the device words stay."""
        from ..detection.structural import _DEVICE_WORDS
        parts = re.split(r"([-_'’]s?[-_ ]?|[-_ ])", original)
        out = []
        for p in parts:
            if (p.isalpha() and len(p) >= 3 and p.lower() not in _DEVICE_WORDS):
                new = self.people.name(p.capitalize())
                p = new.split()[0] if p[:1].isupper() else new.split()[0].lower()
            out.append(p)
        return "".join(out)

    def _gen_age_like(self, original: str) -> str:
        m = re.search(r"\d+", original)
        n = int(m.group())
        step = self._rng.randint(1, 3) * self._rng.choice((-1, 1))
        new = n + step if n + step >= 1 else n + abs(step)
        return original[:m.start()] + str(new) + original[m.end():]

    # ── dates of birth (audit D2) ──────────────────────────────────────────────

    @staticmethod
    def _read_date(text: str):
        """The fields of a date as written: ``(fields, kind, ambiguous)`` with
        fields {"year"|"month"|"day"|"monthname"|"weekday": match}, kind
        "ymd", "ym", "y" or "m"; None when it is not a date."""
        months = list(_MONTH_RE.finditer(text))
        nums = list(re.finditer(r"(\d+)(st|nd|rd|th)?", text, re.IGNORECASE))
        if len(months) > 1 or len(nums) > 3 or (not nums and not months):
            return None
        f: Dict[str, "re.Match"] = {}
        wd = _WEEKDAY_RE.search(text)
        if wd and not (months and months[0].start() <= wd.start() < months[0].end()):
            f["weekday"] = wd
        if months:
            f["monthname"] = months[0]
        year = next((n for n in nums if len(n.group(1)) == 4), None)
        if year is None:
            apos = next((n for n in nums if n.start() and text[n.start() - 1] in "'’"), None)
            if apos is not None:
                year = apos
            elif len(nums) == 3 or (months and len(nums) == 2):
                year = nums[-1]
        rest = [n for n in nums if n is not year]
        if year is not None:
            f["year"] = year
        ambiguous = False
        if months:
            if len(rest) > 1:
                return None
            if rest:
                f["day"] = rest[0]
        elif len(rest) == 2:
            a, b = int(rest[0].group(1)), int(rest[1].group(1))
            if year is not None and nums[0] is year:          # ISO: year, month, day
                f["month"], f["day"] = rest
            elif a > 12 >= b:
                f["day"], f["month"] = rest
            elif b > 12 >= a:
                f["month"], f["day"] = rest
            elif a <= 12 and b <= 12:
                f["month"], f["day"] = rest
                ambiguous = True
            else:
                return None
        elif len(rest) == 1:
            if year is None:
                return None
            f["month"] = rest[0]
        elif rest:
            return None
        if "year" not in f:
            return (f, "m", False) if months and not nums else None
        if not ("month" in f or "monthname" in f):
            return f, "y", False
        return f, ("ymd" if "day" in f else "ym"), ambiguous

    def _gen_dob_like(self, original: str) -> str:
        """The date moved by 30 days to two years, never into the future, in
        the original's exact format: field order, separators, zero padding,
        ordinal suffix, month-name style and case, two-digit year, weekday."""
        read = self._read_date(original)
        if read is None:
            return self._dob_legacy(original)
        f, kind, ambiguous = read
        rng = self._rng
        today = datetime.date.today()

        def val(name):
            return int(f[name].group(1))

        if kind == "m":
            month = _MONTHS.index(next(m for m in _MONTHS
                                       if m[:3].lower() == f["monthname"].group()[:3].lower()))
            new_month = (month + rng.choice((-1, 1)) * rng.randint(1, 3)) % 12 + 1
            return self._render_date(original, f, None, new_month, None)
        y_raw = f["year"].group(1)
        year = int(y_raw)
        if len(y_raw) == 2:
            year += 1900 if year > today.year % 100 else 2000
        if "monthname" in f:
            word = f["monthname"].group()[:3].lower()
            month = next(i for i, m in enumerate(_MONTHS, 1) if m[:3].lower() == word)
        elif "month" in f:
            month = val("month")
        if kind == "y":
            step = rng.randint(1, 2)
            new_year = year + step if year + step <= today.year else year - step
            return self._render_date(original, f, new_year, None, None)
        if kind == "ym":
            if not 1 <= month <= 12:
                return self._dob_legacy(original)
            step = rng.randint(1, 24)
            idx = year * 12 + month - 1
            new = idx + step if idx + step <= today.year * 12 + today.month - 1 else idx - step
            return self._render_date(original, f, new // 12, new % 12 + 1, None)
        try:
            date = datetime.date(year, month, val("day"))
        except ValueError:
            return self._dob_legacy(original)
        for _ in range(100):
            delta = datetime.timedelta(days=rng.randint(30, 730))
            new = date + delta if date + delta <= today else date - delta
            if not ambiguous or new.day <= 12:
                break
        return self._render_date(original, f, new.year, new.month, new.day, new)

    @staticmethod
    def _render_date(original: str, f, year, month, day, date=None) -> str:
        edits = []

        def number(name, value):
            m = f[name]
            raw = m.group(1)
            if name == "year":
                text = f"{value % 100:02d}" if len(raw) == 2 else str(value)
            else:
                text = f"{value:02d}" if len(raw) == 2 else str(value)
            if m.group(2):
                suf = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(
                    value % 10, "th")
                text += suf.upper() if m.group(2).isupper() else suf
            edits.append((m.start(), m.end(), text))

        if year is not None:
            number("year", year)
        if month is not None and "month" in f:
            number("month", month)
        if day is not None and "day" in f:
            number("day", day)
        if month is not None and "monthname" in f:
            m = f["monthname"]
            word = m.group()
            full = _MONTHS[month - 1]
            abbr = len(word) < len(next(x for x in _MONTHS if x[:3].lower() == word[:3].lower()))
            edits.append((m.start(), m.end(), _case_like(word, full[:3] if abbr else full)))
        if date is not None and "weekday" in f:
            m = f["weekday"]
            name = _WEEKDAYS[date.weekday()]
            edits.append((m.start(), m.end(),
                          _case_like(m.group(), name if len(m.group()) > 4 else name[:3])))
        out = original
        for s, e, t in sorted(edits, reverse=True):
            out = out[:s] + t + out[e:]
        return out

    def _dob_legacy(self, original: str) -> str:
        """A date the reader cannot parse: every number replaced in place
        (a year by at most two years), month names swapped."""
        rng = self._rng
        if not re.search(r"\d", original):
            if _MONTH_RE.search(original):
                return _MONTH_RE.sub(lambda m: _case_like(m.group(), rng.choice(
                    [x for x in _MONTHS if x[:3].lower() != m.group()[:3].lower()])), original)
            return self._fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y")
        nums = list(re.finditer(r"\d+", original))
        year_m = next((n for n in nums if len(n.group()) == 4), None)
        if year_m is None and nums and (original[nums[-1].start() - 1:nums[-1].start()] == "'"
                                        or len(nums) >= 3):
            year_m = nums[-1]
        out, pos = [], 0
        for n in nums:
            out.append(original[pos:n.start()])
            g = n.group()
            if n is year_m:
                y = int(g) + rng.choice((-1, 1)) * rng.randint(1, 2)
                out.append(f"{y % 100:02d}" if len(g) == 2 else str(y))
            else:
                v = int(g)
                new = rng.randint(1, 12) if v <= 12 else rng.randint(13, 28)
                out.append(f"{new:0{len(g)}d}" if g.startswith("0") or len(g) == 2 and v < 10
                           else str(new))
            pos = n.end()
        out.append(original[pos:])
        return re.sub(r"(?i)(\d+)(st|nd|rd|th)\b", _ordinal, "".join(out))

    def _gen_gender(self, original: str = "") -> str:
        """Different gender term in the same grammatical form as *original*.

        Readable sentences survive ('I am a female nurse' → 'I am a male
        nurse'); the original term is never returned. A term that agrees
        with a named person of the message is not masked at all (detection,
        ``gender_follows_name``)."""
        return _swap_gender(original, self._rng)

    # ── Dispatch ───────────────────────────────────────────────────────────────

    _IDENTITY_TYPES = frozenset({"person", "PERSON", "email"})
    _GENERATORS: Dict[str, str] = {
        "ssn":               "_gen_ssn_like",
        "phone_us":          "_gen_phone_like",
        "phone_uk":          "_gen_phone_like",
        "phone_intl":        "_gen_phone_like",
        "credit_card":       "_gen_card_like",
        "ip_address":        "_gen_ip_like",
        "zip_us":            "_gen_zip_like",
        "postcode_uk":       "_shape_like",
        "api_key":           "_gen_api_key_like",
        "crypto":            "_gen_crypto_like",
        "us_bank_number":    "_gen_bank_like",
        "us_driver_license": "_shape_like",
        "iban":              "_gen_iban_like",
        "implicit_location": "_gen_place",
        "GPE":               "_gen_place",
        "LOC":               "_gen_place",
        "ORG":               "_gen_institution",
        "FAC":               "_gen_institution",
        "url":               "_gen_url_like",
        "handle":            "_gen_handle_like",
        "age":               "_gen_age_like",
        "hostname":          "_gen_hostname_like",
        "dob":               "_gen_dob_like",
    }

    def generate(
        self,
        entity: DetectedEntity,
        address_mode: str = "shift",
        address_shift_range: int = 1,
        forbidden: FrozenSet[str] = frozenset(),
    ) -> str:
        if entity.type == "gender_indicator":
            surrogate = self._gen_gender(entity.text)
        elif entity.type == "address":
            surrogate = self._gen_address(
                entity,
                mode=address_mode,
                shift_range=address_shift_range,
                forbidden=forbidden,
            )
        elif entity.type in ("person", "PERSON"):
            # one surrogate person per real person: not "unique" per value
            surrogate = self.people.name(entity.text.strip(), self._context, fresh=self._fresh)
        elif entity.type == "email":
            surrogate = self.people.email(entity.text.strip(), fresh=self._fresh)
        else:
            name = self._GENERATORS.get(entity.type)
            if name is None:
                name = "_shape_like" if re.search(r"[^\W_]", entity.text) else None
            if name is None:
                surrogate = self._unique(lambda: self._fake.bothify("??##??##"))
            else:
                fn = getattr(self, name)
                surrogate = self._unique(lambda: fn(entity.text.strip()))
        logger.debug(f"[MimicGen] {entity.type}: generated a surrogate")
        return surrogate

    def generate_all(
        self,
        entities: List[DetectedEntity],
        address_mode: str = "shift",
        address_shift_range: int = 1,
        forbidden: Optional[Set[str]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate surrogates for all entities.

        Args:
            entities:            Confirmed PII entities.
            address_mode:        "shift", "replace" or "coarse" (an "auto" policy must be
                                 resolved by the caller before this point).
            address_shift_range: Max house-number delta for shift mode.
            forbidden:           Extra strings surrogates must never equal
                                 (e.g. shadow-map originals from prior turns).
            text:                The message the entities come from. A surrogate
                                 that already appears in it (whole value, any
                                 case) is avoided when another can be drawn:
                                 restoring it would also rewrite that text
                                 (audit I5). Equality with an original, or
                                 (except for addresses) containing one as a
                                 whole value, is never accepted.

        People come first, longest names first, then e-mails, so "Sarah",
        "Ms. Mitchell" and ``sarah.m@…`` link to the surrogate of "Sarah
        Mitchell" in the same message.
        """
        blocked: Set[str] = set(forbidden or ())
        # A shifted address must never equal ANOTHER real address in the same
        # message ("789 X" and "790 X" both present → shift must dodge).
        blocked |= {e.text.strip() for e in entities if e.type == "address"}
        # No surrogate may equal any original in the message (J4): a value
        # "replaced" by itself is sent verbatim.
        originals = {e.text.strip().lower() for e in entities}
        # Nor any earlier original (low-entropy values are not unique).
        equal_blocked = originals | {f.strip().lower() for f in blocked if not is_low_entropy(f)}
        # Nor contain one as a whole value: "Daniel Kowalczyk" → "Emily
        # Daniel" would send the separately detected "Daniel" verbatim.
        # Addresses keep their street and city by design (shift policy).
        contained = sorted({o for o in originals | {f.strip().lower() for f in blocked}
                            if len(o) >= 3 and not is_low_entropy(o)}, key=len)

        def contains_original(surrogate: str) -> bool:
            # words: as a whole word ("Ann" in "Joanna" is fine); structured
            # values (URL, e-mail, number): anywhere ("…/team/sarah" in
            # "…/team/sarahjohnson" is not)
            folded = surrogate.lower()
            return any(o in folded and o != folded
                       and (not o.replace(" ", "").isalpha() or occurs(surrogate, o))
                       for o in contained)

        self._context = text or ""
        self._avoid = set(equal_blocked)
        self.people.observe([e.text for e in entities]
                            + [f for f in blocked if not is_low_entropy(f)], self._context)

        def order(item):
            i, ent = item
            if ent.type in ("person", "PERSON"):
                return (0, -len(ent.text.split()), i)
            return (1 if ent.type == "email" else 2, 0, i)

        mapping: Dict[str, str] = {}
        chosen: Set[str] = set()
        try:
            for _, ent in sorted(enumerate(entities), key=order):
                key = ent.text.strip()
                if key in mapping:
                    continue
                identity = ent.type in self._IDENTITY_TYPES
                fallback = None
                for attempt in range(_MAX_ORIGINAL_RETRIES):
                    if identity:
                        self.people.begin()
                        self._fresh = attempt > 0
                    surrogate = self.generate(
                        ent,
                        address_mode=address_mode,
                        address_shift_range=address_shift_range,
                        forbidden=frozenset(blocked),
                    )
                    reject = surrogate.strip().lower() in equal_blocked or (
                        ent.type != "address" and (
                            contains_original(surrogate)
                            or len(key) >= 3 and key.lower() in surrogate.lower()))
                    # a surrogate issued earlier stands for another original
                    # (originals already mapped are reused by the caller)
                    reject = reject or identity and self._taken(surrogate)
                    if not reject and (surrogate in chosen
                                       or (text is not None and occurs(text, surrogate))):
                        fallback = fallback or surrogate
                        reject = True
                    if reject:
                        if identity:
                            self.people.rollback()
                        continue
                    break
                else:
                    if fallback is None:
                        raise RuntimeError(
                            f"could not generate a surrogate for a {ent.type} entity "
                            f"that differs from every original in the message"
                        )
                    logger.warning(f"[MimicGen] {ent.type}: every candidate surrogate is already "
                                   f"used in this message; using one anyway")
                    surrogate = fallback
                mapping[key] = surrogate
                chosen.add(surrogate)
                self.used_surrogates.add(surrogate)
                blocked.add(surrogate)
                self._avoid.add(surrogate.strip().casefold())
        finally:
            self._fresh = False
            self._context = ""
        logger.info(f"[MimicGen] Generated {len(mapping)} surrogate mappings")
        return mapping
