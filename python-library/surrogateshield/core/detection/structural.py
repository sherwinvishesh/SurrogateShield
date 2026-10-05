"""
detection/structural.py — Pass S: structure- and frame-driven detection
(audit I8 / I13).

NER misses PII whose evidence is the *shape* of the message rather than the
word itself: a CSV column, a chat speaker, a name the message already gave
in full, a lowercase or non-English place after "lives in" / "vivo en" /
"住在", a name after "我叫" / "मेरा नाम", a payee after "PAYMENT TO", a
nickname after "DM", a street written the French or Portuguese way. Each
rule here needs its own structural evidence; none fires on a bare word.

``detect`` returns ``(added, removed)``. Every added entity carries
``source="structural"``. *removed* are existing entities that a typed span
supersedes (a CSV surname NER called GPE, a PERSON span that swallowed
"je m'appelle").
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

from ..entities import DetectedEntity
from . import pattern_scan
from .geo_data import MAJOR_CITIES, US_STATES
from .public_names import NOT_NAMES, PUBLIC_ORGS, PUBLIC_PEOPLE

logger = logging.getLogger(__name__)

SOURCE = "structural"
_GEO = {"GPE", "LOC", "FAC", "LOCATION"}

# (sentence, start, end) -> True when the span names a place.
PlaceVerifier = Callable[[str, int, int], bool]


def _ent(text: str, s: int, e: int, typ: str, score: float = 0.9) -> DetectedEntity:
    return DetectedEntity(text[s:e], s, e, typ, score, SOURCE)


def _overlaps(s: int, e: int, ents: Iterable[DetectedEntity]) -> bool:
    return any(x.start < e and s < x.end for x in ents)


# ── 1. intro prefixes swallowed into a PERSON span ───────────────────────────

_INTRO = re.compile(
    r"^(?:je\s+m['’]appelle|me\s+llamo|mi\s+nombre\s+es|mi\s+chiamo|meu\s+nome\s+[ée]"
    r"|ich\s+hei(?:ß|ss)e|my\s+name\s+is|i\s*['’]?m|i\s+am|this\s+is|hi|hello|hey|dear)\s+",
    re.IGNORECASE,
)


def _trim_intros(text: str, ents: Sequence[DetectedEntity]):
    added, removed = [], []
    for e in ents:
        if e.type != "PERSON":
            continue
        m = _INTRO.match(e.text)
        if m and m.end() < len(e.text):
            removed.append(e)
            added.append(DetectedEntity(e.text[m.end():], e.start + m.end(), e.end,
                                        e.type, e.score, e.source))
    return added, removed


# ── 2. surname particles ("Jean-Luc van der Berg") ───────────────────────────

_PARTICLES = (r"van|von|der|den|de|del|della|di|da|du|la|le|bin|ibn|al|el|ten|ter"
              r"|op|dos|das|do|y|zu")
_PARTICLE_TAIL = re.compile(
    r"\s+(?:(?:" + _PARTICLES + r")\s+){1,3}[A-ZÀ-Ý][\w'’\-]+(?:\s+[A-ZÀ-Ý][\w'’\-]+)?"
)


def _extend_particles(text: str, ents: Sequence[DetectedEntity]):
    added, removed = [], []
    for e in ents:
        if e.type != "PERSON":
            continue
        m = _PARTICLE_TAIL.match(text, e.end)
        if not m:
            continue
        end = m.end()
        inside = [x for x in ents if x is not e and e.start <= x.start and x.end <= end]
        if any(x.end > end for x in ents if x is not e and e.end <= x.start < end):
            continue
        removed.extend([e] + inside)
        added.append(DetectedEntity(text[e.start:end], e.start, end, "PERSON",
                                    e.score, e.source))
    return added, removed


# ── 3. name components ("Hannah Reyes" … "Hannah, approved") ─────────────────

# Given names that are also ordinary words: never propagated on their own.
_AMBIGUOUS_COMPONENTS = frozenset("""
will may june april august bill mark grant rose hope joy faith chase drew dawn
summer frank young king long little white black brown green gray grey price
hall wood stone hill lane rich penny sky star gay page ward bell rich
""".split())


def _components(text: str, ents: Sequence[DetectedEntity]):
    persons = [e for e in ents if e.type == "PERSON"]
    added: List[DetectedEntity] = []
    seen: Set[str] = set()
    for p in persons:
        toks = re.findall(r"[^\W\d_][\w'’\-]*", p.text)
        if len(toks) < 2:
            continue
        lower = p.text.islower()
        for tok in toks:
            key = tok.lower()
            if (len(tok) < 3 or key in seen or key in _AMBIGUOUS_COMPONENTS
                    or key in NOT_NAMES or re.fullmatch(_PARTICLES, key)):
                continue
            seen.add(key)
            flags = re.IGNORECASE if lower else 0
            for m in re.finditer(r"(?<![\w@.\-])" + re.escape(tok) + r"(?![\w@\-]|\.\w)", text, flags):
                if not lower and not m.group(0)[:1].isupper():
                    continue
                if _overlaps(m.start(), m.end(), list(ents) + added):
                    continue
                added.append(_ent(text, m.start(), m.end(), "PERSON", 0.85))
    return added, []


# ── 4. CSV / TSV columns ─────────────────────────────────────────────────────

_COLUMN_TYPES = {
    "PERSON": {"first", "first_name", "firstname", "fname", "given", "given_name", "last",
               "last_name", "lastname", "lname", "surname", "family_name", "name",
               "full_name", "fullname", "customer", "customer_name", "contact",
               "contact_name", "payee", "employee", "employee_name", "patient",
               "patient_name", "client", "client_name", "student", "student_name",
               "owner", "member", "member_name", "recipient", "sender", "tenant"},
    "GPE": {"city", "town", "hometown"},
    "zip_us": {"zip", "zipcode", "zip_code", "postal", "postal_code", "postcode"},
    "address": {"address", "street", "address1", "address_1", "address_line_1",
                "street_address", "addr"},
    "dob": {"dob", "birthdate", "birth_date", "date_of_birth", "birthday"},
}
_COL_OF = {c: t for t, cols in _COLUMN_TYPES.items() for c in cols}
_DESC_COLUMNS = {"desc", "description", "memo", "details", "narrative", "payee_desc"}


def _norm_col(c: str) -> str:
    return re.sub(r"[\s\-]+", "_", c.strip().strip('"').lower())


def _csv(text: str, ents: Sequence[DetectedEntity]):
    added, removed = [], []
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    i = 0
    while i < len(lines):
        off, line = lines[i]
        sep = next((c for c in (",", "\t", ";", "|") if line.count(c) >= 2), None)
        cols = [_norm_col(c) for c in line.split(sep)] if sep else []
        if not cols or not all(re.fullmatch(r"[a-z_#]{1,30}", c) for c in cols) or not (
                set(cols) & (set(_COL_OF) | _DESC_COLUMNS)):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j][1].count(sep) == len(cols) - 1 and lines[j][1].strip():
            roff, row = lines[j]
            k = roff
            for col, cell in zip(cols, row.split(sep)):
                raw = cell.strip().strip('"')
                cs = k + cell.index(raw) if raw else k
                ce = cs + len(raw)
                k += len(cell) + 1
                typ = _COL_OF.get(col)
                if not raw:
                    continue
                if col in _DESC_COLUMNS:
                    a, _ = _payments(text, cs, ce)
                    added.extend(a)
                    continue
                if typ is None:
                    continue
                if typ == "zip_us" and not re.fullmatch(r"\d{5}(?:-\d{4})?", raw):
                    continue
                if typ == "PERSON" and not re.fullmatch(r"[^\W\d_][\w'’.\- ]*", raw):
                    continue
                over = [x for x in ents if x.start < ce and cs < x.end]
                if any(x.source == "pattern" for x in over):
                    continue
                removed.extend(over)
                added.append(_ent(text, cs, ce, typ, 0.95))
            j += 1
        i = j
    return added, removed


# ── 5. chat-transcript speakers ──────────────────────────────────────────────

_SPEAKER = re.compile(
    r"^[ \t]*(?:\[[^\]\n]{4,40}\]|\d{1,2}[:.]\d{2}(?:\s*[AaPp][Mm])?\s*[-–]?)?[ \t]*"
    r"([^\W\d_][\w.\-]{1,29}):[ \t]",
    re.MULTILINE,
)
_NOT_SPEAKERS = frozenset("""
me you user assistant system bot ai q a note subject from to date re cc bcc sent tip
warning error info debug step example answer question input output customer agent
support admin moderator mod host guest server client http https ps edit update
""".split())


def _speakers(text: str, ents: Sequence[DetectedEntity]):
    hits = list(_SPEAKER.finditer(text))
    if len(hits) < 2:
        return [], []
    # A transcript, not a form: a speaker is timestamped or speaks twice
    # ("NAME: …\nDOB: …" labels each occur once).
    stamped = {m.group(1) for m in hits if m.group(0).lstrip()[:1] in "[0123456789"}
    counts: dict = {}
    for m in hits:
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    names = {n for n in counts if (n in stamped or counts[n] >= 2)
             and n.lower() not in _NOT_SPEAKERS and n.lower() not in NOT_NAMES}
    added: List[DetectedEntity] = []
    for name in names:
        for m in re.finditer(r"(?<![\w@])" + re.escape(name) + r"(?!\w)", text):
            if not _overlaps(m.start(), m.end(), list(ents) + added):
                added.append(_ent(text, m.start(), m.end(), "PERSON", 0.9))
    return added, []


# ── 6. verb + nickname ("DM Kev", "thank Priya") ─────────────────────────────

_VERBS = (r"text|texted|call|called|email|emailed|ask|asked|tell|told|ping|pinged"
          r"|message|messaged|thank|thanked|thanking|remind|reminded|cc|invite|invited|meet|met"
          r"|loop\s+in|introduce")
_VERB_NAME = re.compile(
    r"(?:\b(?:DM|dm|" + _VERBS + r")"
    # sentence-initial imperative: "Ping Kev about it." — not a title-case
    # heading ("Email Address", "Call Center"), see _NOT_NICKNAMES
    r"|(?:^|(?<=[.!?\n])[ \t]*)(?:" + "|".join(
        v[:1].upper() + v[1:] for v in _VERBS.split("|")) + r"))"
    r"\s+([A-Z][a-z]{1,15})\b(?![ \t]+[A-Z])(?![ \t]*:)",
    re.MULTILINE,
)
_NOT_NICKNAMES = frozenset("""
mom dad mum mommy daddy grandma grandpa nana papa sis bro support siri alexa
google hr it me you them him her everyone everybody team the this that monday tuesday
wednesday thursday friday saturday sunday january february march april may june july
august september october november december god jesus santa uber lyft
address center centre centers list number log logs history list form forms details
info support service services us now today tomorrow tonight later back again
""".split())


def _verb_frames(text: str, ents: Sequence[DetectedEntity]):
    added = []
    for m in _VERB_NAME.finditer(text):
        name = m.group(1)
        key = name.lower()
        if (key in _NOT_NICKNAMES or key in NOT_NAMES or key in PUBLIC_ORGS
                or key in PUBLIC_PEOPLE):
            continue
        if not _overlaps(m.start(1), m.end(1), list(ents) + added):
            added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.85))
    return added, []


# ── 7. payee after a payment verb ("VENMO PAYMENT TO MARISOL ORTEGA") ────────

_PAYMENT = re.compile(
    r"(?i:\b(?:payment|pmt|transfer|xfer|zelle|venmo|paypal|cash\s*app|sent|paid|deposit|refund)"
    r"\s+(?:to|from)\s+)"
    r"((?:[A-Z][a-z'’\-]+|[A-Z][A-Z'’\-]+)(?:[ \t]+(?:[A-Z][a-z'’\-]+|[A-Z][A-Z'’\-]+)){1,2})\b"
)
_LEGAL = re.compile(r"(?i)\b(?:inc|llc|ltd|corp|co|company|bank|group|plc|gmbh)\.?$")


def _payments(text: str, start: int = 0, end: Optional[int] = None):
    added = []
    for m in _PAYMENT.finditer(text, start, len(text) if end is None else end):
        name = m.group(1)
        if name.lower() in PUBLIC_ORGS or _LEGAL.search(name) or any(
                w.lower() in PUBLIC_ORGS for w in name.split()):
            continue
        added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.9))
    return added, []


# ── 8. places after a residence / work cue, and non-English frames ───────────

_EN_RESIDENCE = re.compile(
    r"(?i:\b(?:live[sd]?|living|reside[sd]?|residing|grew\s+up|raised|born|staying|stays?"
    r"|based|located|moved|moving|relocat(?:ed|ing))(?:\s+(?:alone|now|here|back|currently"
    r"|together|out))?\s+(?:in|to|near|outside(?:\s+of)?)\s+"
    r"|\b(?:work|works|working|job|shifts?)\b[^.!?\n]{0,60}?\bin\s+"
    # "from" only after a person or a self/third-person description:
    # "sarah mitchell from austin", "I'm from boise" — not "stop it from …"
    r"|\b(?:i\s*['’]?m|im|i\s+am|we\s*['’]?re|we\s+are|originally|(?:he|she|they)\s*['’]?(?:s|re)"
    r"|is|are|was|were|comes?|came)\s+from\s+)"
)
_PERSON_FROM = re.compile(r"\s+from\s+")
_LOWER_PLACE = re.compile(r"([a-z][a-z'’\-]{2,}(?:\s+[a-z][a-z'’\-]{2,})?)\b")
_PLACE_STOP = frozenset("""
the a an my his her our their your this that it there here home town work school
college fear love debt poverty peace hell bed person general advance order total
time front back public private company office house apartment city country state
postcode postal zip zipcode code area district neighborhood neighbourhood region suburb
suburbs unit building block flat room floor street road avenue lane village rural urban
downtown uptown midtown center centre north south east west central
""".split())

# Non-English residence frames: a capitalised place after the verb.
_INTL_RESIDENCE = re.compile(
    r"(?i:\b(?:vivo|vive|vivimos|resido|reside)\s+en\s+|\bsoy\s+de\s+"
    r"|\bj['’]habite\s+(?:à|a|en)\s+|\bhabite\s+à\s+|\bje\s+suis\s+de\s+"
    r"|\b(?:moro|mora|moramos)\s+(?:em|no|na)\s+|\bsou\s+d[eoa]\s+"
    r"|\b(?:abito|vivo)\s+a\s+|\bsono\s+di\s+"
    r"|\bwohne\s+in\s+|\bkomme\s+aus\s+)"
    r"([A-ZÀ-Ý][\w'’\-]+(?:[ \t]+[A-ZÀ-Ý][\w'’\-]+){0,2})"
)
_HAN = r"[\u4e00-\u9fff]"
_ZH_PLACE = re.compile(r"(?:住在|来自|住址是|地址是)(" + _HAN + r"{2,15})")
_ZH_NAME = re.compile(r"(?:我叫|我的名字是|名字叫|我是)(" + _HAN + r"{2,4})(?=[，,。.！!；;\s]|$)")
_HI_NAME = re.compile(r"मेरा\s+नाम\s+(\S+(?:\s+\S+){0,2}?)\s+है")
_HI_PLACE = re.compile(r"(?:मैं|हम)\s+(\S+)\s+में\s+(?:रहता|रहती|रहते)")


def _sentence_bounds(text: str, s: int, e: int) -> Tuple[int, int]:
    a = max(text.rfind(c, 0, s) for c in ".!?\n") + 1
    ends = [i for i in (text.find(c, e) for c in ".!?\n") if i != -1]
    return a, (min(ends) + 1 if ends else len(text))


def _places(text: str, ents: Sequence[DetectedEntity],
            verify: Optional[PlaceVerifier]):
    added: List[DetectedEntity] = []

    def add(s: int, e: int, typ: str) -> None:
        if not _overlaps(s, e, list(ents) + added):
            added.append(_ent(text, s, e, typ, 0.9))

    cue_ends = [c.end() for c in _EN_RESIDENCE.finditer(text)]
    cue_ends += [m.end() for p in ents if p.type == "PERSON"
                 for m in [_PERSON_FROM.match(text, p.end)] if m]
    for cue_end in sorted(set(cue_ends)):
        m = _LOWER_PLACE.match(text, cue_end)
        if not m:
            continue
        words = m.group(1).split()
        # longest first: "el paso" before "el"
        for n in (len(words), 1) if len(words) > 1 else (1,):
            cand = " ".join(words[:n])
            if any(w in _PLACE_STOP or w.endswith("ing") for w in cand.split()):
                continue
            s = m.start(1)
            e = s + len(cand)
            ok = cand in MAJOR_CITIES or cand in US_STATES
            if not ok and verify is not None:
                a, b = _sentence_bounds(text, s, e)
                sent = text[a:b]
                ok = verify(sent, s - a, e - a)
            if ok:
                add(s, e, "GPE")
                break
    for m in _INTL_RESIDENCE.finditer(text):
        add(m.start(1), m.end(1), "GPE")
    for m in _ZH_PLACE.finditer(text):
        add(m.start(1), m.end(1), "GPE")
    for m in _HI_PLACE.finditer(text):
        add(m.start(1), m.end(1), "GPE")
    return added, []


def _names_intl(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []
    for rx in (_ZH_NAME, _HI_NAME):
        for m in rx.finditer(text):
            if not _overlaps(m.start(1), m.end(1), list(ents) + added):
                added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.9))
    return added, []


# ── 9. European street addresses ─────────────────────────────────────────────

_W = r"[A-ZÀ-Ý][\w'’\-]+"
_JOIN = r"(?:(?:de|des|du|de\s+la|d['’]|da|do|dos|das|del|della|di|von|der)\s*)?"
_INTL_STREET = re.compile(
    # fr: "14 rue des Lilas", "3 bis avenue Victor Hugo"
    r"\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|chemin|all[ée]e|impasse"
    r"|place|quai|route|cours)\s+" + _JOIN + _W + r"(?:\s+" + _W + r"){0,3}"
    # pt/es/it: "Rua Augusta 1508", "Calle Mayor, 12", "Via Roma 5"
    r"|\b(?:Rua|Avenida|Av\.|Travessa|Alameda|Calle|Paseo|Plaza|Carrer|Via|Viale|Piazza|Corso)"
    r"\s+" + _JOIN + _W + r"(?:\s+" + _JOIN + _W + r"){0,3},?\s+(?:n[º°o]\.?\s*)?\d{1,5}[A-Za-z]?\b"
    # de: "Hauptstraße 5", "Lindenweg 12a"
    r"|\b[A-ZÄÖÜ][\wäöüß]+(?:stra(?:ß|ss)e|str\.|weg|platz|allee|gasse|ring|damm)\s+\d{1,4}[a-z]?\b"
)
_W_ANY = r"[^\W\d_][\w'’\-]*"
_CITY_W = r"[A-ZÀ-ÝČŠŽŘĽĎŤŇ][^\W\d_][\w'’\-]*(?:[ \-][A-ZÀ-ÝČŠŽŘĽĎŤŇ][^\W\d_][\w'’\-]*)?"
# a postcode written before or after its town, or in brackets after one:
# "79098 Freiburg", "101 Reykjavík", "040 01 Košice", "Hamilton 3204",
# "Zwolle (8011 PK)". Only after a street on the same line, or bracketed.
_NOT_NL_SUFFIX = r"(?!AD|BC|CE|AM|PM|OK|US|UK|EU|TV|PC|GB|MB|KB|HP|KM|MM|CM)"
_POSTCODE_TOWN = re.compile(
    r"[ \t]*,?[ \t]*(?:"
    r"(?P<pc>\d{3}[ ]\d{2}|\d{4}[ ]?" + _NOT_NL_SUFFIX + r"[A-Z]{2}|\d{3,5}|[A-Z]{1,2}-?\d{4,5})[ \t]+(?P<t1>" + _CITY_W + r")"
    r"|(?P<t2>" + _CITY_W + r")[ \t]+(?P<pc2>\d{4,5})(?!\d)"
    r")(?![\w\-])"
)
_BRACKET_POSTCODE = re.compile(r"\((\d{4}[ ]?" + _NOT_NL_SUFFIX + r"[A-Z]{2})\)")
# number-after streets with no suffix word, confirmed by the postcode-town
# that follows: "Laugavegur 52, 101 Reykjavík", "17 Hlavná, 040 01 Košice"
_BARE_STREET = re.compile(
    r"\b(?:" + _CITY_W + r")[ \t]+\d{1,4}[a-z]?(?=[ \t]*,[ \t]*\d{3})"
    r"|\b\d{1,4}[a-z]?[ \t]+(?:" + _CITY_W + r")(?=[ \t]*,[ \t]*\d{3})"
)
_ADDRESS_CUE = re.compile(
    r"(?i)\b(?:address|adresse|addr|direcci[oó]n|endere[cç]o|indirizzo|shipping|ship\s+to"
    r"|deliver\w*|send\s+(?:it\s+)?to|mail\s+to|lives?|living|reside\w*|wohne\w*|abito"
    r"|vivo|moro|bor|b[ýy]v\w*|heima|postal)\b")
# "in via Emilia Est 211": the Italian street word after "in", lower-case
_IT_LOWER_STREET = re.compile(
    r"(?<=\bin )(?:via|viale|piazza|corso|vicolo|largo)\s+" + _W + r"(?:\s+" + _W + r"){0,3},?\s+\d{1,4}[A-Za-z]?\b"
)
# a lower-case British street whose suffix is also a noun ("14 birchwood
# close"), confirmed by a postcode later on the line
_LOWER_UK_STREET = re.compile(
    r"\b\d{1,4}[a-z]?\s+[a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+)?\s+(?:close|grove|way|green|hill|rise"
    r"|walk|gardens|view|park|chase|mews|crescent|terrace|lane|road|street|drive|avenue|place|court|row)\b"
    r"(?=[^\n]{0,40}\b[a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2}\b)", re.IGNORECASE
)
# a flat or unit number in front of a street address: "flat 4, 70 Cowley Road"
_UNIT_BEFORE = re.compile(
    r"(?i)\b(?:flat|apt\.?|apartment|unit|suite|ste\.?)\s*#?\s*\d{1,4}[a-z]?\s*,?\s*$")


def _address_parts(text: str, ents: Sequence[DetectedEntity]):
    """Postcode-towns, bracketed postcodes and unit numbers around a street
    address, and number-after streets that a postcode-town confirms."""
    added: List[DetectedEntity] = []

    def _add(s, e):
        if s < e and not any(x.start < e and s < x.end and (x.type == "address" or x.source == "pattern")
                             for x in list(ents) + added):
            added.append(_ent(text, s, e, "address", 0.95))

    for rx in (_BARE_STREET, _IT_LOWER_STREET, _LOWER_UK_STREET):
        for m in rx.finditer(text):
            if rx is _BARE_STREET:
                words = re.findall(r"[^\W\d_]+", m.group())
                if (any(w.lower() in pattern_scan._NOT_NAME_WORD for w in words)
                        or not _ADDRESS_CUE.search(text, max(0, m.start() - 120), m.start())):
                    continue                # "Page 12, 2024 Report"
            _add(m.start(), m.end())
    streets = [x for x in list(ents) + added if x.type == "address"]
    for st in streets:
        m = _POSTCODE_TOWN.match(text, st.end)
        town = m and (m.group("t1") or m.group("t2"))
        if m and not any(w.lower() in pattern_scan._NOT_NAME_WORD
                         for w in re.findall(r"[^\W\d_]+", town)):
            _add(m.start("pc") if m.group("pc") else m.start("t2"), m.end())
        ls = text.rfind("\n", 0, st.start) + 1
        u = _UNIT_BEFORE.search(text, ls, st.start)
        if u:
            _add(u.start(), u.end() - (len(u.group()) - len(u.group().rstrip(" ,"))))
    for m in _BRACKET_POSTCODE.finditer(text):
        _add(m.start(1), m.end(1))
    removed = [x for x in ents if x.source != "pattern"
               and any(a.start <= x.start and x.end <= a.end for a in added)]
    return added, removed


_INTL_UNIT = re.compile(
    r"(?i:\b(?:apto|apartamento|apt|piso|depto|dpto|appartement|appt|[ée]tage|wohnung|interno"
    r"|int\.|sala|bloco|bloque|escalera|esc\.)\.?\s*)(?:n[º°o]\.?\s*)?\d{1,4}[A-Za-z]?\b"
)


def _streets(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []
    for rx in (_INTL_STREET, _INTL_UNIT):
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            over = [x for x in list(ents) + added if x.start < e and s < x.end]
            if any(x.type == "address" or x.source == "pattern" for x in over):
                continue
            added.append(_ent(text, s, e, "address", 0.95))
    removed = [x for x in ents if x.source != "pattern"
               and any(a.start <= x.start and x.end <= a.end for a in added)]
    return added, removed


# ─── entry point ─────────────────────────────────────────────────────────────

def detect(
    text: str,
    entities: Sequence[DetectedEntity],
    *,
    place_verifier: Optional[PlaceVerifier] = None,
    skip_locations: bool = False,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """Run every structural rule; returns ``(added, removed)``.

    Rules run in order on the running entity set, so a trimmed or merged
    PERSON is what the component rule propagates. *place_verifier* confirms a
    lowercase place candidate that is not in the gazetteer (the pipeline
    passes a spaCy check on the truecased sentence); without it only the
    gazetteer is used. *skip_locations* (service-query mode) disables the
    place rules.
    """
    current = list(entities)
    added_all: List[DetectedEntity] = []
    removed_all: List[DetectedEntity] = []
    rules = [
        lambda t, es: _trim_intros(t, es),
        lambda t, es: _extend_particles(t, es),
        lambda t, es: _streets(t, es),
        lambda t, es: _address_parts(t, es),
        lambda t, es: _csv(t, es),
        lambda t, es: _speakers(t, es),
        lambda t, es: _payments(t),
        lambda t, es: _names_intl(t, es),
        lambda t, es: _verb_frames(t, es),
        lambda t, es: _components(t, es),
    ]
    if not skip_locations:
        rules.insert(3, lambda t, es: _places(t, es, place_verifier))
    for rule in rules:
        added, removed = rule(text, current)
        removed_ids = {id(x) for x in removed}
        current = [x for x in current if id(x) not in removed_ids]
        added = [a for a in added if not _overlaps(a.start, a.end, current)]
        current.extend(added)
        added_all = [x for x in added_all if id(x) not in removed_ids] + added
        removed_all.extend(x for x in removed if x not in added_all)
        for a in added:
            logger.debug("[SentinelLayer] Pass S %s: %r", a.type, a.text)
    return added_all, [x for x in removed_all if any(x is e for e in entities)]
