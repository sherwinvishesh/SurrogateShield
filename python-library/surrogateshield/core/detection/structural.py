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
from functools import lru_cache
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
    seen: Set[Tuple[str, bool]] = set()
    for p in persons:
        toks = re.findall(r"[^\W\d_][\w'’\-]*", p.text)
        if len(toks) < 2:
            continue
        lower = p.text.islower()
        for tok in toks:
            key = tok.lower()
            # "JAVIER MONTOYA" must not hide the "Javier" of a later
            # "Javier Montoya": seen is keyed on the exact token
            if (len(tok) < 3 or (tok, lower) in seen or key in _AMBIGUOUS_COMPONENTS
                    or key in NOT_NAMES or re.fullmatch(_PARTICLES, key)):
                continue
            seen.add((tok, lower))
            # an all-caps or lower-case name also finds its title-case form
            flags = re.IGNORECASE if lower or tok.isupper() else 0
            for m in re.finditer(r"(?<![\w@.\-])" + re.escape(tok) + r"(?![\w@\-]|\.\w)", text, flags):
                if not lower and not m.group(0)[:1].isupper():
                    continue
                start = m.start()
                # "van der Merwe" from "Pieter van der Merwe": the particles
                # the name carries come along
                own = re.search(r"(?i)((?:\b(?:" + _PARTICLES + r")\s+)+)" + re.escape(tok) + r"\b", p.text)
                if own:
                    pm = re.search(r"(?i)(?:\b(?:" + _PARTICLES + r")\s+)+$", text[max(0, start - 30):start])
                    if pm:
                        start -= len(pm.group())
                if _overlaps(start, m.end(), list(ents) + added):
                    if start == m.start() or _overlaps(m.start(), m.end(), list(ents) + added):
                        continue
                    start = m.start()
                added.append(_ent(text, start, m.end(), "PERSON", 0.85))
    return added, []


# ── 3b. display names and greetings ──────────────────────────────────────────

_ANGLE_EMAIL = re.compile(r"\s*<([^@\s<>]+)@[^\s<>]+>")
_GREETING = re.compile(
    r"(?m)(?:^|(?<=[.!?]\s))[ \t]*(?:Hi|Hello|Hey|Dear|Hallo|Hej|Hoi|Bonjour|Hola|Ciao|Olá|Oi)"
    r"[ \t]+([A-ZÀ-ÖØ-Þ][^\W\d_][\w'’\-]*)(?=[ \t]*[,!:\n])")
_NOT_GREETED = frozenset("all team everyone everybody there sir madam folks guys "
                         "friends again world support hr admin".split())


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if c.isalnum())


def _display_names(text: str, ents: Sequence[DetectedEntity]):
    """A company/place-typed name that is an e-mail display name whose local
    part spells it ("Ifeoma Chukwu <ifeoma@chukwustudio.com>") is a person;
    so is a name greeted at the start of a line or sentence ("Hi Ifeoma,")."""
    added: List[DetectedEntity] = []
    removed: List[DetectedEntity] = []
    for e in ents:
        if e.type not in ("ORG", "GPE", "LOC", "FAC") or e.source == "pattern":
            continue
        m = _ANGLE_EMAIL.match(text, e.end)
        local = _fold(m.group(1)) if m else ""
        toks = [_fold(t) for t in re.findall(r"[^\W\d_]+", e.text)]
        if local and any(len(t) >= 3 and t in local for t in toks):
            removed.append(e)
            added.append(_ent(text, e.start, e.end, "PERSON", 0.9))
    for m in _GREETING.finditer(text):
        name = m.group(1)
        if name.lower() in _NOT_GREETED or name.lower() in NOT_NAMES:
            continue
        s, t = m.start(1), m.end(1)
        over = [x for x in ents if x.start < t and s < x.end]
        if any(x.type == "PERSON" or x.source == "pattern" for x in over):
            continue
        removed.extend(x for x in over if x not in removed)
        added.append(_ent(text, s, t, "PERSON", 0.85))
    return added, removed


# ── 3c. named institutions ───────────────────────────────────────────────────

_CAP_W = r"[A-ZÀ-ÝČŠŽ][^\W\d_][\w'’\-]*"
_INST_PART = r"(?:(?:de|del|da|do|dos|das|di|du|des|de\s+la|der|des|von)\s+)?"
# "Clinique Saint-Vincent", "Escola Básica de Alvalade", "Université Laval"
_INSTITUTION = re.compile(
    r"\b(?:Clinique|H[ôo]pital|Centre\s+[Hh]ospitalier|[ÉE]cole|Escola|Lyc[ée]e|Coll[èe]ge"
    r"|Universit[ée]|Universidad|Universidade|Universit[äa]t|Universit[àa]|Hospital|Cl[íi]nica"
    r"|Klinik|Krankenhaus|Ospedale|Scuola|Colegio|Col[ée]gio|Instituto|Liceo|Gymnasium|Grundschule"
    r"|Gesamtschule|Realschule|Studio\s+(?:[Mm]edico|[Dd]entistico|[Ll]egale|[Nn]otarile)"
    r"|Farmacia|Parrocchia|Poliambulatorio|Ambulatorio|Pharmacie|Parroquia|Paróquia|Praxis"
    r"|(?-i:(?<=\b[a-zà-ÿ]{2} )|(?<=\b[a-zà-ÿ]{3} )|(?<=\b[a-zà-ÿ]{4} )|(?<=\bl['’]))"
    r"(?:coll[èe]ge|[ée]cole|lyc[ée]e|scuola|escola|colegio|liceo|clinique|cl[íi]nica|universit[àée]))"
    r"(?:\s+(?:B[áa]sica|Secund[áa]ria|Primaria|Prim[áa]ria|Primaire|[ÉE]l[ée]mentaire|Elementare"
    r"|Superior|Municipal|Nacional|Estadual|Statale|Priv[ée]e?|Saint|Sainte|Santa|San|São"
    r"|b[áa]sica|secund[áa]ria|primaria|prim[áa]ria|primaire|[ée]l[ée]mentaire|elementare|media"
    r"|secondaria|statale|privada|privata|priv[ée]e?))?"
    r"\s+" + _INST_PART + _CAP_W + r"(?:[ \t]+" + _INST_PART + _CAP_W + r"){0,3}"
)
_INSTITUTION_LOWER = re.compile(
    r"\b(?:universit[ée]|universidad|universidade|universit[äa]t|clinique|escola|[ée]cole|lyc[ée]e)"
    r"\s+" + _INST_PART + r"([^\W\d_][\w'’\-]{2,})(?=\s*[?.!,;)]|\s*$)"
)
# "my doctor at riverside family clinic" from a writer who capitalises nothing
_EN_LOWER_INSTITUTION = re.compile(
    r"\b(?:at|from|to)\s+(?:the\s+)?((?:[a-z][a-z'’\-]+\s+){1,3}"
    r"(?:clinic|hospital|medical\s+cent(?:er|re)|health\s+cent(?:er|re)|surgery|elementary(?:\s+school)?"
    r"|middle\s+school|high\s+school|primary\s+school|academy|practice|pharmacy|parish|church"
    r"|pediatrics|paediatrics|dental))\b"
)
_EN_INSTITUTION = re.compile(
    r"\b((?:St\.?|Saint|Our\s+Lady\s+of)\s+[A-Z][a-z]+(?:['’]s)?"
    r"|[A-Z][a-z'’]+(?:\s+[A-Z][a-z'’]+){0,2}|[A-Z]{3,}(?:\s+[A-Z]{3,}){0,2})"
    r"\s+(?:Parish|Church|Cathedral|Chapel|Practice|Surgery|Pharmacy|Pediatrics|Paediatrics"
    r"|Dental(?:\s+(?:Clinic|Practice|Care|Group|Surgery))?|Medical\s+(?:Practice|Group|Cent(?:er|re))"
    r"|Family\s+(?:Practice|Medicine)|Health\s+Cent(?:er|re)|Primary\s+School|Elementary(?:\s+School)?"
    r"|High\s+School|Academy|PARISH|CHURCH|PRACTICE|PHARMACY|DENTAL|CLINIC)\b")
_BANK_WORDS = frozenset("PAYROLL DIRECT DEPOSIT DEP PAYMENT PMT TRANSFER XFER POS ACH DEBIT "
                        "CREDIT PURCHASE REFUND SALARY CHECKCARD CARD VISA".split())
_GENERIC_INST_WORD = frozenset("""
a an the my our your his her their this that same new old local nearest nearby closest
urgent walk-in walk free dental vet veterinary eye children's childrens kids women's
womens public private community mental sexual fertility pain sleep cancer va er any
another some every one big small main family general medical health care primary
specialist specialty city county state best good cheap private regional teaching
""".split())


_INST_STOP = frozenset("""
is are was were be been has have had will can and or but the a an of for to in on at
by with closed open opens near general central regional national university hospital
school college clinic medical
""".split())


def _generic_inst(word: str) -> bool:
    """"dermatology clinic", "pediatric hospital": a specialty, not a name."""
    w = word.lower()
    return w in _GENERIC_INST_WORD or bool(re.search(
        r"(?:ology|ologic(?:al)?|iatric|iatry|ics|ist|ists|ic|ical|ary|ive|ent|ant|al|ed|ing)$", w))


def _lowercase_writer(text: str) -> bool:
    return bool(re.search(r"(?<![\w'’])i(?:['’]?m)?\s+(?:am|was|have|had|do|did|don['’]?t|can|will"
                          r"|need|want|just|got|think|feel|graduated|live|work|went)\b|(?<![\w'’])im\b", text)) \
        or not re.search(r"\b[A-Z][a-z]", text)


def _institutions(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []
    removed: List[DetectedEntity] = []

    def add(s, e):
        over = [x for x in list(ents) + added if x.start < e and s < x.end]
        if any(x.source == "pattern" or x.type == "PERSON" and x.source != "slm"
               and not (s <= x.start and x.end <= e) for x in over):
            return
        removed.extend(x for x in over if x in ents and x not in removed)
        added.append(_ent(text, s, e, "ORG", 0.9))

    for m in _INSTITUTION.finditer(text):
        words = re.findall(r"[^\W\d_]+", m.group())
        _saints = ("saint", "sainte", "santa", "san", "são", "santo", "sant")
        if any((w.lower() in pattern_scan._NOT_NAME_WORD and w.lower() not in _saints)
               or w.lower() in _INST_STOP for w in words[1:]):
            continue                        # "The Hospital Is Closed", "Hospital General"
        add(m.start(), m.end())
    for m in _EN_INSTITUTION.finditer(text):
        words = m.group(1).split()
        s0 = m.start()
        while words and words[0] in _BANK_WORDS:    # "PAYROLL HEXAGON DENTAL"
            s0 = text.index(words[1], s0 + len(words[0])) if len(words) > 1 else m.end(1)
            words = words[1:]
        if not words:
            continue
        first = words[0].lower().rstrip(".")
        if first in ("st", "saint") or m.group(1).startswith("Our Lady") or not (
                first == "our" or _generic_inst(first) or first in pattern_scan._NOT_NAME_WORD
                or first in NOT_NAMES):
            add(s0, m.end())
    lower = None
    for rx in (_INSTITUTION_LOWER, _EN_LOWER_INSTITUTION):
        for m in rx.finditer(text):
            if lower is None:
                lower = _lowercase_writer(text)
            if not lower:
                break
            name = m.group(1).split()
            generic = _generic_inst(name[0]) if rx is _EN_LOWER_INSTITUTION else (
                name[0].lower() in _GENERIC_INST_WORD)
            saint = name[0].lower().rstrip(".") in ("st", "saint") and len(name) > 2
            if not saint and (generic or name[0].lower() in pattern_scan._NOT_NAME_WORD):
                continue
            if rx is _EN_LOWER_INSTITUTION and all(_generic_inst(w) for w in name[:-1]):
                continue
            add(m.start() if rx is _INSTITUTION_LOWER else m.start(1), m.end(1))
    return added, removed


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


# role mailboxes and generic handle words: never a person's name
_ROLE_WORDS = frozenset("""
info admin support sales contact help team hello noreply reply donotreply mail office billing
service services security jobs careers press news marketing accounts webmaster postmaster
root notifications alerts newsletter enquiries inquiries reception user test dev official
real the official gaming games tv live media studio shop store app bot svc srv sys api ci
sa daemon cron backup deploy prod staging
""".split())


def _fold(w: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if not unicodedata.combining(c))


def _stop_en() -> frozenset:
    from .relation_gate import _stop_words
    return _stop_words()["en"]


# words a file name adds to a person's name ("Kateryna_Bondar_Resume_2026")
_FILE_WORDS = frozenset("""
resume cv final draft copy scan scanned report letter invoice doc docs photo img image
screenshot backup old new edit edited signed version rev updated latest
""".split())


@lru_cache(maxsize=1)
def _common_words() -> frozenset:
    """English function words and Faker's common-word list."""
    from .relation_gate import _stop_words
    from faker.providers.lorem.en_US import Provider as Lorem
    return frozenset(Lorem.word_list) | _stop_words()["en"]


def _local_part_names(text: str, ents: Sequence[DetectedEntity]):
    """The words of a detected e-mail, @handle or profile-URL slug name the
    person elsewhere in the message: "hui.min.tan68@…" makes "TAN HUI MIN"
    a name, "@farshad.m" makes "not again farshad" one. A single word must
    be at least three letters and no function or mailbox word. A middle
    initial may stand inside the name ("Cauã I. Pacheco"), and a name a model
    found only part of ("Acuña" of "Zoé Acuña") is taken whole."""
    common = _common_words()
    parts: Set[str] = set()
    lone: Set[str] = set()                 # may stand alone as a name
    for e in ents:
        if e.type == "email":
            local = e.text.split("@")[0]
        elif e.type == "handle":
            local = e.text.lstrip("@").split("#")[0]
        elif e.type == "url":
            local = re.split(r"[/?#]", e.text.rstrip("/"))[-1]
        else:
            continue
        segs = [w for w in re.split(r"[._\-+\d]+", _fold(local)) if w and w not in _FILE_WORDS]
        if any(w in _ROLE_WORDS for w in segs):
            continue                        # "svc_payroll", "support.emea": no person
        for n, w in enumerate(segs):
            if len(w) >= 3 and w not in _ROLE_WORDS and w not in _stop_en():
                parts.add(w)
                if w not in common and (len(w) >= 4 or (n == 0 and len(segs) > 1)):
                    lone.add(w)
    if not parts:
        return [], []
    added: List[DetectedEntity] = []
    removed: List[DetectedEntity] = []
    toks = list(re.finditer(r"[^\W\d_]+(?:[\-'’][^\W\d_]+)*", text))

    def spelled(k: int) -> bool:
        return all(_fold(p) in parts for p in re.split(r"[\-'’]", toks[k].group()))

    def initial(k: int) -> bool:            # "Cauã I. Pacheco"
        t = toks[k]
        return (len(t.group()) == 1 and t.group().isupper() and text[t.end():t.end() + 1] == "."
                and k + 1 < len(toks) and spelled(k + 1))

    k = 0
    while k < len(toks):
        run = []
        while k < len(toks) and (spelled(k) or run and initial(k)):
            gap = text[run[-1].end():toks[k].start()] if run else " "
            if not (re.fullmatch(r"[ \t]+", gap) or initial(k - 1) and re.fullmatch(r"\.[ \t]+", gap)):
                break
            run.append(toks[k])
            k += 1
        while run and len(run[-1].group()) == 1 and not spelled(toks.index(run[-1])):
            run.pop()                           # no trailing initial
        if not run:
            k += 1
            continue
        s, e = run[0].start(), run[-1].end()
        single = len(run) == 1
        if single and (_fold(run[0].group()) not in lone or run[0].group().lower() in _NOT_NICKNAMES):
            continue
        over = [x for x in list(ents) + added if x.start < e and s < x.end]
        if not over:
            added.append(_ent(text, s, e, "PERSON", 0.9))
        elif not single and all(x.type == "PERSON" and x.source != "pattern" and s <= x.start
                                and x.end <= e and any(x is y for y in ents) for x in over):
            # a model found part of the name ("Acuña" of "Zoé Acuña")
            removed.extend(over)
            added.append(_ent(text, s, e, "PERSON", 0.9))
    return added, removed


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
mr mrs ms mx miss dr prof sir madam
""".split())


# a device named after its owner: "Galaxy-S23-Ritika", "Ritikas-iPhone",
# "DESKTOP-ritika" — the whole host name is the owner's
_DEVICE_WORDS = frozenset("""
iphone ipad ipod galaxy pixel macbook mbp mba imac mac mini pro air max plus ultra laptop
desktop pc phone tablet android surface thinkpad xps kindle echo watch tv chromebook
note tab redmi oneplus huawei xiaomi samsung apple dell hp lenovo asus acer oppo vivo
nokia moto motorola sony xperia nintendo switch ps xbox
""".split())
_DEVICE_HOST = re.compile(r"(?<![\w.@/:\-])([A-Za-z][\w]*(?:(?:[-_]|['’]s[-_ ])[\w]+){1,3})(?![\w@:/\-])")


def _device_hosts(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []
    removed: List[DetectedEntity] = []
    for m in _DEVICE_HOST.finditer(text):
        host = m.group(1)
        own = re.match(r"([A-Za-z]+)['’]s[-_ ]", host)
        if own and own.group(1).lower() in _DEVICE_WORDS:
            continue                        # "MacBook's serial": the device owns
        parts = [p for p in re.split(r"[-_ ]|['’]s\b", host) if p]
        dev = [p for p in parts if p.lower() in _DEVICE_WORDS or re.fullmatch(r"[A-Za-z]{1,3}\d+\w*", p)]
        names = [p for p in parts if p not in dev]
        if (len(names) != 1 or not dev or not any(p.lower() in _DEVICE_WORDS for p in dev)
                or not names[0].isalpha() or len(names[0]) < 3
                or names[0].lower() in NOT_NAMES or names[0].lower() in _NOT_NICKNAMES
                or names[0].lower() in _common_words()):
            continue
        s, e = m.span(1)
        inside = [x for x in ents if s <= x.start and x.end <= e]
        if _overlaps(s, e, [x for x in ents if x not in inside] + added):
            continue
        added.append(_ent(text, s, e, "hostname", 0.9))
        removed.extend(inside)          # "Ritika" inside it is the same person
    return added, removed


# "Contact Bríd on 086 422 7731", "reach Kofi at kofi@x.io": a name with
# the way to reach it right after
_CONTACT_ON = re.compile(
    r"\b(?i:contact|reach|ring|call|text|whatsapp|message|email)\s+"
    r"([A-ZÀ-Ý][a-zà-ÿ'’\-]{1,15}(?:[ \t]+[A-ZÀ-Ý][a-zà-ÿ'’\-]{1,20})?)"
    r"\s+(?:on|at|via)\s+(?=[+(]?\d|\S+@)")


def _verb_frames(text: str, ents: Sequence[DetectedEntity]):
    added = []
    for m in _CONTACT_ON.finditer(text):
        words = m.group(1).lower().split()
        if any(w in _NOT_NICKNAMES or w in NOT_NAMES or w in PUBLIC_ORGS for w in words):
            continue
        if not _overlaps(m.start(1), m.end(1), list(ents) + added):
            added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.85))
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
    r"((?:[A-Z][a-z'’\-]+|[A-Z][A-Z'’\-]+|[A-Z]\.?(?=[ \t]+[A-Z]{2}))"   # "M OKAFOR"
    r"(?:[ \t]+(?:[A-Z][a-z'’\-]+|[A-Z][A-Z'’\-]+)){1,2})\b"
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
_LOWER_PLACE = re.compile(r"([a-zß-ÿ][a-zß-ÿ'’\-]{2,}(?:\s+[a-zß-ÿ][a-zß-ÿ'’\-]{2,})?)(?![\w'’\-])")
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


# "my sister ines,", "a minha filha Beatriz", "meine Tochter Lena"
_KIN_ANY = (
    pattern_scan._KIN[:-1] + r"|irm[ãa]o?|filh[ao]|m[ãa]e|pai|espos[ao]|marido|namorad[ao]"
    r"|hij[ao]|herman[ao]|madre|padre|novi[ao]|fille|fils|s(?:œ|oe)ur|fr[èe]re|femme|mari"
    r"|copine|copain|tochter|sohn|schwester|bruder|frau|mann|freundin|freund|dochter"
    r"|zoon|zus|broer|vriendin|vriend)"
)
_KIN_NAME = re.compile(
    r"(?i:\b(?:my|our|his|her|their|(?:a\s+|o\s+)?minha|(?:a\s+|o\s+)?meu|mi|mis|mon|ma"
    r"|mein|meine|meinem|meiner|mijn)\s+(?:(?:little|older|younger|big|baby|best|eldest"
    r"|youngest|kleine|petite|petit|mayor|menor|mais\s+nova|mais\s+velha)\s+)?"
    + _KIN_ANY + r"s?\s*,?\s+)"
    r"([^\W\d_][\w'’\-]{1,20})(?![\w'’\-])"
)
_KIN_NAME_STOP = frozenset("""
is was has had and or but the a an to too also again here there now today who
which that this just still always never will would can could should did does do
lives lived works worked turns turned says said asked told got gets wants needs
loves likes thinks keeps kept went goes came comes made makes took takes in on at
for of with from by about after before when if because so as is's isn't wasn't
tem é e de da do que se está foi und ist hat war en het is et est a le la les
""".split())
# a dash sign-off at the end of a message or line: "... is 4091. -tash"
_SIGN_OFF = re.compile(r"(?m)(?:^|(?<=\s))[-–—~]\s?([A-Za-z][a-z]{2,15}|[A-Z][a-z]{1,15})[ \t]*$")
_NOT_SIGN_OFF = frozenset("ish esque ly ness able ful less like sama san kun chan".split())


def _kin_names(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []
    lower_writer = None
    for m in _KIN_NAME.finditer(text):
        name = m.group(1)
        key = name.lower()
        if key in _KIN_NAME_STOP or key in NOT_NAMES or key in _NOT_NICKNAMES:
            continue
        if re.search(r"[-’'][a-z]", name) and name[:1].isupper():
            continue                        # "my wife Facebook-stalks", not "Ana-Maria"
        if not name[:1].isupper():
            if lower_writer is None:
                lower_writer = not any(w[:1].isupper()
                                       for w in re.findall(r"[^\W\d_][\w'’\-]*", text))
            # a lower-case name only from a writer who capitalises nothing,
            # and only with a comma or a pronoun after it ("my sister ines, she")
            if not lower_writer or not re.match(r"\s*[,;:!?)]|\s+(?:who|she|he)\b", text[m.end(1):]):
                continue
        if not _overlaps(m.start(1), m.end(1), list(ents) + added):
            added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.85))
    for m in _SIGN_OFF.finditer(text):
        if m.group(1).lower() in _NOT_SIGN_OFF or m.group(1).lower() in NOT_NAMES:
            continue
        if text[m.end():].strip():
            continue                        # only the last line of the message
        if not _overlaps(m.start(1), m.end(1), list(ents) + added):
            added.append(_ent(text, m.start(1), m.end(1), "PERSON", 0.8))
    return added, []


# ── 7c. affiliation, age and contact-block frames ───────────────────────────

# "estudia en Redfern Physiotherapy", "trabalho na Copperleaf", "arbeite bei
# Kestrel Analytics": the place a person studies or works, in the languages
# whose NER gate drops model ORGs.
_AFFIL_VERB = (
    r"(?i:\b(?:estudia|estudio|estudian|estudiaba|trabaja|trabajo|trabajan|trabajaba)"
    r"\s+(?:en|para)\s+(?:el\s+|la\s+)?"
    r"|\b(?:estuda|estudo|estudava|trabalha|trabalho|trabalhava)\s+(?:na|no|em|para)\s+"
    r"|\b(?:travaille|travaillais|travaillait|étudie|étudiais)\s+(?:à|a|chez|pour|au)\s+(?:la\s+|l['’])?"
    r"|\b(?:lavora|lavoro|lavoravo|studia|studio)\s+(?:presso|da|al|alla|per)\s+"
    r"|\b(?:arbeite|arbeitet|arbeitete|studiere|studiert)\s+(?:bei|für|an\s+der|am)\s+(?:der\s+)?"
    r"|\b(?:werk|werkt|studeer|studeert)\s+(?:bij|aan|voor)\s+(?:de\s+|het\s+)?)"
)
_AFFIL_NAME = _CAP_W + r"(?:[ \t]+(?:(?:&|y|e|et|und|en|de|del|di|da|do|du|van)[ \t]+)?" + _CAP_W + r"){0,4}"
_AFFIL = re.compile(_AFFIL_VERB + r"(" + _AFFIL_NAME + r")")

# "<Name> tiene 25 años", "Noa a 71 ans", "Jonas ist 41 Jahre alt"
_NAME_AGE = re.compile(
    r"(?<![\w@#./-])(" + _CAP_W + r"(?:[ \t]+" + _CAP_W + r")?)[ \t]+"
    r"(?:tiene|tenía|tem|tinha|a|avait|ha|aveva|hat|hatte|ist|heeft|is|was|turned|turns)"
    r"[ \t]+\d{1,3}[ \t]+(?:años|anos|ans|anni|jahre|jaar|years?[ \t]+old)\b", re.I)
_NAME_AGE_STOP = frozenset("""
el ella él yo ele ela il elle er sie es he she it lui lei hij zij mein meine mijn mon ma
mi su tu who this that there which mio mia sua suo nostro nostra unser unsere notre
""".split())

# An all-caps name: "FOLAKE BANERJEE" on the line above a street, or
# "URGENT: ULRIKE REYES (875.228.2393)" before a bracketed contact detail.
_CAPS_NAME = r"[A-ZÀ-Ý][A-ZÀ-Ý'’\-]+(?:[ \t]+[A-ZÀ-Ý][A-ZÀ-Ý'’\-]+){1,2}"
_CAPS_BEFORE_CONTACT = re.compile(r"(?:^|(?<=[:;,\n])[ \t]*)(" + _CAPS_NAME + r")[ \t]*\(")
_BLOCK_LINE = re.compile(r"[ \t]*([^\W\d_][\w'’.\-]*(?:[ \t]+[^\W\d_][\w'’.\-]*){1,3})[ \t]*")
_NOT_CAPS_NAME = frozenset("""
urgent call contact phone tel fax mobile cell office home work email please note attn
customer service hotline help desk support ship shipping billing bill deliver delivery
mailing return address to from my our the name re fwd subject order refund invoice
account dear sir madam team dept department
""".split())

# "im bongani from moncton": one given name after "I'm", then "from" — words
# that are no name ("im back from", "im fresh from") are listed.
_IM_FROM = re.compile(r"(?i:\b(?:i\s*['’]?m|i\s+am)[ \t]+)([^\W\d_][\w'’\-]{2,15})[ \t]+from[ \t]+\S")
_NOT_IM_FROM = frozenset("""
back home free far away fresh just originally coming calling writing texting tired sick
different safe sore here there out up down done also still now really actually not too
so very only literally apart separate divorced estranged banned blocked exempt immune
protected absent missing gone retired fired suspended released discharged hiding running
glad happy proud sorry new young old native local based located currently straight right
recovering visiting moving travelling traveling returning flying driving graduating
transferring escaping fleeing a an the from your his her their our
""".split())


# "Can Lumen Credit Union fire Bilal Park for …": the object of a personnel
# verb is a person, its subject the employer — whatever type NER guessed.
_PERSONNEL = (r"(?:fire|fired|firing|hire|hired|hiring|promote|promoted|demote|demoted"
              r"|discipline|disciplined|terminate|terminated|dismiss|dismissed|lay\s+off|laid\s+off)")
_PERSONNEL_FRAME = re.compile(
    r"(?:\b(?i:can|could|did|does|will|would|should|may|might)[ \t]+("
    + _CAP_W + r"(?:[ \t]+" + _CAP_W + r"){0,3})[ \t]+(?i:legally[ \t]+)?)?"
    r"\b(?i:" + _PERSONNEL + r")[ \t]+(" + _CAP_W + r"(?:[ \t]+" + _CAP_W + r"){1,2})"
    r"(?=[ \t]+(?i:for|because|after|over|without|while)\b|[ \t]*[.?!,;]|$)")


def _caps_ok(words: Sequence[str]) -> bool:
    return not any(w.lower().strip("'’.-") in _NOT_CAPS_NAME or w.lower() in NOT_NAMES
                   or _LEGAL.search(w) or w.lower() in _GENERIC_INST_WORD for w in words)


def _contact_frames(text: str, ents: Sequence[DetectedEntity]):
    from . import relation_gate as rg       # relation_gate imports this module
    added: List[DetectedEntity] = []

    def add(s, e, typ, score=0.85):
        cand = _ent(text, s, e, typ, score)
        if s < e and not _overlaps(s, e, list(ents) + added) and not rg.is_junk(cand, text):
            added.append(cand)

    for m in _AFFIL.finditer(text):
        name = m.group(1)
        words = name.split()
        if words[0].lower() in NOT_NAMES or words[0].lower() in _INST_STOP:
            continue
        typ = "GPE" if name.lower() in MAJOR_CITIES else "ORG"
        add(m.start(1), m.end(1), typ)
    for m in _NAME_AGE.finditer(text):
        words = m.group(1).split()
        if words[0].lower() in _NAME_AGE_STOP or words[0].lower() in NOT_NAMES:
            continue
        if words[0].lower() in PUBLIC_PEOPLE or m.group(1).lower() in PUBLIC_ORGS:
            continue
        add(m.start(1), m.end(1), "PERSON")
    # all-caps names in a contact block
    for m in _CAPS_BEFORE_CONTACT.finditer(text):
        words = m.group(1).split()
        close = text.find(")", m.end())
        inside = [e for e in ents if e.source == "pattern" and m.end() - 1 <= e.start < (close if close > 0 else len(text))]
        if inside and _caps_ok(words):
            add(m.start(1), m.end(1), "PERSON")
    starts = {e.start for e in ents if e.type in ("address", "street", "ADDRESS", "FAC")}
    pos = 0
    for line in text.split("\n"):
        nxt = pos + len(line) + 1
        lm = _BLOCK_LINE.fullmatch(line)
        if lm and line.strip() and nxt in starts and not line.rstrip().endswith(":"):
            words = lm.group(1).split()
            if _caps_ok(words) and all(w[:1].isupper() for w in words):
                add(pos + lm.start(1), pos + lm.end(1), "PERSON")
        pos = nxt
    for m in _PERSONNEL_FRAME.finditer(text):
        obj = m.group(2).split()
        if _caps_ok(obj) and not any(w.lower() in PUBLIC_PEOPLE for w in obj):
            add(m.start(2), m.end(2), "PERSON")
            if m.group(1) and not rg.is_public_org(m.group(1)) and m.group(1).split()[0].lower() not in (
                    "they", "we", "i", "you", "he", "she", "my", "our", "the", "a", "an"):
                add(m.start(1), m.end(1), "ORG")
    lower_writer = _lowercase_writer(text)
    for m in _IM_FROM.finditer(text):
        name = m.group(1)
        key = name.lower()
        if key in _NOT_IM_FROM or key in NOT_NAMES or key in _NOT_NICKNAMES:
            continue
        if re.search(r"(?:ing|ed|ly)$", key):
            continue
        if not (name[:1].isupper() or lower_writer):
            continue
        add(m.start(1), m.end(1), "PERSON")
    return added, []


# A company line under the name in a sign-off ("Thanks,\nAndrei Ruiz\n
# Orbital Freight\n+1-379-…"). A job-title line is skipped, not masked.
_TITLE_LINE = re.compile(r"[ \t]*((?:[A-Z][\w'’.\-]*|&)(?:[ \t]+(?:[A-Z][\w'’.\-]*|&|of|and|for|the|de|y)){0,5})[ \t]*")


def _signature_orgs(text: str, ents: Sequence[DetectedEntity]):
    from . import relation_gate as rg
    if "\n" not in text:
        return [], []
    added: List[DetectedEntity] = []
    lines, starts, pos = text.split("\n"), [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    persons = [e for e in ents if e.type == "PERSON"]
    contacts = [e for e in ents if e.source == "pattern" and e.type in ("email", "phone_us", "phone_uk",
                "phone_intl", "phone", "url")]
    for i, line in enumerate(lines):
        if not any(p.start >= starts[i] and p.end <= starts[i] + len(line) and
                   not line.replace(p.text, "").strip(" \t,") for p in persons):
            continue                        # a line that holds only a name
        closing = any(rg._CLOSING.match(l) for l in lines[max(0, i - 2):i])
        for j in range(i + 1, min(i + 3, len(lines))):
            lm = _TITLE_LINE.fullmatch(lines[j])
            if not lm or _overlaps(starts[j], starts[j] + len(lines[j]), ents):
                break
            core = lm.group(1)
            if rg._JOB_TITLE.search(core) or core.lower() in NOT_NAMES:
                continue                    # "Senior Analyst"
            follows = j + 1 < len(lines) and any(starts[j + 1] <= c.start < starts[j + 1] + len(lines[j + 1])
                                                 for c in contacts)
            if closing or follows:
                cand = _ent(text, starts[j] + lm.start(1), starts[j] + lm.end(1), "ORG", 0.85)
                if not rg.is_junk(cand, text):
                    added.append(cand)
            break
    return added, []


_ORG_LABEL = re.compile(
    r"(?im)^[ \t\-*•]*(?:employer|company|organi[sz]ation|workplace|business(?:\s+name)?|firm"
    r"|employer\s+name|company\s+name|empresa|empleador|arbeitgeber|employeur|datore\s+di\s+lavoro"
    r"|entreprise|firma|werkgever)[ \t]*[:=][ \t]*"
    r"([^\n,;]{2,60}?)[ \t]*(?=,|;|$)")
_LEGAL_SUFFIX = re.compile(
    r"((?:[A-ZÀ-Ý][\w&'’\-]*[ \t]+){1,4})(?:S\.A\.C\.|S\.A\.S\.?|S\.A\.|S\.R\.L\.|S\.r\.l\.|S\.L\.U?\.?"
    r"|S\.p\.A\.|Ltda\.?|GmbH|B\.V\.|N\.V\.|Pty\.?[ \t]+Ltd\.?|Sdn\.?[ \t]+Bhd\.?|Pvt\.?[ \t]+Ltd\.?"
    r"|K\.K\.|A/S|SARL|S\.?à\s?r\.?l\.?|Kft\.?|s\.r\.o\.|sp\.[ \t]*z[ \t]*o\.o\.|Oy|AB|AG)(?![\w])")
_PAYROLL = re.compile(r"\b(?:PAYROLL|DIRECT[ \t]+DEP(?:OSIT)?|SALARY|PAYCHECK)[ \t]+(?:FROM[ \t]+)?"
                      r"([A-Z][A-Z&'’\-]+(?:[ \t]+[A-Z][A-Z&'’\-]+){0,3})")
_CALLED = re.compile(
    r"\b(?i:shop|caf[eé]|coffee\s+shop|restaurant|bar|pub|bakery|salon|gym|store|company|business"
    r"|startup|firm|daycare|nursery|agency|practice)"
    # "the coffee shop where i work its called X"
    r"(?:[^.\n]{0,40}?\b(?i:it['’]?s|it\s+is)(?=[ \t]+(?i:called|named)))?"
    r"[ \t]+(?i:(?:is|was|i\s+work\s+at\s+is)[ \t]+)?(?i:called|named)[ \t]+[\"“']?"
    # the longest run up to a preposition, never across a clause word
    r"((?:[\w'’&\-]+)(?:(?![ \t]+(?i:and|but|so|which|where|what|who|when|i|we|it|is|was|are"
    r"|on|in|at|near|by)\b)[ \t]+[\w'’&\-]+){0,4})[\"”']?"
    r"(?=[ \t]+(?i:on|in|at|near|by|and|but|so|which|that|where|what|who|downtown)\b|[ \t]*[,.;!?)\n]|$)")


def _org_frames(text: str, ents: Sequence[DetectedEntity]):
    from . import relation_gate as rg
    added: List[DetectedEntity] = []

    def add(s, e):
        while e > s and text[e - 1] in " \t.'\"”" and not (     # keep "S.A.C."
                text[e - 1] == "." and re.search(r"\b[A-Za-z]\.[A-Za-z]\.$", text[s:e])):
            e -= 1
        if e - s < 2 or _overlaps(s, e, list(ents) + added):
            return
        cand = _ent(text, s, e, "ORG", 0.85)
        if not rg.is_public_org(cand.text) and cand.text.lower() not in NOT_NAMES:
            added.append(cand)

    for m in _ORG_LABEL.finditer(text):
        v = m.group(1).strip()
        if v.lower() not in ("n/a", "na", "none", "self", "self-employed", "unemployed", "retired",
                             "student", "-", "tbd", "same"):
            add(m.start(1), m.start(1) + len(m.group(1).rstrip()))
    for m in _LEGAL_SUFFIX.finditer(text):
        add(m.start(1), m.end())
    for m in _PAYROLL.finditer(text):
        add(m.start(1), m.end(1))
    for m in _CALLED.finditer(text):
        v, quoted = m.group(1), text[m.start(1) - 1:m.start(1)] in "\"“'"
        stop = re.match(r"[ \t]+[a-z]+\b", text[m.end(1):])
        if quoted or v[:1].isupper() or (stop and _lowercase_writer(text)):
            add(m.start(1), m.end(1))
    for m in re.finditer(r"(?m)^[ \t]*([^|\n]{3,60}?)[ \t]+\|[ \t]+([^|\n]{3,60}?)[ \t]*$", text):
        lm = _TITLE_LINE.fullmatch(m.group(2))
        if (rg._JOB_TITLE.search(m.group(1)) and lm and not rg._JOB_TITLE.search(m.group(2))
                and len(m.group(2).split()) >= 2):
            add(m.start(2), m.end(2))           # "Events Coordinator | Lantern Hall Collective"
    return added, []


_KANA = r"[\u4e00-\u9fff\u3040-\u30ff々]"
# "はじめまして、佐々木 美咲です" — a self-introduction after a greeting
_JA_NAME = re.compile(r"(?:はじめまして|初めまして|私の名前は|名前は)[、,]?\s*"
                      r"(" + _KANA + r"{1,4}(?:[ 　]" + _KANA + r"{1,4})?)(?:です|と申します|といいます)")
_JA_PLACE = re.compile(r"(" + r"[\u4e00-\u9fff々ァ-ヶー]{1,6})(?:に住んで|在住|出身)")
# "李梅的身份证号", "在北京协和医院工作"
_ZH_PII_AFTER = r"(?=的(?:身份证|电话|手机|地址|邮箱|护照|生日|银行卡|病历))"
_JA_ORG = re.compile(r"(?<=[ぁ-んるたのはがでにを、。「（\s])([\u4e00-\u9fff々ァ-ヶー]{1,8}[ぁ-ん]{0,4}"
                     r"(?:小学校|中学校|高等学校|高校|大学|幼稚園|保育園|病院|医院|クリニック|株式会社))")
_JA_HONORIFIC = re.compile(r"(?<=[のとは、。「\s])([\u4e00-\u9fff々]{2,3})(?=くん|君|さん|ちゃん|様|先生)")
_JA_NOT_NAME = frozenset("皆様 皆さん お客 奥様 先生 担任 校長 店長 部長 課長 社長 医者 患者 生徒 同級 友達".split())
# "一家叫“蓝鲸数智”的小公司"
_ZH_CALLED_ORG = re.compile(r"叫[“\"「]([^”\"」]{2,20})[”\"」]的?(?:小|大)?(?:公司|店|餐厅|学校|机构|工作室|诊所)")
_ZH_ROLE = (r"(?:房东|老板|经理|老师|医生|同事|朋友|邻居|室友|丈夫|妻子|老公|老婆|儿子|女儿|妈妈|爸爸"
            r"|哥哥|姐姐|弟弟|妹妹|客户|律师|主任|上司|领导|男朋友|女朋友)")
_ZH_ORG = re.compile(r"在(" + _HAN + r"{2,10}?(?:医院|诊所|学校|大学|中学|小学|幼儿园|公司|银行))"
                     r"(?:工作|上班|上学|读书|就诊|看病|住院)")
# "深圳南山区桃园路88号5栋1203", "成都高新区": a city (Faker's zh_CN list), a
# district and a numbered road; a function character never starts a name
_ZH_NAME_CH = r"[^\W\d_A-Za-z在住是我你他她们的了和与到从于为把被叫个家这那一]"
_ZH_ROAD = (r"(?P<road>" + _ZH_NAME_CH + r"{1,4}?(?:路|街|大道|巷|弄|胡同)\d{1,5}号"
            r"(?:\d{1,4}(?:栋|幢|座|单元|号楼|楼|层|室|号))*(?:\d{1,5}室?)?)")
_ZH_DISTRICT = r"(?P<dist>" + _ZH_NAME_CH + r"{1,3}?[区县])"
_HI_KIN = (r"(?:मेरी|मेरा|मेरे|हमारी|हमारा|हमारे)\s+(?:माँ|मां|माता|माताजी|पिता|पिताजी|पापा|मम्मी|बहन"
           r"|भाई|बेटी|बेटा|पत्नी|पति|दोस्त|सहेली|दादी|दादा|नानी|नाना|चाचा|चाची|मामा|मामी|बुआ"
           r"|ससुर|सास|पड़ोसी|बॉस)\s+")
# words that end the name after a kin word: case markers, verbs, "years"
_HI_STOP = frozenset("का की के ने को से में है हैं था थी थे और भी तो जी साल वर्ष बहुत अभी कल आज "
                     "एक दो डॉक्टर".split())
_HI_CITIES = ("दिल्ली मुंबई लखनऊ कोलकाता चेन्नई बेंगलुरु बैंगलोर हैदराबाद पुणे जयपुर कानपुर पटना "
              "भोपाल इंदौर अहमदाबाद वाराणसी आगरा नागपुर सूरत चंडीगढ़ देहरादून रांची गुवाहाटी").split()
_HI_CITY = re.compile(r"(?<!\S)(" + "|".join(_HI_CITIES) + r")(?=\s+(?:में|से|का|की|के)(?!\S))")


@lru_cache(maxsize=1)
def _lexicons():
    """CJK surnames and Hindi given / family names from Faker's locale lists."""
    from faker.providers.person.zh_CN import Provider as ZH
    from faker.providers.person.hi_IN import Provider as HI
    zh = sorted({n for n in ZH.last_names if 1 <= len(n) <= 2}, key=len, reverse=True)
    hi_first = frozenset(tuple(HI.first_names_male) + tuple(HI.first_names_female))
    hi_last = frozenset(tuple(HI.last_names) + ("देवी", "कुमारी", "कुमार", "बाई"))
    return zh, hi_first, hi_last


@lru_cache(maxsize=1)
def _zh_address():
    from faker.providers.address.zh_CN import Provider as A
    cities = sorted(set(A.cities), key=len, reverse=True)
    city = r"(?P<city>(?:" + "|".join(cities) + r")市?)"
    # a city alone stays ("北京烤鸭"); with a district or road it is an address
    return (re.compile(city + _ZH_DISTRICT + r"?" + _ZH_ROAD + r"?(?<=[区县\d室号])"),
            re.compile(_ZH_DISTRICT + r"?" + _ZH_ROAD))


def _names_intl(text: str, ents: Sequence[DetectedEntity]):
    added: List[DetectedEntity] = []

    def add(s, e, typ):
        if s < e and not _overlaps(s, e, list(ents) + added):
            added.append(_ent(text, s, e, typ, 0.9))

    for rx in (_ZH_NAME, _HI_NAME, _JA_NAME):
        for m in rx.finditer(text):
            add(m.start(1), m.end(1), "PERSON")
    if re.search(_HAN, text):
        zh, _, _ = _lexicons()
        rx = re.compile(r"(?:^|(?<=[，,。：:；;\s]))((?:" + "|".join(zh) + r")" + _HAN + r"{1,2}?)" + _ZH_PII_AFTER)
        for m in rx.finditer(text):
            add(m.start(1), m.end(1), "PERSON")
        for m in _ZH_ORG.finditer(text):
            add(m.start(1), m.end(1), "ORG")
        for m in _ZH_CALLED_ORG.finditer(text):
            add(m.start(1), m.end(1), "ORG")
        role = re.compile(_ZH_ROLE + r"((?:" + "|".join(zh) + r")" + _HAN + r"{1,2})(?=[，,。：:；;、\s]|$)")
        for m in role.finditer(text):
            add(m.start(1), m.end(1), "PERSON")
        for m in _JA_ORG.finditer(text):
            add(m.start(1), m.end(1), "ORG")
        for m in _JA_HONORIFIC.finditer(text):
            if m.group(1) not in _JA_NOT_NAME and not m.group(1).endswith(("生", "長")):
                add(m.start(1), m.end(1), "PERSON")
        for m in _JA_PLACE.finditer(text):
            add(m.start(1), m.end(1), "GPE")
        for m in (m for rx in _zh_address() for m in rx.finditer(text)):
            for g, typ in (("city", "GPE"), ("dist", "GPE"), ("road", "address")):
                if m.groupdict().get(g):
                    add(m.start(g), m.end(g), typ)
    if re.search(r"[\u0900-\u097f]", text):
        _, hi_first, hi_last = _lexicons()
        for m in re.finditer(_HI_KIN + r"([\u0900-\u097f]+(?:\s+[\u0900-\u097f]+){0,2})", text):
            toks = list(re.finditer(r"[\u0900-\u097f]+", m.group(1)))
            n = next((i for i, t in enumerate(toks) if t.group() in _HI_STOP), len(toks))
            # a kin word then a name: the words before a case marker or verb,
            # if the run ends there (not "मेरे पिता बहुत …")
            # one word then a copula is a predicate ("मेरी माँ बीमार हैं")
            copula = n == 1 and n < len(toks) and toks[1].group() in ("है", "हैं", "था", "थी", "थे")
            if n and not copula and (n < len(toks) or re.match(r"\s+(?:\d|[,।]|[\u0900-\u097f]+\s)", text[m.end():])) \
                    and toks[0].group() not in _HI_STOP:
                add(m.start(1) + toks[0].start(), m.start(1) + toks[n - 1].end(), "PERSON")
        # a city in a message that already names a person
        if any(e.type == "PERSON" for e in list(ents) + added):
            for m in _HI_CITY.finditer(text):
                add(m.start(1), m.end(1), "GPE")
    return added, []


# ── 9. European street addresses ─────────────────────────────────────────────

_W = r"[A-ZÀ-Ý][\w'’\-]+"
_JOIN = r"(?:(?:de|des|du|de\s+la|d['’]|da|do|dos|das|del|della|di|von|der)\s*)?"
_INTL_STREET = re.compile(
    # fr: "14 rue des Lilas", "3 bis avenue Victor Hugo"
    r"\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|chemin|all[ée]e|impasse"
    r"|place|quai|route|cours)\s+" + _JOIN + _W + r"(?:\s+" + _W + r"){0,3}"
    # pt/es/it/pl: "Rua Augusta 1508", "Calle Mayor, 12", "Via Roma 5", "Bv. San Juan 1120"
    r"|(?<![\w.])(?:Rua|Avenida|Avda\.|Av\.|Travessa|Trav\.|Alameda|Estrada|Largo|Pra[çc]a|Rodovia"
    r"|Calle|Paseo|Plaza|Carrer|Camino|Carrera|Bulevar|Bv\.|Bvd\.|Blvd\.|Via|Viale|Piazza|Corso"
    r"|Vicolo|Strada|ul\.|ulica|Aleja)"
    r"\s+" + _JOIN + _W + r"(?:\s+" + _JOIN + _W + r"){0,3},?\s+(?:n[º°o]\.?\s*)?\d{1,5}[A-Za-z]?\b"
    # number first: "12 Rua do Sol", "8 Via dei Mille"
    r"|\b\d{1,4}[A-Za-z]?,?\s+(?:Rua|Avenida|Travessa|Alameda|Estrada|Largo|Pra[çc]a|Calle|Via|Viale"
    r"|Piazza|Corso|Strada)\s+" + _JOIN + _W + r"(?:\s+" + _JOIN + _W + r"){0,3}"
    # de/nl/nordic/fi compounds: "Hauptstraße 5", "Lindenweg 12a", "Kerkstraat 14",
    # "Storgata 41B", "Nørregade 3", "Mannerheimintie 12"
    r"|\b(?!(?:Spring|String|Offspring|Boring|Swing|Sing|Bring|Wing|Sling|Sting|Spa)\b)"
    r"[A-ZÄÖÜÆØÅ][\wäöüßæøå]+(?:stra(?:ß|ss)e|str\.|weg|platz|allee|gasse|ring|damm|straat|laan"
    r"|gracht|plein|kade|singel|dijk|steeg|dreef|gata|gatan|gaten|vägen|veien|vegen|vej|gade"
    r"|stræde|katu|tie)\s+\d{1,4}[A-Za-z]?\b"
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
    # a locality may sit between: "Gandhi Nagar, Almora 263601" (IN PIN)
    r"|(?:(?P<loc>" + _CITY_W + r")[ \t]*,[ \t]*)?(?P<t2>" + _CITY_W + r")[ \t]+(?P<pc2>\d{4,6})(?!\d)"
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
# the town between a street and a UK postcode
_TOWN_UK_POSTCODE = re.compile(
    r"[ \t]*,[ \t]*([A-Za-z][a-z'’\-]+(?:[ \-][A-Za-z][a-z'’\-]+)?)[ \t]*,?[ \t]+"
    r"(?=[A-Za-z]{1,2}\d[A-Za-z\d]?[ \t]?\d[A-Za-z]{2}\b)")
# a flat or unit number in front of a street address: "flat 4, 70 Cowley Road"
_UNIT_BEFORE = re.compile(
    r"(?i)\b((?:flat|apt\.?|apartment|unit|suite|ste\.?)\s*#?\s*\d{1,4}[a-z]?)\s*(?:,|\bat\b|\bof\b)?\s*$")
# floor / door / unit after a street: "Calle Mayor 17, 3º B", "Travessa do
# Carmo 8, 2.º Esq.", "Lindenallee 3a, 2. OG links", "Piso 4 Dpto B"
_UNIT_AFTER = re.compile(
    r"[ \t]*,?[ \t]*(?:"
    r"\d{1,2}[ \t]?\.?[ \t]?[º°ª]\.?(?:[ \t]*(?:[A-H](?![\w])|Esq\.?|Dto\.?|Dta\.?|Dir\.?|Izq(?:da)?\.?"
    r"|Dcha\.?|Fte\.?|Frente|Puerta[ \t]+\w{1,3}))?"
    r"|\d{1,2}\.[ \t]*(?:OG|Stock|Etage|Obergeschoss|EG|DG|UG)(?:[ \t]+(?:links|rechts|mitte|Mitte))?"
    r"|(?i:piso|planta|dpto|depto|puerta|pta|apto|apt|esc|escalera|bloque|portal|unit|flat|suite|ste"
    r"|wohnung|whg|top|stiege|bus|bo[iî]te)\.?[ \t]*#?[ \t]*(?=[\w]{0,4}\d|[A-Z]\b)\w{1,4}\b"
    r")")
_STREET_POSTCODE = re.compile(r"[^\n]{0,40}?\b[A-ZÀ-Ý][\w'’\-]+[ \t]*\((\d{4,6})\)")


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
        end = st.end
        for _ in range(3):                  # "Piso 4 Dpto B", "3º B"
            u = _UNIT_AFTER.match(text, end)
            if not u or u.end() == end:
                break
            _add(u.start() + len(u.group()) - len(u.group().lstrip(" \t,")), u.end())
            end = u.end()
        b = _STREET_POSTCODE.match(text, end)
        if b:
            _add(b.start(1), b.end(1))       # "Ponferrada (24401)"
        m = _POSTCODE_TOWN.match(text, end)
        town = m and (m.group("t1") or (m.group("loc") or "") + " " + m.group("t2"))
        if m and not any(w.lower() in pattern_scan._NOT_NAME_WORD
                         for w in re.findall(r"[^\W\d_]+", town)):
            _add(m.start("pc") if m.group("pc") else m.start("loc") if m.group("loc")
                 else m.start("t2"), m.end())
        t = _TOWN_UK_POSTCODE.match(text, st.end)
        if t and t.group(1).lower() not in pattern_scan._NOT_NAME_WORD:
            _add(t.start(1), t.end(1))     # "14 birchwood close, wokingham rg40 2hd"
        ls = text.rfind("\n", 0, st.start) + 1
        u = _UNIT_BEFORE.search(text, ls, st.start)
        if u:
            _add(u.start(1), u.end(1))
    for m in _BRACKET_POSTCODE.finditer(text):
        _add(m.start(1), m.end(1))
    removed = [x for x in ents if x.source != "pattern"
               and any(a.start <= x.start and x.end <= a.end for a in added)]
    return added, removed


_INTL_UNIT = re.compile(
    r"(?i:\b(?:apto|apartamento|apt|piso|depto|dpto|appartement|appt|[ée]tage|wohnung|interno"
    r"|int\.|sala|bloco|bloque|escalera|esc\.)\.?\s*)(?:n[º°o]\.?\s*)?\d{1,4}[A-Za-z]?\b"
    # Indian house numbers: "H.No. 3-118", "Door No: 12/4", "Plot No. 45A"
    r"|\b(?:H\.?[ ]?No|House[ ]No|Flat[ ]No|Plot[ ]No|Door[ ]No|D\.[ ]?No)\.?[ ]?[:\-]?[ ]?\d[\w/\-]{0,9}"
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
        # before _places: "im olumide from valparaíso" needs the person first
        lambda t, es: _contact_frames(t, es),
        lambda t, es: _streets(t, es),
        lambda t, es: _address_parts(t, es),
        lambda t, es: _csv(t, es),
        lambda t, es: _speakers(t, es),
        lambda t, es: _local_part_names(t, es),
        lambda t, es: _device_hosts(t, es),
        lambda t, es: _payments(t),
        lambda t, es: _names_intl(t, es),
        lambda t, es: _kin_names(t, es),
        lambda t, es: _verb_frames(t, es),
        lambda t, es: _display_names(t, es),
        lambda t, es: _institutions(t, es),
        lambda t, es: _signature_orgs(t, es),
        lambda t, es: _org_frames(t, es),
        lambda t, es: _components(t, es),
    ]
    if not skip_locations:
        rules.insert(4, lambda t, es: _places(t, es, place_verifier))
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
