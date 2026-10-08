"""The PIITagger's training messages (V3 §3.3, "the first large use of our
injection pipeline as a data factory").

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.data --n 40000 --seed 1 --out bench/tagger/build/data/train-40k.jsonl

Every message is assembled here from ``bank.py`` (carriers, placements,
hard negatives, all written for this project) and values from the
**training half** of the identity pools (``identities.identity(pool="train")``,
§5.3), so no evaluation-pool word is ever a positive. No real text is used
(DETECTOR_PROGRESS D3-3). Each record is ``{id, text, spans, meta}`` with
``spans`` = ``[start, end, tag]`` character offsets; the trainer turns them
into BIO token labels for whatever tokenizer it uses.

Tags are our 13 protect types, with ADDRESS split into STREET / CITY /
REGION / POSTCODE so that a partial address is visible (the detector joins
adjacent parts back into one ADDRESS). Two value formats are held out of
training entirely (``HELD_OUT``) so generalisation to an unseen format can be
measured on dev. Augmentations (case, missing spaces, typos in the carrier,
Markdown, line breaks inside a value, negatives next to values) are applied
to recorded fractions of the messages (``AUG``).

VERSION 2 (V4 §3.1 D) adds the "age" layout: ages next to the people they
belong to (a sign-off "Ana Ruiz, 34", a header line, mid-sentence, bracketed,
"of <city>", Reddit's "(34F)", worded, a form row), lines of the same shape
whose number is not an age (labelled O), and AGE in signature blocks. Its
templates (``bank.AGE_*``) are worded apart from the AGE probe's
(tests/test_V4_tagger_age_data.py). It also trims spaces off address parts.
The V4 model trains on ``train-40k-v4.jsonl`` / ``val-2k-v4.jsonl``; the
VERSION 1 files stay as the record of ``pii-tagger-dv3s-40k``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bench.realdata import identities as ids
from bench.realdata.common import ROOT, TASKS
from bench.tagger import bank

TYPES = ids.TYPES
ADDRESS_TAGS = ("STREET", "CITY", "REGION", "POSTCODE")
TAGS = ("PERSON", "ORG", *ADDRESS_TAGS, "LOCATION", "AGE", "DATE_OF_BIRTH", "EMAIL", "PHONE", "HANDLE", "URL",
        "NETWORK", "ID", "CREDENTIAL")
LABELS = ("O",) + tuple(f"{p}-{t}" for t in TAGS for p in ("B", "I"))
# (type, fmt) never generated for training; dev measures them as unseen formats
HELD_OUT = (("PHONE", "words"), ("DATE_OF_BIRTH", "month-ordinal"))
BUILD = ROOT / "bench" / "tagger" / "build"
PUBLIC_PEOPLE = ROOT / "python-library" / "surrogateshield" / "core" / "detection" / "public_people.txt"
VERSION = 2                         # 2: ages of named people (V4 §3.1 D: the "age" layout, AGE in signatures)

LAYOUTS = {"intro": .24, "inline": .18, "signature": .14, "form": .14, "json": .07, "env": .06, "table": .04,
           "story": .03, "none": .24, "age": .05}
# the "age" layout's forms (bank.AGE_*): a name and an age closing a sign-off, a labelled line, mid-sentence, ...
AGE_LAYOUTS = {"signoff": .24, "line": .16, "mid": .14, "bracket": .14, "reddit": .12, "from": .08, "worded": .07,
               "form": .05}
AGE_NEAR = .4        # share of "age" messages with a same-shape line whose number is not an age
# fraction of messages each augmentation is applied to
AUG = {"lower": .07, "upper": .02, "nospace": .06, "typos": .15, "markdown": .06, "linebreak": .05,
       "negatives": .30, "third_party": .20, "greeting": .45, "closing": .35}
SHIFT = .30          # identities drawn with shift formats (as Phase 3's format-shift prompts)


# ── pieces ───────────────────────────────────────────────────────────────────

@dataclass
class Piece:
    text: str
    tag: Optional[str] = None         # a TAG, or None (O)
    kind: str = "c"                   # c carrier, v value, n hard negative


def join(pieces: Sequence[Piece]) -> Tuple[str, List[List]]:
    text, spans, pos = [], [], 0
    for p in pieces:
        if p.tag is not None and p.text:
            spans.append([pos, pos + len(p.text), p.tag])
        text.append(p.text)
        pos += len(p.text)
    return "".join(text), spans


# ── values ───────────────────────────────────────────────────────────────────

def address_parts(value: str, country: str) -> List[Piece]:
    """*value* (``identities.address``) as STREET / CITY / REGION / POSTCODE
    pieces with the layout's separators between them (O)."""
    country = country if country in ids.COUNTRIES else "US"
    _loc, layout, cities = ids.COUNTRIES[country]
    for town, region in cities + ids.EXTRA_CITIES.get(country, []):
        rx, names = "", []
        for lit, name in re.findall(r"([^{]*)(?:\{(\w+)\})?", layout):
            rx += re.escape(lit)
            if name == "street":
                rx += r"(?P<street>.+)"
            elif name == "city":
                rx += f"(?P<city>{re.escape(town)})"
            elif name == "region":
                rx += f"(?P<region>{re.escape(region)})"
            elif name == "postcode":
                rx += r"(?P<postcode>.*?)"
        m = re.fullmatch(rx.rstrip(r"\ ").rstrip(","), value) or re.fullmatch(rx, value)
        if m is None:
            continue
        cuts = sorted((m.start(g), m.end(g), g.upper()) for g in m.groupdict() if m.group(g))
        out, pos = [], 0
        for s, e, g in cuts:
            s, e = s + len(value[s:e]) - len(value[s:e].lstrip()), e - len(value[s:e]) + len(value[s:e].rstrip())
            if s > pos:
                out.append(Piece(value[pos:s]))
            out.append(Piece(value[s:e], g, "v"))
            pos = e
        if pos < len(value):
            out.append(Piece(value[pos:]))
        return out
    raise ValueError(f"cannot split an address of {country}")


def value_pieces(v: dict, country: str) -> List[Piece]:
    if v["type"] == "ADDRESS":
        return address_parts(v["value"], country)
    return [Piece(v["value"], v["type"], "v")]


def draw_identity(rng: random.Random, types: Sequence[str], task: str) -> dict:
    """A training-pool identity with none of the held-out formats."""
    for _ in range(50):
        ident = ids.identity(rng, types, task, shift=rng.random() < SHIFT, pool="train")
        if not any((v["type"], v["fmt"]) in HELD_OUT for v in ident["values"]):
            return ident
    raise RuntimeError("no identity without a held-out format")


def third_party(rng: random.Random) -> Tuple[str, List[str]]:
    c = ids.Ctx(rng, False, "train")
    name = c.first if rng.random() < .3 else f"{c.first} {c.last}"
    return name, sorted(c.pool_tokens)


# ── hard negatives ───────────────────────────────────────────────────────────

def _train_only(name: str) -> bool:
    return ids.in_pool(ids.tokens(name), "train")


_PARTICLES = {"of", "the", "de", "da", "do", "dos", "das", "di", "del", "della", "du", "la", "le", "van", "von", "der",
              "den", "ten", "ter", "bin", "ibn", "al", "el", "y", "e", "und", "zu"}
_ROMAN = re.compile(r"(?i)^(?:i{1,3}|iv|v|vi{1,3}|ix|x{1,3}i{0,3}v?)$")


def _title(w: str, i: int) -> str:
    if i and w in _PARTICLES:
        return w
    if i and _ROMAN.fullmatch(w):
        return w.upper()
    return re.sub(r"(^|[\-'’.])([a-z])", lambda m: m.group(1) + m.group(2).upper(), w)


@lru_cache(maxsize=None)
def public_people() -> Tuple[str, ...]:
    """Widely known people (Wikidata, CC0), title-cased, every pool word of
    them in the training half (so no evaluation-pool word is in training)."""
    out = []
    for line in PUBLIC_PEOPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## surnames"):
            break
        if not line or line.startswith("#"):
            continue
        name = " ".join(_title(w, i) for i, w in enumerate(line.split()))
        if _train_only(name):
            out.append(name)
    return tuple(out)


@lru_cache(maxsize=None)
def places() -> Tuple[str, ...]:
    cities = {c for v in ids.COUNTRIES.values() for c, _r in v[2]} | {c for v in ids.EXTRA_CITIES.values() for c, _r in v}
    bad = [p for p in bank.PLACES if p in cities]
    if bad:
        raise ValueError(f"topic places that a value could be: {bad}")
    return tuple(p for p in bank.PLACES if _train_only(p))


@lru_cache(maxsize=None)
def fiction_names() -> Tuple[str, ...]:
    """Characters' names: first names from the training half, alone or with a title."""
    rng = random.Random(7)
    seen = set()
    for _ in range(4000):
        c = ids.Ctx(rng, False, "train")
        seen.add(c.first)
    return tuple(sorted(seen))


SLOT = re.compile(r"\{(PUBLIC|BRAND|PLACE|FICTION|IDENT|LIB|ACRO|NUM|YEAR|DATE|MONEY|VERSION|LANG|MONTH)\}")


def negative(rng: random.Random, slot: str) -> str:
    if slot == "PUBLIC":
        return rng.choice(public_people())
    if slot == "BRAND":
        return rng.choice(bank.BRANDS)
    if slot == "PLACE":
        return rng.choice(places())
    if slot == "FICTION":
        name = rng.choice(fiction_names())
        return rng.choice((name, name, name, f"Captain {name}", f"Lady {name}", f"Sir {name}", f"Dr. {name}"))
    if slot == "IDENT":
        return rng.choice(bank.IDENTS)
    if slot == "LIB":
        return rng.choice(bank.LIBRARIES)
    if slot == "ACRO":
        return rng.choice(bank.ACRONYMS)
    if slot == "NUM":
        return str(rng.choice((rng.randint(2, 99), rng.randint(2, 20), rng.randint(100, 5000))))
    if slot == "YEAR":
        return str(rng.randint(1900, 2027))
    if slot == "DATE":
        y, m, d = rng.randint(1900, 2027), rng.randint(1, 12), rng.randint(1, 28)
        return rng.choice((f"{m:02d}/{d:02d}/{y}", f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}",
                           f"{bank.MONTH_NAMES[m - 1]} {d}, {y}", f"{d} {bank.MONTH_NAMES[m - 1]} {y}",
                           f"{bank.MONTH_NAMES[m - 1]} {d}", f"the {ids.ORDINALS[d - 1]} of {bank.MONTH_NAMES[m - 1]}"))
    if slot == "MONEY":
        n = rng.choice((rng.randint(5, 99), rng.randint(100, 9999), rng.randint(10, 500) * 100))
        return rng.choice((f"${n:,}", f"£{n:,}", f"€{n:,}", f"{n} dollars", f"{n} EUR", f"USD {n:,}"))
    if slot == "VERSION":
        return rng.choice((f"{rng.randint(1, 4)}.{rng.randint(0, 12)}", f"{rng.randint(0, 3)}.{rng.randint(0, 20)}"
                           f".{rng.randint(0, 9)}", f"v{rng.randint(1, 22)}"))
    if slot == "LANG":
        return rng.choice(bank.LANGUAGES)
    if slot == "MONTH":
        return rng.choice(bank.MONTH_NAMES)
    raise KeyError(slot)


def fill(rng: random.Random, template: str) -> List[Piece]:
    """*template* with every negative slot drawn (kind ``n``, tag O)."""
    out, pos = [], 0
    for m in SLOT.finditer(template):
        if m.start() > pos:
            out.append(Piece(template[pos:m.start()]))
        out.append(Piece(negative(rng, m.group(1)), None, "n"))
        pos = m.end()
    if pos < len(template):
        out.append(Piece(template[pos:]))
    return out


# ── placements ───────────────────────────────────────────────────────────────

ID_NAMES = {"ssn": ("SSN", "social security number", "SSN#"), "ssn-nodash": ("SSN", "social security number"),
            "card": ("card number", "credit card number", "debit card"), "card-spaced": ("card number", "card"),
            "iban": ("IBAN", "bank account (IBAN)", "account IBAN"), "iban-spaced": ("IBAN", "bank account"),
            "passport": ("passport number", "passport no."), "nino": ("National Insurance number", "NI number"),
            "nhs": ("NHS number",), "policy": ("policy number", "member ID", "insurance ID"),
            "license": ("driver's licence number", "license number", "DL number"),
            "student": ("student ID", "employee ID", "staff number")}
CRED_NAMES = {"password": ("password", "passcode", "login password"), "hex": ("API key", "secret key", "token"),
              "zqk": ("API key", "access token", "token"), "secret": ("secret", "recovery code", "app password")}
NET_NAMES = {"ipv4": ("IP", "IP address", "server IP"), "ipv6": ("IPv6 address", "IP"), "mac": ("MAC address",)}
KIND_TYPES = ("ID", "CREDENTIAL", "NETWORK")
GENERIC_KEYS = {"ID": ("ID", "ID number", "Reference", "Account number", "Customer ID", "Number", "Ref"),
                "CREDENTIAL": ("Secret", "Key", "Token", "Credentials"),
                "NETWORK": ("Address", "Host", "Device")}
KIND_TEMPLATES = ["My {name} is {v}.", "{Name}: {v}", "Here's my {name}: {v}", "my {name} is {v}",
                  "The {name} is {v}.", "{Name} - {v}", "I'll give you my {name}: {v}."]


def kind_name(v: dict, rng: random.Random) -> Optional[str]:
    table = {"ID": ID_NAMES, "CREDENTIAL": CRED_NAMES, "NETWORK": NET_NAMES}.get(v["type"])
    return rng.choice(table[v["fmt"]]) if table and v["fmt"] in table else None


def own_sentence(rng: random.Random, v: dict, country: str) -> List[Piece]:
    name = kind_name(v, rng)
    if name and rng.random() < .6:
        t = rng.choice(KIND_TEMPLATES).replace("{name}", name).replace("{Name}", name[:1].upper() + name[1:])
    else:
        t = rng.choice(bank.OWN[v["type"]])
    a, b = t.split("{v}", 1)
    return [Piece(a)] + value_pieces(v, country) + [Piece(b)]


def closed_sentence(rng: random.Random, v: dict, country: str) -> List[Piece]:
    """A placement that does not run on (for after the body, or before another sentence)."""
    for _ in range(20):
        s = own_sentence(rng, v, country)
        if not ends_open(s):
            return s
    return s + [Piece(".")]


def ends_open(pieces: Sequence[Piece]) -> bool:
    """A placement that runs on into the next sentence ("As a 34-year-old,")."""
    tail = "".join(p.text for p in pieces).rstrip()
    return bool(tail) and pieces[-1].tag is None and tail[-1] not in ".!?):" and not tail.endswith("}")


def lower_first(pieces: List[Piece]) -> List[Piece]:
    if pieces and pieces[0].kind == "c" and pieces[0].tag is None:
        t = pieces[0].text
        m = re.match(r"\s*([A-Z])([a-z'’]*)", t)
        if m and m.group(0).strip() not in ("I", "I'm", "I've", "I'd", "I'll"):
            pieces[0] = Piece(t[:m.start(1)] + m.group(1).lower() + t[m.end(1):])
    return pieces


def sep(rng: random.Random) -> str:
    return rng.choice((" ", " ", " ", "\n", "\n\n", "  "))


def form_key(rng: random.Random, v: dict) -> str:
    """A field name for *v*: an ID, credential or network value is named by its kind ("IBAN") or generically."""
    if v["type"] in KIND_TYPES:
        name = kind_name(v, rng)
        if name is None or rng.random() < .3:
            return rng.choice(GENERIC_KEYS[v["type"]])
        return name[:1].upper() + name[1:]
    return rng.choice(bank.FORM_KEYS[v["type"]])


def json_key(rng: random.Random, v: dict) -> str:
    if v["type"] in KIND_TYPES:
        name = kind_name(v, rng)
        if name is None or rng.random() < .3:
            name = rng.choice(GENERIC_KEYS[v["type"]])
        name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return rng.choice((name, re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)))
    return rng.choice(bank.JSON_KEYS[v["type"]])


def field_line(rng: random.Random, v: dict, country: str, key: str, style: int) -> List[Piece]:
    vp = value_pieces(v, country)
    if style == 0:
        return [Piece(f"{key}: ")] + vp
    if style == 1:
        return [Piece(f"- {key}: ")] + vp
    if style == 2:
        return [Piece(f"**{key}:** ")] + vp
    if style == 3:
        return [Piece(f"{key} - ")] + vp
    return [Piece(f"{key}:\t")] + vp


def form_block(rng: random.Random, values: Sequence[dict], country: str) -> List[Piece]:
    style = rng.randrange(5)
    out: List[Piece] = []
    if rng.random() < .4:
        out.append(Piece(rng.choice(("My details:\n", "Details:\n", "Here is the form:\n", "Applicant info\n",
                                     "Customer details\n", "Please fill this in for me:\n", "Info:\n"))))
    for i, v in enumerate(values):
        key = form_key(rng, v)
        out += field_line(rng, v, country, key, style)
        if i < len(values) - 1:
            out.append(Piece("\n"))
    return out


def json_block(rng: random.Random, values: Sequence[dict], country: str) -> List[Piece]:
    yaml = rng.random() < .3
    out: List[Piece] = [Piece("" if yaml else "{\n")]
    for i, v in enumerate(values):
        key = json_key(rng, v)
        if yaml:
            out += [Piece(f"{key}: ")] + value_pieces(v, country) + [Piece("\n")]
        else:
            out += [Piece(f'  "{key}": "')] + value_pieces(v, country) + [Piece('"' + ("," if i < len(values) - 1 else "") + "\n")]
    out.append(Piece("" if yaml else "}"))
    wrap = rng.choice(bank.CODE_WRAPPERS[:4] if not yaml else ("```yaml\n{block}\n```", "{block}"))
    a, b = wrap.split("{block}")
    return [Piece(a)] + out + [Piece(b)]


def env_block(rng: random.Random, values: Sequence[dict], country: str) -> List[Piece]:
    style = rng.randrange(3)
    out: List[Piece] = []
    for v in values:
        keys = bank.ENV_KEYS.get(v["type"]) or [k.upper() for k in bank.JSON_KEYS[v["type"]]]
        key = rng.choice(keys)
        if style == 0:
            out += [Piece(f"{key}=")] + value_pieces(v, country) + [Piece("\n")]
        elif style == 1:
            out += [Piece(f'{key.lower()} = "')] + value_pieces(v, country) + [Piece('"\n')]
        else:
            out += [Piece(f'const {key.lower()} = \'')] + value_pieces(v, country) + [Piece("';\n")]
    wrap = rng.choice(("```\n{block}```", "```python\n{block}```", "```js\n{block}```", "{block}", "```bash\n{block}```"))
    a, b = wrap.split("{block}")
    return [Piece(a)] + out + [Piece(b)]


def table_block(rng: random.Random, values: Sequence[dict], country: str) -> List[Piece]:
    keys = [form_key(rng, v) for v in values]
    out = [Piece("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n| ")]
    for i, v in enumerate(values):
        out += value_pieces(v, country) + [Piece(" | " if i < len(values) - 1 else " |")]
    return out


SIG_PREFIX = {"PHONE": ("", "Tel: ", "M: ", "Mobile: ", "Phone: ", "T "), "EMAIL": ("", "E: ", "Email: "),
              "URL": ("", "Web: ", "W: "), "HANDLE": ("", "Twitter: "), "ADDRESS": ("",), "ORG": ("",),
              "PERSON": ("",), "LOCATION": ("",), "AGE": ("", "Age: ", "age ")}
SIG_ORDER = ("PERSON", "AGE", "ORG", "ADDRESS", "LOCATION", "PHONE", "EMAIL", "URL", "HANDLE")


def signature_block(rng: random.Random, values: Sequence[dict], country: str) -> Tuple[List[Piece], List[dict]]:
    """A sign-off and the values a signature holds, one per line; returns the rest."""
    inside = sorted((v for v in values if v["type"] in SIG_ORDER), key=lambda v: SIG_ORDER.index(v["type"]))
    rest = [v for v in values if v not in inside]
    out = [Piece(rng.choice(bank.SIGN_OFFS) + "\n")]
    for i, v in enumerate(inside):
        title, prefix = "", rng.choice(SIG_PREFIX[v["type"]])
        if v["type"] == "ORG" and rng.random() < .5:
            title = rng.choice(("Senior Analyst, ", "Project Manager | ", "Office Manager at ", "Student, ",
                                "Founder, ", "Head of Sales — "))
        if v["type"] == "AGE" and not prefix:
            if i and inside[i - 1]["type"] == "PERSON":          # on the name's line: "Ana Ruiz, 34", "Ana Ruiz (34)"
                join = rng.choice((", ", ", ", " ("))
                out[-1] = Piece(join)                             # the separator after the name
                out += value_pieces(v, country) + ([Piece(")")] if join == " (" else [])
                out += [Piece("\n")] if i < len(inside) - 1 else []
                continue
            prefix = "Age: "                                      # a bare number on its own line is not an age
        out += [Piece(title + prefix)] + value_pieces(v, country)
        if i < len(inside) - 1:
            out.append(Piece(rng.choice(("\n", "\n", " | ")) if v["type"] in ("PHONE", "EMAIL", "URL") else "\n"))
    return out, rest


# ── ages of named people (V4 §3.1 D) ─────────────────────────────────────────

def _age_value(rng: random.Random, lo: int = 18, hi: int = 89, worded: bool = False) -> dict:
    n = rng.randint(lo, hi)
    return {"value": ids.number_words(n) if worded else str(n), "type": "AGE", "fmt": "words" if worded else "plain"}


def _slots(rng: random.Random, template: str, people: Sequence[ids.Ctx], ages: Sequence[dict],
           kin: Sequence[str] = bank.RELATIONS) -> List[Piece]:
    """*template* (bank.AGE_*) as pieces: names PERSON, cities LOCATION, ages AGE, negative slots O;
    a relation ``{r}`` comes from *kin*."""
    out: List[Piece] = []
    for part in re.split(r"(\{(?:n2?|f2?|c|v2?|r|k)\})", template):
        if not part:
            continue
        key = part[1:-1] if part.startswith("{") and part.endswith("}") else None
        if key is None:
            out += fill(rng, part)
            continue
        p = people[1] if key.endswith("2") and key != "v2" else people[0]
        if key in ("n", "n2"):
            out.append(Piece(f"{p.first} {p.last}", "PERSON", "v"))
        elif key in ("f", "f2"):
            out.append(Piece(p.first, "PERSON", "v"))
        elif key == "c":
            out.append(Piece(ids.city(p)[0], "LOCATION", "v"))
        elif key in ("v", "v2"):
            out.append(Piece(ages[key == "v2"]["value"], "AGE", "v"))
        elif key == "r":
            out.append(Piece(rng.choice(kin)))
        else:                                                      # {k}: a number that is not an age
            out.append(Piece(str(rng.choice((rng.randint(2, 99), rng.randint(10, 99))))))
    return out


def age_block(rng: random.Random) -> Tuple[List[Piece], List[str], List[dict], str]:
    """One "age" layout (``AGE_LAYOUTS``): its pieces, the pool tokens of its
    names, its ages, and which form it took. Names and cities belong to new
    training-half people (``identities.Ctx(pool="train")``)."""
    kind = rng.choices(list(AGE_LAYOUTS), weights=list(AGE_LAYOUTS.values()))[0]
    people = [ids.Ctx(rng, False, "train"), ids.Ctx(rng, False, "train")]
    table = {"line": bank.AGE_LINES, "mid": bank.AGE_MID, "bracket": bank.AGE_BRACKET, "from": bank.AGE_FROM,
             "reddit": bank.AGE_REDDIT, "worded": bank.AGE_WORDED}.get(kind)
    t = rng.choice(table) if table else ""
    who, key = rng.choice(bank.AGE_FORM) if kind == "form" else ("", "")
    child = (t or who).startswith(bank.AGE_CHILDREN) or ("{f}" in t and rng.random() < .6)  # named by first name
    lo, hi = (2, 17) if child else (18, 89)
    ages = [_age_value(rng, lo, hi, worded=kind == "worded"), _age_value(rng, lo, hi)]
    if kind == "signoff":
        pieces = [Piece(rng.choice(bank.SIGN_OFFS) + "\n"), Piece(f"{people[0].first} {people[0].last}", "PERSON", "v"),
                  Piece(", "), Piece(ages[0]["value"], "AGE", "v")]
        r = rng.random()
        if r < .1:
            pieces.append(Piece("."))
        elif r < .25:
            pieces += [Piece("\n"), Piece(ids.city(people[0])[0], "LOCATION", "v")]
        elif r < .32:
            pieces.append(Piece("\nSent from my phone"))
    elif kind == "form":
        sep_ = rng.choice((": ", " - ", ":\t"))
        pieces = [Piece(who + sep_), Piece(f"{people[0].first} {people[0].last}", "PERSON", "v"),
                  Piece(rng.choice(("\n", "\n", " | ", ", ")) + key + sep_), Piece(ages[0]["value"], "AGE", "v")]
    else:
        pieces = _slots(rng, t, people, ages, bank.CHILD_KIN if child else bank.RELATIONS)
    named = [people[0]] if kind in ("signoff", "form") or re.search(r"\{[nfc]\}", t) else []
    named += [people[1]] if re.search(r"\{[nf]2\}", t) else []
    toks = sorted({tok for c in named for tok in c.pool_tokens})
    return pieces, toks, ages[:2 if "{v2}" in t else 1], kind


def age_near(rng: random.Random) -> Tuple[List[Piece], List[str]]:
    """A line in an "age" shape whose number is not an age (bank.AGE_NEAR; numbers O)."""
    p = ids.Ctx(rng, False, "train")
    t = rng.choice(bank.AGE_NEAR)
    pieces = _slots(rng, t, [p, p], [])
    return pieces, sorted(p.pool_tokens) if "{n}" in t else []


# ── one message ──────────────────────────────────────────────────────────────

def choose_types(rng: random.Random, task: str) -> List[str]:
    pri = ids.PRIORS.get(task, ids.BASE)
    chosen = [t for t in TYPES if rng.random() < pri[t] * .8]
    for a, b in ids.EXCLUSIVE:
        if a in chosen and b in chosen:
            chosen.remove(rng.choice((a, b)))
    rng.shuffle(chosen)
    return chosen[:rng.choice((1, 1, 2, 2, 3, 3, 4, 5))] or [rng.choice(TYPES)]


def body(rng: random.Random, task: str, negatives_only: bool = False) -> List[Piece]:
    if negatives_only and rng.random() < .5:
        return fill(rng, rng.choice(bank.NEGATIVE_ONLY))
    return fill(rng, rng.choice(bank.BODIES[task]))


def message(rng: random.Random, i: int) -> dict:
    task = rng.choice(TASKS)
    layout = rng.choices(list(LAYOUTS), weights=list(LAYOUTS.values()))[0]
    aug: List[str] = []
    pool_tokens: set = set()
    values: List[dict] = []
    country = "US"
    if layout not in ("none", "age") or (layout == "age" and rng.random() < .5):
        types = choose_types(rng, task)
        if layout == "env":
            types = [t for t in types if t in bank.ENV_KEYS] or ["CREDENTIAL"]
        ident = draw_identity(rng, types, task)
        values = list(ident["values"])
        country = ident["country"]
        pool_tokens.update(ident["pool_tokens"])
        rng.shuffle(values)
    main = body(rng, task, negatives_only=not values)
    head: List[Piece] = []
    tail: List[Piece] = []
    ending: List[Piece] = []          # an "age" sign-off or line that closes the message
    extra: dict = {}
    if rng.random() < AUG["greeting"]:
        head.append(Piece(rng.choice(bank.GREETINGS) + rng.choice((",", "!", "", ".")) + sep(rng)))
        aug.append("greeting")

    if layout in ("intro", "inline", "story"):
        k = len(values) if layout != "inline" else rng.randint(0, len(values))
        intro, later = values[:k] if layout == "intro" else [], values if layout != "intro" else values[k:]
        if layout == "inline":
            intro, later = values[:k], values[k:]
        last: List[Piece] = []
        for v in intro:
            last = own_sentence(rng, v, country)
            head += last + [Piece(" ")]
        if last and ends_open(last):
            main = lower_first(main)
        elif head:
            head[-1] = Piece(sep(rng))
        for v in later:
            tail += [Piece(sep(rng))] + closed_sentence(rng, v, country)
    elif layout == "signature":
        sig, rest = signature_block(rng, values, country)
        for v in rest:
            head += closed_sentence(rng, v, country) + [Piece(" ")]
        tail += [Piece(rng.choice(("\n\n", "\n", "\n\n")))] + sig
    elif layout in ("form", "json", "env", "table"):
        block = {"form": form_block, "json": json_block, "env": env_block, "table": table_block}[layout](
            rng, values, country)
        if rng.random() < .3:
            head += block + [Piece("\n\n")]
        else:
            tail += [Piece(rng.choice(("\n\n", "\n", "\n\n")))] + block
    elif layout == "age":
        block, toks, ages, kind = age_block(rng)
        pool_tokens.update(toks)
        extra = {"age_form": kind, "age_values": [["AGE", a["fmt"]] for a in ages]}
        if kind == "signoff":
            values = [v for v in values if v["type"] != "PERSON"]
        for v in values:
            head += closed_sentence(rng, v, country) + [Piece(" ")]
        if kind != "signoff" and rng.random() < .4:                  # opens the message
            head = block + [Piece(rng.choice(("\n", "\n\n") if kind in ("line", "form", "from") else ("\n", " ")))] + head
        elif kind in ("signoff", "line", "form", "from"):
            ending = [Piece(rng.choice(("\n\n", "\n")))] + block
        else:
            tail += [Piece(sep(rng))] + block
        if rng.random() < AGE_NEAR:
            near, ntoks = age_near(rng)
            pool_tokens.update(ntoks)
            head = near + [Piece("\n")] + head
            extra["age_near"] = True

    if rng.random() < AUG["third_party"]:
        name, toks = third_party(rng)
        pool_tokens.update(toks)
        t = rng.choice(bank.OTHER_PERSON).replace("{rel}", rng.choice(bank.RELATIONS))
        a, b = t.split("{v}", 1)
        piece = [Piece(a), Piece(name, "PERSON", "v"), Piece(b)]
        if t.startswith(("Dear", "Hi {v}")):
            head = piece + [Piece("\n")] + head
        else:
            tail += [Piece(sep(rng))] + piece
        aug.append("third_party")
    if rng.random() < AUG["negatives"]:
        neg = fill(rng, rng.choice(bank.NEGATIVE_ONLY))
        if rng.random() < .5:
            tail += [Piece(sep(rng))] + neg
        else:
            main = neg + [Piece(sep(rng))] + main
        aug.append("negatives")
    if layout != "signature" and not ending and rng.random() < AUG["closing"]:
        tail.append(Piece(sep(rng) + rng.choice(bank.CLOSINGS)))
        aug.append("closing")
    pieces = head + main + tail + ending
    pieces = augment(rng, pieces, aug)
    text, spans = join(pieces)
    text, spans = _strip(text, spans)
    return {"id": f"tg-{i:06d}", "text": text, "spans": spans,
            "meta": {"task": task, "layout": layout, "aug": aug, "country": country,
                     "values": [[v["type"], v["fmt"]] for v in values] + extra.pop("age_values", []),
                     "pool_tokens": sorted(pool_tokens), **extra}}


def _strip(text: str, spans: List[List]) -> Tuple[str, List[List]]:
    lead = len(text) - len(text.lstrip())
    text2 = text.strip()
    return text2, [[s - lead, e - lead, t] for s, e, t in spans if s - lead >= 0 and e - lead <= len(text2)]


# ── augmentations ────────────────────────────────────────────────────────────

_KEYS = "qwertyuiopasdfghjklzxcvbnm"


def typo(rng: random.Random, word: str) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    k = rng.randrange(4)
    if k == 0:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if k == 1:
        return word[:i] + word[i + 1:]
    if k == 2:
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice(_KEYS) + word[i + 1:]


def augment(rng: random.Random, pieces: List[Piece], aug: List[str]) -> List[Piece]:
    if rng.random() < AUG["typos"]:
        aug.append("typos")
        carrier = [p for p in pieces if p.kind == "c" and p.tag is None and re.search(r"[a-z]{4}", p.text)]
        for p in rng.sample(carrier, min(len(carrier), rng.randint(1, 3))):
            words = re.findall(r"[A-Za-z]{4,}", p.text)
            if words:
                w = rng.choice(words)
                p.text = p.text.replace(w, typo(rng, w), 1)
    if rng.random() < AUG["nospace"]:
        aug.append("nospace")
        for p in pieces:
            if p.kind == "c" and p.tag is None:
                p.text = re.sub(r"([.,!?]) (?=[A-Za-z])", lambda m: m.group(1) if rng.random() < .7 else m.group(0),
                                p.text)
    if rng.random() < AUG["markdown"]:
        vals = [i for i, p in enumerate(pieces) if p.tag is not None and p.tag not in ADDRESS_TAGS]
        if vals:
            aug.append("markdown")
            i = rng.choice(vals)
            mark = rng.choice(("`", "**", "*", "_"))
            pieces = pieces[:i] + [Piece(mark)] + [pieces[i]] + [Piece(mark)] + pieces[i + 1:]
    if rng.random() < AUG["linebreak"]:
        vals = [p for p in pieces if p.tag in ("PERSON", "ORG", "STREET") and " " in p.text.strip()]
        if vals:
            aug.append("linebreak")
            p = rng.choice(vals)
            cut = [m.start() for m in re.finditer(" ", p.text)]
            j = rng.choice(cut)
            p.text = p.text[:j] + "\n" + p.text[j + 1:]
    r = rng.random()
    if r < AUG["lower"]:
        aug.append("lower")
        for p in pieces:
            low = p.text.lower()
            if len(low) == len(p.text):
                p.text = low
    elif r < AUG["lower"] + AUG["upper"]:
        aug.append("upper")
        for p in pieces:
            up = p.text.upper()
            if len(up) == len(p.text):
                p.text = up
    return pieces


# ── corpus ───────────────────────────────────────────────────────────────────

def build(n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    return [message(rng, i) for i in range(n)]


def data_hash(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["text"], r["spans"]], ensure_ascii=False).encode())
    return h.hexdigest()


def stats(rows: Sequence[dict]) -> dict:
    tags = Counter(t for r in rows for _s, _e, t in r["spans"])
    return {"messages": len(rows), "with_values": sum(bool(r["spans"]) for r in rows),
            "spans_by_tag": dict(sorted(tags.items())),
            "layouts": dict(sorted(Counter(r["meta"]["layout"] for r in rows).items())),
            "tasks": dict(sorted(Counter(r["meta"]["task"] for r in rows).items())),
            "augmentations": dict(sorted(Counter(a for r in rows for a in r["meta"]["aug"]).items())),
            "formats": dict(sorted(Counter(f"{t}:{f}" for r in rows for t, f in r["meta"]["values"]).items())),
            "pool_tokens": len({t for r in rows for t in r["meta"]["pool_tokens"]})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = build(a.n, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"version": VERSION, "n": a.n, "seed": a.seed, "data_sha256": data_hash(rows), "held_out": HELD_OUT,
            "layouts": LAYOUTS, "aug": AUG, "shift": SHIFT, "labels": LABELS, **stats(rows)}
    Path(str(a.out) + ".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in ("n", "data_sha256", "with_values", "spans_by_tag")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
