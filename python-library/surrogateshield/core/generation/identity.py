"""
generation/identity.py — one surrogate person per real person (audit D1, D2,
D4, I7).

Every given name and surname a conversation uses maps to ONE surrogate token,
so "Sarah Mitchell", "Sarah", "Ms. Mitchell", "SARAH MITCHELL" and
``sarah.mitchell@gmail.com`` all become the same surrogate person, in this
message and in later ones. The surrogate keeps what an answer may depend on
and nothing that identifies:

* gender — the first name comes from the same gendered list as the original's
  (Faker's per-locale lists); a title ("Mr.", "Ms.") decides when the name is
  unisex or unknown;
* locale and script — "Rahul Verma" gets an Indian name, "李娜" a Chinese one,
  "Zhou Yan" a romanised Chinese one in the same family-name-first order;
* form — titles, suffixes, particles, initials, hyphens, case and the number
  of name words are kept; no title or suffix is ever added;
* e-mail — the local part is rebuilt from the linked surrogate in the
  original's style (``sarah.mitchell`` → ``mia.lopez``, ``smitchell`` →
  ``mlopez``); free-mail domains are kept, other domains get one fake domain
  per conversation with the same suffix.

Surrogate tokens are never ordinary English words ("Grace", "Brown"), never
shorter than three letters (Latin script), and never a word any original of
the conversation uses. All draws use the caller's seeded RNG.
"""

from __future__ import annotations

import importlib
import re
import string
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..consistency import COMMON_WORD_NAMES

# ── name lists ───────────────────────────────────────────────────────────────

_LATIN_LOCALES = ("en_US", "en_GB", "en_IN", "es_ES", "es_MX", "fr_FR", "de_DE", "it_IT",
                  "pt_BR", "nl_NL", "sv_SE", "tr_TR", "yo_NG", "ig_NG", "sw", "tw_GH")
_SCRIPT_LOCALES = {"Cyrl": ("ru_RU",), "Grek": ("el_GR",), "Deva": ("hi_IN",),
                   "Hani": ("zh_CN", "ja_JP"), "Kana": ("ja_JP",), "Hang": ("ko_KR",),
                   "Arab": ("ar_AA",), "Hebr": ("he_IL",), "Thai": ("th_TH",)}
# Romanised Chinese / Japanese names, often written family name first.
_ROMANISED = ("zh_Latn", "ja_Latn")
_FAMILY_FIRST = frozenset({"zh_CN", "ja_JP", "ko_KR", "zh_Latn", "ja_Latn"})
_DEFAULT_LOCALE = "en_US"

TITLES = {"mr": "m", "mrs": "f", "ms": "f", "miss": "f", "mx": None, "dr": None,
          "prof": None, "professor": None, "sir": "m", "madam": "f", "dame": "f",
          "lady": "f", "lord": "m", "rev": None, "fr": None, "mme": "f", "mlle": "f",
          "herr": "m", "frau": "f", "sr": None, "sra": "f", "srta": "f"}
SUFFIXES = frozenset("jr sr ii iii iv md phd dds dvm esq rn cpa".split())
_PARTICLES = frozenset("van von de der den da di du del della la le bin binti bint al el ben ibn "
                       "dos das do ter ten zu af".split())
_GENDER_PRONOUNS = {"m": re.compile(r"(?i)\b(?:he|him|his|himself)\b"),
                    "f": re.compile(r"(?i)\b(?:she|her|hers|herself)\b")}
_LATIN_NAME = re.compile(r"[^\W\d_]{3,}(?:['’][^\W\d_]+)?")


def _seq(value) -> Tuple[Tuple[str, ...], Optional[Tuple[float, ...]]]:
    if value is None:
        return (), None
    if isinstance(value, dict):
        return tuple(value.keys()), tuple(float(w) for w in value.values())
    return tuple(value), None


def _usable(name: str, latin: bool) -> bool:
    if not name or any(c.isspace() for c in name):
        return False
    if not latin:
        return True
    return bool(_LATIN_NAME.fullmatch(name)) and name.casefold() not in COMMON_WORD_NAMES


@dataclass(frozen=True)
class _List:
    names: Tuple[str, ...]
    weights: Optional[Tuple[float, ...]]

    @staticmethod
    def make(value, latin: bool, exclude: frozenset = frozenset()) -> "_List":
        """Usable names of *value*; names in *exclude* (casefold) are dropped
        unless that would leave fewer than ten."""
        names, weights = _seq(value)
        keep = [i for i, n in enumerate(names) if _usable(n, latin)]
        only = [i for i in keep if names[i].casefold() not in exclude]
        if len(only) >= 10:
            keep = only
        return _List(tuple(names[i] for i in keep),
                     tuple(weights[i] for i in keep) if weights else None)

    def where(self, keep) -> "_List":
        """The names for which *keep(casefold name)* holds, if ten or more do."""
        idx = [i for i, n in enumerate(self.names) if keep(n.casefold())]
        if len(idx) < 10:
            return self
        return _List(tuple(self.names[i] for i in idx),
                     tuple(self.weights[i] for i in idx) if self.weights else None)

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class _Pool:
    locale: str
    latin: bool
    given: Dict[str, _List]          # "m", "f", "x" (unisex or unknown)
    family: Dict[str, _List]         # "x", plus "m"/"f" where surnames are gendered
    gender_of: Dict[str, str]        # casefold given name → "m" / "f" / "x"
    family_set: frozenset            # casefold surnames (unfiltered)

    def given_for(self, gender: str) -> _List:
        return self.given.get(gender) or self.given["x"]

    def family_for(self, gender: str) -> _List:
        return self.family.get(gender) or self.family["x"]


def _provider(locale: str):
    return importlib.import_module(f"faker.providers.person.{locale}").Provider


def _make_pool(locale: str, male, female, last, last_m=None, last_f=None, latin=True) -> _Pool:
    m_names, _ = _seq(male)
    f_names, _ = _seq(female)
    m_set = {n.casefold() for n in m_names}
    f_set = {n.casefold() for n in f_names}
    gender_of = {n: ("x" if n in f_set else "m") for n in m_set}
    gender_of.update({n: "f" for n in f_set - m_set})
    # a gendered draw uses names of that gender only ("Ashley" is in both)
    m_list = _List.make(male, latin, frozenset(f_set))
    f_list = _List.make(female, latin, frozenset(m_set))
    both = [n for n in _List.make(male, latin).names if n.casefold() in f_set]
    x_list = _List(tuple(both), None) if len(both) >= 10 else _List(
        m_list.names + f_list.names, None)
    family = {"x": _List.make(last, latin)}
    if last_m is not None and last_f is not None:
        family["m"], family["f"] = _List.make(last_m, latin), _List.make(last_f, latin)
    family_set = frozenset(n.casefold() for src in (last, last_m, last_f) for n in _seq(src)[0])
    return _Pool(locale, latin, {"m": m_list, "f": f_list, "x": x_list}, family, gender_of,
                 family_set)


@lru_cache(maxsize=1)
def _pools() -> Dict[str, _Pool]:
    pools: Dict[str, _Pool] = {}
    for loc in _LATIN_LOCALES:
        P = _provider(loc)
        pools[loc] = _make_pool(loc, P.first_names_male, P.first_names_female, P.last_names,
                                getattr(P, "last_names_male", None),
                                getattr(P, "last_names_female", None))
    zh_tw, zh_cn, ja = _provider("zh_TW"), _provider("zh_CN"), _provider("ja_JP")
    # pinyin given names are not gendered in Faker's lists
    pools["zh_Latn"] = _make_pool(
        "zh_Latn", zh_cn.first_romanized_names, zh_cn.first_romanized_names,
        tuple(dict.fromkeys(tuple(zh_cn.last_romanized_names) + tuple(zh_tw.last_romanized_names))))
    pools["ja_Latn"] = _make_pool("ja_Latn", ja.first_romanized_names_male,
                                  ja.first_romanized_names_female, ja.last_romanized_names)
    for locs in _SCRIPT_LOCALES.values():
        for loc in locs:
            if loc in pools:
                continue
            P = _provider(loc)
            pools[loc] = _make_pool(loc, P.first_names_male, P.first_names_female, P.last_names,
                                    getattr(P, "last_names_male", None),
                                    getattr(P, "last_names_female", None), latin=False)
    # a gendered draw uses only names of that gender in every locale
    # (en_GB lists "Ashley" as male, en_US as female)
    genders: Dict[str, set] = {}
    for pool in pools.values():
        for cf, g in pool.gender_of.items():
            genders.setdefault(cf, set()).add(g)
    for pool in pools.values():
        for g in ("m", "f"):
            pool.given[g] = pool.given[g].where(lambda cf, g=g: genders.get(cf) == {g})
    return pools


# ── scripts, locale and gender ───────────────────────────────────────────────

def _script(text: str) -> str:
    han = False
    for c in text:
        if not c.isalpha():
            continue
        cp = ord(c)
        if 0x0400 <= cp <= 0x04FF:
            return "Cyrl"
        if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
            return "Grek"
        if 0x0900 <= cp <= 0x097F:
            return "Deva"
        if 0x3040 <= cp <= 0x30FF:
            return "Kana"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            return "Hang"
        if 0x0600 <= cp <= 0x06FF:
            return "Arab"
        if 0x0590 <= cp <= 0x05FF:
            return "Hebr"
        if 0x0E00 <= cp <= 0x0E7F:
            return "Thai"
        if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
            han = True
    return "Hani" if han else "Latn"


def _choose(given: Sequence[str], family: Sequence[str], script: str,
            east_ok: bool) -> Tuple[str, bool]:
    """(locale, family_first) that best explains the name words. *given* and
    *family* are the casefold words in western order (first … last)."""
    pools = _pools()
    if script != "Latn":
        cands = _SCRIPT_LOCALES[script]
        if len(cands) == 1:
            return cands[0], cands[0] in _FAMILY_FIRST
        words = list(given) + list(family)
        best = max(cands, key=lambda L: sum(w in pools[L].family_set or w in pools[L].gender_of
                                            for w in words))
        return best, True
    best, best_score, east = _DEFAULT_LOCALE, 0, False
    for loc in _LATIN_LOCALES + _ROMANISED:
        p = pools[loc]
        orders = [(given, family, False)]
        if east_ok and loc in _FAMILY_FIRST and given and family:
            orders.append((family, given, True))      # "Zhou Yan": Zhou is the family name
        for g, f, is_east in orders:
            score = 2 * any(w in p.gender_of for w in g) + 2 * any(w in p.family_set for w in f)
            if score > best_score:
                best, best_score, east = loc, score, is_east
    return best, east


def given_gender(word: str, locale: Optional[str] = None) -> Optional[str]:
    """"m" / "f" for a given name found in only male or only female lists,
    "x" for a unisex one, None when unknown."""
    cf = word.casefold().rstrip(".")
    pools = _pools()
    if locale in pools and cf in pools[locale].gender_of:
        return pools[locale].gender_of[cf]
    seen = {p.gender_of[cf] for p in pools.values() if cf in p.gender_of}
    if not seen:
        return None
    return seen.pop() if len(seen) == 1 else "x"


def _known_given(cf: str) -> bool:
    return any(cf in p.gender_of for p in _pools().values())


def _known_family(cf: str) -> bool:
    return any(cf in p.family_set for p in _pools().values())


# ── parsing a name ───────────────────────────────────────────────────────────

# combining marks (Devanagari and other Indic vowel signs, decomposed
# accents) are not \w: without them "रमेश" would read as "रम" + "श"
_MARKS = "".join(chr(c) for c in range(0x0300, 0x1100)
                 if unicodedata.category(chr(c)).startswith("M"))
_LETTERS = r"[^\W\d_](?:[^\W\d_]|[" + _MARKS + r"])*"
_WORD = re.compile(_LETTERS + r"(?:['’]" + _LETTERS + r")*\.?")


@dataclass
class _Word:
    start: int
    end: int
    text: str
    kind: str = "name"          # name / title / suffix / particle / initial
    role: str = ""              # given / family (name and initial words)
    slot: int = -1

    @property
    def core(self) -> str:
        return self.text.rstrip(".")


def _parse(text: str) -> List[_Word]:
    words = [_Word(m.start(), m.end(), m.group()) for m in _WORD.finditer(text)]
    lead = True
    for i, w in enumerate(words):
        cf = w.core.casefold()
        if lead and cf in TITLES and len(words) > 1:
            w.kind = "title"
            continue
        lead = False
        if i > 0 and cf in SUFFIXES and (w.core.isupper() or w.core[:1].isupper()):
            w.kind = "suffix"
        elif w.core.islower() and cf in _PARTICLES and len(words) > 1:
            w.kind = "particle"
        elif len(w.core) == 1 and w.core.isalpha() and _script(w.core) == "Latn":
            w.kind = "initial"
    # name words joined by a hyphen alone share a slot ("Okafor-Smith")
    slot = -1
    prev: Optional[_Word] = None
    for w in words:
        if w.kind not in ("name", "initial"):
            prev = None
            continue
        if not (prev is not None and text[prev.end:w.start] == "-"):
            slot += 1
        w.slot = slot
        prev = w
    return words


def _case_like(model: str, value: str) -> str:
    if len(model) > 1 and model.isupper():
        return value.upper()
    if model.islower():
        return value.lower()
    return value


def _ascii(value: str) -> str:
    norm = unicodedata.normalize("NFKD", value)
    return "".join(c for c in norm if c.isascii() and c.isalnum())


# ── the registry ─────────────────────────────────────────────────────────────

class People:
    """The surrogate people of one conversation (one per :class:`MimicGen`).

    ``name()`` and ``email()`` draw new tokens from the shared RNG and record
    them; ``begin()`` / ``rollback()`` undo the draws of a surrogate the
    caller rejected."""

    def __init__(self, rng) -> None:
        self._rng = rng
        self._tokens: Dict[Tuple[str, str], str] = {}     # (role, casefold original) → token
        self._owner: Dict[str, Tuple[str, str]] = {}       # casefold token → (role, original)
        self._gender: Dict[str, str] = {}                  # casefold given original → gender
        self._locale: Dict[str, str] = {}                  # casefold original word → locale
        self._pairs: List[Tuple[str, str]] = []            # (given, family) originals, newest last
        self._domains: Dict[str, str] = {}
        self._originals: Set[str] = set()                  # casefold words of every original
        self._message_words: Set[str] = set()
        self._journal: List[tuple] = []
        self._fresh = False             # a retry: memoised tokens are not reused

    # ── conversation state ──────────────────────────────────────────────────

    def observe(self, originals: Iterable[str], message: str = "") -> None:
        """Note the originals of a message (and earlier ones) and its words:
        no new surrogate token equals one of them."""
        for value in originals:
            self._originals.update(w.casefold() for w in re.findall(r"[^\W\d_]+", value))
        self._message_words = {w.casefold() for w in re.findall(r"[^\W\d_]+", message)}

    def learn(self, surrogate: str, original: str) -> None:
        """Re-register a person surrogate issued before (a reopened session)."""
        so, oo = _parse(surrogate), _parse(original)
        sn = [w for w in so if w.kind == "name"]
        on = [w for w in oo if w.kind == "name"]
        if not 2 <= len(on) <= 3 or len(sn) != len(on) or _script(original) != "Latn":
            return
        if not all(w.text[:1].isupper() for w in sn + on):
            return
        if any(a.core.casefold() == b.core.casefold() for a, b in zip(sn, on)):
            return                                   # a shifted address, not a person
        if not (_known_given(on[0].core.casefold()) or _known_family(on[-1].core.casefold())):
            return
        for role, s, o in (("given", sn[0], on[0]), ("family", sn[-1], on[-1])):
            key, cf = (role, o.core.casefold()), s.core.casefold()
            if key in self._tokens or cf in self._owner:
                return
        for role, s, o in (("given", sn[0], on[0]), ("family", sn[-1], on[-1])):
            self._store((role, o.core.casefold()), s.core)
        self._originals.update(w.core.casefold() for w in on)
        self._pairs.append((on[0].core.casefold(), on[-1].core.casefold()))

    def begin(self) -> None:
        self._journal = []

    def rollback(self) -> None:
        for kind, *args in reversed(self._journal):
            if kind == "token":
                key, cf = args
                self._tokens.pop(key, None)
                self._owner.pop(cf, None)
            elif kind == "pair":
                self._pairs.remove(args[0])
            elif kind == "domain":
                self._domains.pop(args[0], None)
            elif kind == "gender":
                self._gender.pop(args[0], None)
            elif kind == "locale":
                self._locale.pop(args[0], None)
        self._journal = []

    def _store(self, key: Tuple[str, str], token: str) -> None:
        self._tokens[key] = token
        self._owner[token.casefold()] = key
        self._journal.append(("token", key, token.casefold()))

    def _set(self, table: Dict[str, str], name: str, key: str, value: str) -> None:
        if key not in table:
            table[key] = value
            self._journal.append((name, key))

    # ── drawing tokens ──────────────────────────────────────────────────────

    def _free(self, cand: str, original_cf: str) -> bool:
        cf = cand.casefold()
        # an empty original (the empty label of "x@y..com") is in every
        # string; testing it made every candidate taken and _draw spin forever
        return not (cf in self._owner or cf in self._originals or cf in self._message_words
                    or original_cf and (original_cf in cf or cf in original_cf))

    def _draw(self, role: str, original_cf: str, locale: str, gender: str,
              length: Optional[int] = None) -> str:
        rng = self._rng
        pools = _pools()
        for loc in (locale, _DEFAULT_LOCALE):
            pool = pools[loc]
            lst = pool.given_for(gender) if role == "given" else pool.family_for(gender)
            if not len(lst):
                continue
            for attempt in range(80):
                cand = (rng.choices(lst.names, lst.weights)[0] if lst.weights
                        else rng.choice(lst.names))
                if length is not None and len(cand) != length and attempt < 40:
                    continue
                if self._free(cand, original_cf):
                    return cand
        while True:                                         # every listed name is taken
            cand = rng.choice(string.ascii_uppercase) + "".join(
                rng.choice("aeiou" if i % 2 else "bcdfghklmnprstvz") for i in range(5))
            if self._free(cand, original_cf):
                return cand

    def _token(self, role: str, original: str, locale: str, gender: str) -> str:
        key = (role, original.casefold())
        known = self._tokens.get(key)
        if known is not None and not self._fresh and known.casefold() not in self._originals:
            return known
        latin = _pools()[locale].latin
        token = self._draw(role, key[1], locale, gender,
                           None if latin else len(original))
        if known is None:
            self._store(key, token)
        # else: the memoised token became a real name of this conversation —
        # a one-off token (not stored) keeps the two people apart
        return token

    def _initial(self, letter: str, family_cf: Optional[str]) -> str:
        if family_cf is not None:
            for g, f in reversed(self._pairs):
                tok = self._tokens.get(("given", g))
                if f == family_cf and g[:1] == letter.casefold() and tok:
                    return tok[0]
        choices = [c for c in string.ascii_uppercase if c != letter.upper()]
        return self._rng.choice(choices)

    # ── names ───────────────────────────────────────────────────────────────

    def _roles(self, words: List[_Word], text: str) -> Tuple[List[_Word], bool, bool]:
        """Assign given/family to name words; returns (name words, titled,
        family_first)."""
        names = [w for w in words if w.kind in ("name", "initial")]
        titled = any(w.kind == "title" for w in words)
        slots = sorted({w.slot for w in names})
        if not slots:
            return names, titled, False
        if len(slots) == 1:
            if names[0].kind == "name" and _script(names[0].text) in ("Hani", "Hang", "Kana") \
                    and len(names) == 1:
                return names, titled, True                  # "李娜": split in _cjk
            cf = names[0].core.casefold()
            if titled:
                role = "family"
            elif ("given", cf) in self._tokens:
                role = "given"
            elif ("family", cf) in self._tokens:
                role = "family"
            elif not _known_given(cf) and _known_family(cf):
                role = "family"
            else:
                role = "given"
            for w in names:
                w.role = role
            return names, titled, False
        first_slot = slots[0]
        # "Mitchell, Sarah"
        first_end = max(w.end for w in names if w.slot == first_slot)
        nxt = min(w.start for w in names if w.slot != first_slot)
        comma = "," in text[first_end:nxt]
        given = [w.core.casefold() for w in names if w.slot == first_slot and w.kind == "name"]
        family = [w.core.casefold() for w in names if w.slot == slots[-1] and w.kind == "name"]
        known_east = any(("family", g) in self._tokens for g in given) and not any(
            ("given", g) in self._tokens for g in given)
        _, east = _choose(given, family, "Latn", east_ok=not comma)
        east = (east or known_east) and not comma
        for w in names:
            if comma or east:
                w.role = "family" if w.slot == first_slot else "given"
            else:
                w.role = "family" if w.slot == slots[-1] else "given"
        return names, titled, east

    def _locale_for(self, names: List[_Word], text: str) -> str:
        for w in names:
            loc = self._locale.get(w.core.casefold())
            if loc:
                return loc
        script = _script(text)
        given = [w.core.casefold() for w in names if w.role == "given" and w.kind == "name"]
        family = [w.core.casefold() for w in names if w.role == "family" and w.kind == "name"]
        if script != "Latn":
            if len(names) == 1 and not given and not family:   # "李娜", "山田太郎"
                w = names[0].core.casefold()
                family = [w[:1], w[:2]]
            return _choose(given, family, script, east_ok=True)[0]
        if not given and not family:
            return _DEFAULT_LOCALE
        # roles are already ordered; score the locale on them as they stand
        pools = _pools()
        best, best_score = _DEFAULT_LOCALE, 0
        for loc in _LATIN_LOCALES + _ROMANISED:
            p = pools[loc]
            score = 2 * any(g in p.gender_of for g in given) + 2 * any(f in p.family_set
                                                                       for f in family)
            if score > best_score:
                best, best_score = loc, score
        return best

    def _gender_for(self, words: List[_Word], names: List[_Word], locale: str,
                    context: str) -> str:
        for w in words:
            if w.kind == "title" and TITLES.get(w.core.casefold()):
                return TITLES[w.core.casefold()]
        given = [w for w in names if w.role == "given" and w.kind == "name"]
        if given:
            cf = given[0].core.casefold()
            if cf in self._gender:
                return self._gender[cf]
            g = given_gender(cf, locale)
            if g in ("m", "f"):
                return g
        if context:
            hits = [g for g, pat in _GENDER_PRONOUNS.items() if pat.search(context)]
            if len(hits) == 1:
                return hits[0]
        if not given:      # a bare surname: its pair's gender, if known
            fam = [w.core.casefold() for w in names if w.role == "family"]
            for g, f in reversed(self._pairs):
                if fam and f == fam[-1] and g in self._gender:
                    return self._gender[g]
        return "x"

    def _cjk(self, word: str, locale: str, gender: str) -> str:
        """"李娜" → family + given drawn separately, same lengths."""
        pool = _pools()[locale]
        cut = next((n for n in (2, 1) if len(word) > n and word[:n].casefold() in pool.family_set),
                   1 if len(word) > 1 else 0)
        fam, giv = word[:cut], word[cut:]
        if gender == "x" and giv:
            gender = pool.gender_of.get(giv.casefold(), "x")
        out = self._token("family", fam, locale, "x") if fam else ""
        if giv:
            out += self._token("given", giv, locale, gender)
        for part in (fam, giv):
            if part:
                self._set(self._locale, "locale", part.casefold(), locale)
        return out

    def name(self, text: str, context: str = "", *, fresh: bool = False) -> str:
        """The surrogate for a person name (any form). *fresh*: the caller
        rejected the memoised surrogate — draw new tokens for this one."""
        self._fresh = fresh
        try:
            return self._name(text, context)
        finally:
            self._fresh = False

    def _name(self, text: str, context: str) -> str:
        words = _parse(text)
        names, titled, east = self._roles(words, text)
        if not any(w.kind == "name" for w in names):
            out, pos = [], 0
            for w in words:
                if w.kind == "initial":
                    out += [text[pos:w.start], self._initial(w.core, None) + w.text[1:]]
                    pos = w.end
            return "".join(out) + text[pos:] if out else text
        locale = self._locale_for(names, text)
        gender = self._gender_for(words, names, locale, context)
        given = next((w.core.casefold() for w in names if w.role == "given" and w.kind == "name"),
                     None)
        family = next((w.core.casefold() for w in reversed(names)
                       if w.role == "family" and w.kind == "name"), None)
        if given is not None and gender != "x":
            self._set(self._gender, "gender", given, gender)
        out, pos = [], 0
        for w in words:
            if w.kind not in ("name", "initial"):
                continue
            out.append(text[pos:w.start])
            if w.kind == "initial":
                rep = self._initial(w.core, family if w.role == "given" else None) + w.text[1:]
            elif len(names) == 1 and east and _script(w.text) in ("Hani", "Hang", "Kana"):
                rep = self._cjk(w.text, locale, gender)
            else:
                tok = self._token(w.role, w.core, locale, gender if w.role == "given"
                                  else gender if gender in ("m", "f") else "x")
                self._set(self._locale, "locale", w.core.casefold(), locale)
                rep = _case_like(w.core, tok) + w.text[len(w.core):]
            out.append(rep)
            pos = w.end
        out.append(text[pos:])
        if given is not None and family is not None and (given, family) not in self._pairs:
            self._pairs.append((given, family))
            self._journal.append(("pair", (given, family)))
        return "".join(out)

    # ── e-mail ──────────────────────────────────────────────────────────────

    _ROLE_LOCALS = frozenset("""info admin support contact sales hello office team noreply no-reply
        billing hr jobs careers help service enquiries inquiries mail webmaster postmaster security
        press media accounts finance marketing reception booking bookings orders""".split())

    def _components(self, runs: List[str]) -> Optional[List[Tuple[str, str]]]:
        """Read alpha runs as parts of a known person: [(part, original)],
        part one of G (given), F (family), g / f (initials)."""
        pats1 = (("G", "F"), ("G",), ("F",), ("g", "F"), ("G", "f"), ("F", "G"), ("F", "g"))
        for g, f in reversed(self._pairs):
            val = {"G": g, "F": f, "g": g[:1], "f": f[:1]}
            if len(runs) == 1:
                for pat in pats1:
                    if "".join(val[p] for p in pat) == runs[0]:
                        return [(p, g if p in "Gg" else f) for p in pat]
            else:
                parts = []
                for r in runs:
                    p = next((p for p in ("G", "F") if val[p] == r), None)
                    if p is None and len(r) == 1:
                        p = next((p for p in ("g", "f") if val[p] == r), None)
                    if p is None:
                        break
                    parts.append([(p, g if p in "Gg" else f)])
                else:
                    return [x for part in parts for x in part]
        return None

    def _render_part(self, part: str, original: str, run: str) -> str:
        if part == "?":                                  # an initial of nobody known
            return self._rng.choice(string.ascii_uppercase if run.isupper()
                                    else string.ascii_lowercase)
        role = "given" if part in "Gg" else "family"
        given = original if role == "given" else next(
            (g for g, f in reversed(self._pairs) if f == original), "")
        locale = self._locale.get(original, _DEFAULT_LOCALE)
        tok = _ascii(self._token(role, original, locale, self._gender.get(given, "x"))) or "x"
        tok = tok.upper() if run.isupper() and len(run) > 1 else (
            tok.capitalize() if run[:1].isupper() else tok.lower())
        return tok if part in "GF" else tok[0]

    def email(self, text: str, *, fresh: bool = False) -> str:
        """The surrogate for an e-mail address, linked to its person."""
        local, at, domain = text.rpartition("@")
        if not at:
            return text
        self._fresh = fresh
        try:
            return self.local(local) + "@" + self.domain(domain)
        finally:
            self._fresh = False

    def surname(self, word: str) -> str:
        """The surrogate surname for *word* (an organisation word shares the
        surrogate of the person surname it equals)."""
        return self._token("family", word, self._locale.get(word.casefold(), _DEFAULT_LOCALE), "x")

    def local(self, local: str) -> str:
        """A user name (e-mail local part, handle, URL slug) rebuilt from the
        linked surrogate person in the same style."""
        if local.casefold() in self._ROLE_LOCALS:
            return local
        local, plus, tag = local.partition("+")          # "sarah+news": the tag is kept
        pieces = re.findall(r"[A-Z]?[^\W\d_A-Z]+|[A-Z]+(?![^\W\d_A-Z])|[^\W\d_]+|\d+|[\W_]+",
                            local)
        runs = [p.casefold() for p in pieces if p[:1].isalpha()]
        comps = (self._components(runs) or self._new_person(runs)) if runs else None
        # one component list per alpha run: a single run holds them all
        per_run = [comps] if comps is not None and len(runs) == 1 else (
            [[c] for c in comps] if comps is not None else None)
        out, k = [], 0
        for p in pieces:
            if p[:1].isalpha():
                if per_run is None:
                    out.append(self._slug_like(p))
                else:
                    out.append("".join(self._render_part(pt, o, p) for pt, o in per_run[k]))
                k += 1
            elif p.isdigit():
                out.append(self._digits_like(p))
            else:
                out.append(p)
        return "".join(out) + plus + tag

    def _new_person(self, runs: List[str]) -> Optional[List[Tuple[str, str]]]:
        """An e-mail of a person not named in the conversation: read the
        local part with the name lists and register the person, so a later
        "Sarah Mitchell" gets the same surrogate."""
        def person(g: str, f: Optional[str]) -> None:
            gg = given_gender(g)
            if gg in ("m", "f"):
                self._set(self._gender, "gender", g, gg)
            if f is not None and (g, f) not in self._pairs:
                self._pairs.append((g, f))
                self._journal.append(("pair", (g, f)))

        if len(runs) == 2:
            g, f = runs
            if len(g) >= 3 and _known_given(g):
                if len(f) == 1:
                    person(g, None)
                    return [("G", g), ("?", f)]
                if len(f) >= 3:
                    person(g, f)
                    return [("G", g), ("F", f)]
            if len(g) == 1 and len(f) >= 3 and _known_family(f):
                return [("?", g), ("F", f)]
            return None
        if len(runs) != 1:
            return None
        r = runs[0]
        if len(r) >= 3 and _known_given(r):
            person(r, None)
            return [("G", r)]
        if len(r) >= 4 and _known_family(r[1:]):
            return [("?", r[0]), ("F", r[1:])]
        if len(r) >= 3 and _known_family(r):
            return [("F", r)]
        for cut in range(3, len(r) - 2):
            g, f = r[:cut], r[cut:]
            if _known_given(g) and _known_family(f):
                person(g, f)
                return [("G", g), ("F", f)]
        return None

    def _slug_like(self, run: str) -> str:
        if len(run) <= 2:
            return "".join(self._rng.choice(string.ascii_lowercase) for _ in run)
        tok = _ascii(self._draw("given", run.casefold(), _DEFAULT_LOCALE, "x")).lower()
        return tok.upper() if run.isupper() else tok

    def _digits_like(self, digits: str) -> str:
        rng = self._rng
        if len(digits) == 4 and digits[:2] in ("19", "20"):
            return digits[:2] + "".join(str(rng.randint(0, 9)) for _ in range(2))
        for _ in range(100):
            out = "".join(str(rng.randint(0, 9)) for _ in digits)
            if out != digits and (len(digits) == 1 or (digits[0] == "0") == (out[0] == "0")):
                return out
        return str((int(digits) + 1) % 10 ** len(digits)).zfill(len(digits))

    # ── domains ─────────────────────────────────────────────────────────────

    def domain(self, domain: str) -> str:
        cf = domain.casefold()
        if cf in FREEMAIL:
            return domain
        if cf in self._domains:
            return self._domains[cf]
        labels = domain.split(".")
        k = 2 if (len(labels) >= 3 and labels[-2].lower() in _SECOND_LEVEL
                  and len(labels[-1]) == 2) else 1
        i = max(0, len(labels) - k - 1)
        label = labels[i]
        for _ in range(100):
            words = [_ascii(self._draw("family", label.casefold(), _DEFAULT_LOCALE, "x")).lower()
                     for _ in range(2 if "-" in label else 1)]
            new = "-".join(words)
            new = new.upper() if label.isupper() else new
            out = ".".join(labels[:i] + [new] + labels[i + 1:])
            if out.casefold() not in FREEMAIL and out.casefold() != cf \
                    and out.casefold() not in self._domains.values():
                break
        self._domains[cf] = out
        self._journal.append(("domain", cf))
        return out


FREEMAIL = frozenset("""gmail.com googlemail.com yahoo.com yahoo.co.uk yahoo.ca yahoo.fr yahoo.de
    yahoo.co.in ymail.com outlook.com hotmail.com hotmail.co.uk hotmail.fr live.com msn.com
    icloud.com me.com mac.com aol.com proton.me protonmail.com pm.me gmx.com gmx.de gmx.net web.de
    mail.com yandex.ru yandex.com mail.ru qq.com 163.com 126.com naver.com hanmail.net
    rediffmail.com zoho.com fastmail.com tutanota.com hey.com orange.fr free.fr libero.it
    t-online.de btinternet.com comcast.net verizon.net att.net sbcglobal.net cox.net""".split())
_SECOND_LEVEL = frozenset("co com ac org net gov edu ne or".split())


def name_gender(text: str) -> Optional[str]:
    """"m" / "f" when a person name is unambiguously gendered: its title
    ("Mr.", "Ms.") or its first given name (in only one gender's lists)."""
    words = _parse(text)
    for w in words:
        if w.kind == "title" and TITLES.get(w.core.casefold()):
            return TITLES[w.core.casefold()]
    names = [w for w in words if w.kind == "name"]
    if not names:
        return None
    script = _script(text)
    if script in ("Hani", "Hang", "Kana") and len(names) == 1:
        return None
    first = names[0].core
    if len(names) >= 2:
        loc, east = _choose([names[0].core.casefold()], [names[-1].core.casefold()], script,
                            east_ok=True)
        if east:
            first = names[1].core
    g = given_gender(first)
    return g if g in ("m", "f") else None
