"""
Canonicaliser — alternative *views* of a message for PatternScan
(PROMPT_FOR_OPUS_V3 §3.2).

People write structured PII the way they speak it: "six one seven, two eight
four, three three nine one", "jdoe at fastmail dot com", "I'm twenty-nine",
"born on the third of March 1991", or with full-width digits and look-alike
letters. Each view rewrites one such habit into the form the regexes know,
keeping an offset map back to the original text; PatternScan runs on the
view, and a hit that covers a rewritten stretch is mapped back to the
original span and reported with the view's name (``DetectedEntity.view``).
The hit also keeps its written form (``DetectedEntity.canonical``, a
``Canon``): MimicGen draws the surrogate for that form and ``render_like``
writes it back the way the message spelled it, so "six one seven, ..." gets a
surrogate phone number in words and "jdoe at fastmail dot com" a surrogate
address spelled the same way.

Conservative by construction: a view hit is still a PatternScan hit, so it
has passed the type's validator and context checks (Luhn, IBAN, e-mail
grammar, birth cue, age cue) on the rewritten text; a hit that overlaps an
entity found on the original text is dropped; addresses are left to the
address parser on the original text.

Views (``VIEWS``): ``worded`` (number words to digits), ``spelled`` (``dot`` /
``at`` / ``(at)`` / ``[dot]`` inside e-mail-shaped stretches), ``dates``
(ordinal dates to ISO), ``folded`` (full-width forms, look-alike letters,
zero-width characters), ``joined`` (spaced or hyphenated digit groups
joined) and ``casefold``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..entities import DetectedEntity


@dataclass(frozen=True)
class View:
    """A rewritten copy of a message. ``starts[i]`` / ``ends[i]`` give the
    original span the view's character ``i`` came from; ``changed`` lists the
    rewritten stretches in view coordinates."""
    name: str
    text: str
    starts: Tuple[int, ...]
    ends: Tuple[int, ...]
    changed: Tuple[Tuple[int, int], ...]

    def original(self, s: int, e: int) -> Tuple[int, int]:
        return self.starts[s], self.ends[e - 1]

    def touches_change(self, s: int, e: int) -> bool:
        return any(a < e and s < b for a, b in self.changed)


@dataclass(frozen=True)
class Canon:
    """A view hit as the view wrote it: ``text``, and for each rewritten
    piece inside it ``(c_start, c_end, o_start, o_end)``, the piece in
    ``text`` and in the entity's original text (both local offsets)."""
    text: str
    pieces: Tuple[Tuple[int, int, int, int], ...]


def rewrite(name: str, text: str, edits: Iterable[Tuple[int, int, str]]) -> Optional[View]:
    """Apply non-overlapping ``(start, end, replacement)`` edits; None when
    nothing changes."""
    out: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    changed: List[Tuple[int, int]] = []
    pos = 0
    for s, e, rep in sorted(edits):
        if s < pos or text[s:e] == rep:
            continue
        for i in range(pos, s):
            out.append(text[i]); starts.append(i); ends.append(i + 1)
        if rep:
            changed.append((len(out), len(out) + len(rep)))
        elif out:
            changed.append((len(out) - 1, len(out)))       # a deletion marks its left neighbour
        for ch in rep:
            out.append(ch); starts.append(s); ends.append(e)
        if not rep and ends:
            ends[-1] = e                                   # the neighbour now spans what was removed
        pos = e
    if not changed:
        return None
    for i in range(pos, len(text)):
        out.append(text[i]); starts.append(i); ends.append(i + 1)
    return View(name, "".join(out), tuple(starts), tuple(ends), tuple(changed))


# ── numbers in words ─────────────────────────────────────────────────────────

ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_N: Dict[str, int] = {**{w: i for i, w in enumerate(ONES)}, **{w: 10 + i for i, w in enumerate(TEENS)},
                      **{w: 20 + 10 * i for i, w in enumerate(TENS)}}
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
_ORD: Dict[str, int] = {w: i + 1 for i, w in enumerate(ORDINALS)}
_ORD.update({"thirtieth": 30})
for _t, _base in (("twenty", 20), ("thirty", 30)):
    for _i, _w in enumerate(ORDINALS[:9 if _t == "twenty" else 1]):
        _ORD[f"{_t} {_w}"] = _base + _i + 1

_ONE = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
_TEEN = r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
_TEN = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_ORD_ONE = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)"
_TWO = rf"(?:{_TEN}(?:[ \t]*-[ \t]*|[ \t]+){_ONE}\b|{_TEN}\b(?![ \t]*-?[ \t]*{_ORD_ONE}\b)|{_TEEN}\b)"
# Seven or more single digits in words: a phone number, an ID read aloud.
_DIGIT_RUN = re.compile(rf"(?i)\b{_ONE}(?:(?:[ \t]*,[ \t]*|[ \t]*-[ \t]*|[ \t]+){_ONE}){{6,}}\b")
_DIGIT_SEP = re.compile(r"[ \t]*,[ \t]*|[ \t]*-[ \t]*|[ \t]+")
_CARDINAL = re.compile(rf"(?i)\b{_TWO}")
_ONE_BEFORE_UNIT = re.compile(rf"(?i)\b{_ONE}\b(?=[ \t]*-?[ \t]*(?:years?|yrs?)\b)")


def number_value(words: str) -> Optional[int]:
    """'twenty-nine' → 29, 'fifteen' → 15, 'seven' → 7; None otherwise."""
    parts = [w for w in re.split(r"[\s\-]+", words.lower()) if w]
    if len(parts) == 1:
        return _N.get(parts[0])
    if len(parts) == 2 and parts[0] in TENS and parts[1] in ONES[1:]:
        return _N[parts[0]] + _N[parts[1]]
    return None


def worded(text: str) -> Optional[View]:
    edits: List[Tuple[int, int, str]] = []
    for m in _DIGIT_RUN.finditer(text):
        groups = [""]
        pos = m.start()
        for d in re.finditer(rf"(?i){_ONE}", m.group()):
            sep = m.group()[pos - m.start():d.start()] if pos > m.start() else ""
            if "," in sep:
                groups.append("")
            groups[-1] += str(_N[d.group().lower()])
            pos = m.start() + d.end()
        edits.append((m.start(), m.end(), " ".join(groups)))
    taken = [(s, e) for s, e, _r in edits]
    for rx in (_CARDINAL, _ONE_BEFORE_UNIT):
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            n = number_value(m.group())
            if n is not None:
                edits.append((m.start(), m.end(), str(n)))
                taken.append(m.span())
    return rewrite("worded", text, edits)


# ── e-mail addresses spelled out ─────────────────────────────────────────────

_TLDS = frozenset("""com org net edu gov io co uk de fr es it nl be ch at se no dk fi ie pt pl cz ru ua in jp cn kr
au nz ca us mx br ar cl za ng ke eu info biz me dev app ai xyz online site tech email mail""".split())
_FREE_MAIL = frozenset("""gmail googlemail yahoo ymail outlook hotmail live msn icloud me mac aol proton protonmail pm
fastmail gmx web mail zoho yandex tutanota tuta hey posteo mailbox""".split())
# words that stand before an ordinary " at " ("look at gmail dot com")
_AT_WORDS = frozenset("""me him her us them you it one this that these those here there home work back look
looking looked arrive arrived stare staring laugh laughing point pointing aim aiming stay staying live
living meet meeting see seen sign signed log logged go went get got be is was are were am in on up
out off down over by and or but not also only now then just right""".split())
_LBR = r"[\(\[\{<]"
_RBR = r"[\)\]\}>]"
_DOT = rf"(?:[ \t]*{_LBR}[ \t]*dot[ \t]*{_RBR}[ \t]*|[ \t]+dot[ \t]+|\.)"
_AT = rf"(?:[ \t]*{_LBR}[ \t]*at[ \t]*{_RBR}[ \t]*|[ \t]+at[ \t]+|[ \t]*@[ \t]*)"
_PART = r"[A-Za-z0-9](?:[A-Za-z0-9_+\-]*[A-Za-z0-9])?"
_SPELLED = re.compile(rf"(?i)(?<![\w.@])(?P<local>{_PART}(?:{_DOT}{_PART}){{0,3}}){_AT}"
                      rf"(?P<domain>{_PART}(?:{_DOT}{_PART}){{1,3}})(?![\w@])")
_SPELLED_TOKEN = re.compile(rf"(?i){_LBR}[ \t]*(?:at|dot)[ \t]*{_RBR}|[ \t]dot[ \t]|[ \t]at[ \t]|[ \t]@|@[ \t]")
_BARE_AT = re.compile(r"(?i)[ \t]+at[ \t]+")
_MAIL_CUE = re.compile(r"(?i)(?:e-?mail|mail|inbox|contact|reach|write\s+to|address|addr)\b[^\n]{0,40}$")
_DOT_SPLIT = re.compile(rf"(?i){_DOT}")


def _spelled_ok(m: "re.Match", text: str) -> bool:
    local, domain = m.group("local"), m.group("domain")
    labels = [x for x in _DOT_SPLIT.split(domain) if x]
    if not _SPELLED_TOKEN.search(m.group()) or labels[-1].lower() not in _TLDS:
        return False
    at = m.group()[len(local):len(m.group()) - len(domain)]
    if not _BARE_AT.fullmatch(at):
        return True                                   # "(at)", "[at]", "@": written as an address
    # a bare " at " is ordinary English ("look at gmail dot com"): ask for an
    # address-shaped local part, a mail cue just before it, or a webmail
    # domain behind a word that is not one that stands before "at"
    parts = [x for x in _DOT_SPLIT.split(local) if x]
    shaped = len(parts) > 1 or re.search(r"[\d_+]", local)
    webmail = labels[0].lower() in _FREE_MAIL and local.lower() not in _AT_WORDS
    return bool(shaped or webmail or _MAIL_CUE.search(text[max(0, m.start() - 60):m.start()]))


def spelled(text: str) -> Optional[View]:
    edits, pos = [], 0
    # search, not finditer: a rejected match ("me at jane dot doe") must not
    # consume the address that starts inside it ("jane dot doe at …")
    while (m := _SPELLED.search(text, pos)) is not None:
        if not _spelled_ok(m, text):
            pos = m.start() + 1
            continue
        local = ".".join(x for x in _DOT_SPLIT.split(m.group("local")) if x)
        domain = ".".join(x for x in _DOT_SPLIT.split(m.group("domain")) if x)
        edits.append((m.start(), m.end(), f"{local}@{domain}"))
        pos = m.end()
    return rewrite("spelled", text, edits)


# ── ordinal dates ────────────────────────────────────────────────────────────

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_ORD_WORDS = "|".join(sorted((k.replace(" ", r"[ \t]*-?[ \t]*") for k in _ORD), key=len, reverse=True))
_DAY = rf"(?:{_ORD_WORDS}|\d{{1,2}}(?:st|nd|rd|th))"
_YEAR_WORDS = (rf"(?:(?:nineteen|twenty)[ \t\-]+(?:{_TEN}(?:[ \t\-]+{_ONE})?|{_TEEN}|oh[ \t\-]+{_ONE}|hundred)"
               rf"|two[ \t]+thousand(?:[ \t]+and)?(?:[ \t\-]+(?:{_TEN}(?:[ \t\-]+{_ONE})?|{_TEEN}|{_ONE}))?)")
_YEAR = rf"(?:\d{{4}}|{_YEAR_WORDS})"
_DAY_OF = re.compile(rf"(?i)\b(?:the[ \t]+)?(?P<d>{_DAY})[ \t]+(?:day[ \t]+)?of[ \t]+(?P<m>{_MONTH})[ \t]*,?[ \t]*"
                     rf"(?:in[ \t]+)?(?P<y>{_YEAR})\b")
_MONTH_DAY = re.compile(rf"(?i)\b(?P<m>{_MONTH})[ \t]+(?:the[ \t]+)?(?P<d>{_DAY})[ \t]*,?[ \t]*(?P<y>{_YEAR})\b")


def year_value(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    words = [w for w in re.split(r"[\s\-]+", s) if w and w != "and"]
    if words[:2] == ["two", "thousand"]:
        rest = number_value(" ".join(words[2:])) if words[2:] else 0
        return None if rest is None else 2000 + rest
    head = _N.get(words[0])
    if head not in (19, 20):
        return None
    tail = words[1:]
    if tail == ["hundred"]:
        return head * 100
    if tail[:1] == ["oh"] and len(tail) == 2 and tail[1] in ONES:
        return head * 100 + _N[tail[1]]
    n = number_value(" ".join(tail))
    return None if n is None or n < 10 else head * 100 + n


def day_value(s: str) -> Optional[int]:
    s = re.sub(r"[\s\-]+", " ", s.strip().lower())
    if s[:1].isdigit():
        return int(re.match(r"\d+", s).group())
    return _ORD.get(s)


def dates(text: str) -> Optional[View]:
    edits, taken = [], []
    for rx in (_DAY_OF, _MONTH_DAY):
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            d, y = day_value(m.group("d")), year_value(m.group("y"))
            mon = next(i + 1 for i, name in enumerate(_MONTHS) if name.startswith(m.group("m").lower().rstrip(".")[:3]))
            if d and y and 1 <= d <= 31 and 1000 <= y <= 2999:
                edits.append((m.start(), m.end(), f"{y:04d}-{mon:02d}-{d:02d}"))
                taken.append(m.span())
    return rewrite("dates", text, edits)


# ── look-alike characters ────────────────────────────────────────────────────

_ZERO_WIDTH = frozenset("​‌‍⁠﻿­")
_HOMOGLYPHS = dict(zip(
    "АВЕКМНОРСТХаеорсухіјѕԁԛԝΑΒΕΖΗΙΚΜΝΟΡΤΥΧαο",
    "ABEKMHOPCTXaeopcyxijsdqwABEZHIKMNOPTYXao"))


def folded(text: str) -> Optional[View]:
    edits = []
    for i, ch in enumerate(text):
        if ch.isascii():
            continue
        if ch in _ZERO_WIDTH:
            edits.append((i, i + 1, ""))
        elif ch in _HOMOGLYPHS:
            edits.append((i, i + 1, _HOMOGLYPHS[ch]))
        else:
            n = unicodedata.normalize("NFKC", ch)
            if n != ch and n.isascii() and n.isprintable():
                edits.append((i, i + 1, n))
    return rewrite("folded", text, edits)


# ── digit groups, case ───────────────────────────────────────────────────────

# space or hyphen groups only: dotted groups are versions and paths ("/10.1.1234.567")
_GROUPS = re.compile(r"(?<![\w.\-/])\d{2,5}(?:[ \-]\d{1,5}){1,5}(?![\w]|[.\-]\d)")


def joined(text: str) -> Optional[View]:
    edits = []
    for m in _GROUPS.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 9 <= len(digits) <= 19 and len({c for c in m.group() if not c.isdigit()}) == 1:
            edits.append((m.start(), m.end(), digits))
    return rewrite("joined", text, edits)


def casefold(text: str) -> Optional[View]:
    low = text.lower()
    if low == text or len(low) != len(text):
        return None
    return rewrite("casefold", text, [(i, i + 1, c) for i, (c, o) in enumerate(zip(low, text)) if c != o])


VIEWS: Dict[str, Callable[[str], Optional[View]]] = {
    "worded": worded, "spelled": spelled, "dates": dates, "folded": folded, "joined": joined, "casefold": casefold,
}
# run_cascade's default (``canonical_views=None``); chosen on dev, see
# bench/results/attribution_dev.json and DETECTOR_REPORT.md
DEFAULT_VIEWS: Tuple[str, ...] = ("worded", "spelled", "dates", "folded", "joined")


def scan_views(
    text: str,
    taken: Sequence[DetectedEntity],
    views: Sequence[str],
    scanner: Callable[[str], List[DetectedEntity]],
    skip_types: Iterable[str] = ("address",),
) -> List[DetectedEntity]:
    """PatternScan hits found only on a view, in original coordinates, each
    tagged with its view. *taken* are the entities found on the original
    text; a view hit overlapping one (or an earlier view's hit) is dropped."""
    skip = set(skip_types)
    spans = [(e.start, e.end) for e in taken]
    out: List[DetectedEntity] = []
    for name in views:
        view = VIEWS[name](text)
        if view is None:
            continue
        for h in scanner(view.text):
            if h.type in skip or not view.touches_change(h.start, h.end):
                continue
            # a hit on part of a rewritten stretch takes all of it: the
            # original span covers the whole stretch either way
            cs, ce = h.start, h.end
            for a, b in view.changed:
                if a < ce and cs < b:
                    cs, ce = min(cs, a), max(ce, b)
            s, e = view.original(cs, ce)
            if any(s < b and a < e for a, b in spans):
                continue
            pieces = tuple((a - cs, b - cs, view.starts[a] - s, view.ends[b - 1] - s)
                           for a, b in view.changed if cs <= a and b <= ce)
            out.append(replace(h, text=text[s:e], start=s, end=e, parsed=None, view=name,
                               canonical=Canon(view.text[cs:ce], pieces)))
            spans.append((s, e))
    out.sort(key=lambda x: x.start)
    return out


# ── surrogates back in the message's spelling ────────────────────────────────

def number_words(n: int, joiner: str = "-") -> str:
    """29 → 'twenty-nine' (or 'twenty nine'), 105 → 'one hundred and five'."""
    if n < 0:
        raise ValueError(n)
    if n >= 100:
        head, rest = divmod(n, 100)
        if head >= 10:
            raise ValueError(n)
        return f"{ONES[head]} hundred" + (f" and {number_words(rest, joiner)}" if rest else "")
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    tens, ones = divmod(n, 10)
    return TENS[tens - 2] + (joiner + ONES[ones] if ones else "")


def ordinal_words(n: int, joiner: str = "-") -> str:
    """3 → 'third', 21 → 'twenty-first', 30 → 'thirtieth'."""
    if 1 <= n <= 20:
        return ORDINALS[n - 1]
    if n == 30:
        return "thirtieth"
    tens, ones = divmod(n, 10)
    if tens in (2, 3) and ones:
        return TENS[tens - 2] + joiner + ORDINALS[ones - 1]
    raise ValueError(n)


def year_words(y: int, like: str = "", joiner: str = "-") -> str:
    """1991 → 'nineteen ninety-one', 1905 → 'nineteen oh five', 2004 → 'two
    thousand and four' ('two thousand four' when *like* has no 'and')."""
    low = like.lower()
    if 2000 <= y < 2100 and (y < 2010 or "thousand" in low):
        rest = y - 2000
        if not rest:
            return "two thousand"
        return "two thousand " + ("and " if " and " in low or not low else "") + number_words(rest, joiner)
    head, rest = divmod(y, 100)
    if not 10 <= head <= 99:
        raise ValueError(y)
    tail = "hundred" if not rest else f"oh {ONES[rest]}" if rest < 10 else number_words(rest, joiner)
    return f"{number_words(head, joiner)} {tail}"


def _joiner(piece: str) -> str:
    return " " if re.search(r"[A-Za-z][ \t]+[A-Za-z]", piece) and not re.search(r"[A-Za-z]-[A-Za-z]", piece) else "-"


def _case(model: str, value: str) -> str:
    letters = [c for c in model if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return value.upper()
    words = re.findall(r"[A-Za-z]+", model)
    if len(words) > 1 and all(w[0].isupper() for w in words):
        return re.sub(r"[A-Za-z]+", lambda m: m.group().capitalize(), value)
    if letters and letters[0].isupper():
        return value[:1].upper() + value[1:]
    return value


def _render_worded(original: str, canon: Canon, sur: str) -> Optional[str]:
    c_runs = list(re.finditer(r"\d+", canon.text))
    s_runs = list(re.finditer(r"\d+", sur))
    if len(c_runs) != len(s_runs):
        return None
    edits = []
    for cs, ce, os_, oe in canon.pieces:
        idx = [i for i, m in enumerate(c_runs) if cs <= m.start() and m.end() <= ce]
        if not idx:
            continue
        a, b = s_runs[idx[0]].start(), s_runs[idx[-1]].end()
        piece, digits = original[os_:oe], re.sub(r"\D", "", sur[a:b])
        words = list(re.finditer(rf"(?i)\b{_ONE}\b", piece))
        if _DIGIT_RUN.search(piece):
            if len(words) == len(digits):
                new = list(piece)
                for w, d in reversed(list(zip(words, digits))):
                    new[w.start():w.end()] = _case(w.group(), ONES[int(d)])
                edits.append((a, b, "".join(new)))
            else:
                edits.append((a, b, _case(piece, " ".join(ONES[int(d)] for d in digits))))
        else:
            edits.append((a, b, _case(piece, number_words(int(digits), _joiner(piece)))))
    for a, b, rep in sorted(edits, reverse=True):
        sur = sur[:a] + rep + sur[b:]
    return sur


def _render_spelled(original: str, canon: Canon, sur: str) -> Optional[str]:
    m = _SPELLED.search(original)
    if m is None or sur.count("@") != 1:
        return None
    local_o, domain_o = m.group("local"), m.group("domain")
    at = original[m.end("local"):m.start("domain")]
    dot_l = next((d.group() for d in _DOT_SPLIT.finditer(local_o)), None)
    dot_d = next((d.group() for d in _DOT_SPLIT.finditer(domain_o)), ".")
    local_s, domain_s = sur.split("@")
    return ((dot_l or dot_d).join(local_s.split(".")) + at + dot_d.join(domain_s.split(".")))


def _render_dates(original: str, canon: Canon, sur: str) -> Optional[str]:
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", sur)
    m = _DAY_OF.search(original) or _MONTH_DAY.search(original)
    if iso is None or m is None:
        return None
    y, mo, d = (int(x) for x in iso.groups())
    day_o, mon_o, year_o = m.group("d"), m.group("m"), m.group("y")
    if day_o[:1].isdigit():
        suf = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        day = f"{d}{suf.upper() if day_o[-2:].isupper() else suf}"
    else:
        day = _case(day_o, ordinal_words(d, _joiner(day_o)))
    full = _MONTHS[mo - 1]
    bare = mon_o.rstrip(".")
    month = _case(bare, full if len(bare) > 3 or bare.lower() == "may" else full[:3]) + mon_o[len(bare):]
    year = str(y) if year_o.isdigit() else _case(year_o, year_words(y, year_o, _joiner(year_o)))
    new = original
    for g, rep in sorted((("y", year), ("m", month), ("d", day)), key=lambda x: -m.start(x[0])):
        new = new[:m.start(g)] + rep + new[m.end(g):]
    # the date in words where the drawn surrogate has it in ISO form
    return sur[:iso.start()] + new[m.start():m.end() + len(new) - len(original)] + sur[iso.end():]


def _render_joined(original: str, canon: Canon, sur: str) -> Optional[str]:
    digits = re.sub(r"\D", "", sur)
    if sum(c.isdigit() for c in original) != len(digits):
        return None
    it = iter(digits)
    return "".join(next(it) if c.isdigit() else c for c in original)


_RENDER: Dict[str, Callable[[str, Canon, str], Optional[str]]] = {
    "worded": _render_worded, "spelled": _render_spelled, "dates": _render_dates, "joined": _render_joined,
}


def render_like(view: str, original: str, canon: Canon, surrogate: str) -> str:
    """*surrogate*, drawn for ``canon.text``, written the way *original* was
    (number words, spelled-out e-mail, ordinal date, grouped digits). Falls
    back to the surrogate as drawn when the shapes do not line up."""
    fn = _RENDER.get(view)
    if fn is None:
        return surrogate
    try:
        out = fn(original, canon, surrogate)
    except (ValueError, IndexError, KeyError):
        out = None
    return out or surrogate
