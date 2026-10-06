"""
detection/org_assembly.py — Pass O: an organisation's name read whole (V3 §3.5).

The model stages cut a firm's name into pieces. "Singh, Adams and Ellis"
comes back as three PERSONs and the "and" between them goes out as typed;
"Jacobi Jäckel GmbH & Co. KG" loses its "KG", "Langan and Sons" its "and
Sons"; "Smith and Sons" is dropped as an implausible ORG (a lower-case word)
and "Leeds Community College" is masked as "College". Pass S
(structural.py) knows several of these shapes, but it runs after the gate
and never takes a span another stage already holds.

This pass reads the name where something says it is an organisation and
then resolves: the whole name replaces every candidate inside it, whatever
its source or type (a model span that runs past the name's edge is taken
into it, so no character loses its cover).

Evidence:

  label   a form or JSON field for one: "Employer:", "Organisation:",
          "School/Employer:", "Org:", "Company name:", "employer": "…"
  work    a work relation right before it: "I work at", "working for",
          "employed by", "hired by", "I'm a nurse at"
  link    a sign-off or a person next to it: "— Ana Ruiz, …", "Ana Ruiz |
          … |", a short line under a name or a closing, "<person> from …"
  form    the name carries its legal form ("GmbH & Co. KG", "S.L.N.E",
          "e.V.", "PLC", "Pty Ltd"; the ISO 20275 entity legal forms, written
          down here, not downloaded) or a family-firm tail ("and Sons",
          "e Filhos", "& Zonen", "y asociados")

What each shape of name needs:

  "Guerra", "Fenlon-McBreen"       label; or work and a model span on it
  "Singh, Adams and Ellis"         label, work or link
  "Lucknow Middle School"          label, work or link
  "Roach PLC", "Freitas e Filhos"  nothing more; a bare code that is also a
                                   state or a word ("NV", "AG", "SA", "AB")
                                   needs label, work or link
  "Ana Ruiz and Tom Bell"          label (two full names are people, not a
                                   firm, anywhere else)

A person followed by a comma, a bar or a dash links only on a line the size
of a signature, and only a full name does: in prose and in citations the
comma goes on listing people.

A name is capitalised words, with name particles ("van", "de", "von"),
joined by ", " and a final coordinator ("and", "&", "und", "et", "e", "y",
"en"), then its legal form or its kind ("Middle School"). A list of known
cities or one with a public company in it is not a firm.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from ..entities import DetectedEntity
from .geo_data import MAJOR_CITIES
from .public_names import NOT_NAMES, PUBLIC_ORGS

SOURCE = "structural"

# ── the parts of a name ──────────────────────────────────────────────────────

_UP = "A-ZÀ-ÖØ-ÞČŠŽŁ"
_WORD = re.compile(rf"[{_UP}][\w'’]*(?:-[{_UP}][\w'’]*)*")
_PARTICLE = re.compile(
    r"(?:van|von|vom|zu|zum|zur|de|del|della|der|den|di|da|das|do|dos|du|la|le|ten|ter|bin|al|el)"
    r"(?=[ \t]+[" + _UP + "])")

# Entity legal forms (ISO 20275 style), longest first where one is the
# start of another. Case as written: "AG" is a form, "ag" is not.
_LEGAL = (
    r"GmbH[ \t]*&[ \t]*Co\.?[ \t]*KG(?:aA)?|AG[ \t]*&[ \t]*Co\.?[ \t]*KG|GmbH|gGmbH|mbH|KGaA|OHG|GbR"
    r"|UG(?:[ \t]*\(haftungsbeschränkt\))?|e\.[ \t]?V\.|e\.[ \t]?G\.|e\.[ \t]?K\.|PartG"
    r"|Inc\.?|Incorporated|LLC|L\.L\.C\.|LLP|L\.L\.P\.|PLLC|L\.P\.|Ltd\.?|Limited|PLC|[Pp]lc"
    r"|Corp\.?|Corporation|Pty\.?[ \t]+Ltd\.?|Pvt\.?[ \t]+Ltd\.?|Pte\.?[ \t]+Ltd\.?|Sdn\.?[ \t]+Bhd\.?"
    r"|S\.A\.[ \t]?de[ \t]C\.V\.|S\.A\.S\.?|S\.A\.C\.|S\.A\.U\.|S\.A\.|S/A|SAS|SARL|S\.A\.R\.L\.?|SASU"
    r"|EURL|S\.L\.N\.E\.?|S\.L\.U\.|S\.L\.|S\.[ \t]?Coop\.?|S\.C\.|Ltda\.?|Lda\.?|EIRELI"
    r"|S\.p\.A\.?|SpA|S\.r\.l\.?|Srl|s\.r\.l\.|S\.a\.s\.|S\.n\.c\.|B\.V\.?|N\.V\.?|BVBA|VZW|V\.O\.F\.?"
    r"|A/S|ApS|ASA|Oyj|Kft\.?|Zrt\.?|Nyrt\.?|s\.r\.o\.|a\.s\.|d\.o\.o\.|sp\.[ \t]*z[ \t]*o\.[ \t]?o\.|K\.K\."
)
# the same forms as a bare code: a state, a word or an abbreviation as well
_AMBIG = r"AG|KG|SE|SA|NV|BV|AB|AS|Oy|eG|SC|ME"
# a family or partnership firm
_FAMILY = (
    r"(?:and|&)[ \t]+(?:Sons?|Daughters?|Co\.?(?![ \t]*KG)|Company|Partners|Associates|Brothers)"
    r"|Bros\.?|Brothers|e[ \t]+(?:Filhos|Figli|figli|Irmãos)|y[ \t]+(?:asociados|Asociados|Hijos|Cía\.?)"
    r"|(?:et|&)[ \t]+(?:Fils|Cie\.?|associés|Associés)|(?:und|&)[ \t]+Söhne|(?:en|&)[ \t]+Zonen|&[ \t]+Zn\.?"
    r"|Hermanos|Hnos\.?|Irmãos"
)
# what a school, college or hospital is
_KIND = (
    r"(?:(?:Community|Junior|Technical|State|City|Sixth[ \t]+Form|Further[ \t]+Education)[ \t]+)?College"
    r"|University|Polytechnic|Academy|Institute(?:[ \t]+of[ \t]+Technology)?"
    r"|(?:(?:Primary|Secondary|Middle|High|Elementary|Grammar|Junior|Senior|Public|Prep(?:aratory)?"
    r"|Comprehensive|Montessori|Catholic|Charter|International|Nursery|Infant|Day|Medical|Law"
    r"|Business)[ \t]+)?School|(?:General[ \t]+|Memorial[ \t]+|Children['’]s[ \t]+)?Hospital"
    r"|Clinic|Medical[ \t]+Cent(?:er|re)|Health[ \t]+Cent(?:er|re)|Surgery|Kindergarten"
)
_END = r"(?![\w&])"
_TAIL = re.compile(rf"[ \t]*,?[ \t]+(?:(?P<legal>{_LEGAL})|(?P<ambig>{_AMBIG})|(?P<family>{_FAMILY})"
                   rf"|(?P<kind>{_KIND})){_END}")
_TAIL_GLUED = re.compile(rf"[ \t]*(?P<family>&[ \t]*(?:Co\.?(?![ \t]*KG)|Sons?|Partners|Associates)){_END}")
_COORD = re.compile(r"[ \t]*,?[ \t]+(?:and|&|und|et|e|y|en|og|och)[ \t]+|[ \t]*&[ \t]*")
_COMMA = re.compile(r"[ \t]*,[ \t]*")

# capitalised words that never start a name or a piece of one
_NOT_START = frozenset("""
i we you he she they it the a an my our your his her their this that these those
thanks thank regards best cheers sincerely hi hello dear yours kind warm
mr mrs ms dr prof sir madam phone tel mobile email e-mail fax address name dob
employer company organisation organization org school website linkedin
none self retired unemployed student freelance freelancer homemaker various tbd n/a
monday tuesday wednesday thursday friday saturday sunday january february march
april may june july august september october november december christmas easter
""".split()) | NOT_NAMES


class Name:
    __slots__ = ("start", "end", "pieces", "form", "kind")

    def __init__(self, start, end, pieces, form, kind):
        self.start, self.end = start, end
        self.pieces = pieces            # [(start, end)] of the name words
        self.form = form                # "legal" / "ambig" / "family" / None
        self.kind = kind                # "School", "College", … or None

    @property
    def is_list(self) -> bool:
        return len(self.pieces) > 1


def _piece(text: str, pos: int) -> Optional[int]:
    """End of a run of up to five name words at *pos* (particles inside)."""
    end, words = None, 0
    while words < 5:
        m = _PARTICLE.match(text, pos)
        if m:
            pos = re.compile(r"[ \t]+").match(text, m.end()).end()
        m = _WORD.match(text, pos)
        if not m or _TAIL.match(" " + text[m.start():m.start() + 40]) or \
                re.match(r"-[^\W\d_]", text[m.end():m.end() + 2]):
            break
        if m.group().lower().rstrip("'’") in _NOT_START or text[m.end():m.end() + 1] == ":":
            break
        end, words = m.end(), words + 1
        nxt = re.compile(r"[ \t]+").match(text, end)
        if not nxt:
            break
        pos = nxt.end()
    return end


def _strip_particles(piece: str) -> str:
    return re.sub(r"^(?:(?:van|von|vom|zu|de|del|della|der|den|di|da|das|do|dos|du|la|le|ten|ter"
                  r"|bin|al|el)[ \t]+)+", "", piece)


def read_name(text: str, pos: int) -> Optional[Name]:
    """The organisation name that starts at *pos*, or None."""
    first = _piece(text, pos)
    if first is None:
        return None
    pieces, end, settled = [(pos, first)], first, first
    form = kind = None
    while len(pieces) < 6:
        if _TAIL.match(text, end) or _TAIL_GLUED.match(text, end):
            break
        m = _COORD.match(text, end) or _COMMA.match(text, end)
        if not m:
            break
        nxt = _piece(text, m.end())
        if nxt is None:
            break
        if m.re is _COMMA and any(re.search(r"[ \t]", _strip_particles(text[a:b]))
                                  for a, b in pieces + [(m.end(), nxt)]):
            break                       # "— Tamsin Orr, Hollen, …": a comma list is of single names
        pieces.append((m.end(), nxt))
        end = nxt
        if m.re is _COORD:
            settled = end               # "A, B and C": the commas are settled
    if settled < end:                   # "Acme, Seattle": no coordinator, no list
        pieces = [p for p in pieces if p[1] <= settled]
        end = settled
    for _ in range(2):
        m = _TAIL.match(text, end) or _TAIL_GLUED.match(text, end)
        if not m:
            break
        g = m.lastgroup
        if g == "kind":
            kind = m.group("kind")
        elif form is None or form == "ambig":
            form = g
        end = m.end()
    if len(text[pos:end]) > 90:
        return None
    return Name(pos, end, pieces, form, kind)


# ── where a name stands ──────────────────────────────────────────────────────

_LABEL = re.compile(
    r"(?i)(?:^|[\s|,;{(\"'])(?:[\w ]{1,20}/[ \t]*)*"
    r"(?:employer|employer[ \t]+name|company|company[ \t]+name|organi[sz]ation|org|firm|firm[ \t]+name"
    r"|workplace|place[ \t]+of[ \t]+work|business|business[ \t]+name|school|university|college"
    r"|agency|practice|empresa|empleador|arbeitgeber|employeur|entreprise|firma|werkgever"
    r"|datore[ \t]+di[ \t]+lavoro)"
    r"(?:[ \t]*/[ \t]*[\w ]{1,20})*[\"']?[ \t]*[:=][ \t]*[\"']?[ \t]*$")
_WORK = re.compile(
    r"(?i)(?:\b(?:work(?:s|ed|ing)?|employed|intern(?:ed|ing)?|hired|contracted|placed|job|position"
    r"|role|post)(?:[ \t]+as[ \t]+(?:a|an)[ \t]+[\w \-]{1,30}?)?"
    r"|\b(?:i['’]?m|i[ \t]+am|im)[ \t]+(?:a|an)[ \t]+[\w \-]{1,30}?)"
    r"[ \t]+(?:at|for|by)[ \t]+(?:the[ \t]+)?$"
    r"|(?i:\b(?:my|our)[ \t]+(?:employer|company|firm|school|workplace)(?:[ \t]+is|[ \t]*[,:])?[ \t]+)$")
# a sign-off line: "— Ana Ruiz, …", "-- Ana Ruiz | …"
_SIGN_OFF = re.compile(rf"^[ \t]*(?:—|–|--|-)[ \t]*(?:[{_UP}][\w'’.\-]*[ \t]+){{0,3}}[{_UP}][\w'’.\-]*"
                       r"[ \t]*[,|][ \t]*(?:[^,|\n]*[,|][ \t]*)?$|^[ \t]*(?:—|–|--)[ \t]*$")
_PERSON_GAP = re.compile(r"[ \t]*(?:,|\||—|–|-|@)[ \t]*|[ \t]*,?[ \t]+(?:from|at|of|with)[ \t]+")
_WORD_GAP = re.compile(r"[ \t]*,?[ \t]+(?:from|at|of|with)[ \t]+")
# what may sit right before a name in a link: a person or a contact detail
_CONTACT = frozenset({"PERSON", "email", "phone_us", "phone_uk", "phone_intl", "phone"})
_CLOSING = re.compile(
    r"(?i)^\s*(?:regards|best(?:\s+regards|\s+wishes)?|kind\s+regards|warm\s+regards|many\s+thanks"
    r"|thanks|thank\s+you|cheers|sincerely|respectfully|yours|cordialement|saludos|atenciosamente)\b")


def _line(text: str, pos: int) -> Tuple[int, int]:
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e == -1 else e)


def slot(text: str, start: int, anchors: Sequence[DetectedEntity] = (),
         end: Optional[int] = None) -> Optional[str]:
    """What the text right before *start* says about a name there: "label",
    "work", "link", "closing" (the line right under a sign-off word) or None.
    *anchors* are the message's people and contact details ("Ana Ruiz from
    …", "ana@x.org, from …"). A name that ends at *end* in front of ":" is a
    field's key ("Budget: $25k"), not its value: None."""
    if end is not None and re.match(r"[ \t]*[:=]", text[end:]):
        return None
    ls, le = _line(text, start)
    # every look is bounded: a long line is no field, sign-off or signature
    if _LABEL.search(text[max(ls, start - 160):start]):
        return "label"
    if _WORK.search(text[max(0, start - 80):start]):
        return "work"
    before = text[ls:start] if start - ls <= 200 else None
    if before is not None and (_SIGN_OFF.match(before) or (
            "|" in before and le - ls <= 600 and len(text[ls:le].split()) <= 16)):
        return "link"
    for a in anchors:
        if not (a.type in _CONTACT and ls <= a.start and a.end <= start and start - a.end <= 20):
            continue
        gap = text[a.end:start]
        if _WORD_GAP.fullmatch(gap):
            return "link"
        # "Ana Ruiz, …" / "ana@x.org | …" on a signature-sized line; in prose,
        # or after a first name alone, a comma goes on listing people
        # ("Maria, Tom and Natalia", "(Lund, Okafor-Bell, & Ruiz, 2019)")
        if _PERSON_GAP.fullmatch(gap) and le - ls <= 600 and len(text[ls:le].split()) <= 16 \
                and (a.type != "PERSON" or len(a.text.split()) >= 2):
            return "link"
    if before is not None and not before.strip() and le - ls <= 300 and \
            len(text[ls:le].split()) <= 8 and ls > 0:
        cut = ls - 1
        for _ in range(4):
            cut = text.rfind("\n", 0, cut)
            if cut < 0:
                break
        prev = text[cut + 1:ls - 1].split("\n")
        prev_start = ls - sum(len(l) + 1 for l in prev[-2:])
        if any(a.type == "PERSON" and prev_start <= a.start and a.end <= ls for a in anchors):
            return "link"
        if any(_CLOSING.match(l) for l in prev):
            # right under "Thanks," people sign their own names ("Mike and Sarah")
            return "link" if not _CLOSING.match(prev[-1]) else "closing"
    return None


def has_form(name: str) -> bool:
    """"Seifert Trubin e.V.", "Freitas e Filhos": a name that ends in its
    legal form or a family-firm tail (not a bare code such as "NV")."""
    m = re.search(rf"[ \t]*,?[ \t]+(?:{_LEGAL}|{_FAMILY}){_END}[ \t.]*$", name)
    return bool(m and m.start() > 0)


# ── the pass ─────────────────────────────────────────────────────────────────

_NAME_START = re.compile(rf"(?<![\w'’&.\-])(?=[{_UP}]|(?:van|von|de|del|der|den|di|da|du|la|le|ten|ter)[ \t]+[{_UP}])")


def _not_a_firm(text: str, name: Name) -> bool:
    words = [text[s:e] for s, e in name.pieces]
    if any(w.lower() in PUBLIC_ORGS for w in words):
        return True
    if name.is_list and name.form is None and name.kind is None:
        return all(w.lower() in MAJOR_CITIES for w in words)
    return False


def _accept(name: Name, where: Optional[str], supported: bool, people: bool = False) -> bool:
    """*people*: two or more of the pieces are full names ("Ana Ruiz and Tom
    Bell"), which a firm without a legal form or kind word is not."""
    if where == "label":
        return True
    if name.form in ("legal", "family"):
        return True
    if people and not name.kind:
        return False
    if where == "work":
        return name.is_list or bool(name.form or name.kind) or supported
    if where == "link":
        return name.is_list or bool(name.form or name.kind)
    if where == "closing":
        return len(name.pieces) > 2 or bool(name.form or name.kind)
    return False


def assemble(text: str, entities: Sequence[DetectedEntity],
             opaque: Sequence[Tuple[int, int]] = ()) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """(added, superseded): organisation names read whole, and the
    candidates of *entities* each one replaces."""
    anchors = [e for e in entities if e.type in _CONTACT]
    fixed = [e for e in entities if e.source == "pattern"]
    added: List[DetectedEntity] = []
    superseded: List[DetectedEntity] = []
    pos = 0
    for m in _NAME_START.finditer(text):
        if m.start() < pos:
            continue
        prev = re.search(r"([^\W\d_][\w'’]*)[ \t]+$", text[max(0, m.start() - 40):m.start()])
        if prev and prev.group(1)[0].isupper() and prev.group(1).lower() not in _NOT_START:
            continue                    # "Smith" in "John Smith"
        name = read_name(text, m.start())
        if name is None:
            continue
        s, e = name.start, name.end
        if any(a < e and s < b for a, b in opaque) or any(x.start < e and s < x.end for x in fixed):
            continue
        if _not_a_firm(text, name):
            pos = e
            continue
        over = [x for x in entities if x.source != "pattern" and x.start < e and s < x.end]
        where = slot(text, s, anchors)
        people = sum(len(_strip_particles(text[a:b]).split()) >= 2 for a, b in name.pieces) >= 2
        if not _accept(name, where, bool(over), people):
            continue
        if name.form == "ambig" and where is None:
            continue
        s = min([s] + [x.start for x in over])
        e = max([e] + [x.end for x in over])
        added.append(DetectedEntity(text[s:e], s, e, "ORG", 0.9, SOURCE))
        superseded.extend(x for x in over if x not in superseded)
        pos = e
    return added, superseded
