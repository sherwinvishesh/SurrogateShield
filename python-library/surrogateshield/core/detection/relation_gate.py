"""
detection/relation_gate.py — Pass R: is this name personal? (audit I12)

NER tags every proper noun: acronyms ("SLA", "W-4"), programming terms
("Django", "Python 3.12.4"), greetings ("Regards"), public figures ("Taylor
Swift"), public companies ("Nvidia") and places that are only the topic of a
question ("the population of Tempe"). Replacing those breaks the question and
protects nobody. The rule the J2 corpus (bench/realworld/GUIDE.md) encodes is
that an organisation or place is personal only when it is **tied to a person**:

    "I work at Mercy General"             ORG   tied   → masked
    "my son goes to Kyrene Middle School" ORG   tied   → masked
    "Compare Microsoft, Apple and Nvidia" ORG   public → kept
    "How far is Tempe from Phoenix?"      GPE   topic  → kept
    "Liam is 7 and we live near Reno"     GPE   tied   → masked

Decision for a NER-type entity (ORG / GPE / LOC / FAC / PERSON):

1. junk — an acronym or code token, a version or code fragment, a greeting
   or closing → kept (for any type; a US state code is left to the others).
2. PERSON — kept only when it is a public figure (gazetteer or an epithet
   such as "Emperor Meiji", "Peter the Great") with no personal cue before
   it ("my coworker …"). Every other PERSON stays masked.
3. ORG / GPE / LOC / FAC — masked when *tied*: a relation cue before it
   ("live in", "work at", "my school", "I'm from"), a relation after it
   ("… where I live", "… hired me"), a person right before it ("Hannah Reyes
   from Acme"), a form/JSON/CSV field label ("City:", "employer"), or a
   signature / e-mail header block. Otherwise:
     · a public organisation or product (gazetteer, whatever NER type it
       got: "on Discord" is often a GPE) → kept;
     · the message identifies a private person (anchored) → masked
       (a place in such a message is most likely that person's place);
     · else → kept.

Dropped entities are returned so the evaluator can report them under the
documented policy ``not_tied_to_person`` (eval_metrics.POLICY_REASONS).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from ..entities import DetectedEntity
from .geo_data import US_STATE_ABBREVS
from .public_names import NOT_NAMES, PUBLIC_ORGS, PUBLIC_PEOPLE

GATED_TYPES = frozenset({"ORG", "GPE", "LOC", "FAC", "PERSON"})
_PLACE_ORG = frozenset({"ORG", "GPE", "LOC", "FAC"})

# Direct identifiers of a private person (pattern types + non-public PERSON).
_ANCHOR_TYPES = frozenset({
    "email", "phone_intl", "phone_us", "phone_uk", "ssn", "dob", "address",
    "credit_card", "us_driver_license", "passport", "id_number",
    "us_bank_number", "iban", "handle", "url", "age", "credential", "PERSON",
})

# ── junk ─────────────────────────────────────────────────────────────────────

_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
_ACRONYM = re.compile(r"^(?=[^a-z]*[A-Z])[A-Z0-9][A-Z0-9&./\-]{0,5}$|^[A-Z]-[a-z]{2,5}$")
_CODE = re.compile(
    r"[()=`\"<>{}\[\];\\]"                  # code punctuation
    r"|\d+\.\d+"                            # version number
)
# model numbers (M404n) and snake_case — never a place/org, but the same
# shape on a PERSON is likely a handle ("sarahm92"), so it stays masked
_MODEL = re.compile(r"_|\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z\d]{3,}\b")


def _core(text: str) -> str:
    return _POSSESSIVE.sub("", text.strip()).strip(" .,;:!?")


# Pronouns, auxiliaries and verbs that never sit inside a person's name
# (particles such as "de", "du", "la", "van", "bin" do, so they are not listed).
_CLAUSE_WORD = re.compile(
    r"(?<![\w'])(?:i|you|we|they|is|are|was|were|want|need|have|has|the|and|to"
    r"|je|tu|nous|vous|veux|veut|est|suis|sont|c'est|ich|wir|ist|und"
    r"|yo|quiero|tengo|es|eu|quero|sou)(?![\w'])"
)


def is_junk(ent: DetectedEntity, text: str = "") -> bool:
    core = _core(ent.text)
    if not core:
        return True
    if core.lower() in NOT_NAMES:
        return True
    if ent.type == "PERSON" and core == core.lower() and _CLAUSE_WORD.search(core):
        return True                         # "je veux vérifier la clé" is a clause
    if _CODE.search(core):
        return True
    if ent.type != "PERSON" and _MODEL.search(core):
        return True
    if _ACRONYM.match(core) and not (text and _shouting(text)):
        letters = sum(c.isalpha() for c in core)
        if ent.type == "PERSON" and letters > 3:
            return False                    # "NOOR" in caps is still a name
        return not (ent.type in ("GPE", "LOC") and core in US_STATE_ABBREVS)
    return False


def _shouting(text: str) -> bool:
    """All-caps message: an upper-case token is not evidence of an acronym."""
    letters = [c for c in text if c.isalpha() and c.isascii()]
    return len(letters) >= 20 and sum(c.isupper() for c in letters) > 0.6 * len(letters)


# ── public figures ───────────────────────────────────────────────────────────

_EPITHET = re.compile(
    r"^(?:Emperor|Empress|King|Queen|Pope|Saint|St\.|Prince|Princess|Sultan"
    r"|Czar|Tsar|Pharaoh|Kaiser|Shah)\s+[A-Z]"
    r"|\s+the\s+(?:Great|Terrible|Elder|Younger|Conqueror|Wise|Bold|Magnificent)$"
)
_PERSONAL_BEFORE_PERSON = re.compile(
    r"(?i)\b(?:my|our|his|her|their|your)\s+(?:[\w'’\-]+\s+){0,2}$"
    r"|\b(?:named|called|dear|hi|hello|hey|dr|mr|mrs|ms|mx)\.?\s*,?\s*$"
)


# Public products that are also given names — never treated as public when
# NER calls them a PERSON ("Chase said he'd call").
_NAME_LIKE_ORGS = frozenset({
    "chase", "mercedes", "claude", "alexa", "siri", "ford", "wise", "prime",
    "edge", "opera", "box", "go", "rust", "swift", "word", "teams", "switch",
    "x", "node", "spring", "orange", "delta", "united", "target", "discover",
    "signal", "threads", "gemini", "ruby", "julia", "dell", "kia", "lg",
    "tesla", "angular", "vue", "flask", "notion", "square", "shell", "visa",
    "harvard", "stanford", "berkeley", "oxford", "cambridge", "yale",
})


def is_public_person(ent: DetectedEntity, text: str) -> bool:
    core = _core(ent.text)
    low = core.lower()
    public = (low in PUBLIC_PEOPLE or bool(_EPITHET.search(core))
              or (low in PUBLIC_ORGS and low not in _NAME_LIKE_ORGS))
    if not public:
        return False
    before = text[max(0, ent.start - 40):ent.start]
    return not _PERSONAL_BEFORE_PERSON.search(before)


# ── ties ─────────────────────────────────────────────────────────────────────

_FILLER = (r"(?:\s+(?:in|at|near|to|outside|around|from|for|with|by|of|on|into|the|a|an|downtown|central|greater|north|south|east|west|beautiful|sunny|rural|suburban|small|little|called|named"
           r"|à|au|aux|en|em|no|na|a|bei|im|in|dans|près\s+de|de|del|da|do|la|le|el)){0,3}\s*$")

_CUE_BEFORE = re.compile(
    r"(?i)(?:"
    # living / location
    r"\b(?:live[sd]?|living|reside[sd]?|residing|resident|based|located|staying"
    r"|stuck|settled|grew\s+up|raised|born|moved|moving|move|relocat\w*"
    r"|commut\w*|hometown|home\s+town|neighbou?rhood|zip|postcode)"
    r"|\b(?:i'?m|i\s+am|im|we'?re|we\s+are|he'?s|he\s+is|she'?s|she\s+is"
    r"|they'?re|they\s+are|currently|still|here|back\s+home)\s+(?:in|at|near|from|outside|based)"
    r"|\b(?:originally|come|comes|came|hails?|is|are|was|were)\s+from"
    # work / study / care
    r"|\b(?:work(?:s|ed|ing)?|employed|employer|employee|job|intern(?:ing|ship)?"
    r"|hired|joined|joining|start(?:s|ed|ing)?\s+(?:a\s+)?(?:job|work|role)"
    r"|manager|engineer|nurse|teacher|developer|analyst|director|doctor|surgeon"
    r"|stud(?:y|ies|ied|ying)|student|attend(?:s|ed|ing)?|enrolled|graduat\w*"
    r"|teach(?:es|ing)?|volunteer(?:s|ed|ing)?|member|patient|treated|admitted"
    r"|goes|go|went|going)\b"
    r"|\b(?:named|called|(?:letter|email|e-mail|note|message|resignation)\s+from"
    r"|on\s+behalf\s+of|signed\s+by|representing|represents|employed\s+by)"
    # possessive relation
    r"|\b(?:my|our|his|her|their|my\s+\w+'s)\s+(?:own\s+)?(?:city|town|village|area"
    r"|neighbou?rhood|school|college|university|uni|employer|company|firm|office"
    r"|workplace|clinic|hospital|doctor|dentist|practice|gym|church|mosque|temple"
    r"|synagogue|bank|landlord|team|club|daycare|kindergarten|job|work|agency"
    r"|startup|lab|department|plant|warehouse|store|shop|restaurant|salon|branch"
    r"|apartment|building|complex|hoa|district|county|parish|ward)\b[^.!?\n]{0,15}?"
    # other languages (es/fr/de/pt/it)
    r"|\b(?:vivo|vive|vivimos|trabajo|trabaja|estudio|j'habite|habite|je\s+vis"
    r"|travaille|wohne|wohnt|arbeite|arbeitet|moro|mora|trabalho|abito|lavoro)"
    r")" + _FILLER
)
_CUE_AFTER = re.compile(
    r"(?i)^(?:'s)?[^.!?\n]{0,20}?\b(?:where\s+(?:i|we|he|she|they|my)\b"
    r"|hired\s+me|offered\s+me|employs\s+me|is\s+(?:my|our|his|her|their)\s+"
    r"(?:employer|school|company|hometown|home|clinic|doctor)"
    r"|-based|\s+based\b)"
)
_FIELD_LABEL = re.compile(
    r"(?i)\b(?:city|town|location|employer|company|organi[sz]ation|org|school"
    r"|university|college|workplace|hometown|home\s*town|residence|clinic"
    r"|hospital|practice|place\s+of\s+(?:work|birth)|birthplace|ciudad|ville"
    r"|stadt|cidade|lugar|from)\s*[\"']?\s*[:=]\s*[\"']?\s*$"
)
_CSV_LABEL = re.compile(
    r"(?i)^\s*[\"']?(?:city|town|location|employer|company|organi[sz]ation|org"
    r"|school|university|college|workplace|hometown|clinic|hospital|state|country)"
    r"[\"']?\s*$"
)
# A company with a legal-form suffix that is not a public brand is a small,
# identifying firm ("DataPulse Inc", "Hollis Property Group") — kept masked.
_LEGAL_SUFFIX = re.compile(
    r"(?i)\b(?:inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|gmbh|ag|sa|s\.a"
    r"|srl|bv|plc|pty|llp|lp|pllc|group|holdings|partners|solutions|associates"
    r"|consulting|studio|studios|labs)\.?$"
)
_CLOSING = re.compile(
    r"(?i)^\s*(?:regards|best(?:\s+regards)?|kind\s+regards|warm\s+regards"
    r"|thanks|thank\s+you|cheers|sincerely|respectfully|yours|cordialement"
    r"|saludos|atenciosamente|--|—|sent\s+from)\b"
)
_HEADER = re.compile(r"(?i)^\s*(?:from|to|cc|bcc|reply-to|sender)\s*:")
# "<person> from|at|of|with|representing <X>", "<tied org> in <city>"
_LINK_BETWEEN = re.compile(
    r"(?i)\s*,?\s*(?:(?:who|a|an|the|our|is|was|works?|worked|working|now|currently"
    r"|based|lives?|living|\w+(?:er|or|ist|ant|ent|ing))\s+){0,3}"
    r"(?:from|at|of|with|for|in|near|representing|by|out\s+of)\s+(?:the\s+)?"
)
_PERSON_LINK = re.compile(r"(?:,\s*|\s+(?:from|at|of|with|@|—|-|–|\|)\s+)$")


def _line_bounds(text: str, pos: int) -> Tuple[int, int]:
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e == -1 else e)


def _sentence_before(text: str, pos: int) -> str:
    cut = max(text.rfind(c, 0, pos) for c in ".!?\n")
    # don't cut on abbreviations/decimals inside the window
    return text[cut + 1:pos] if pos - cut <= 120 else text[pos - 120:pos]


def _csv_column_label(text: str, ent: DetectedEntity) -> bool:
    ls, le = _line_bounds(text, ent.start)
    line = text[ls:le]
    if line.count(",") < 2:
        return False
    col = text[ls:ent.start].count(",")
    for header in text[:ls].split("\n"):
        cells = header.split(",")
        if len(cells) == line.count(",") + 1:
            return bool(_CSV_LABEL.match(cells[col]))
    return False


def _in_signature_or_header(text: str, ent: DetectedEntity,
                            persons: List[DetectedEntity]) -> bool:
    ls, le = _line_bounds(text, ent.start)
    line = text[ls:le]
    if _HEADER.match(line):
        return True
    if "\n" not in text or len(line.split()) > 8 or line.rstrip().endswith(("?", "!")):
        return False
    lines = text[:ls].split("\n")[-4:]
    if any(_CLOSING.match(l) for l in lines):
        return True
    # a short line right after a line that holds a person / contact detail
    prev_start = ls - sum(len(l) + 1 for l in lines[-2:])
    return any(p.start >= prev_start and p.end <= ls for p in persons)


# Identifiers whose issuer/brand is itself revealing ("Visa 4111…", "my
# Chase account 0123…") — an ORG directly before one belongs to the person.
_ISSUED_TYPES = frozenset({"credit_card", "us_bank_number", "iban", "id_number",
                           "passport", "us_driver_license"})


def is_tied(ent: DetectedEntity, text: str, persons: List[DetectedEntity],
            context: Iterable[DetectedEntity] = ()) -> bool:
    for c in context:
        if (c.type in _ISSUED_TYPES and 0 <= c.start - ent.end <= 60
                and not re.search(r"[.!?\n]", text[ent.end:c.start])):
            return True
    before = _sentence_before(text, ent.start)
    if _CUE_BEFORE.search(before):
        return True
    ls, _ = _line_bounds(text, ent.start)
    if _FIELD_LABEL.search(text[ls:ent.start]):
        return True
    after = text[ent.end:ent.end + 40]
    if _CUE_AFTER.match(after):
        return True
    for p in persons:
        if 0 <= ent.start - p.end <= 4 and _PERSON_LINK.search(text[p.end:ent.start] or " "):
            return True
        if 0 <= ent.start - p.end <= 40 and _LINK_BETWEEN.fullmatch(text[p.end:ent.start]):
            return True
    if _csv_column_label(text, ent):
        return True
    return _in_signature_or_header(text, ent, persons)


# ── the pass ─────────────────────────────────────────────────────────────────

def gate(
    text: str,
    entities: Iterable[DetectedEntity],
    context: Iterable[DetectedEntity] = (),
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """Split *entities* into (kept, dropped). *context* are other entities of
    the same message (used for anchoring and person links, never dropped)."""
    entities = list(entities)
    context = list(context)
    dropped: List[DetectedEntity] = []

    def _drop(e):
        dropped.append(e)
        return False

    # 1–2: junk and public people first — they never anchor anything
    first = []
    for e in entities:
        if e.type not in GATED_TYPES:
            first.append(e)
        elif is_junk(e, text):
            _drop(e)
        elif e.type == "PERSON" and is_public_person(e, text):
            _drop(e)
        else:
            first.append(e)

    survivors = first + [c for c in context if c not in entities]
    persons = [e for e in survivors if e.type == "PERSON"]
    anchored = any(e.type in _ANCHOR_TYPES for e in survivors)

    # ties propagate: "Tariq Mansoor representing Aramco out of Flagstaff",
    # "a manager at Dangote Group in Burlington"
    tied: List[DetectedEntity] = []
    pending = [e for e in first if e.type in _PLACE_ORG]
    changed = True
    while changed:
        changed = False
        for e in list(pending):
            if is_tied(e, text, persons + tied, survivors):
                tied.append(e)
                pending.remove(e)
                changed = True
    tied_ids = {id(e) for e in tied}

    kept = []
    for e in first:
        if e.type not in _PLACE_ORG or id(e) in tied_ids:
            kept.append(e)
        elif (e.type == "ORG" and _LEGAL_SUFFIX.search(_core(e.text))
              and _core(e.text).lower() not in PUBLIC_ORGS):
            kept.append(e)
        elif _core(e.text).lower() in PUBLIC_ORGS:
            _drop(e)
        elif anchored:
            kept.append(e)
        else:
            _drop(e)
    return kept, dropped
