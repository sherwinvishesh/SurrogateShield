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
from dataclasses import replace as _dc_replace
from functools import lru_cache
from importlib import import_module
from typing import Dict, Iterable, List, Optional, Tuple

from ..entities import DetectedEntity
from .geo_data import US_STATE_ABBREVS
from .pattern_scan import _KIN as _KIN_WORDS
from .public_names import NOT_NAMES, PUBLIC_ORGS, PUBLIC_PEOPLE, WORD_NAMES, is_listed_public_person
from . import org_assembly

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


def _unquote(core: str) -> str:
    """Matching outer quotes removed: '"Kalinda Whitehorse"' -> 'Kalinda Whitehorse'."""
    m = re.fullmatch(r"([\"'“‘«`])([^\"'“”‘’«»`]+)([\"'”’»`])", core)
    return m.group(2) if m else core


def _core(text: str) -> str:
    return _POSSESSIVE.sub("", text.strip()).strip(" .,;:!?")


# Pronouns, auxiliaries and verbs that never sit inside a person's name
# (particles such as "de", "du", "la", "van", "bin" do, so they are not listed).
_CLAUSE_WORD = re.compile(
    r"(?<![\w'])(?:i|you|we|they|is|are|was|were|want|need|have|has|the|and|to"
    r"|je|tu|nous|vous|veux|veut|est|suis|sont|c'est|ich|wir|ist|und"
    r"|yo|quiero|tengo|es|eu|quero|sou|como|hai|hoon|hain|mein|karna|kya|aur)(?![\w'])"
)


# ── non-English frames around a name (en NER on de / nl / hi / pt text) ─────

# Lower-case words that never start or end a name: "ich bin Jörg Baumgartner",
# "Mera naam Rohit Bhardwaj hai", "ik ben Sanne", "PAN se".
_EDGE_WORDS = frozenset("""
ich bin und ik ben je uit naam hai hoon hain mein main se ka ki ke ko aur kya
karna nome é me llamo soy suis est como sono chiamo heet heiße heisse i am im
is and the name called named from at with to for in of on hi hello hey dear
hallo bonjour hola oi ciao tiene tenía tem tinha vive mora estudia trabaja está
""".split())
# The same frame words written with a capital at the start of a sentence
# ("Ich heiße …", "Hallo, ik ben …"); "Ben", "Main" and "Bin" are names.
_EDGE_CAPITAL = frozenset("ich ik je hallo bonjour hola hi hello hey dear naam".split())
# Possessives that make the next word a noun ("Mein Vermieter", "Mijn BSN").
# "Mia" and "Ma" are names, so they are not listed.
# titles before a name: never part of it ("Don" and "Sr" are names or
# ambiguous, so they are not listed)
_TITLE_WORDS = frozenset("""
mr mrs ms mx dr prof father fr sister brother rev reverend pastor imam rabbi professor
professora doutor doutora dra dott dottor dottore dottoressa signor signora señor señora sra
srta herr frau monsieur madame mme mlle maître mevrouw meneer dhr mevr
""".split())
_ARTICLES = frozenset("il lo la le el o a os as der die das the".split())
_POSSESSIVE_DET = frozenset("""
my mein meine meinem meinen meiner unser unsere mijn onze mon mes notre mi mis
nuestro nuestra meu minha nosso nossa mio nostro nostra mera meri mere hamara
hamari
""".split())
_NAME_PARTICLES = frozenset("""
de del della der den di da das do dos du la le van von zu ten ter bin binti
ibn al el y e wa ap af av st
""".split())


_COORD = frozenset("and und en et y e & + / , og och i".split())


# A Gaelic surname mutated after "Ó", "Ní", "Mac" or "Uí" ("Ó hÉinniú", "Uí
# nGallchóir"): a lower-case prefix, then the capitalised name
_IRISH_MUTATION = re.compile(r"(?:h|n|t|bh|bp|dt|gc|mb|ng)(?=[^\W\d_])")


def mutated_name(w: str) -> bool:
    """A mutated Irish name word ("hÉinniú", "nGallchóir")."""
    m = _IRISH_MUTATION.match(w)
    return bool(m) and w[m.end():m.end() + 1].isupper()


def _is_name_word(w: str) -> bool:
    return w[:1].isupper() or w.lower() in _NAME_PARTICLES or mutated_name(w)


def trim_person(ent: DetectedEntity, text: str):
    """A model PERSON cut down to the name inside a sentence frame, or None
    when no name is left. ``ent`` itself when nothing changes.

    "ich bin Jörg Baumgartner" -> "Jörg Baumgartner"; "Mera naam Rohit
    Bhardwaj hai" -> "Rohit Bhardwaj"; "Mein Vermieter" (my landlord) and
    "Kun je mijn bezwaarschrift aan de Belastingdienst" -> None. A lower-case
    remainder is left to is_junk; one stray lower-case word ("maria Lopez")
    keeps the whole span (fail closed)."""
    toks = list(re.finditer(r"\S+", ent.text))
    i, j = 0, len(toks)
    had_possessive = False
    while i < j:
        w = toks[i].group()
        if w.lower().rstrip(".") in _TITLE_WORDS and i + 1 < j:
            pass                            # "Father Eamon Kirwan" -> "Eamon Kirwan"
        elif (w.lower() in _ARTICLES and i + 2 < j
              and toks[i + 1].group().lower().rstrip(".") in _TITLE_WORDS):
            pass                            # "il dottor Bellandi" -> "Bellandi"
        elif w.lower() in _POSSESSIVE_DET:
            had_possessive = True
        elif not (w.lower() in _EDGE_WORDS and (w.islower() or w.lower() in _EDGE_CAPITAL)):
            break
        i += 1
    while j > i and toks[j - 1].group().islower() and toks[j - 1].group() in _EDGE_WORDS:
        j -= 1
    if (i < j and toks[i].group()[:1].isupper() and text != text.lower()
            and (i, j) != (0, len(toks))):
        # after a title or frame: a lower-case verb ending the span is not
        # the name ("professora Renata pediu" -> "Renata")
        while (j - 1 > i and toks[j - 1].group().islower()
               and toks[j - 1].group() not in _NAME_PARTICLES):
            j -= 1
    if i == j:
        return None
    words = [t.group() for t in toks[i:j]]
    if not any(w[:1].isupper() for w in words):
        return ent                          # lower-case chat: is_junk decides
    if had_possessive and len(words) == 1:
        return None                         # "Mein Vermieter": my landlord
    loose = [w for w in words if not _is_name_word(w)]
    if loose:
        runs, run = [], []
        for k in range(i, j):
            w = toks[k].group()
            if w[:1].isupper() or mutated_name(w) or (run and w.lower() in _NAME_PARTICLES):
                run.append(k)
            else:
                runs.append(run)
                run = []
        runs.append(run)
        named = [r for r in runs if any(toks[k].group()[:1].isupper() for k in r)]
        if all(w.lower() in _COORD for w in loose):
            return ent                      # "Ngozi Adeyemi + Chidi Adeyemi"
        for r in runs:
            while r and not (toks[r[-1]].group()[:1].isupper()
                             or mutated_name(toks[r[-1]].group())):
                r.pop()                     # no trailing particle
        long_runs = [r for r in runs if len(r) >= 2]
        if len(long_runs) == 1:
            i, j = long_runs[0][0], long_runs[0][-1] + 1   # "Kun je Anna Müller bellen"
        elif long_runs:
            return ent                      # two names in a clause: keep it all
        elif len(loose) >= 2 or any(w.lower() in _EDGE_WORDS or _CLAUSE_WORD.fullmatch(w.lower())
                                    for w in loose):
            return None                     # a clause with a capital in it
        else:
            return ent
    if (i, j) == (0, len(toks)):
        return ent
    s, e = ent.start + toks[i].start(), ent.start + toks[j - 1].end()
    if text[s:e] != ent.text[toks[i].start():toks[j - 1].end()]:
        return ent                          # offsets disagree: fail closed
    return _dc_replace(ent, text=text[s:e], start=s, end=e)


# "user.getFirstName", "UserDTO.lastName" — a member access, never a name
_MEMBER_ACCESS = re.compile(r"\w\.[a-z]+[A-Z]\w*|^[a-z]+\.[a-z_]\w*$")
_DEGREES = frozenset("msc bsc ba ma mba phd dphil mphil md llb llm bed med beng meng ms bs "
                     "mres pgce bcom mcom btech mtech".split())
_JOB_TITLE = re.compile(
    r"(?i)\b(?:analyst|engineer|developer|designer|scientist|manager|specialist|coordinator"
    r"|assistant|officer|director|consultant|administrator|architect|technician|associate"
    r"|representative|intern|lead|executive|accountant)$")
# the noun of an automated sender ("Chase Alert", "Amazon Support")
_SENDER_NOUNS = frozenset("alert alerts notification notifications notice support team "
                          "service services update updates reminder security billing".split())
# Single capitalised common nouns NER calls a place or company ("Port of
# entry", "Passport C7HX…", "met at Uni", "Budget: $25k"); with a number
# they are a room or a port ("Room 1208", "Port 22").
_COMMON_NOUNS = frozenset("""
port passport uni university college school budget room suite floor gate terminal
platform office lobby reception kitchen garden visa account invoice receipt order
ticket station airport hotel hospital clinic church court bank branch department
dept building block level unit hall campus library pharmacy embassy consulate
""".split())
_DE_ARTICLE = re.compile(r"(?i)\b(?:der|die|das|den|dem|des|ein|eine|einen|einem|einer|keine?n?"
                         r"|meine?[nmrs]?|deine?[nmrs]?|seine?[nmrs]?|ihre?[nmrs]?|unse?re?[nmrs]?)\s+$")
_DE_TEXT = re.compile(r"(?i)\b(?:und|ich|nicht|ist|wie|kann|mein|meine|will|zurück\w*|was|wir)\b")


def is_common_noun(ent: DetectedEntity, text: str) -> bool:
    """A place/company that is an ordinary noun: dropped unless tied."""
    core = _core(ent.text)
    words = core.split()
    if not words or not words[0].lower() in _COMMON_NOUNS and not (
            len(words) == 1 and _DE_ARTICLE.search(text[max(0, ent.start - 12):ent.start])
            and len(_DE_TEXT.findall(text)) >= 2):
        return False                        # "die Kaution" in German
    return len(words) == 1 or (len(words) == 2 and re.fullmatch(r"[A-Z]?\d{1,5}[A-Z]?", words[1]) is not None)


# ── non-English messages: function words inside a model span ──────────────

_LANGS = ("de", "fr", "es", "it", "pt", "nl")


@lru_cache(maxsize=1)
def _stop_words() -> dict:
    """spaCy's stop-word list per language (shipped with spaCy, offline)."""
    out = {l: frozenset(import_module(f"spacy.lang.{l}.stop_words").STOP_WORDS)
           for l in _LANGS + ("en",)}
    return out


# Romanised Hindi / Urdu function words: "mera naam … hai, Nagpur se" is
# not Italian although "se", "hai" and "ma" are Italian stop words.
_HINGLISH = frozenset("""
hai hain hoon tha thi the mera meri mere mujhe mujhko naam ka ki ke ko se aur
nahi nahin kya kyun karna karo chahiye bhai yaar ap aap apna hum tum wala wali
""".split())


@lru_cache(maxsize=256)
def message_language(text: str):
    """The language of a message by its function words, or None for English,
    romanised Hindi, or when unsure: at least three stop words of one of
    de/fr/es/it/pt/nl, and more of them than English or Hinglish ones."""
    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.lower())
    if len(words) < 4:
        return None
    stops = _stop_words()
    counts = {l: sum(w in stops[l] and (l == "en" or w not in stops["en"]) for w in words)
              for l in _LANGS + ("en",)}      # "in", "me", "die" count for English
    best = max(_LANGS, key=lambda l: counts[l])
    other = max(counts["en"], sum(w in _HINGLISH for w in words))
    return best if counts[best] >= 3 and counts[best] >= 2 * other else None


# Stop words that are also given names or surnames (from Faker's name lists
# for 24 locales): capitalised, they stay names ("Ik ben Ben", "Sara").
_STOP_NAMES = frozenset("""
allen aller anders ben bent dan elke lang lange gross gute mir einer tage aura bas
font mille sans sur ante bien bueno dar dias diez ella esa grande hasta mas mia mio
nada pero salvo sean sola tan uno alla anni bravo chi coll dalla ebbe essi forza li
lo mai mie mila otto prima sara sue tali volta ali estes esteve nova sem moet om uwe
""".split())
# Words that open a German noun phrase or close a request, never a name
_DE_OPENERS = frozenset("der die das den dem des ein eine einen einem einer sie er es wir ich".split())
_DE_NOUN = re.compile(r"(?:ung|heit|keit|schaft|it[äa]t|ismus|nummer|vertrag|gesch[äa]fte?)$")


def is_foreign_fragment(ent: DetectedEntity, text: str) -> bool:
    """A model span in a de/fr/es/it/pt/nl message that is part of a clause,
    not a name: "Kannst du diese", "Ele nasceu", "Der Vertrag", "Rufen Sie",
    "een bezwaarbrief aan de gemeente", "Kündigung". Only model spans (an
    English NER model reading another language); a capitalised name inside
    the span ("ich bin Jörg Baumgartner") was cut out by trim_person."""
    if ent.source in ("pattern", "structural") or not text or text == text.lower():
        return False
    lang = message_language(text)
    if lang is None:
        return False
    stop = _stop_words()[lang]
    found = list(re.finditer(r"[^\W\d_]+(?:['’][^\W\d_]+)?", ent.text))
    toks = [m.group() for m in found]
    if not toks:
        return False

    def name_like(k: int) -> bool:          # a capitalised word that is no function word
        t = toks[k]
        return t[0].isupper() and (t.lower() not in stop or t.lower() in _STOP_NAMES)

    last_low = max((k for k, t in enumerate(toks) if t[0].islower() and t in stop), default=-1)
    if lang == "de":                        # every German noun is capitalised: only a
        if any(name_like(k) for k in range(last_low + 1, len(toks))) and last_low >= 0:
            return False                    # word after a preposition ("bei Kraftwerk")
    elif any(name_like(k) and not re.search(r"(?:^|[.!?¿¡:\n]\s*)$",
                                            text[:ent.start + found[k].start()])
             for k in range(len(toks))):
        return False                        # "Nagpur se": a name mid-sentence
    low = [t for t in toks if t[0].islower()]
    caps = [t for t in toks if t[0].isupper()]
    if low and any(t not in _NAME_PARTICLES for t in low) and any(t in stop for t in low):
        return True                         # a clause: "mir helfen", "al mio medico"
    if caps and all(t.lower() in stop and t.lower() not in _STOP_NAMES for t in caps) and (
            low or len(caps) == 1):
        return True                         # "Ele nasceu", "Mi aiuti", "Mon"
    if lang == "de":
        if toks[0].lower() in _DE_OPENERS and len(toks) <= 3:
            return True                     # "Der Vertrag"
        if len(toks) >= 2 and toks[-1].lower() in ("sie", "du", "ihr", "wir", "uns"):
            return True                     # "Rufen Sie"
        if len(toks) == 1 and _DE_NOUN.search(toks[0].lower()):
            return True                     # "Kündigung", "Mitgliedsnummer"
    if len(toks) == 1 and low and ent.type == "PERSON":
        return toks[0] in stop              # a lower-case function word alone
    return False


# A title alone is not a name ("Mr" of "Mr. Okonkwo-Hale" when NER splits it).
_HONORIFICS = frozenset("""
mr mrs ms mx miss dr prof sir madam herr frau mme mlle sr sra srta dott sig
""".split())
# An object pronoun after a sentence-initial word makes that word a verb:
# "Ping me on …", "Bel me even op …", "Ruf mich an". Not in French, Spanish,
# Italian or Portuguese, where the pronoun comes before the verb ("Ana me
# ligou", "Marie me dit").
_OBJECT_PRONOUN = {None: re.compile(r"[ \t]+(?:me|us)\b"),
                   "de": re.compile(r"[ \t]+(?:mich|uns)\b"),
                   "nl": re.compile(r"[ \t]+(?:me|mij|ons)\b")}
_SENTENCE_START = re.compile(r"(?:^|[.!?\n]\s*|^\s*[-*•]\s*)$")


_COLUMN_NAME = re.compile(
    r"(?i)^(?:(?:first|last|full|user|sur|given|family)?\s*_?names?|age|e-?mail|phone|mobile|tel"
    r"|date|dob|id|address|city|town|country|role|title|status|type|notes?|amount|qty"
    r"|department|dept|team|company|employer|school|class|grade|gender|salary|start|end)$")


def _header_cells(text: str) -> set:
    """Cells of the first row of a CSV / TSV / pipe table: column names
    ("Name,Age,Allergy,Parent phone"), never values. The row has no digit,
    three or more cells, one of them a usual column name, and the next line
    has as many separators. A headerless table ("Maria Lopez,Sales,Berlin")
    has no header row."""
    cells = set()
    lines = text.split("\n")
    for k in range(len(lines) - 1):
        row, nxt = lines[k], lines[k + 1]
        prev = lines[k - 1] if k else ""
        for sep in (",", "\t", ";", "|"):
            n = row.count(sep)
            if (n >= 2 and nxt.count(sep) == n and prev.count(sep) != n
                    and not any(c.isdigit() for c in row)):
                row_cells = {c.strip(" *`") for c in row.split(sep) if c.strip()}
                if any(_COLUMN_NAME.match(c) for c in row_cells):
                    cells |= row_cells
    return cells


def _defined_terms(text: str) -> set:
    """Defined terms of a contract: ("Tenant"), ("the Landlord")."""
    return {m.group(1) for m in re.finditer(
        r"\(\s*[\"“'‘](?:the\s+)?([A-Z][a-z]+)[\"”'’]\s*\)", text)}


def is_junk(ent: DetectedEntity, text: str = "") -> bool:
    core = _core(ent.text)
    if not core or not any(c.isalpha() for c in core):
        return True                         # "&" from "Mr. & Mrs. Castellano"
    if core.lower() in NOT_NAMES:
        return True
    if is_foreign_fragment(ent, text):
        return True
    if core.lower() in _HONORIFICS or core.lower() in _POSSESSIVE_DET:
        return True                         # "Mr", "Dr", "mera"
    pair = core.split()
    if (ent.type == "PERSON" and len(pair) == 2 and core == core.lower() and pair[0] in _POSSESSIVE_DET
            and text != text.lower()):
        return True                         # "mera account" in a cased message
    if text and " " not in core and ent.source not in ("pattern", "structural"):
        pronoun = _OBJECT_PRONOUN.get(message_language(text))
        if (pronoun and core[:1].isupper() and _SENTENCE_START.search(text[:ent.start])
                and pronoun.match(text, ent.end)):
            return True                     # "Ping me", "Bel me"
        if core in _header_cells(text) or core in _defined_terms(text):
            return True                     # "Name,Age,Allergy", ("Tenant")
    words = core.split()
    if _QUESTION_WORD.match(core):
        return True                         # "Combien de calories", "¿Cuáles son …"
    if ent.type == "PERSON" and len(words) >= 2 and any(c.isdigit() for c in core):
        return True                         # "mas faltam 8 itens", "ORD-562939 chegou"
    if (ent.type == "PERSON" and core == core.lower() and text and text != text.lower()
            and _ARTICLE.match(core)):
        return True                         # "las ventajas" in a cased sentence
    if ent.type == "PERSON" and core == core.lower() and _CLAUSE_WORD.search(core):
        return True                         # "je veux vérifier la clé" is a clause
    if _CODE.search(_unquote(core)) or _MEMBER_ACCESS.search(core):
        return True                         # a quoted name ("Kalinda Whitehorse") is not code
    if core.lower() in _DEGREES:
        return True                         # "MSc from Politehnica"
    if ent.type != "PERSON" and len(words) >= 2 and _JOB_TITLE.search(core) and text and re.match(
            r"\s+(?:role|position|job|post|vacancy|opening|internship)\b", text[ent.end:ent.end + 20], re.I):
        return True                         # "a Data Analyst role at Spotify"
    if ent.type == "PERSON" and len(words) >= 2 and words[-1].lower() in _SENDER_NOUNS:
        return True                         # "Chase Alert: verify …"
    if ent.type != "PERSON" and _MODEL.search(core):
        return True
    if _ACRONYM.match(core) and not (text and _shouting(text)):
        letters = sum(c.isalpha() for c in core)
        if ent.type == "PERSON" and letters > 3:
            return False                    # "NOOR" in caps is still a name
        return not (ent.type in ("GPE", "LOC") and core in US_STATE_ABBREVS)
    return False


# A span that opens with a question word is a question, not a name or org.
_QUESTION_WORD = re.compile(
    r"(?i)^[¿¡]?(?:combien|pourquoi|comment|quand|quel(?:le)?s?|lequel|où"
    r"|cu[aá]l(?:es)?|cu[aá]nt[oa]s?|qu[eé]|c[oó]mo|d[oó]nde|qui[eé]n(?:es)?|por\s+qu[eé]"
    r"|quais|qual|quantos?|onde|porque|warum|wieso|weshalb|welche[rsnm]?|wann|wieviel"
    r"|perch[eé]|quanti|dove|chi)(?![\w'’])"
)
_ARTICLE = re.compile(r"^(?:las|los|el|la|les|le|des|die|der|das|den|dem|os|as|una|une|ein|eine)\s")


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


# A person named as the author, subject or star of a public work: "the plot
# of Chimamanda Ngozi Adichie's …", "novels similar to Haruki Murakami's …",
# "Which Shah Rukh Khan film …", "who was Mumtaz Mahal?"
_WORK_BEFORE = re.compile(
    r"(?i)\b(?:similar\s+to|plot\s+of|summar(?:y|ize|ise)\s+of|(?:novels?|books?|poems?"
    r"|songs?|albums?|films?|movies?|plays?|paintings?|works?|essays?|lyrics)\s+(?:by|of)"
    r"|written\s+by|directed\s+by|painted\s+by|composed\s+by|sung\s+by|starring"
    r"|biography\s+of|history\s+of|quotes?\s+(?:by|from))\s+(?:the\s+)?$"
)
_WORK_AFTER = re.compile(
    r"(?i)^(?:'s)?\s+(?:films?|movies?|songs?|albums?|books?|novels?|poems?|plays?"
    r"|paintings?|shows?|series|quotes?|speech|lyrics|discography|filmography)\b"
)


# "this introductory statement written by Olwethu Dlamini": the author of
# the user's own draft, not of a public work.
_OWN_DRAFT_BY = re.compile(
    r"(?i)\b(?:this|that|my|our|his|her|their|the\s+attached|the\s+following)\s+"
    r"(?:[\w'’\-]+\s+){0,5}(?:statements?|bios?|biography|essays?|letters?|paragraphs?|drafts?"
    r"|introductions?|intros?|e-?mails?|notes?|posts?|messages?|cvs?|resumes?|résumés?|summary"
    r"|profiles?|proposals?|reports?|applications?|speech|story|stories|poems?|texts?)\s+"
    r"(?:written|drafted|prepared|composed|sent|submitted|put\s+together)\s+by\s+$")


def is_public_person(ent: DetectedEntity, text: str) -> bool:
    core = _core(ent.text)
    low = core.lower()
    before = text[max(0, ent.start - 40):ent.start]
    if _PERSONAL_BEFORE_PERSON.search(before):
        return False
    if _OWN_DRAFT_BY.search(text[max(0, ent.start - 120):ent.start]):
        return False
    if low in PUBLIC_PEOPLE or is_listed_public_person(core) or _EPITHET.search(core) or (
            low in PUBLIC_ORGS and low not in _NAME_LIKE_ORGS):
        return True
    if " " not in core:
        return False                        # one name alone is never "public"
    after = text[ent.end:ent.end + 30]
    return bool(_WORK_BEFORE.search(before) or _WORK_AFTER.match(after)
                or (re.search(r"(?i)\bwho\s+(?:was|is)\s+$", before)
                    and after.lstrip().startswith("?")))


# Something that points to a person next to a word-name: a greeting or an
# addressee before it, a person's action or relation after it, a sign-off.
_PERSON_BEFORE = re.compile(
    r"(?i)(?:\b(?:(?<!compared\s)(?<!similar\s)(?<!next\s)(?<!close\s)(?<!equal\s)(?<!relative\s)to"
    r"|from|tell|ask|told|asked|met|meet|call|text|email|cc|thanks"
    r"|thank\s+you|dear|hi|hey|hello|love|sincerely|regards)\s*,?\s*$|(?:^|\n)\s*[-–—~]\s*$)"
)
_PERSON_AFTER = re.compile(
    r"(?i)^(?:'s\s+(?:mom|mum|dad|mother|father|wife|husband|partner|son|daughter"
    r"|kid|baby|birthday|party|wedding|phone|number|email|address|house|place|car"
    r"|school|teacher|boss|room|condition|surgery|diagnosis)\b"
    r"|,?\s+(?:said|says|told|asked|wants?|wanted|texted|called|emailed|replied|thinks"
    r"|needs?|lives|lived|works|worked|got|gets|has|had|is\s+(?:my|our|a|an|coming"
    r"|going|sick|pregnant|turning|getting)|was\s+(?:my|our|born|diagnosed)"
    r"|are\s+(?:my|our|coming|going|visiting|staying|both)|were\s+(?:my|our|born)"
    r"|and\s+(?:i|me|my)|will|can|could|would|keeps|just)\b"
    r"|,\s+(?:my|our|his|her|their)\b)"
)


_LIST_BEFORE = re.compile(r"(?:[A-Z][\w'\-]*\s*(?:,\s*(?:and\s+|&\s+|or\s+)?|\s+(?:and|&|or)\s+))+$")
_LIST_AFTER = re.compile(r"(?:\s*(?:,\s*(?:and\s+|&\s+|or\s+)?|\s+(?:and|&|or)\s+)[A-Z][\w'\-]*)+")


def is_word_name(ent: DetectedEntity, text: str) -> bool:
    """A lone PERSON that is also an ordinary word, with no sign of a person
    around it ("What rhymes with Joy, Hunter and Destiny?"), or any lone
    lower-case PERSON after an article ("under the matt")."""
    core = _core(ent.text)
    if " " in core:
        return False
    before = text[max(0, ent.start - 40):ent.start]
    if core.islower() and re.search(r"(?i)\b(?:the|a|an|this|that)\s+$", before):
        return True
    devanagari = re.fullmatch(r"[\u0900-\u097f]+", core) is not None
    if devanagari:
        # NER tags Hindi adjectives as PERSON ("अच्छे" = good); a lone word
        # that is no known Hindi name is a word
        from .structural import _lexicons
        _, hi_first, hi_last = _lexicons()
        if core in hi_first or core in hi_last:
            return False
    elif core.lower() not in WORD_NAMES:
        return False
    # read the evidence around the whole list: "my friends Hope and Will are …"
    lead = _LIST_BEFORE.search(before)
    if lead:
        before = before[:lead.start()]
    after = text[ent.end:ent.end + 80]
    trail = _LIST_AFTER.match(after)
    if trail:
        after = after[trail.end():]
    personal = (_PERSONAL_BEFORE_PERSON.search(before) or _PERSON_BEFORE.search(before)
                or _PERSON_AFTER.match(after)
                or re.search(rf"(?i)\b{_KIN_WORDS}\s+$", before))
    return not personal


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
    r"|apartment|building|complex|hoa|district|county|parish|ward|boss|manager"
    r"|supervisor|coworkers?|co-workers?|colleagues?|midwife|therapist|lawyer|gp"
    r"|landlady|kids?'?|son'?s|daughter'?s)\b[^.!?\n]{0,15}?"
    # "I'm a Unite shop steward at the Dagenham plant", "I'm a nurse at"
    r"|\b(?:i'?m|i\s+am|im|i\s+work\s+as)\s+(?:a|an)\s+[^.!?\n,]{1,40}?\s+(?:at|for|with)"
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
    r"|^\s*[?.!]\s*(?:she|he)\s+(?:is|was|keeps|kept|has|had|said|says|told|sent|texted)\b"
)
# a device named after its owner: "Lucas-iPhone", "Priya's MacBook Pro"
_DEVICE = re.compile(
    r"^[A-Z][a-z]{2,15}(?:['’]?s)?[-_ ](?:iPhone|iPad|MacBook|Mac|Galaxy|Pixel|PC"
    r"|Laptop|Desktop|Phone|Tablet|Watch|Kindle|Echo|TV)\b")
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


# A public company is personal only as an employer or school: "my Chase
# account", "my MacBook's serial", "Vodafone customer number" name a customer
# of a company millions use, and masking it protects nobody.
_PUBLIC_CUE = re.compile(
    r"(?i)(?:\b(?:work(?:s|ed|ing)?|job|employed|intern(?:ing)?|hired|stud(?:y|ying|ied|ent)"
    r"|enrolled|graduated|attend(?:s|ed|ing)?|teach(?:es|ing)?|taught)\b[^.!?\n]{0,25}"
    r"|\b(?:i['’]?m|i\s+am|im)\s+(?:a|an)\s+[^.!?\n,]{1,40}?\s+)"
    r"\b(?:at|for|with|by|from|in)\s+(?:the\s+)?$"
    # "our cloud infrastructure at Microsoft": the writer's employer
    r"|\bour\s+(?!(?:[\w'’\-]+\s+){0,3}?(?:account|subscription|order|plan|card|membership)s?\b)"
    r"(?:[\w'’\-]+\s+){1,4}at\s+$")


_PRODUCT_WORD = re.compile(
    r"(?i)^(?:marketplace|store|shop|pay|music|prime|drive|maps|support|account|card|app"
    r"|plus|premium|business|terminal(?:\s*\d)?|t\d|airport)$")


def is_public_org(core: str) -> bool:
    """"Chase", "Facebook Marketplace", "Heathrow T5"."""
    low = core.lower()
    if low in PUBLIC_ORGS:
        return True
    words = core.split()
    return len(words) == 2 and words[0].lower() in PUBLIC_ORGS and bool(_PRODUCT_WORD.match(words[1]))


def is_tied(ent: DetectedEntity, text: str, persons: List[DetectedEntity],
            context: Iterable[DetectedEntity] = ()) -> bool:
    public = is_public_org(_core(ent.text))
    if _DEVICE.match(_core(ent.text)) and not public:
        return True
    for c in context:
        if (c.type in _ISSUED_TYPES and 0 <= c.start - ent.end <= 60 and not public
                and not re.search(r"[.!?\n]", text[ent.end:c.start])):
            return True
    before = _sentence_before(text, ent.start)
    if (_PUBLIC_CUE if public else _CUE_BEFORE).search(before):
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
    # an organisation in a field, a work relation or a sign-off made for one
    # ("Company name: …", "hired by …", "— Ana Ruiz | … |"); a public one only
    # as an employer or school
    if ent.type == "ORG":
        where = org_assembly.slot(text, ent.start, list(persons) + [
            c for c in context if c.type in org_assembly._CONTACT and c not in persons], ent.end)
        if where in ("label", "work") or where in ("link", "closing") and not public:
            return True
    return _in_signature_or_header(text, ent, persons)


# ── the pass ─────────────────────────────────────────────────────────────────

def gate(
    text: str,
    entities: Iterable[DetectedEntity],
    context: Iterable[DetectedEntity] = (),
    reasons: Optional[Dict[int, str]] = None,
    vouched: Iterable[DetectedEntity] = (),
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """Split *entities* into (kept, dropped). *context* are other entities of
    the same message (used for anchoring and person links, never dropped).
    *reasons*, if given, receives ``id(entity) -> rule`` for every drop
    (``junk``, ``public_person``, ``word_name``, ``public_org``,
    ``common_noun``, ``untied``); it changes nothing else. *vouched*: those
    of *entities* a detector is sure are the writer's (a stage's
    ``gate_above``): a place or an organisation among them counts as tied,
    and still faces every other rule."""
    entities = list(entities)
    context = list(context)
    sure = {id(e) for e in vouched}
    dropped: List[DetectedEntity] = []

    def _drop(e, rule):
        dropped.append(e)
        if reasons is not None:
            reasons[id(e)] = rule
        return False

    # 1–2: junk and public people first — they never anchor anything
    first = []
    for e in entities:
        if e.type not in GATED_TYPES:
            first.append(e)
        elif is_junk(e, text):
            _drop(e, "junk")
        elif e.type == "PERSON" and is_public_person(e, text):
            _drop(e, "public_person")
        elif e.type == "PERSON" and is_word_name(e, text):
            _drop(e, "word_name")
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
        elif (e.type == "ORG" and " " in _core(e.text).strip()
              and (_LEGAL_SUFFIX.search(_core(e.text)) or org_assembly.has_form(e.text.strip()))
              and _core(e.text).lower() not in PUBLIC_ORGS):
            kept.append(e)
        elif is_public_org(_core(e.text)):
            _drop(e, "public_org")
        elif is_common_noun(e, text):
            _drop(e, "common_noun")
        elif anchored or id(e) in sure:
            kept.append(e)
        else:
            _drop(e, "untied")
    return kept, dropped
