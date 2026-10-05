"""
detection/pipeline.py — SentinelLayer

Cascade: PatternScan → EntityTrace → ContextGuard, followed by four
model-output-driven post-processing passes.

Post-processing passes (all use model outputs + linguistic structure,
no keyword lists):

  Pass A — Structural ORG detection
      Regex: "[the/a/an] <name> [corporation|company|corp|inc|ltd|llc…]"
      emits <name> as ORG.  The organisational suffix is the signal that
      the preceding word is a company name, regardless of capitalisation.
      This is structural detection (like PatternScan's address pattern),
      not a list of company names.

  Pass B — Email-username → PERSON reclassification
      If an ORG entity's text is a prefix of a detected email username

  Pass C — PERSON component deduplication
      When entity A ("Mitchell") is a word-component of entity B
      ("Sarah Mitchell"), both PERSON, A is removed.  ResolvePass
      component matching handles standalone surname occurrences from the
      full-name surrogate, giving consistent replacement.

  Pass D — Topical geo-entity filter (revised)
      A GPE/LOC is dropped ONLY if it appears exclusively in query
      sub-clauses.  Appearing in any non-query sub-clause → always kept,
      regardless of whether a PERSON is present.

      Additionally: mid-sentence lowercase geo entities (common-noun
      usages like "phoenix bird" or "springfield field team") are skipped
      via a capitalisation check — proper place names are capitalised in
      English.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import replace as _dc_replace
from typing import Dict, List, Optional, Set, Tuple

from ..entities import DetectedEntity, mask_spans
from . import pattern_scan, entity_trace, context_guard, relation_gate, structural
from .quasi_identifier import score as qi_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# pii_off alias resolution
# ─────────────────────────────────────────────────────────────────────────────

_PII_OFF_ALIASES: Dict[str, Set[str]] = {
    "phone":       {"phone_us", "phone_uk", "phone_intl"},
    "postal_code": {"zip_us", "postcode_uk"},
    "zip":         {"zip_us"},
    "postcode":    {"postcode_uk"},
    "name":        {"PERSON"},
    "names":       {"PERSON"},
    "location":    {"GPE", "LOC"},
    "org":         {"ORG"},
    "facility":    {"FAC"},
    "crypto":      {"crypto"},
    "bank":        {"us_bank_number"},
    "license":     {"us_driver_license"},
    "username":    {"handle"},
    "password":    {"credential"},
}


def resolve_pii_off(pii_off) -> Set[str]:
    """Entity types switched off by *pii_off* (type names or aliases)."""
    exclude: Set[str] = set()
    for item in pii_off or ():
        exclude.update(_PII_OFF_ALIASES.get(item.lower(), {item}))
    return exclude


class _TaggedList(list):
    """list subclass that allows attribute assignment (used for _qi_matches)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Pass A — Structural ORG detection
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURAL_ORG_PATTERN = re.compile(
    r'\b(?:the|a|an)\s+'
    r'([A-Za-z][A-Za-z]*(?:\s+[A-Za-z]+){0,3}?)'
    r'\s+(?:corporation|company|corp|inc|ltd|llc|group|firm|enterprise'
    r'|organization|organisation|associates|holdings|ventures|solutions)\b',
    re.IGNORECASE,
)

# TitleCase run ending in an organisational suffix ("Summit Ridge
# Consulting", "Meridian Capital Group") — no article needed; the case
# pattern plus the suffix word IS the evidence.  Span includes the suffix.
_SUFFIXED_ORG_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Za-z&'’.\-]+\s+){1,4}"
    r"(?:Consulting|Group|Holdings|Corporation|Company|Corp|Inc|Ltd|LLC"
    r"|Associates|Partners|Ventures|Solutions|Enterprises|Industries"
    r"|Capital|Bank|University|College|Hospital|Institute|Labs"
    r"|Technologies|Systems|Foundation|Agency)\b\.?)"
)


def _detect_structural_orgs(
    text: str,
    existing_entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    """
    Emit ORG entities for company names identified by organisational suffixes.

    The structural suffix (corporation, company, etc.) signals that the
    preceding word is used as a company name regardless of capitalisation.
    This catches "target corporation", "the phoenix group", etc. without
    any list of known company names.

    The detected span covers ONLY the name (e.g. "target"), not the suffix,
    so the surrogate replaces only "target" and "corporation" is preserved,
    giving "The RetailHoldings corporation announced…" as expected.
    """
    occupied = {(e.start, e.end) for e in existing_entities}
    new_ents: List[DetectedEntity] = []

    for pattern in (_STRUCTURAL_ORG_PATTERN, _SUFFIXED_ORG_PATTERN):
        for m in pattern.finditer(text):
            name_text  = m.group(1).strip()
            name_start = m.start(1)
            name_end   = name_start + len(name_text)

            # Skip if overlaps with an already-detected entity
            if any(not (name_end <= os or name_start >= oe) for os, oe in occupied):
                continue

            ent = DetectedEntity(
                text=name_text,
                start=name_start,
                end=name_end,
                type="ORG",
                score=0.90,
                source="pattern",
            )
            new_ents.append(ent)
            occupied.add((name_start, name_end))
            logger.debug(
                f"[SentinelLayer] Pass A structural ORG: {name_text!r}"
            )

    return new_ents


# ─────────────────────────────────────────────────────────────────────────────
# Pass E — Structural PERSON detection (case-degenerate text)
# ─────────────────────────────────────────────────────────────────────────────
#
# NER models are trained on properly-cased text; real users type lowercase
# chat ("this is deshawn washington") and ALL-CAPS forms ("APPLICANT:
# WASHINGTON, DESHAWN M").  These frames detect the *syntactic position* a
# name occupies (introduction frames, form labels, titles, login fields) —
# structural detection, not name lists.

# Tokens that can never be part of a person name (function words, chat slang,
# channel labels).  This is an English function-word list — the same category
# of list as address_parser._STREET_STOPWORDS — not a list of names.
_PERSON_STOPWORDS = frozenset({
    "i", "im", "u", "ur", "a", "an", "the", "and", "or", "but", "so", "if",
    "my", "me", "we", "us", "our", "you", "your", "he", "she", "it", "they",
    "them", "his", "her", "its", "their", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "not", "no", "yes", "very", "really",
    "just", "still", "here", "there", "now", "then", "today", "tomorrow",
    "yesterday", "on", "in", "at", "to", "for", "of", "with", "from", "by",
    "about", "into", "over", "under", "after", "before", "again", "also",
    "btw", "thx", "pls", "plz", "lol", "omg", "ok", "okay", "hey", "yo",
    "hi", "hello", "sup", "please", "thanks", "thank", "regards", "cheers",
    "own", "new", "old", "only", "other", "sure", "sorry", "done", "good",
    "glad", "happy", "busy", "late", "early", "ready", "free", "fine",
    "well", "right", "wrong", "interested", "available", "confident",
    "sincerely", "best", "sent", "get", "got", "going", "gonna", "wanna",
    "need", "want", "waiting", "back", "soon", "asap", "guys", "team",
    "all", "everyone", "sir", "madam", "dear",
    # things that appear after contact-ish form labels but are never names
    "text", "message", "phone", "email", "call", "mail", "cell", "mobile",
    "whatsapp", "sms", "same", "above", "below", "none", "na", "n/a",
    "unknown", "see", "attached", "pending", "tbd",
    # frequent title-frame objects ("mr. president", "dr. appointment")
    "president", "appointment", "office", "visit",
    # patient-status words ("Patient: Stable Condition")
    "stable", "condition", "improving", "critical", "discharged",
    "admitted", "recovering",
    # frequent user/login-frame objects ("user guide", "login page")
    "guide", "manual", "name", "id", "error", "account", "interface",
    "data", "input", "profile", "settings", "page", "form",
    "permissions", "credentials", "password", "passwords", "access",
    "accounts", "admin", "root", "guest", "roles", "rights", "groups",
    "files", "logs", "sessions", "preferences", "activity", "management",
})

_NAME_TOKEN = r"[A-Za-z][A-Za-z'’.\-]*"

# E1: introduction frames — weak ("this is", "i'm") need a 2-token full name;
#     strong ("my name is") accept a single token.
_INTRO_STRONG = re.compile(
    rf"\bmy\s+name(?:'s|\s+is)\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})",
    re.IGNORECASE,
)
_INTRO_WEAK = re.compile(
    rf"\b(?:i\s+am|i'm|im|this\s+is)\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})",
    re.IGNORECASE,
)

# E2: "<name> here," self-identification at message/sentence start.
_HERE_FRAME = re.compile(
    rf"(?:^|[.!?\n]\s*)({_NAME_TOKEN}\s+{_NAME_TOKEN})\s+here\b",
    re.IGNORECASE,
)

# E3: form labels.  Name-family labels ("Name:", "Patient Name:") say "a
# name follows", so lowercase values are believable; role labels
# ("Customer:", "Referral:") additionally require the value to be
# case-marked as a name.  A strong separator ([:=]) licenses the frame
# anywhere in the text ("…intake system: Name - Sunita Rathod, Email -…");
# the ambiguous dash separator only counts at a line/clause start.
_FORM_LABEL_NAME_CORE = (
    r"(?:(?:patient|full|legal|first|last|middle|maiden|account\s+holder"
    r"|card\s*holder|mother'?s\s+maiden)\s+)?name"
)
_FORM_LABEL_ROLE_CORE = (
    r"(?:customer|applicant|beneficiary|claimant|insured|guarantor"
    r"|patient|referral|emergency\s+contact)"
)

# (regex, value-must-be-case-marked)
_FORM_LABEL_PATTERNS = [
    (re.compile(
        rf"\b{_FORM_LABEL_NAME_CORE}\s*[:=]\s*([^\n]{{2,60}})",
        re.IGNORECASE), False),
    (re.compile(
        rf"(?:^|\n|[.:;,]\s)\s*{_FORM_LABEL_NAME_CORE}\s*[\-–]\s*([^\n]{{2,60}})",
        re.IGNORECASE), False),
    (re.compile(
        rf"\b{_FORM_LABEL_ROLE_CORE}\s*[:=]\s*([^\n]{{2,60}})",
        re.IGNORECASE), True),
    (re.compile(
        rf"(?:^|\n|[.:;,]\s)\s*{_FORM_LABEL_ROLE_CORE}\s*[\-–]\s*([^\n]{{2,60}})",
        re.IGNORECASE), True),
]

# E4: title frames in lowercase text ("mr. thompson") — properly-cased
# titles are already handled by NER, so this only fires on lowercase names.
_TITLE_FRAME = re.compile(
    r"\b(?:mr|mrs|ms|mx|dr|prof)\.?\s+((?-i:[a-z])[a-z'’\-]{2,})\b",
    re.IGNORECASE,
)

# E5: login/username frames in logs ("user asmith connected").  Prose like
# "failed login attempts from …" or "a user who declared …" must never
# match, so the value needs log-shaped evidence: an explicit separator
# ([:=]), a digit/._- inside the value, or a log verb right after it.
_USER_FRAME = re.compile(
    r"\b(?:(?:user(?:name)?|uid|logged\s+in\s+as)\s*[:=]?\s+"
    r"|login\s*[:=]\s*)"  # bare "login <word>" is prose ("login attempts")
    r"((?-i:[a-z])[a-z0-9._\-]{2,20})\b",
    re.IGNORECASE,
)
_USER_VALUE_SHAPE = re.compile(r"[0-9._\-]")
_USER_CONTINUATION = re.compile(
    r"^\s+(?:connected|logged|from|authenticated|signed|attempted|session|ssh)\b",
    re.IGNORECASE,
)


def _token_core(t: str) -> str:
    return t.lower().strip(".,'’-")


def _token_ok(t: str) -> bool:
    core = _token_core(t)
    return (
        core not in _PERSON_STOPWORDS
        and not core.endswith("ly")             # adverbs are never names
        and len(core) >= 2
    )


_INITIAL_RE = re.compile(r"[A-Za-z]\.?,?$")

# Form-field labels that end a name value ("Name - Fatou Diarra, Email - …")
_FIELD_LABEL_BREAKERS = frozenset({
    "email", "e-mail", "phone", "tel", "telephone", "mobile", "cell", "fax",
    "dob", "ssn", "address", "direct", "line", "contact", "id", "account",
})


def _person_tokens_ok(tokens: List[str]) -> bool:
    return bool(tokens) and all(_token_ok(t) for t in tokens)


def _verbish(t: str) -> bool:
    """Gerunds/participles ("heading", "updated") — prose, not names."""
    core = _token_core(t)
    return core.endswith("ing") or core.endswith("ed")


def _trim_trailing_stopwords(tokens: List[str]) -> List[str]:
    while tokens and _token_core(tokens[-1]) in _PERSON_STOPWORDS:
        tokens = tokens[:-1]
    return tokens


def _name_case_marked(tokens: List[str]) -> bool:
    """ALL-CAPS or Every-Word-Capitalized — how forms case-mark names."""
    return all(t[0].isupper() for t in tokens if t[0].isalpha())


def _form_label_candidate(raw: str, require_case: bool) -> Optional[str]:
    """Trim a form-field value down to a plausible name, or None."""
    tokens = []
    for tok in raw.split():
        # a name never contains digits, @, parentheses, field separators,
        # or dotted compounds ("example.com"); a bare dash or a following
        # field label ("…, Email - …") ends the name
        if any(ch.isdigit() for ch in tok) or any(
            ch in tok for ch in "(@|/#<>[]"
        ) or re.search(r"[A-Za-z0-9]\.[A-Za-z]", tok):
            break
        if tok in ("-", "–", "—") or _token_core(tok) in _FIELD_LABEL_BREAKERS:
            break
        tokens.append(tok)
    tokens = _trim_trailing_stopwords(tokens[:5])
    while tokens and tokens[-1].rstrip(",") != tokens[-1]:
        # a trailing comma marks the end of the field value
        tokens[-1] = tokens[-1].rstrip(",")
        if not tokens[-1]:
            tokens.pop()
        break
    if not (1 <= len(tokens) <= 4):
        return None
    if require_case and not _name_case_marked(tokens):
        return None
    cleaned = [t.rstrip(",") for t in tokens]
    # every token must be name-plausible; single-letter initials are fine
    # ("WASHINGTON, DESHAWN M")
    if not all(_token_ok(t) or _INITIAL_RE.fullmatch(t) for t in cleaned):
        return None
    if not any(len(t.strip(".,'’-")) >= 3 for t in cleaned):
        return None
    return " ".join(tokens)


def _detect_structural_persons(
    text: str,
    existing_entities: List[DetectedEntity],
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Emit PERSON entities for names in structural frames that case-sensitive
    NER misses.  Returns (new_entities, superseded_existing) — an existing
    PERSON strictly contained in a new wider span is superseded (the wider
    span wins so "WASHINGTON, DESHAWN M" replaces "WASHINGTON, DESHAWN").
    """
    candidates: List[Tuple[int, int, str]] = []

    for m in _INTRO_STRONG.finditer(text):
        toks = _trim_trailing_stopwords(m.group(1).split())
        if toks and _person_tokens_ok(toks):
            span_text = " ".join(toks)
            candidates.append((m.start(1), m.start(1) + len(span_text), span_text))

    for m in _INTRO_WEAK.finditer(text):
        toks = _trim_trailing_stopwords(m.group(1).split())
        # weak frames ("this is X") need a full 2+-token name, no gerunds
        # anywhere ("im heading home") and no participle in first position
        # ("this is expected behavior")
        # a properly-cased "I'm"/"I am" says the text is not case-degenerate,
        # so the name must be capitalised ("I'm risk averse." is not a name)
        cased_frame = m.group(0)[:1] == "I"
        if (len(toks) >= 2 and _person_tokens_ok(toks)
                and not any(_token_core(t).endswith("ing") for t in toks)
                and not _verbish(toks[0])
                and not (cased_frame and not toks[0][:1].isupper())):
            span_text = " ".join(toks)
            candidates.append((m.start(1), m.start(1) + len(span_text), span_text))

    for m in _HERE_FRAME.finditer(text):
        toks = m.group(1).split()
        if (len(toks) >= 2 and _person_tokens_ok(toks)
                and not any(_verbish(t) for t in toks)):
            candidates.append((m.start(1), m.end(1), m.group(1)))

    for label_re, require_case in _FORM_LABEL_PATTERNS:
        for m in label_re.finditer(text):
            cand = _form_label_candidate(m.group(1), require_case=require_case)
            if cand:
                start = m.start(1) + m.group(1).find(cand.split()[0])
                candidates.append((start, start + len(cand), cand))

    for m in _TITLE_FRAME.finditer(text):
        tok = m.group(1)
        if _person_tokens_ok([tok]) and not _verbish(tok):
            candidates.append((m.start(1), m.end(1), tok))

    for m in _USER_FRAME.finditer(text):
        tok = m.group(1)
        log_shaped = (
            _USER_VALUE_SHAPE.search(tok)
            or _USER_CONTINUATION.match(text[m.end(1):m.end(1) + 20])
        )
        if log_shaped and _person_tokens_ok([tok]) and not _verbish(tok):
            candidates.append((m.start(1), m.end(1), tok))

    # E6: initial-led PERSON extension — NER often clips ALL-CAPS names to
    # the initial ("P. BRIGGS" from "LATOYA P. BRIGGS"); reattach the
    # immediately preceding all-caps name word.
    for ent in existing_entities:
        if ent.type != "PERSON" or not re.match(r"^[A-Z]\.\s", ent.text):
            continue
        prefix = text[:ent.start]
        pm = re.search(r"([A-Z][A-Z'’\-]{2,})\s$", prefix)
        if pm and _person_tokens_ok([pm.group(1)]):
            candidates.append(
                (pm.start(1), ent.end, text[pm.start(1):ent.end])
            )

    # E7: clipped-name extension — NER trained on common names clips
    # unusual first names ("Katarzyna Wojciechowska" → only
    # "Wojciechowska").  Reattach an adjacent capitalized word that is not
    # sentence-initial, not a function word, and not part of another entity.
    _TITLES = {"mr", "mrs", "ms", "mx", "dr", "prof"}
    for ent in existing_entities:
        if ent.type not in ("PERSON", "ORG") or " " in ent.text:
            continue
        if not ent.text[:1].isupper():
            continue
        prefix = text[:ent.start]
        pm = re.search(r"([A-Z][a-z'’\-]{1,20})\s$", prefix)
        if not pm:
            continue
        tok = pm.group(1)
        if (_token_core(tok) in _TITLES or not _person_tokens_ok([tok])
                or _verbish(tok)):
            continue
        head = prefix[:pm.start(1)].rstrip()
        if not head or head[-1] in ".!?":
            continue  # sentence-initial capitalisation is not name evidence
        if any(e.end > pm.start(1) and e.start < pm.end(1)
               for e in existing_entities if e is not ent):
            continue  # the word already belongs to another entity
        candidates.append(
            (pm.start(1), ent.end, text[pm.start(1):ent.end])
        )

    new_ents: List[DetectedEntity] = []
    superseded: List[DetectedEntity] = []
    claimed: List[Tuple[int, int]] = []

    for start, end, span_text in sorted(candidates, key=lambda c: (c[0], -(c[1]))):
        if text[start:end] != span_text:
            continue  # offset bookkeeping failed — never emit a wrong span
        if any(not (end <= s or start >= e) for s, e in claimed):
            continue
        blocking = None
        for ent in existing_entities:
            if ent in superseded:
                continue
            if not (end <= ent.start or start >= ent.end):
                # overlap: a name-like entity strictly inside the new span is
                # upgraded — case-degenerate NER often mistakes a lowercase
                # surname for a place ("washington" GPE inside "deshawn
                # washington").  Any other overlap blocks the candidate.
                if (ent.type in ("PERSON", "GPE", "ORG", "LOC", "FAC")
                        and ent.start >= start and ent.end <= end
                        and (ent.end - ent.start) < (end - start)):
                    superseded.append(ent)
                else:
                    blocking = ent
                    break
        if blocking is not None:
            continue
        new_ents.append(DetectedEntity(
            text=span_text, start=start, end=end,
            type="PERSON", score=0.82, source="pattern",
        ))
        claimed.append((start, end))
        logger.debug(f"[SentinelLayer] Pass E structural PERSON: {span_text!r}")

    return new_ents, superseded


# ─────────────────────────────────────────────────────────────────────────────
# Pass F — ORG plausibility filter
# ─────────────────────────────────────────────────────────────────────────────
#
# NER models over-fire ORG on acronyms ("TSA", "AWS", "ISO"), product/brand
# mentions ("GitHub token", "My Amex") and word fragments.  An ORG is only
# personally identifying when it plausibly names the user's affiliation.

_ORG_SUFFIX_TOKENS = frozenset({
    "inc", "llc", "ltd", "corp", "corporation", "company", "co", "group",
    "holdings", "ventures", "solutions", "associates", "partners",
    "enterprises", "industries", "capital", "bank", "university", "college",
    "school", "hospital", "health", "healthcare", "clinic", "institute",
    "labs", "laboratories", "technologies", "systems", "media", "airlines",
    "insurance", "foundation", "agency", "consulting", "firm", "gmbh", "plc",
    "board", "bureau", "authority", "commission", "council", "department",
    "ministry",
})

def _org_is_plausible(ent: DetectedEntity, text: str) -> bool:
    """
    Keep every ORG that is case-marked as a proper name (the project's
    philosophy is that organisations ARE maskable PII — "Visa", "NHS",
    "Microsoft").  Drop only structural junk NER produces:

      • names containing an all-digit token ("ISO 27001", "SOC 2") —
        those are standards/models, not organisations
      • names whose words are not case-marked ("phoenix program",
        "lient") — proper names are capitalised in English
    """
    if ent.source == "pattern":
        return True  # structural Pass A ORGs carry their own evidence
    words = ent.text.split()
    if any(w.isdigit() for w in words):
        return False
    # an explicit organisational suffix is convincing regardless of case
    # ("the national insurance board", "Meridian Capital Group")
    if any(w.strip(".,").lower() in _ORG_SUFFIX_TOKENS for w in words):
        return True
    if not all(w[0].isupper() for w in words if w[0].isalpha()):
        return False
    # acronym immediately followed by a bare number in the text is a
    # standard/spec ("SOC 2", "ISO 27001"), not an organisation
    if re.match(r"\s+\d+\b", text[ent.end:ent.end + 8]):
        return False
    # letter-digit codes are gates/seats/models ("B22", "14C"), never orgs
    if re.fullmatch(r"[A-Z]{1,2}\d{1,3}[A-Z]?", ent.text):
        return False
    # a single word tagged ORG by the word-piece model ONLY because it is
    # capitalized at a sentence start ("Dispute filed…", "Renewal invoice…")
    if ent.source == "slm" and len(words) == 1:
        head = text[:ent.start].rstrip()
        if not head or head[-1] in ".!?\n":
            return False
    # properly-cased name with no counter-evidence → keep
    return True


def _filter_implausible_orgs(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    result = []
    for ent in entities:
        if ent.type == "ORG" and not _org_is_plausible(ent, text):
            logger.debug(
                f"[SentinelLayer] Pass F dropping implausible ORG: {ent.text!r}"
            )
            continue
        result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass G — Adjacent-PERSON merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge_adjacent_persons(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """
    NER sometimes splits one name into two entities ("Priya" + "Nambiar").
    Two PERSON entities separated by exactly one space are one name.
    """
    persons = sorted(
        (e for e in entities if e.type == "PERSON"), key=lambda e: e.start
    )
    others = [e for e in entities if e.type != "PERSON"]

    merged: List[DetectedEntity] = []
    for ent in persons:
        if merged and text[merged[-1].end:ent.start] == " ":
            prev = merged[-1]
            merged[-1] = DetectedEntity(
                text=text[prev.start:ent.end],
                start=prev.start,
                end=ent.end,
                type="PERSON",
                score=max(prev.score, ent.score),
                source=prev.source,
            )
            logger.debug(
                f"[SentinelLayer] Pass G merged PERSONs → {merged[-1].text!r}"
            )
        else:
            merged.append(ent)
    return others + merged


# ─────────────────────────────────────────────────────────────────────────────
# Pass H — Card-brand ORG (word immediately preceding a card number)
# ─────────────────────────────────────────────────────────────────────────────

_CARD_LABEL_WORDS = frozenset({
    "card", "number", "account", "cc", "no", "num", "ending", "payment",
})


def _detect_card_brand_orgs(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """
    A capitalized word directly before a card number is the card's brand
    ("Mastercard 5425233430109903") — organisational PII by this project's
    philosophy.  Structural: brand-before-number, no brand list.
    """
    new_ents: List[DetectedEntity] = []
    occupied = [(e.start, e.end) for e in entities]
    for ent in entities:
        if ent.type != "credit_card":
            continue
        m = re.search(r"([A-Z][A-Za-z]{2,15})\s$", text[:ent.start])
        if not m:
            continue
        word = m.group(1)
        if word.lower() in _CARD_LABEL_WORDS:
            continue
        s, e = m.start(1), m.end(1)
        if any(not (e <= os or s >= oe) for os, oe in occupied):
            continue
        new_ents.append(DetectedEntity(
            text=word, start=s, end=e, type="ORG",
            score=0.88, source="pattern",
        ))
        occupied.append((s, e))
        logger.debug(f"[SentinelLayer] Pass H card-brand ORG: {word!r}")
    return new_ents


# ─────────────────────────────────────────────────────────────────────────────
# Pass B — Email-username → PERSON reclassification
# ─────────────────────────────────────────────────────────────────────────────

def _reclassify_email_username_orgs(
    entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    """
    Reclassify ORG entities whose text is a name-prefix of a detected email.

    spaCy sees masked text after PatternScan — the email is hidden — so it
    sometimes labels the standalone first name as ORG.  This pass uses the
    email detection output (from PatternScan) to correct that labelling.

    Example: "[redacted]@gmail.com" detected → username "[redacted]"
             "abhi" detected as ORG → "abhi" is prefix of username
             → reclassify "abhi" as PERSON
    """
    email_usernames: Set[str] = set()
    for ent in entities:
        if ent.type == "email" and "@" in ent.text:
            email_usernames.add(ent.text.split("@")[0].lower())

    if not email_usernames:
        return entities

    result = []
    for ent in entities:
        if ent.type == "ORG" and len(ent.text) >= 3:
            ent_lower = ent.text.lower()
            for username in email_usernames:
                if username.startswith(ent_lower):
                    ent = _dc_replace(ent, type="PERSON", score=0.88)
                    logger.debug(
                        f"[SentinelLayer] Pass B ORG→PERSON (email prefix): "
                        f"{ent.text!r}"
                    )
                    break
        result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass C — PERSON component deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_person_components(
    entities: List[DetectedEntity],
) -> List[DetectedEntity]:
    """
    Remove PERSON entities that are word-components of longer PERSON entities.

    When "Mitchell" and "Sarah Mitchell" are both detected:
      • Keep "Sarah Mitchell"
      • Remove "Mitchell" (ResolvePass component matching will handle
        standalone "Mitchell" occurrences using the full-name surrogate,
        giving consistent output like "Clark" from "Jessica Clark")

    Decision is purely structural (word-set containment), no name lists.
    """
    persons = [e for e in entities if e.type == "PERSON"]
    others  = [e for e in entities if e.type != "PERSON"]

    to_remove: Set[str] = set()
    for short in persons:
        short_words = set(short.text.split())
        for long_ent in persons:
            if short.text != long_ent.text:
                long_words = set(long_ent.text.split())
                if short_words <= long_words:
                    to_remove.add(short.text)
                    logger.debug(
                        f"[SentinelLayer] Pass C removing component PERSON "
                        f"{short.text!r} (subset of {long_ent.text!r})"
                    )
                    break

    return others + [e for e in persons if e.text not in to_remove]


# ─────────────────────────────────────────────────────────────────────────────
# Pass D — Topical geo-entity filter
# ─────────────────────────────────────────────────────────────────────────────

_CLAUSE_SPLIT = re.compile(
    r'[.!?]+\s+'
    r'|\s+(?:and|or|but|however|yet|because|since|although|while|when'
    r'|therefore|whereas|unless|though|despite|nevertheless|so)\s+'
    r'|,\s+',
    re.IGNORECASE,
)

_QUERY_FRAME = re.compile(
    r'^\s*'
    r'(?:please\s+|could\s+you\s+(?:please\s+)?|can\s+you\s+(?:please\s+)?'
    r'|would\s+you\s+(?:please\s+)?)?'
    r'(?:'
    r'give\s+me|tell\s+me|show\s+me|find\s+me|help\s+me|get\s+me'
    r'|look\s+up|search\s+for|look\s+for|explain|list|describe'
    r'|summarize|summarise|compare|recommend|suggest|advise'
    r'|what\s+(?:is|are|was|were|would|can|do|does|did)'
    r'|how\s+(?:do|does|did|can|much|many|to|would)'
    r'|which\s+(?:is|are|was|were|would)'
    r'|where\s+(?:is|are|can|do|should|would)'
    r'|when\s+(?:is|are|was|were|do|does|did|can)'
    r'|who\s+(?:is|are|was|were|can|would|do|does)'
    r'|why\s+(?:is|are|was|were|do|does|did|would|should)'
    r'|is\s+there|are\s+there'
    r'|i\s+(?:want|need|would\s+like|\'d\s+like)\s+(?:to\s+know|to\s+find|'
    r'to\s+learn|to\s+understand|information|details|advice|help)'
    r')',
    re.IGNORECASE,
)

_GEO_FILTERABLE = {"GPE", "LOC"}

# Direct identifiers of a person. A place that shares a message with one of
# these is that person's place ("Carlos Mendez … in Naperville"), so the
# topical-query exemption below does not apply to it.
_PERSONAL_ANCHOR_TYPES = {
    "PERSON", "email", "phone_intl", "phone_us", "phone_uk", "ssn", "dob",
    "address", "credit_card", "us_driver_license", "passport", "id_number",
    "us_bank_number", "iban", "handle", "url", "age", "credential",
}


def _has_personal_anchor(*entity_lists: List[DetectedEntity]) -> bool:
    return any(e.type in _PERSONAL_ANCHOR_TYPES for ents in entity_lists for e in ents)


def _all_sub_clauses(text: str) -> List[str]:
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _contains_entity(entity_text: str, clause: str) -> bool:
    return entity_text.lower() in clause.lower()


def _is_proper_capitalized(entity_text: str, text: str) -> bool:
    """
    Return True if this entity should be treated as a proper noun.

    English proper nouns are capitalised.  A geo entity whose surface form
    starts with a lowercase letter and appears mid-sentence is almost
    certainly a common-noun usage ("phoenix bird", "springfield field team"),
    not the name of a place.

    This is a linguistic structure rule — it uses no geographic keyword lists.
    """
    if entity_text[0].isupper():
        return True  # capitalised → proper noun ✓

    # Lowercase entity — check if it's at the start of a sentence/clause
    idx = text.find(entity_text)
    if idx == -1:
        idx = text.lower().find(entity_text.lower())
    if idx == -1:
        return True  # not found → conservative: keep

    prefix = text[:idx].rstrip()
    if not prefix or prefix[-1] in ".!?;":
        return True  # sentence-start → might be proper noun despite lowercase

    return False  # lowercase, mid-sentence → common noun usage → skip


def _filter_topical_geo_entities(
    entities: List[DetectedEntity],
    text: str,
    anchored: bool = False,
) -> tuple[List, List]:
    """
    Remove GPE/LOC entities that are query topics rather than personal refs.

    Revised rule (no PERSON co-occurrence requirement):
      • Geo entity appears ONLY in query sub-clauses → topical → DROPPED
      • Geo entity appears in any non-query sub-clause → personal/narrative
        → KEPT regardless of whether a PERSON entity is present

    This means:
      "give me tax benefits of Wyoming"         → query only → dropped
      "from the ashes of Phoenix"               → non-query  → kept ✓
      "The city of Springfield is beautiful"    → non-query  → kept ✓
      "What restaurants are near London?"       → query only → dropped ✓
      "Revanth lives in Wyoming"                → non-query  → kept ✓

    Person anchor: when the message also names a person or carries one of
    their direct identifiers (``anchored=True``), the query-only exemption is
    off — "find Carlos Mendez from NovaBuild in Naperville" keeps Naperville.

    The capitalisation check (_is_proper_capitalized) additionally filters
    mid-sentence lowercase usages ("phoenix bird", "springfield field team").

    Returns:
        (kept_entities, skipped_entities)
    """
    geo_ents   = [e for e in entities if e.type in _GEO_FILTERABLE]
    other_ents = [e for e in entities if e.type not in _GEO_FILTERABLE]

    if not geo_ents:
        return entities, []

    all_clauses     = _all_sub_clauses(text)
    clause_is_query = [bool(_QUERY_FRAME.match(c)) for c in all_clauses]

    skipped: List[DetectedEntity] = []
    result = list(other_ents)

    for geo_ent in geo_ents:
        # Capitalisation check: lowercase mid-sentence = common noun → skip
        if not _is_proper_capitalized(geo_ent.text, text):
            logger.debug(
                f"[SentinelLayer] Pass D: lowercase geo skipped (not proper noun): "
                f"{geo_ent.text!r}"
            )
            skipped.append(geo_ent)
            continue

        # Query-clause filter
        in_query    = False
        in_personal = False

        for i, clause in enumerate(all_clauses):
            if _contains_entity(geo_ent.text, clause):
                if clause_is_query[i]:
                    in_query = True
                else:
                    in_personal = True

        if in_query and not in_personal and not anchored:
            logger.debug(
                f"[SentinelLayer] Pass D: topical geo (query-only): {geo_ent.text!r}"
            )
            skipped.append(geo_ent)
            continue

        logger.debug(
            f"[SentinelLayer] Pass D: geo kept (non-query context): {geo_ent.text!r}"
        )
        result.append(geo_ent)

    return result, skipped


# ─────────────────────────────────────────────────────────────────────────────
# ORG→GPE reclassification (narrow preps only)
# ─────────────────────────────────────────────────────────────────────────────

_LOCATION_PREPS = {
    "in", "near", "live", "lives", "lived", "grew", "born", "raised",
    "moved", "relocate", "relocated", "residing", "reside",
    "hometown", "birthplace", "based",
}

_GEO_TYPES = {"GPE", "LOC", "FAC"}


def _reclassify_location_orgs(
    entities: List[DetectedEntity],
    original_text: str,
) -> List[DetectedEntity]:
    result = []
    for ent in entities:
        if ent.type == "ORG":
            ctx = original_text[max(0, ent.start - 50): ent.start].lower()
            if _LOCATION_PREPS & set(ctx.split()):
                ent = _dc_replace(ent, type="GPE", score=0.85)
                logger.debug(f"[SentinelLayer] Reclassified ORG→GPE: {ent.text!r}")
        result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main cascade
# ─────────────────────────────────────────────────────────────────────────────

def _spacy_place_verifier(model_name: str):
    """Pass S place check: does spaCy tag the candidate GPE/LOC once it is
    capitalised ("she lives alone in Flagstaff")? Only reached for a
    lowercase word right after a residence or work cue."""
    def verify(sentence: str, s: int, e: int) -> bool:
        cand = sentence[s:e]
        cased = " ".join(w[:1].upper() + w[1:] for w in cand.split(" "))
        if len(cased) != len(cand):
            return False
        probe = sentence[:s] + cased + sentence[e:]
        probe = probe[:1].upper() + probe[1:]
        doc = entity_trace._get_nlp(model_name)(probe)
        return any(ent.label_ in ("GPE", "LOC") and ent.start_char <= s and e <= ent.end_char
                   for ent in doc.ents)
    return verify


_TERM_GENDER = {"m": frozenset("male man boy guy he/him he him m gentleman".split()),
                "f": frozenset("female woman girl lady she/her she her f".split())}


def gender_follows_name(entities):
    """(entities to mask, gender terms kept as typed): a gender term of the
    same gender as a named person among *entities* is kept (Pass G)."""
    if not any(e.type == "gender_indicator" for e in entities):
        return list(entities), []
    from ..generation.identity import name_gender
    named = {name_gender(e.text) for e in entities if e.type in ("PERSON", "person")}
    named.discard(None)
    follows = [e for e in entities
               if e.type == "gender_indicator" and term_gender(e.text) in named]
    gone = {id(e) for e in follows}
    return [e for e in entities if id(e) not in gone], follows


def term_gender(text: str) -> Optional[str]:
    """"m" / "f" for a gender term ("female", "gender: M", "she/her"), else None."""
    term = re.sub(r"(?i)^(?:gender|sex|pronouns?)\s*[:=]?\s*", "", text.strip()).lower()
    return next((g for g, terms in _TERM_GENDER.items() if term in terms), None)


def run_cascade(
    text: str,
    skip_values: Optional[Set[str]] = None,
    skip_location_entities: bool = False,
    timings: Optional[Dict[str, float]] = None,
    *,
    pii_off=None,
    spacy_model: str = "en_core_web_lg",
    context_guard_enabled: bool = True,
    entity_trace_high_threshold: float = 0.85,
    entity_trace_low_threshold: float = 0.60,
    context_guard_threshold: float = 0.70,
    entity_trace_fallback_threshold: float = 0.65,
    context_guard_model: str = "dslim/distilbert-NER",
    context_guard_device: int = -1,
    use_entity_trace: bool = True,
    use_context_guard: Optional[bool] = None,
    use_post_passes: bool = True,
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Execute the full SentinelLayer cascade then apply post-processing passes.

    Args:
        text:                   Raw user message.
        skip_values:            Surrogate strings to skip in PatternScan.
        skip_location_entities: Suppress ALL geo entities (service-query mode).
        timings:                If given, filled with wall-clock milliseconds
                                of THIS pass: pattern_scan_ms, entity_trace_ms,
                                context_guard_ms, post_passes_ms.
        pii_off:                PII type names/aliases to drop from the result
                                (detected, then deliberately not replaced).
        spacy_model:            spaCy model for EntityTrace.
        context_guard_enabled:  Run ContextGuard; when off, borderline NER
                                entities at or above
                                entity_trace_fallback_threshold are promoted.
        entity_trace_high_threshold / entity_trace_low_threshold:
                                EntityTrace confirmed / borderline cut-offs.
        context_guard_threshold: ContextGuard confirmation cut-off.
        context_guard_model / context_guard_device:
                                HuggingFace model and device for ContextGuard.
        use_entity_trace:       False skips spaCy NER (ablation, audit A9).
        use_context_guard:      None follows context_guard_enabled; a bool
                                overrides it (ablation).
        use_post_passes:        False skips structural passes A, B, C, E–H.
                                Pass D (topical geo policy) always runs.
    """
    _clock = time.perf_counter
    _t = _clock()

    def _lap(name: str) -> None:
        nonlocal _t
        now = _clock()
        if timings is not None:
            timings[name] = round((now - _t) * 1000, 3)
        _t = now

    confirmed: List[DetectedEntity] = []
    needs_confirmation: List[DetectedEntity] = []
    all_skipped: List[DetectedEntity] = []

    # ── Stage 1: PatternScan ──────────────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 1: PatternScan")
    pattern_results = pattern_scan.scan(text, skip_values=skip_values)
    confirmed.extend(pattern_results)
    remaining_text = mask_spans(text, pattern_results)
    # URLs are opaque to the NER stages (audit I1): a model never sees
    # "github.com/Microsoft" or a query string, so it cannot tag a fragment.
    opaque = pattern_scan.opaque_spans(text)
    remaining_text = mask_spans(remaining_text, [
        DetectedEntity(text[s:e], s, e, "opaque", 0.0, "url") for s, e in opaque])
    _lap("pattern_scan_ms")

    # ── Stage 2: EntityTrace ──────────────────────────────────────────────────
    logger.info("[SentinelLayer] Stage 2: EntityTrace")
    if use_entity_trace:
        ner_confirmed, ner_borderline = entity_trace.trace(
            remaining_text,
            existing_entities=confirmed,
            spacy_model=spacy_model,
            high_threshold=entity_trace_high_threshold,
            low_threshold=entity_trace_low_threshold,
        )
    else:
        ner_confirmed, ner_borderline = [], []
    ner_confirmed  = _reclassify_location_orgs(ner_confirmed,  text)
    ner_borderline = _reclassify_location_orgs(ner_borderline, text)

    if skip_location_entities:
        ner_confirmed_filtered  = [e for e in ner_confirmed  if e.type in _GEO_TYPES]
        ner_borderline_filtered = [e for e in ner_borderline if e.type in _GEO_TYPES]
        ner_confirmed  = [e for e in ner_confirmed  if e.type not in _GEO_TYPES]
        ner_borderline = [e for e in ner_borderline if e.type not in _GEO_TYPES]
        all_skipped.extend(ner_confirmed_filtered)
        all_skipped.extend(ner_borderline_filtered)

    confirmed.extend(ner_confirmed)
    remaining_text = mask_spans(remaining_text, ner_confirmed)
    _lap("entity_trace_ms")

    # ── Stage 3: ContextGuard ─────────────────────────────────────────────────
    if use_context_guard is None:
        use_context_guard = context_guard_enabled
    if use_context_guard:
        logger.info("[SentinelLayer] Stage 3: ContextGuard")
        slm_confirmed, slm_uncertain = context_guard.guard(
            remaining_text=remaining_text,
            borderline_entities=ner_borderline,
            model_name=context_guard_model,
            enabled=True,
            confidence_threshold=context_guard_threshold,
            device=context_guard_device,
        )
        if skip_location_entities:
            slm_confirmed = [e for e in slm_confirmed if e.type not in _GEO_TYPES]
            slm_uncertain = [e for e in slm_uncertain if e.type not in _GEO_TYPES]
        confirmed.extend(slm_confirmed)
        needs_confirmation.extend(slm_uncertain)
    else:
        promoted = [
            e for e in ner_borderline
            if e.score >= entity_trace_fallback_threshold
        ]
        if promoted:
            confirmed.extend(promoted)
    _lap("context_guard_ms")

    # A model span may run across a line break and come back with the break
    # normalised to a space ("University of Leicester\nORCID"): its text no
    # longer matches the message, so no edit could be planned for it and the
    # name would go out unmasked. Re-anchor every span on the message itself.
    confirmed          = _snap_to_words(_anchor_to_lines(confirmed, text), text)
    needs_confirmation = _snap_to_words(_anchor_to_lines(needs_confirmation, text), text)

    if use_post_passes:
        # ── Pass A: Structural ORG detection ─────────────────────────────────────
        structural_orgs = _detect_structural_orgs(text, confirmed)
        if structural_orgs:
            logger.info(
                f"[SentinelLayer] Pass A: +{len(structural_orgs)} structural ORG(s)"
            )
            confirmed.extend(structural_orgs)

        # ── Pass E: Structural PERSON detection (case-degenerate text) ───────────
        structural_persons, superseded = _detect_structural_persons(
            text, confirmed + needs_confirmation,
        )
        if superseded:
            confirmed          = [e for e in confirmed          if e not in superseded]
            needs_confirmation = [e for e in needs_confirmation if e not in superseded]
        if structural_persons:
            logger.info(
                f"[SentinelLayer] Pass E: +{len(structural_persons)} structural PERSON(s)"
            )
            confirmed.extend(structural_persons)

        # ── Pass F: ORG plausibility filter ──────────────────────────────────────
        confirmed          = _filter_implausible_orgs(confirmed, text)
        needs_confirmation = _filter_implausible_orgs(needs_confirmation, text)

        # ── Pass G: Adjacent-PERSON merge ────────────────────────────────────────
        confirmed          = _merge_adjacent_persons(confirmed, text)
        needs_confirmation = _merge_adjacent_persons(needs_confirmation, text)

        # ── Pass H: Card-brand ORG detection ─────────────────────────────────────
        brand_orgs = _detect_card_brand_orgs(confirmed, text)
        if brand_orgs:
            confirmed.extend(brand_orgs)

        # ── Pass B: Email-username → PERSON reclassification ─────────────────────
        confirmed          = _reclassify_email_username_orgs(confirmed)
        needs_confirmation = _reclassify_email_username_orgs(needs_confirmation)

        # ── Pass C: PERSON component deduplication ────────────────────────────────
        confirmed          = _deduplicate_person_components(confirmed)
        needs_confirmation = _deduplicate_person_components(needs_confirmation)

    # Structural passes read the raw text; nothing they find inside a URL is
    # kept (the URL is either a pattern entity already or not PII).
    if opaque:
        confirmed          = _outside_opaque(confirmed, opaque)
        needs_confirmation = _outside_opaque(needs_confirmation, opaque)

    # ── Pass D: Topical geo-entity filter ─────────────────────────────────────
    if not skip_location_entities:
        anchored = _has_personal_anchor(confirmed, needs_confirmation)
        confirmed,          skipped_confirmed = _filter_topical_geo_entities(confirmed,          text, anchored)
        needs_confirmation, skipped_nc        = _filter_topical_geo_entities(needs_confirmation, text, anchored)
        all_skipped = skipped_confirmed + skipped_nc

    # ── Pass R: relation gate (I12) — names not tied to a person stay ────────
    # verbatim: acronyms, code, greetings, public figures/companies and
    # places that are only the topic.  Pattern entities are never gated.
    skip_reasons = {}
    # a model PERSON spoken inside a sentence frame is cut to the name
    # ("ich bin Jörg Baumgartner", "Mera naam Rohit Bhardwaj hai"); a frame
    # with no name in it ("Mein Vermieter", "Kun je mijn …") is dropped
    for bucket in ("confirmed", "needs_confirmation"):
        ents, framed = [], []
        for e in (confirmed if bucket == "confirmed" else needs_confirmation):
            t = (relation_gate.trim_person(e, text)
                 if e.type == "PERSON" and e.source != "pattern" else e)
            if t is None:
                framed.append(e)
                skip_reasons[(e.start, e.end)] = "sentence_frame"
            else:
                ents.append(t)
        all_skipped = list(all_skipped) + framed
        if bucket == "confirmed":
            confirmed = ents
        else:
            needs_confirmation = ents
    for bucket in ("confirmed", "needs_confirmation"):
        ents = confirmed if bucket == "confirmed" else needs_confirmation
        # NER/SLM entities, plus structural PERSONs (Pass E) — those only
        # face the junk and public-figure checks ("Emperor Meiji").
        gated = [e for e in ents if e.source != "pattern" or e.type == "PERSON"]
        gated_ids = {id(e) for e in gated}
        others = [e for e in list(confirmed) + list(needs_confirmation)
                  if id(e) not in gated_ids]
        kept, dropped = relation_gate.gate(text, gated, others)
        kept_ids = {id(e) for e in kept}
        ents = [e for e in ents if id(e) not in gated_ids or id(e) in kept_ids]
        for e in dropped:
            skip_reasons[(e.start, e.end)] = "not_tied_to_person"
        all_skipped = list(all_skipped) + dropped
        if bucket == "confirmed":
            confirmed = ents
        else:
            needs_confirmation = ents

    # ── Pass S: structural / frame detection (I8, I13) — CSV columns, chat
    # speakers, name components, particles, residence cues, payees, zh/hi
    # frames, European streets. Runs after the gate: each rule carries its
    # own evidence of a tie to a person.
    if use_post_passes:
        # Only confirmed entities block a structural span; an uncertain one
        # it overlaps is superseded (it would never be replaced unattended).
        added, removed = structural.detect(
            text, list(confirmed),
            place_verifier=_spacy_place_verifier(spacy_model) if use_entity_trace else None,
            skip_locations=skip_location_entities,
        )
        if removed:
            gone = {id(e) for e in removed}
            confirmed          = [e for e in confirmed          if id(e) not in gone]
            needs_confirmation = [e for e in needs_confirmation if id(e) not in gone]
        if added:
            added = _outside_opaque(added, opaque) if opaque else added
            needs_confirmation = [e for e in needs_confirmation if not any(
                a.start < e.end and e.start < a.end for a in added)]
            confirmed = list(confirmed) + added
            logger.info(f"[SentinelLayer] Pass S: +{len(added)} structural entit(ies)")

    # ── Pass G: a gender term that agrees with a named person (D1) ─────────
    # The person's surrogate keeps their gender, so "Sarah … female" →
    # "Mia … male" would only make the message incoherent. A term of the
    # same gender as a named person of the message stays as typed: the
    # surrogate name already says as much. Any other term is still masked.
    confirmed, follows = gender_follows_name(confirmed)
    for e in follows:
        skip_reasons[(e.start, e.end)] = "gender_follows_name"
    if follows:
        all_skipped = list(all_skipped) + follows

    # ── Quasi-identifier combination scoring ──────────────────────────────────
    confirmed = _TaggedList(confirmed)  # wrap to allow attribute assignment
    qi_matches = qi_score(confirmed)
    if qi_matches:
        for match in qi_matches:
            logger.info(
                f"[SentinelLayer] Quasi-ID risk: {match.combination_name} "
                f"(fields: {match.matched_fields}, risk: {match.risk_level})"
            )
    confirmed._qi_matches = qi_matches
    confirmed._skipped_entities = all_skipped
    confirmed._skip_reasons = skip_reasons

    # ── pii_off filtering ─────────────────────────────────────────────────────
    if pii_off:
        exclude_types = resolve_pii_off(pii_off)

        old_qi      = confirmed._qi_matches
        old_skipped = confirmed._skipped_entities
        confirmed = _TaggedList([e for e in confirmed if e.type not in exclude_types])
        confirmed._qi_matches       = old_qi
        confirmed._skipped_entities = old_skipped
        confirmed._skip_reasons     = skip_reasons
    _lap("post_passes_ms")

    logger.info(
        f"[SentinelLayer] Final → "
        f"confirmed={len(confirmed)}, "
        f"needs_confirmation={len(needs_confirmation)}"
    )
    return confirmed, needs_confirmation


def _anchor_to_lines(entities: List[DetectedEntity], text: str) -> List[DetectedEntity]:
    """Each model entity's text is the message slice at its offsets, cut to
    the line holding most of its letters (an entity never spans a line
    break). Pattern entities are exact already and pass through."""
    out = []
    for e in entities:
        if e.source == "pattern" or not (0 <= e.start < e.end <= len(text)):
            out.append(e)
            continue
        raw = text[e.start:e.end]
        if raw == e.text and "\n" not in raw:
            out.append(e)
            continue
        best = None
        for m in re.finditer(r"[^\n]+", raw):
            seg = m.group()
            s = e.start + m.start() + (len(seg) - len(seg.lstrip()))
            t = e.start + m.end() - (len(seg) - len(seg.rstrip()))
            letters = sum(c.isalpha() for c in text[s:t])
            if s < t and (best is None or letters > best[0]):
                best = (letters, s, t)
        if best is None or best[0] == 0:
            continue
        out.append(_dc_replace(e, text=text[best[1]:best[2]], start=best[1], end=best[2]))
    return out


_JOINER = "-'’"
_QUOTES = "\"'“”‘’«»`"
# a name word next to a one-word PERSON: "Hyun-woo Park", "Jean-Luc d'Aubigné"
_NAME_NEXT = re.compile(r" ((?:[dDlL]['’]|[OÓ]['’ ]|Mac|Mc)?[A-ZÀ-ÖØ-Þ][^\W\d_]+(?:[\-'’][^\W\d_]+)*)(?![\w\-'’@.]\w)")
_PARTICLES_BEFORE = re.compile(r"(?:\b(?:[Vv]an|[Vv]on|[Dd]e|[Dd]er|[Dd]en|[Tt]er|[Dd]i|[Dd]a|[Dd]u|[Ll]a|[Ll]e|[Zz]u) )+$")
_NOT_NAME_NEXT = frozenset("""
i im i'm monday tuesday wednesday thursday friday saturday sunday january
february march april may june july august september october november december
thanks thank please hi hello dear regards best cheers sorry yes no ok okay
can could would should will is are was were has have had do does did
""".split())
# a company word after a one-word ORG: "Brightwater Logistics"
_ORG_TAIL = re.compile(
    r" (Ltd|Limited|Inc|LLC|GmbH|AG|SA|BV|Pty|PLC|Co|Corp|Group|Logistics|Roofing|"
    r"Grocers|Clinic|Consulting|Solutions|Studio|Studios|Services|Holdings|Partners|"
    r"Construction|Plumbing|Electric|Dental|Bakery|Labs|Systems|Freight|Design)\b\.?")


def _snap_to_words(entities: List[DetectedEntity], text: str) -> List[DetectedEntity]:
    """Model spans end on word boundaries: a span cut inside a word ("Hyun"
    of "Hyun-woo", "n Adeyemi-" of "Oluwaseun Adeyemi-Clarke") grows to the
    whole word, outer quotes are dropped, a one-word PERSON takes the name
    word or particles next to it ("van der Merwe", "Jean-Luc d'Aubigné") and
    a one-word ORG its company word. Every letter of a name it touches is
    replaced: a partial cover leaks the rest (I1, J2)."""
    out = []
    for e in entities:
        if e.source == "pattern" or not (0 <= e.start < e.end <= len(text)):
            out.append(e)
            continue
        s, t = e.start, e.end
        while s < t and (text[s] in _QUOTES or text[s].isspace()):
            s += 1
        while t > s and (text[t - 1] in _QUOTES or text[t - 1] in _JOINER or text[t - 1].isspace()):
            t -= 1
        if s == t:
            out.append(e)
            continue
        while s > 0 and (text[s - 1].isalnum()
                         or (text[s - 1] in _JOINER and s > 1 and text[s - 2].isalpha()
                             and text[s].isalpha())):
            s -= 1
        while t < len(text) and (text[t].isalnum()
                                 or (text[t] in _JOINER and t + 1 < len(text)
                                     and text[t + 1].isalpha() and text[t - 1].isalpha()
                                     and not re.match(r"['’]s\b", text[t:]))):
            t += 1
        if re.search(r"[^\W\d_]['’]s$", text[s:t]):
            t -= 2                          # possessive: "Thandiwe's" -> "Thandiwe"
        if e.type == "PERSON" and " " not in text[s:t].strip():
            m = _NAME_NEXT.match(text, t)
            if (m and m.group(1).lower() not in _NOT_NAME_NEXT
                    and m.group(1).lower() not in pattern_scan._NOT_NAME_WORD):
                t = m.end(1)
            m = _PARTICLES_BEFORE.search(text, max(0, s - 24), s)
            before = text[:m.start()].split()[-1:] if m else []
            if (m and (m.start() == 0 or not text[m.start() - 1].isalnum())
                    and (re.search(r"[VvTt][oae]n\b", m.group())          # van, von, ter, ten
                         or not before or not before[0][:1].islower())):
                s = m.start()                # not "in der Lindenstraße"
        elif e.type == "ORG" and " " not in text[s:t].strip():
            m = _ORG_TAIL.match(text, t)
            if m:
                t = m.end(1)
        if (s, t) == (e.start, e.end) and text[s:t] == e.text:
            out.append(e)
        else:
            out.append(_dc_replace(e, text=text[s:t], start=s, end=t))
    return out


def _outside_opaque(entities: List[DetectedEntity], spans) -> List[DetectedEntity]:
    """Drop every non-pattern entity that overlaps an opaque (URL) span."""
    return [e for e in entities
            if e.source == "pattern" or not any(e.start < b and a < e.end for a, b in spans)]


def deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """Remove duplicate entities by text, keeping the highest-scored one."""
    seen: dict = {}
    for ent in entities:
        key = ent.text.strip()
        if key not in seen or ent.score > seen[key].score:
            seen[key] = ent
    result = _TaggedList(seen.values())
    result.sort(key=lambda e: e.start)
    if hasattr(entities, "_qi_matches"):
        result._qi_matches = entities._qi_matches
    if hasattr(entities, "_skipped_entities"):
        result._skipped_entities = entities._skipped_entities
    if hasattr(entities, "_skip_reasons"):
        result._skip_reasons = entities._skip_reasons
    return result