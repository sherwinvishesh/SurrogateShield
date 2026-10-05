"""
consistency.py — one surrogate per original across turns, and the rules
that keep a long conversation's map from corrupting later answers
(audit I3, I4, I5).

* :func:`is_low_entropy` — values such as a bare age ("72"), a gender term or
  a one/two-character token recur in ordinary text. Their surrogates are not
  treated as quoted-back surrogates in later turns, and are restored only in
  the answer to the turn that sent them.
* :func:`occurs` / :func:`glued` — whole-value, case-insensitive matching
  that never fires inside a longer word or number ("Lee" in "Leeds", "8" in
  "8.5"). Han, kana and Thai run words together, so they never glue a value
  to its context. Masking (``plan_substitutions``) and restoring use it.
* :func:`assign_surrogates` — reuse the surrogate an original already has in
  the conversation (exact, then case-insensitive with the case adapted),
  generate the rest, and never pick a surrogate that already appears in the
  message.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

_AGE = re.compile(
    r"(?i)^\d{1,3}(?:\s*(?:years?|yrs?|y/?o)(?:\s*old)?|-years?-old)?$")
_GENDER_LABEL = re.compile(r"(?i)^(?:gender|sex|pronouns?)\s*[:=]?\s*")
_GENDER_TERMS = frozenset({
    "male", "female", "man", "woman", "non-binary", "nonbinary", "non-binary person",
    "he/him", "she/her", "they/them", "m", "f", "x",
})
_CLITIC = re.compile(r"(?:'s|’s|'|’)$")


def is_low_entropy(value: str) -> bool:
    """True for values that recur in ordinary text: a bare age, a gender
    term (with or without a ``gender:`` label), or one or two characters."""
    v = value.strip()
    if len(v) <= 2 or _AGE.match(v):
        return True
    return _GENDER_LABEL.sub("", v).strip().lower() in _GENDER_TERMS


def strip_clitic(value: str) -> str:
    """``"Sarah Mitchell's"`` → ``"Sarah Mitchell"``; ``"Jennings'"`` → ``"Jennings"``."""
    return _CLITIC.sub("", value)


def wordchar(c: str) -> bool:
    """A letter or digit of a script that separates words with spaces."""
    if not c.isalnum():
        return False
    cp = ord(c)
    return not (0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
                or 0xF900 <= cp <= 0xFAFF or 0x0E00 <= cp <= 0x0E7F or 0x20000 <= cp <= 0x2FA1F)


def _digit_run(text: str, i: int, step: int) -> int:
    j = i
    while 0 <= j < len(text) and text[j].isdigit():
        j += step
    return abs(j - i)


def glued(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is part of a longer word or number:
    a letter/digit on either side, a decimal or time ("8" in "8.5", "1:30")
    or a thousands group ("1" in "1,000", not "8142" in the CSV row
    "8142,38046").

    A hyphen is a boundary: masking replaces "Tom Okafor" inside
    "Tom Okafor-Smith" (privacy first), so restoring must accept it too.
    Masking and restoring use this one rule."""
    if start >= end:
        return True
    first, last = text[start], text[end - 1]
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    if (wordchar(before) and wordchar(first)) or (wordchar(after) and wordchar(last)):
        return True
    if last.isdigit() and after in ".:" and _digit_run(text, end + 1, 1):
        return True
    if first.isdigit() and before in ".:" and _digit_run(text, start - 2, -1):
        return True
    # thousands: "1,000" — a comma followed by exactly three digits
    if last.isdigit() and after == "," and _digit_run(text, end + 1, 1) == 3:
        return True
    if (first.isdigit() and before == "," and _digit_run(text, start - 2, -1)
            and _digit_run(text, start, 1) == 3):
        return True
    return False


def find_all(text: str, value: str, *, ignore_case: bool = True) -> List[re.Match]:
    """Every whole-value occurrence of *value* in *text*."""
    if not value:
        return []
    pat = re.compile(re.escape(value), re.IGNORECASE if ignore_case else 0)
    return [m for m in pat.finditer(text) if not glued(text, m.start(), m.end())]


def occurs(text: str, value: str) -> bool:
    return bool(find_all(text, value))


def match_case(surface: str, model: str, value: str) -> str:
    """*value* in the case *surface* uses relative to *model* (the form the
    value was registered under): ALL CAPS and all lower are carried over."""
    if surface == model:
        return value
    if surface.isupper() and not model.isupper():
        return value.upper()
    if surface.islower() and not model.islower():
        return value.lower()
    return value


def surrogate_for(shadow, original: str) -> Optional[str]:
    """The surrogate *shadow* already issued for *original*: exact first,
    then case-insensitively with the case adapted ("SARAH MITCHELL" →
    "JANE DOE")."""
    found = shadow.lookup_original(original)
    if found is not None:
        return found
    folded = original.casefold()
    for known in shadow.originals():
        if known.casefold() == folded:
            return match_case(original, known, shadow.lookup_original(known))
    return None


def quoted_back(mappings: Iterable[str]) -> Set[str]:
    """Surrogates a later message may quote back (detection skips them).
    Low-entropy surrogates are excluded: a "70" or "female" in a later
    message is a value of its own."""
    return {s for s in mappings if not is_low_entropy(s)}


def assign_surrogates(entities, text: str, mimic, shadows: Sequence = (), *,
                      forbidden: Optional[Set[str]] = None, **generate_kwargs) -> Dict[str, str]:
    """original → surrogate for *entities* of *text*.

    Reuses the surrogate an original already has in *shadows* (checked in
    order, case-insensitively), unless it is low-entropy and also appears
    elsewhere in *text*. Case variants within *text* ("Sarah Mitchell",
    "SARAH MITCHELL") share one surrogate in the matching case. The rest are
    generated with ``mimic.generate_all(..., text=text)``. Every surrogate
    maps back to exactly one original: a case-adapted form already taken by
    another original is replaced by a fresh surrogate.
    """
    shadows = [sh for sh in shadows if sh is not None]

    def taken(surrogate: str, original: str) -> bool:
        return any(sh.get(surrogate) not in (None, original) for sh in shadows)

    surrogate_map: Dict[str, str] = {}
    pending: Dict[str, list] = {}           # casefold → entities still needing a surrogate
    seen: Set[str] = set()
    for ent in entities:
        key = ent.text.strip()
        if key in seen:
            continue
        seen.add(key)
        existing = next((s for s in (surrogate_for(sh, key) for sh in shadows) if s is not None), None)
        if (existing is not None and not (is_low_entropy(existing) and occurs(text, existing))
                and not taken(existing, key)):
            surrogate_map[key] = existing
        else:
            pending.setdefault(key.casefold(), []).append(ent)

    # One generated surrogate per casefold group, unless a case variant of
    # the value already has one.
    leaders = [g[0] for f, g in pending.items()
               if not any(o.casefold() == f for o in surrogate_map)]
    if leaders:
        surrogate_map.update(mimic.generate_all(leaders, forbidden=forbidden, text=text,
                                                **generate_kwargs))
    used = {s: o for o, s in surrogate_map.items()}
    retry = []
    for group in pending.values():
        for ent in group:
            original = ent.text.strip()
            if original in surrogate_map:
                continue
            base = next(o for o in surrogate_map if o.casefold() == original.casefold())
            adapted = match_case(original, base, surrogate_map[base])
            if used.get(adapted, original) != original or taken(adapted, original):
                retry.append(ent)            # "Sarah mitchell": no distinct case form
            else:
                surrogate_map[original] = adapted
                used[adapted] = original
    if retry:
        surrogate_map.update(mimic.generate_all(retry, forbidden=forbidden, text=text,
                                                **generate_kwargs))
    return surrogate_map
