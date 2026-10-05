"""
reconstruction/resolve.py — ResolvePass

Post-response surrogate-to-original reconstruction (audit E2, I5, I15).

Every pass matches whole values only — never inside a longer word or number
("Lee" ≠ "Leeds", "8" ≠ "8.5", "Lopez" ≠ "Lopez-Garcia") — and every rewritten
span is protected from later passes.

    1. Exact — longest surrogate first, case-insensitive with the case carried
       over ("MIA LOPEZ" → "SARAH MITCHELL"). Lower-case matches only for
       multi-word surrogates or values with a digit/@ (so a surrogate "Will"
       never rewrites "will").
    2. Component — for multi-word surrogates unresolved after Pass 1:
       contiguous n-grams (≥ 2 words) aligned to the original's words with
       difflib, so a partial echo ("790 Crescent Row") restores.
    3. Fuzzy — multi-word, letters-only surrogates of ≥ 6 characters, and only
       when one of their words (≥ 4 letters) is present verbatim; the matched
       span must have the same number of words. Never numbers or single words
       ("Katherine" does not turn "Catherine the Great" into a person).
    4. Given name / surname — "Tell Mia" → "Tell Sarah" for a person-shaped
       surrogate; not before another capitalised word ("Mia Hamm"), not for
       common words ("Grace", "May"), surname only after a title ("Dr. Lopez").

Low-entropy surrogates (a bare age, a gender term, ≤ 2 characters) recur in
ordinary text, so with ``current=`` they are restored only when they were
sent in the current turn (audit I5).

Every failure is logged with its failure type for the research taxonomy:
    exact_miss  — surrogate not found via exact match
    fuzzy_miss  — surrogate not found even via all passes
    fuzzy_hit   — surrogate found only via component/fuzzy match
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..consistency import glued, is_low_entropy, match_case, occurs

logger = logging.getLogger(__name__)

# Default for Pass 3; override per call with resolve(..., fuzzy_threshold=…).
FUZZY_MATCH_THRESHOLD = 85

_TITLES = frozenset("mr mrs ms miss mx dr prof sir madam rev fr".split())
_SUFFIXES = frozenset("jr sr ii iii iv md phd dds dvm esq rn cpa".split())
_ORG_WORDS = frozenset("""inc llc ltd corp co company group partners associates holdings bank
    university college school hospital clinic foundation institute labs systems solutions
    services technologies consulting agency studio center centre church club and of the &""".split())
# Given names that are also ordinary words: a capitalised "Grace" or "May"
# in an answer is not evidence of the person (audit I5).
_COMMON_WORD_NAMES = frozenset("""grace hope faith joy may june april august summer autumn dawn rose
    lily ivy daisy holly iris violet ruby pearl amber crystal sky skye rain storm will mark bill
    bob jack pat sue rob art guy ray don frank grant hunter chase mason carter cooper parker
    taylor bishop king price young rich long little black white brown green gray grey rice hill
    wood lane ford banks bell page booth cash love major miles penny sterling star angel destiny
    harmony trinity justice liberty patience charity precious royal reign sunny honey hazel olive
    sage basil jade jasmine pepper river brook glen dale forest north west south east""".split())
_NAME_TOKEN = re.compile(r"[^\W\d_][^\W\d_'’.-]*(?:['’-][^\W\d_]+)*\.?")
_NEXT_CAPITALISED = re.compile(r"[ \t]+[A-Z]")
_TITLE_BEFORE = re.compile(r"(?i)\b(?:mr|mrs|ms|miss|mx|dr|prof)\.?\s+$")
# A capitalised word right before, which itself is not sentence-initial
# ("Did LeBron James"): the name part belongs to another person's name.
_PREV_NAME = re.compile(r"[^\s.!?:;\"(\[]\s+[A-Z][^\W\d_]*[ \t]+$")


# ─────────────────────────────────────────────
# Failure event dataclass (lightweight)
# ─────────────────────────────────────────────

class ResolutionFailure:
    """Records a single failed or partial resolution for the failure taxonomy."""

    __slots__ = ("surrogate", "original", "failure_type", "context_snippet")

    def __init__(
        self,
        surrogate: str,
        original: str,
        failure_type: str,
        context_snippet: str = "",
    ) -> None:
        self.surrogate = surrogate
        self.original = original
        self.failure_type = failure_type   # 'exact_miss', 'fuzzy_miss', 'fuzzy_hit'
        self.context_snippet = context_snippet


# ─────────────────────────────────────────────
# Span-tracking primitives (shared by all passes)
# ─────────────────────────────────────────────

Spans = List[Tuple[int, int]]


def _overlaps_any(start: int, end: int, spans: Spans) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


def _splice(text: str, start: int, end: int, replacement: str,
            spans: Spans) -> Tuple[str, Spans]:
    """Replace text[start:end] with *replacement*, shifting tracked spans and
    recording the new span as protected."""
    new_text = text[:start] + replacement + text[end:]
    delta = len(replacement) - (end - start)
    updated = [(s + delta, e + delta) if s >= end else (s, e) for s, e in spans]
    updated.append((start, start + len(replacement)))
    return new_text, updated


def _replace_where(
    text: str,
    pattern: "re.Pattern",
    replacement: Callable[[str], str],
    spans: Spans,
    accept: Callable[[str, "re.Match"], bool] = lambda _t, _m: True,
) -> Tuple[str, Spans, int]:
    """Replace every whole-value match of *pattern* outside protected spans
    that *accept* allows; *replacement* maps the matched surface text."""
    hits, pos = 0, 0
    while True:
        m = pattern.search(text, pos)
        if m is None:
            return text, spans, hits
        if (_overlaps_any(m.start(), m.end(), spans) or glued(text, m.start(), m.end())
                or not accept(text, m)):
            pos = m.start() + 1
            continue
        new = replacement(m.group(0))
        text, spans = _splice(text, m.start(), m.end(), new, spans)
        hits += 1
        pos = m.start() + len(new)


def _case_variant_ok(surface: str, surrogate: str) -> bool:
    """Which case variants of a surrogate Pass 1 accepts."""
    if surface == surrogate:
        return True
    if surface.isupper() and sum(c.isalpha() for c in surface) >= 2:
        return True
    if surface == surrogate[:1].upper() + surrogate[1:]:          # sentence start
        return True
    if surface.islower():
        return len(surrogate.split()) > 1 or any(c.isdigit() or c == "@" for c in surrogate)
    return False


# ─────────────────────────────────────────────
# Pass 2 helpers — token alignment
# ─────────────────────────────────────────────

def _aligned_original(opcodes, orig_words: List[str], i: int, j: int) -> Optional[str]:
    """
    Translate the surrogate word range [i, j) into the corresponding original
    words using SequenceMatcher opcodes (a=surrogate words, b=original words).

    Returns None when the range cuts through the middle of a non-equal block
    (no well-defined correspondence).
    """
    parts: List[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if i2 <= i or i1 >= j:
            continue
        if tag == "equal":
            lo, hi = max(i1, i), min(i2, j)
            parts.extend(orig_words[j1 + (lo - i1): j1 + (hi - i1)])
        else:
            if i1 < i or i2 > j:
                return None  # partial overlap with replace/delete/insert block
            parts.extend(orig_words[j1:j2])
    return " ".join(parts) if parts else None


def _ngram_pattern(words: List[str]) -> "re.Pattern":
    """Whitespace-flexible pattern for a run of tokens (boundaries are
    checked by :func:`glued`)."""
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def _snap_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    """Expand [start, end) outward so it never cuts a word in half, then trim
    surrounding whitespace so replacements never eat separating spaces."""
    while start > 0 and start < len(text) and text[start].isalnum() and text[start - 1].isalnum():
        start -= 1
    while end > start and end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _core_words(value: str) -> List[str]:
    """Words of a name without titles and post-nominal suffixes."""
    return [w for w in value.split() if w.strip(".,").lower() not in _TITLES | _SUFFIXES]


def _person_shaped(surrogate: str) -> bool:
    words = _core_words(surrogate)
    return (len(words) >= 2 and all(_NAME_TOKEN.fullmatch(w) and w[0].isupper() for w in words)
            and not any(w.strip(".,").lower() in _ORG_WORDS for w in surrogate.split()))


def _fuzzy_eligible(surrogate: str) -> bool:
    words = surrogate.split()
    return (len(words) >= 2 and len(surrogate) >= 6
            and not any(c.isdigit() for c in surrogate)
            and all(_NAME_TOKEN.fullmatch(w.strip(",")) for w in words))


# ─────────────────────────────────────────────
# ResolvePass
# ─────────────────────────────────────────────

class ResolvePass:
    """
    Reconstructs original PII values in LLM responses (see module docstring).

    Attributes:
        failures: Log of ResolutionFailure events, accumulated across calls.
    """

    def __init__(self) -> None:
        """Initialise ResolvePass with an empty failure log."""
        self.failures: List[ResolutionFailure] = []

    def _mark_hit(self, surrogate: str) -> None:
        for f in reversed(self.failures):
            if f.surrogate == surrogate and f.failure_type == "exact_miss":
                f.failure_type = "fuzzy_hit"
                break

    def resolve(
        self,
        response_text: str,
        shadow_map: Dict[str, str],
        fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
        *,
        current: Optional[Iterable[str]] = None,
        sent: Optional[str] = None,
    ) -> str:
        """
        Reconstruct original values in *response_text* using *shadow_map*.

        Args:
            response_text:   The LLM response text (may contain surrogates).
            shadow_map:      Dict mapping surrogate → original.
            fuzzy_threshold: Minimum rapidfuzz score (0–100) for Pass 3.
            current:         Surrogates sent in the turn being answered. When
                             given, low-entropy surrogates outside it are not
                             restored (audit I5).
            sent:            The masked text the response answers. A bare
                             name part that the user typed there outside any
                             surrogate ("Peter" in "Peter the Great") is the
                             user's own word, so Pass 4 leaves it alone.

        Returns:
            Response string with surrogates replaced by original values.
        """
        if current is not None:
            current = set(current)
            shadow_map = {s: o for s, o in shadow_map.items()
                          if s in current or not is_low_entropy(s)}
        if not shadow_map:
            return response_text

        result = response_text
        protected: Spans = []
        unresolved: Dict[str, str] = {}

        # ── Pass 1: whole-value, case-carrying, longest first ──────────
        for surrogate, original in sorted(shadow_map.items(), key=lambda kv: len(kv[0]),
                                          reverse=True):
            if not surrogate.strip():
                continue
            pattern = re.compile(re.escape(surrogate), re.IGNORECASE)
            result, protected, hits = _replace_where(
                result, pattern, lambda surface: match_case(surface, surrogate, original),
                protected, accept=lambda _t, m: _case_variant_ok(m.group(0), surrogate))
            if hits:
                logger.debug(f"[ResolvePass] Exact hit: {surrogate!r} → {original!r}")
            else:
                unresolved[surrogate] = original
                self.failures.append(ResolutionFailure(
                    surrogate=surrogate, original=original, failure_type="exact_miss",
                    context_snippet=response_text[:120]))

        # Low-entropy values never take part in partial matching.
        unresolved = {s: o for s, o in unresolved.items() if not is_low_entropy(s)}
        if not unresolved:
            return result

        # Partial matches (passes 2–4) never take text the user typed outside
        # a surrogate: "Peter" in "Peter the Great" is the user's own word.
        typed = sent
        if typed is not None:
            for surrogate in sorted(shadow_map, key=len, reverse=True):
                if not is_low_entropy(surrogate):
                    typed = re.sub(re.escape(surrogate), " ", typed, flags=re.IGNORECASE)

        def own_words(_text, m) -> bool:
            return typed is not None and occurs(typed, m.group(0))

        # ── Pass 2: alignment-safe component matching ──────────────────
        for surrogate, original in list(unresolved.items()):
            surrogate_words = surrogate.split()
            original_words = original.split()
            if len(surrogate_words) <= 1:
                continue
            opcodes = difflib.SequenceMatcher(
                None, surrogate_words, original_words, autojunk=False).get_opcodes()
            hit = False
            for n in range(len(surrogate_words), 1, -1):          # longest first, ≥ 2 words
                for i in range(0, len(surrogate_words) - n + 1):
                    aligned = _aligned_original(opcodes, original_words, i, i + n)
                    if aligned is None or aligned == " ".join(surrogate_words[i:i + n]):
                        continue
                    result, protected, hits = _replace_where(
                        result, _ngram_pattern(surrogate_words[i:i + n]),
                        lambda _surface, a=aligned: a, protected,
                        accept=lambda t, m: not own_words(t, m))
                    if hits:
                        logger.debug(f"[ResolvePass] Component hit ({n}-gram) for {surrogate!r}")
                        hit = True
                if hit:
                    break
            if hit:
                self._mark_hit(surrogate)
                del unresolved[surrogate]

        # ── Pass 3: anchored fuzzy matching (names with a typo) ────────
        fuzzy_candidates = {s: o for s, o in unresolved.items() if _fuzzy_eligible(s)}
        if fuzzy_candidates:
            try:
                from rapidfuzz import fuzz
            except ImportError:
                logger.warning("[ResolvePass] rapidfuzz not installed — skipping fuzzy pass")
                fuzzy_candidates = {}
            for surrogate, original in fuzzy_candidates.items():
                span = _find_fuzzy_span(result, surrogate, fuzzy_threshold, protected)
                if span is None or typed is not None and occurs(typed, result[span[0]:span[1]]):
                    continue
                start, end = span
                result, protected = _splice(result, start, end, original, protected)
                logger.debug(f"[ResolvePass] Fuzzy hit for {surrogate!r}")
                self._mark_hit(surrogate)
                del unresolved[surrogate]

        # ── Pass 4: given name / surname of a person-shaped surrogate ──
        other_strings = None
        for surrogate, original in list(unresolved.items()):
            sw, ow = _core_words(surrogate), _core_words(original)
            if not (_person_shaped(surrogate) and len(sw) == len(ow)):
                continue
            if other_strings is None:
                other_strings = [x for pair in shadow_map.items() for x in pair]
            hit = False
            for s_word, o_word, need_title in ((sw[0], ow[0], False), (sw[-1], ow[-1], True)):
                if (s_word == o_word or len(s_word) < 3
                        or s_word.strip(".").lower() in _COMMON_WORD_NAMES
                        or any(s_word in x for x in other_strings if x not in (surrogate, original))
                        or typed is not None and occurs(typed, s_word)):
                    continue

                def accept(text, m, need_title=need_title):
                    if _NEXT_CAPITALISED.match(text, m.end()):
                        return False                     # "Mia Hamm": another person
                    if _PREV_NAME.search(text, 0, m.start()) and not _TITLE_BEFORE.search(
                            text, 0, m.start()):
                        return False                     # "LeBron James": another person
                    return not need_title or bool(_TITLE_BEFORE.search(text, 0, m.start()))
                result, protected, hits = _replace_where(
                    result, re.compile(re.escape(s_word)), lambda _s, o=o_word: o,
                    protected, accept=accept)
                hit = hit or bool(hits)
            if hit:
                logger.debug(f"[ResolvePass] Name-part hit for {surrogate!r}")
                self._mark_hit(surrogate)
                del unresolved[surrogate]

        for surrogate, original in unresolved.items():
            self.failures.append(ResolutionFailure(
                surrogate=surrogate, original=original, failure_type="fuzzy_miss",
                context_snippet=result[:120]))
        return result

    def get_failure_summary(self) -> Dict[str, int]:
        """
        Return counts of each failure type accumulated across all calls.

        Returns:
            Dict with keys 'exact_miss', 'fuzzy_miss', 'fuzzy_hit'.
        """
        summary: Dict[str, int] = {"exact_miss": 0, "fuzzy_miss": 0, "fuzzy_hit": 0}
        for f in self.failures:
            summary[f.failure_type] = summary.get(f.failure_type, 0) + 1
        return summary


# ─────────────────────────────────────────────
# Fuzzy span finder (anchored)
# ─────────────────────────────────────────────

def _find_fuzzy_span(text: str, query: str, threshold: float,
                     protected: Spans = ()) -> Optional[Tuple[int, int]]:
    """
    The best fuzzy occurrence of *query* in *text*, with true offsets, or None.

    Requires one of the query's words (≥ 4 letters) verbatim in *text*
    outside protected spans, then uses rapidfuzz.fuzz.partial_ratio_alignment,
    snaps to word boundaries, and accepts only a span with the same number of
    words that re-scores ≥ *threshold* with fuzz.ratio.
    """
    from rapidfuzz import fuzz

    if not query or not text:
        return None
    anchors = [w.strip(".,'’") for w in query.split() if len(w.strip(".,'’")) >= 4]
    if not any(not _overlaps_any(m.start(), m.end(), protected) and not glued(text, m.start(), m.end())
               for a in anchors for m in re.finditer(re.escape(a), text)):
        return None
    alignment = fuzz.partial_ratio_alignment(query.lower(), text.lower(), score_cutoff=threshold)
    if alignment is None:
        return None
    start, end = _snap_to_word_boundaries(text, alignment.dest_start, alignment.dest_end)
    if (len(text[start:end].split()) != len(query.split())
            or _overlaps_any(start, end, protected) or glued(text, start, end)):
        return None
    if fuzz.ratio(query.lower(), text[start:end].lower()) < threshold:
        return None
    # a capitalised word matches only a capitalised word ("James Orr" is not
    # "James or")
    if any(q[:1].isupper() and not w[:1].isupper()
           for q, w in zip(query.split(), text[start:end].split())):
        return None
    return start, end
