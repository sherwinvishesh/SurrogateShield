# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
reconstruction/logic.py — ResolvePass

Post-response surrogate-to-original reconstruction.

Three passes in sequence:
    1. Exact string replacement — longest surrogate first so substrings never
       clobber longer values; every rewritten span is tracked so later passes
       can never corrupt it.
    2. Alignment-safe component matching — for multi-word surrogates that are
       UNRESOLVED after Pass 1.  Surrogate/original words are aligned with
       difflib.SequenceMatcher (truncation-proof for length-mismatched pairs),
       then contiguous surrogate n-grams (longest first, minimum 2 words) are
       searched with whitespace-flexible word-boundary patterns and replaced
       by the aligned original words.  A guarded single-token fallback covers
       "first name only" echoes.
       IMPORTANT: only processes surrogates in the `unresolved` set —
       running component matching on already-resolved surrogates caused
       silent corruption (e.g. "Ashley" from resolved "Ashley Wise" wrongly
       replacing "Ashley" in "Ashley County" in the same response).
    3. Fuzzy match — rapidfuzz.fuzz.partial_ratio_alignment gives the TRUE
       best-match offsets (v1's sliding window anchored the replacement at
       the window start, garbling output).  The span is snapped to word
       boundaries, sanity-checked for length, and re-verified with
       fuzz.ratio before replacing.

Every failure is logged with its failure type for the research taxonomy:
    exact_miss  — surrogate not found via exact match
    fuzzy_miss  — surrogate not found even via all passes
    fuzzy_hit   — surrogate found only via component/fuzzy match
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Set, Tuple

from config import FUZZY_MATCH_THRESHOLD
from util import get_logger

logger = get_logger(__name__)


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

def _overlaps_any(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


def _splice(
    text: str,
    start: int,
    end: int,
    replacement: str,
    spans: List[Tuple[int, int]],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Replace text[start:end] with *replacement*, shifting tracked spans and
    recording the new span as protected."""
    new_text = text[:start] + replacement + text[end:]
    delta = len(replacement) - (end - start)
    updated = [
        (s + delta, e + delta) if s >= end else (s, e)
        for s, e in spans
    ]
    updated.append((start, start + len(replacement)))
    return new_text, updated


def _replace_exact_tracked(
    text: str,
    needle: str,
    replacement: str,
    spans: List[Tuple[int, int]],
) -> Tuple[str, List[Tuple[int, int]], int]:
    """Replace every occurrence of *needle* outside protected spans."""
    hits = 0
    idx = text.find(needle)
    while idx != -1:
        if _overlaps_any(idx, idx + len(needle), spans):
            idx = text.find(needle, idx + 1)
            continue
        text, spans = _splice(text, idx, idx + len(needle), replacement, spans)
        hits += 1
        idx = text.find(needle, idx + len(replacement))
    return text, spans, hits


def _replace_pattern_tracked(
    text: str,
    pattern: "re.Pattern",
    replacement: str,
    spans: List[Tuple[int, int]],
) -> Tuple[str, List[Tuple[int, int]], int]:
    """Replace every regex match outside protected spans."""
    hits = 0
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if m is None:
            break
        if _overlaps_any(m.start(), m.end(), spans):
            pos = m.start() + 1
            continue
        text, spans = _splice(text, m.start(), m.end(), replacement, spans)
        hits += 1
        pos = m.start() + len(replacement)
    return text, spans, hits


# ─────────────────────────────────────────────
# Pass 2 helpers — token alignment
# ─────────────────────────────────────────────

def _aligned_original(
    opcodes,
    orig_words: List[str],
    i: int,
    j: int,
) -> Optional[str]:
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
    """Whitespace-flexible, word-boundary pattern for a run of tokens."""
    return re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
    )


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


# ─────────────────────────────────────────────
# ResolvePass
# ─────────────────────────────────────────────

class ResolvePass:
    """
    Reconstructs original PII values in LLM responses.

    Uses three passes:
      1. Exact replacement (longest surrogate first, span-tracked)
      2. Alignment-safe component matching — scoped to UNRESOLVED surrogates
         only, to prevent component words of already-resolved surrogates from
         corrupting unrelated text.
      3. Anchored fuzzy matching via rapidfuzz partial_ratio_alignment

    Attributes:
        failures: Log of ResolutionFailure events, accumulated across calls.
    """

    def __init__(self) -> None:
        """Initialise ResolvePass with an empty failure log."""
        self.failures: List[ResolutionFailure] = []

    def resolve(self, response: str, shadow_map: Dict[str, str]) -> str:
        """
        Reconstruct original values in *response* using *shadow_map*.

        Args:
            response:   The LLM response text (may contain surrogates).
            shadow_map: Dict mapping surrogate → original.

        Returns:
            Response string with surrogates replaced by original values.
        """
        if not shadow_map:
            return response

        result = response
        protected: List[Tuple[int, int]] = []
        unresolved: Dict[str, str] = {}

        # ── Pass 1: Exact replacement (longest surrogate first) ────────
        for surrogate, original in sorted(
            shadow_map.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            result, protected, hits = _replace_exact_tracked(
                result, surrogate, original, protected
            )
            if hits:
                logger.debug(f"[ResolvePass] Exact hit: {surrogate!r} → {original!r}")
            else:
                unresolved[surrogate] = original
                logger.debug(f"[ResolvePass] Exact miss: {surrogate!r}")
                self.failures.append(
                    ResolutionFailure(
                        surrogate=surrogate,
                        original=original,
                        failure_type="exact_miss",
                        context_snippet=response[:120],
                    )
                )

        if not unresolved:
            return result

        # ── Pass 2: Alignment-safe component matching ──────────────────
        #
        # Scoped to `unresolved` only (see module docstring for the
        # Ashley-County rationale). Words are aligned with SequenceMatcher,
        # so length-mismatched surrogate/original pairs can never be zipped
        # out of position (the v1 truncation bug).
        component_resolved: Set[str] = set()

        for surrogate, original in list(unresolved.items()):
            surrogate_words = surrogate.split()
            original_words = original.split()

            if len(surrogate_words) <= 1:
                continue  # single-word surrogates handled by Pass 1 / Pass 3

            opcodes = difflib.SequenceMatcher(
                None, surrogate_words, original_words, autojunk=False
            ).get_opcodes()

            hit = False
            # Contiguous n-grams, longest first, minimum 2 tokens.
            for n in range(len(surrogate_words), 1, -1):
                for i in range(0, len(surrogate_words) - n + 1):
                    aligned = _aligned_original(
                        opcodes, original_words, i, i + n
                    )
                    if aligned is None:
                        continue
                    pattern = _ngram_pattern(surrogate_words[i:i + n])
                    result, protected, hits = _replace_pattern_tracked(
                        result, pattern, aligned, protected
                    )
                    if hits:
                        logger.debug(
                            f"[ResolvePass] Component hit ({n}-gram): "
                            f"{' '.join(surrogate_words[i:i + n])!r} → {aligned!r} "
                            f"(surrogate: {surrogate!r})"
                        )
                        hit = True
                if hit:
                    break

            if hit:
                component_resolved.add(surrogate)
                for f in reversed(self.failures):
                    if f.surrogate == surrogate and f.failure_type == "exact_miss":
                        f.failure_type = "fuzzy_hit"
                        break

        for surrogate in component_resolved:
            del unresolved[surrogate]

        if not unresolved:
            return result

        # ── Pass 3: Anchored fuzzy matching ────────────────────────────
        # Runs BEFORE the single-token fallback so a whole-value typo echo
        # ("Jordn Mercer") is repaired as one unit rather than word-by-word.
        try:
            from rapidfuzz import fuzz
            fuzzy_available = True
        except ImportError:
            logger.warning("[ResolvePass] rapidfuzz not installed — skipping fuzzy pass")
            fuzzy_available = False

        if fuzzy_available:
            for surrogate, original in list(unresolved.items()):
                span = _find_fuzzy_span(result, surrogate, FUZZY_MATCH_THRESHOLD)

                if span is not None and not _overlaps_any(span[0], span[1], protected):
                    start, end = span
                    matched_text = result[start:end]
                    result, protected = _splice(result, start, end, original, protected)
                    logger.debug(
                        f"[ResolvePass] Fuzzy hit: {matched_text!r} → {original!r}"
                    )
                    for f in reversed(self.failures):
                        if f.surrogate == surrogate and f.failure_type == "exact_miss":
                            f.failure_type = "fuzzy_hit"
                            break
                    del unresolved[surrogate]

        # ── Pass 4: Guarded single-token fallback (last resort) ────────
        # "First name only" echoes: equal word counts, token ≥3 chars,
        # capitalized, and not a substring of any other shadow key/value.
        final_unresolved: Dict[str, str] = {}

        for surrogate, original in unresolved.items():
            surrogate_words = surrogate.split()
            original_words = original.split()
            hit = False
            if len(surrogate_words) > 1 and len(surrogate_words) == len(original_words):
                other_strings = [
                    s
                    for pair in shadow_map.items()
                    for s in pair
                    if s not in (surrogate, original)
                ]
                for s_word, o_word in zip(surrogate_words, original_words):
                    if s_word == o_word or len(s_word) < 3 or not s_word[0].isupper():
                        continue
                    if any(s_word in other for other in other_strings):
                        continue
                    pattern = _ngram_pattern([s_word])
                    result, protected, hits = _replace_pattern_tracked(
                        result, pattern, o_word, protected
                    )
                    if hits:
                        logger.debug(
                            f"[ResolvePass] Component hit (single token): "
                            f"{s_word!r} → {o_word!r} (surrogate: {surrogate!r})"
                        )
                        hit = True

            if hit:
                for f in reversed(self.failures):
                    if f.surrogate == surrogate and f.failure_type == "exact_miss":
                        f.failure_type = "fuzzy_hit"
                        break
            else:
                final_unresolved[surrogate] = original

        for surrogate, original in final_unresolved.items():
            self.failures.append(
                ResolutionFailure(
                    surrogate=surrogate,
                    original=original,
                    failure_type="fuzzy_miss",
                    context_snippet=result[:120],
                )
            )

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

def _find_fuzzy_span(
    text: str,
    query: str,
    threshold: float,
) -> Optional[Tuple[int, int]]:
    """
    Find the best fuzzy occurrence of *query* in *text* with TRUE offsets.

    Uses rapidfuzz.fuzz.partial_ratio_alignment (one call, no sliding
    window), snaps the returned span outward to word boundaries, rejects
    spans wildly shorter/longer than the query (0.5×–2×), and re-verifies
    the snapped span with fuzz.ratio before accepting.

    Returns:
        (start, end) of the span to replace, or None if no acceptable match.
    """
    from rapidfuzz import fuzz

    if not query or not text:
        return None

    alignment = fuzz.partial_ratio_alignment(
        query.lower(), text.lower(), score_cutoff=threshold
    )
    if alignment is None:
        return None

    start, end = _snap_to_word_boundaries(text, alignment.dest_start, alignment.dest_end)
    span_len = end - start
    if span_len < 0.5 * len(query) or span_len > 2 * len(query):
        return None

    if fuzz.ratio(query.lower(), text[start:end].lower()) < threshold:
        return None

    return start, end
