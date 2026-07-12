"""
reconstruction/resolve.py — ResolvePass

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
       "first name only" echoes.  Scoped to UNRESOLVED surrogates only to
       prevent component words of already-resolved surrogates from corrupting
       unrelated text.
    3. Fuzzy match — rapidfuzz.fuzz.partial_ratio_alignment gives the TRUE
       best-match offsets (v1's sliding window anchored the replacement at
       the window start, garbling output).  The span is snapped to word
       boundaries, sanity-checked for length, and re-verified with
       fuzz.ratio before replacing.  Threshold is configurable
       (config(fuzzy_threshold=…)).
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Span-tracking primitives (shared by all passes)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 helpers — token alignment
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ResolvePass
# ─────────────────────────────────────────────────────────────────────────────

class ResolvePass:
    """Reconstructs original PII values in LLM responses using three passes."""

    def resolve(
        self,
        response_text: str,
        shadow_map: Dict[str, str],
        fuzzy_threshold: int = 85,
    ) -> str:
        """
        Reconstruct original values in *response_text* using *shadow_map*.

        Args:
            response_text:   The LLM response (may contain surrogates).
            shadow_map:      Dict mapping surrogate → original.
            fuzzy_threshold: Minimum rapidfuzz score (0–100) for Pass 3.

        Returns:
            Response string with surrogates replaced by original values.
        """
        if not shadow_map:
            return response_text

        result = response_text
        protected: List[Tuple[int, int]] = []
        unresolved: Dict[str, str] = {}

        # ── Pass 1: Exact replacement (longest surrogate first) ──────────────
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

        if not unresolved:
            return result

        # ── Pass 2: Alignment-safe component matching (UNRESOLVED only) ───────
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

        for surrogate in component_resolved:
            del unresolved[surrogate]

        if not unresolved:
            return result

        # ── Pass 3: Anchored fuzzy matching ───────────────────────────────────
        # Runs BEFORE the single-token fallback so a whole-value typo echo
        # ("Jordn Mercer") is repaired as one unit rather than word-by-word.
        try:
            from rapidfuzz import fuzz  # noqa: F401
            fuzzy_available = True
        except ImportError:
            logger.warning("[ResolvePass] rapidfuzz not installed — skipping fuzzy pass")
            fuzzy_available = False

        if fuzzy_available:
            for surrogate, original in list(unresolved.items()):
                span = _find_fuzzy_span(result, surrogate, fuzzy_threshold)

                if span is not None and not _overlaps_any(span[0], span[1], protected):
                    start, end = span
                    matched_text = result[start:end]
                    result, protected = _splice(result, start, end, original, protected)
                    logger.debug(
                        f"[ResolvePass] Fuzzy hit: {matched_text!r} → {original!r}"
                    )
                    del unresolved[surrogate]

        # ── Pass 4: Guarded single-token fallback (last resort) ───────────────
        # "First name only" echoes: equal word counts, token ≥3 chars,
        # capitalized, and not a substring of any other shadow key/value.
        for surrogate, original in unresolved.items():
            surrogate_words = surrogate.split()
            original_words = original.split()
            if len(surrogate_words) <= 1 or len(surrogate_words) != len(original_words):
                continue
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

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy span finder (anchored)
# ─────────────────────────────────────────────────────────────────────────────

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
