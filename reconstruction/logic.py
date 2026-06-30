# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
reconstruction/logic.py — ResolvePass

Post-response surrogate-to-original reconstruction.

Three passes in sequence:
    1. Exact string replacement — fast, handles the majority of cases.
    2. Component matching — for multi-word surrogates that are UNRESOLVED
       after Pass 1.  Tries each component word at word boundaries.
       IMPORTANT: only processes surrogates in the `unresolved` set —
       running component matching on already-resolved surrogates caused
       silent corruption (e.g. "Ashley" from resolved "Ashley Wise" wrongly
       replacing "Ashley" in "Ashley County" in the same response).
    3. Fuzzy match — for any surrogate still unresolved, uses
       rapidfuzz.fuzz.partial_ratio with threshold 85.

Every failure is logged with its failure type for the research taxonomy:
    exact_miss  — surrogate not found via exact match
    fuzzy_miss  — surrogate not found even via all passes
    fuzzy_hit   — surrogate found only via component/fuzzy match
"""

from __future__ import annotations

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
# ResolvePass
# ─────────────────────────────────────────────

class ResolvePass:
    """
    Reconstructs original PII values in LLM responses.

    Uses three passes:
      1. Exact replacement
      2. Component matching (first/last name split for multi-word surrogates)
         — scoped to UNRESOLVED surrogates only, to prevent component words
         of already-resolved surrogates from corrupting unrelated text.
      3. Fuzzy matching via rapidfuzz

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
        unresolved: Dict[str, str] = {}

        # ── Pass 1: Exact string replacement ──────────────────────────
        for surrogate, original in shadow_map.items():
            if surrogate in result:
                result = result.replace(surrogate, original)
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

        # ── Pass 2: Component matching (UNRESOLVED surrogates only) ───
        #
        # We intentionally iterate over `unresolved` rather than `shadow_map`.
        #
        # Rationale: if "Ashley Wise" was fully resolved in Pass 1, re-running
        # component matching on it would find the word "Ashley" anywhere in the
        # response — including in unrelated tokens like "Ashley County" — and
        # replace it with the original's first name.  This is silent data
        # corruption. By limiting Pass 2 to surrogates that Pass 1 could NOT
        # find, we only handle the legitimate case where Claude used just the
        # first name of a surrogate it was given in full.
        #
        # Trade-off: the case where Claude uses BOTH the full surrogate AND just
        # the first name in the same response is not handled.  This is an
        # acceptable trade-off — the correctness gain from preventing corruption
        # outweighs the marginal resolution loss.
        component_resolved: Set[str] = set()

        for surrogate, original in list(unresolved.items()):
            surrogate_words = surrogate.split()
            original_words  = original.split()

            if len(surrogate_words) <= 1:
                continue  # single-word surrogates handled by Pass 1 / Pass 3

            for s_word, o_word in zip(surrogate_words, original_words):
                pattern = re.compile(r"\b" + re.escape(s_word) + r"\b")
                if pattern.search(result):
                    result = pattern.sub(o_word, result)
                    logger.debug(
                        f"[ResolvePass] Component hit: {s_word!r} → {o_word!r} "
                        f"(surrogate: {surrogate!r})"
                    )
                    component_resolved.add(surrogate)
                    # Reclassify the failure log entry for this surrogate
                    for f in reversed(self.failures):
                        if f.surrogate == surrogate and f.failure_type == "exact_miss":
                            f.failure_type = "fuzzy_hit"
                            break

        # Remove component-resolved surrogates from the unresolved set
        for surrogate in component_resolved:
            del unresolved[surrogate]

        if not unresolved:
            return result

        # ── Pass 3: Fuzzy matching ─────────────────────────────────────
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("[ResolvePass] rapidfuzz not installed — skipping fuzzy pass")
            return result

        final_unresolved: Dict[str, str] = {}

        for surrogate, original in unresolved.items():
            best_score, best_start, best_end = _find_fuzzy_span(result, surrogate)

            if best_score >= FUZZY_MATCH_THRESHOLD and best_start >= 0:
                matched_text = result[best_start:best_end]
                result = result[:best_start] + original + result[best_end:]
                logger.debug(
                    f"[ResolvePass] Fuzzy hit (score={best_score:.1f}): "
                    f"{matched_text!r} → {original!r}"
                )
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
# Fuzzy span finder
# ─────────────────────────────────────────────

def _find_fuzzy_span(
    text: str,
    query: str,
    window_multiplier: float = 2.0,
) -> Tuple[int, int, int]:
    """
    Slide a window over *text* to find the best fuzzy match for *query*.

    The step size is query_len // 8 (previously // 4).  The finer step
    ensures matches that start at non-multiple-of-4 positions are found:
    e.g. for a 16-character surrogate the old step=4 would skip positions
    2, 6, 10, 14; the new step=2 catches all of them.

    Args:
        text:              Text to search within.
        query:             Surrogate string to find.
        window_multiplier: Window size multiplier relative to query length.

    Returns:
        Tuple of (best_score, best_start, best_end).
        best_start = -1 if nothing found above threshold.
    """
    from rapidfuzz import fuzz

    if not query or not text:
        return 0, -1, -1

    query_len = len(query)
    window_size = max(query_len, int(query_len * window_multiplier))
    best_score = 0
    best_start = -1
    best_end = -1

    # Finer step (//8 instead of //4) so matches at any starting position
    # within the text are reliably found.
    step = max(1, query_len // 8)

    for start in range(0, max(1, len(text) - query_len + 1), step):
        end = min(start + window_size, len(text))
        window = text[start:end]
        score = fuzz.partial_ratio(query.lower(), window.lower())
        if score > best_score:
            best_score = score
            best_start = start
            best_end = start + query_len

    return best_score, best_start, best_end