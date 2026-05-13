"""
reconstruction/logic.py — ResolvePass

Post-response surrogate-to-original reconstruction.

Two passes in sequence:
    1. Exact string replacement — fast, handles the majority of cases.
    2. Fuzzy match — for surrogates not found exactly, uses
       rapidfuzz.fuzz.partial_ratio with threshold 85 to catch
       paraphrased or reformatted references.

Every failure is logged with its failure type for the research taxonomy:
    exact_miss  — surrogate not found via exact match
    fuzzy_miss  — surrogate not found even via fuzzy match
    fuzzy_hit   — surrogate found only via fuzzy match (logged for stats)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

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

    Uses exact replacement first, then falls back to fuzzy matching
    for surrogates not found verbatim in the response.

    Attributes:
        failures: Log of ResolutionFailure events from this session,
                  accumulated across all calls to resolve().
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
        unresolved: Dict[str, str] = {}  # surrogates not found by exact pass

        # ── Pass 1: Exact string replacement ──────────────────────
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

        # ── Pass 2: Fuzzy matching ─────────────────────────────────
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning(
                "[ResolvePass] rapidfuzz not installed — skipping fuzzy pass"
            )
            return result

        for surrogate, original in unresolved.items():
            best_score, best_match_start, best_match_end = _find_fuzzy_span(
                result, surrogate
            )

            if best_score >= FUZZY_MATCH_THRESHOLD and best_match_start >= 0:
                # Replace the matched span with the original value
                matched_text = result[best_match_start:best_match_end]
                result = result[:best_match_start] + original + result[best_match_end:]
                logger.debug(
                    f"[ResolvePass] Fuzzy hit (score={best_score}): "
                    f"{matched_text!r} → {original!r}"
                )
                # Update failure log: reclassify as fuzzy_hit
                for f in reversed(self.failures):
                    if f.surrogate == surrogate and f.failure_type == "exact_miss":
                        f.failure_type = "fuzzy_hit"
                        break
            else:
                logger.warning(
                    f"[ResolvePass] Fuzzy miss (best score={best_score}): "
                    f"Could not resolve surrogate {surrogate!r}"
                )
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
            Dict with keys 'exact_miss', 'fuzzy_miss', 'fuzzy_hit'
            and their respective counts.
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

    Uses a sliding window of size len(query) * window_multiplier chars
    to account for reordering and paraphrasing.

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

    step = max(1, query_len // 4)
    for start in range(0, max(1, len(text) - query_len + 1), step):
        end = min(start + window_size, len(text))
        window = text[start:end]
        score = fuzz.partial_ratio(query.lower(), window.lower())
        if score > best_score:
            best_score = score
            best_start = start
            best_end = start + query_len  # approximate end

    return best_score, best_start, best_end
