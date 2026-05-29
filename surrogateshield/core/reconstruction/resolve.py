"""
reconstruction/resolve.py — ResolvePass

Post-response surrogate-to-original reconstruction.

Three passes in sequence:
    1. Exact string replacement — fast, handles the majority of cases.
    2. Component matching — for multi-word surrogates unresolved after Pass 1.
       Scoped to UNRESOLVED surrogates only to prevent component words of
       already-resolved surrogates from corrupting unrelated text.
    3. Fuzzy match — rapidfuzz.fuzz.partial_ratio with configurable threshold.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)


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
            fuzzy_threshold: Minimum rapidfuzz partial_ratio score (0–100).

        Returns:
            Response string with surrogates replaced by original values.
        """
        if not shadow_map:
            return response_text

        result = response_text
        unresolved: Dict[str, str] = {}

        # ── Pass 1: Exact string replacement ──────────────────────────────────
        for surrogate, original in shadow_map.items():
            if surrogate in result:
                result = result.replace(surrogate, original)
                logger.debug(f"[ResolvePass] Exact hit: {surrogate!r} → {original!r}")
            else:
                unresolved[surrogate] = original
                logger.debug(f"[ResolvePass] Exact miss: {surrogate!r}")

        if not unresolved:
            return result

        # ── Pass 2: Component matching (UNRESOLVED surrogates only) ───────────
        component_resolved: Set[str] = set()

        for surrogate, original in list(unresolved.items()):
            surrogate_words = surrogate.split()
            original_words  = original.split()

            if len(surrogate_words) <= 1:
                continue

            for s_word, o_word in zip(surrogate_words, original_words):
                pattern = re.compile(r"\b" + re.escape(s_word) + r"\b")
                if pattern.search(result):
                    result = pattern.sub(o_word, result)
                    logger.debug(
                        f"[ResolvePass] Component hit: {s_word!r} → {o_word!r} "
                        f"(surrogate: {surrogate!r})"
                    )
                    component_resolved.add(surrogate)

        for surrogate in component_resolved:
            del unresolved[surrogate]

        if not unresolved:
            return result

        # ── Pass 3: Fuzzy matching ─────────────────────────────────────────────
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("[ResolvePass] rapidfuzz not installed — skipping fuzzy pass")
            return result

        for surrogate, original in unresolved.items():
            best_score, best_start, best_end = _find_fuzzy_span(result, surrogate)

            if best_score >= fuzzy_threshold and best_start >= 0:
                matched_text = result[best_start:best_end]
                result = result[:best_start] + original + result[best_end:]
                logger.debug(
                    f"[ResolvePass] Fuzzy hit (score={best_score:.1f}): "
                    f"{matched_text!r} → {original!r}"
                )

        return result


def _find_fuzzy_span(
    text: str,
    query: str,
    window_multiplier: float = 2.0,
) -> Tuple[int, int, int]:
    """
    Slide a window over *text* to find the best fuzzy match for *query*.

    Returns:
        Tuple of (best_score, best_start, best_end).
        best_start = -1 if nothing found.
    """
    from rapidfuzz import fuzz

    if not query or not text:
        return 0, -1, -1

    query_len = len(query)
    window_size = max(query_len, int(query_len * window_multiplier))
    best_score = 0
    best_start = -1
    best_end = -1

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
