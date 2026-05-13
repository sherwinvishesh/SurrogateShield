"""
detection/context_guard.py — ContextGuard

Local SLM-based detection of implicit and quasi-identifier PII using
Ollama (phi3:mini). This module contains ONLY the SLM logic.

ContextGuard is called ONLY with:
  1. Borderline entities from EntityTrace (to verify or upgrade)
  2. The remaining_text after PatternScan and EntityTrace have run

It does NOT process the full original message.

Graceful degradation: if Ollama is unavailable, logs a warning and
returns empty lists — the pipeline continues without crashing.

Returns structured JSON parsed into DetectedEntity objects.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from config import CONTEXT_GUARD_MODEL, CONTEXT_GUARD_CONFIDENCE_THRESHOLD
from util import DetectedEntity, get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# System prompt for the SLM
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a privacy detection assistant. Your task is to identify
personally identifiable information (PII) and quasi-identifiers in text fragments.

You will receive either:
1. Text fragments that previous detectors were unsure about (borderline cases)
2. Remaining text after structured PII has already been removed

Identify:
- Implicit PII: information that could identify a person even without a name
  (e.g. "the only surgeon at Springfield General", "my ZIP code is near the stadium")
- Quasi-identifiers: combinations of fields that together enable re-identification
  (e.g. age + employer + city)
- Ambiguous named entities that pattern matching missed

Respond ONLY with a valid JSON object. No explanation, no preamble, no markdown.
Use exactly this schema:
{
  "detected": [
    {
      "text": "exact text span from input",
      "type": "implicit_location|implicit_person|quasi_identifier|PERSON|GPE|LOC|ORG",
      "confidence": 0.0,
      "reason": "brief reason"
    }
  ]
}
If nothing is found, respond with: {"detected": []}"""


def _build_user_prompt(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
) -> str:
    """
    Construct the user-facing prompt for the SLM.

    Args:
        remaining_text:     Text after PatternScan and EntityTrace processing.
        borderline_entities: Entities EntityTrace was uncertain about.

    Returns:
        Formatted prompt string.
    """
    parts = []

    if borderline_entities:
        parts.append("BORDERLINE ENTITIES TO VERIFY:")
        for ent in borderline_entities:
            parts.append(
                f"  - text={ent.text!r}, type={ent.type}, "
                f"score={ent.score:.2f}"
            )

    if remaining_text and remaining_text.strip():
        clean = remaining_text.replace("█", " ").strip()
        if clean:
            parts.append("\nTEXT FRAGMENT TO ANALYSE:")
            parts.append(clean)

    return "\n".join(parts) if parts else "No input provided."


# ─────────────────────────────────────────────
# Main guard function
# ─────────────────────────────────────────────

def guard(
    remaining_text: str,
    borderline_entities: List[DetectedEntity],
) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
    """
    Run the SLM context analysis on remaining text and borderline entities.

    Calls Ollama with phi3:mini. On any failure (connection error, JSON
    parse error, model unavailable) logs a warning and returns empty
    lists — the pipeline continues gracefully.

    Args:
        remaining_text:      Text not covered by PatternScan / EntityTrace.
        borderline_entities: Entities EntityTrace flagged as uncertain.

    Returns:
        Tuple of (confirmed_entities, needs_user_confirmation_entities).
        Both lists contain DetectedEntity objects with source='slm'.
    """
    confirmed: List[DetectedEntity] = []
    uncertain: List[DetectedEntity] = []

    # Nothing to send → skip
    has_text = remaining_text and remaining_text.replace("█", "").strip()
    if not borderline_entities and not has_text:
        logger.debug("[ContextGuard] Nothing to analyse — skipping")
        return confirmed, uncertain

    # Build prompt
    user_prompt = _build_user_prompt(remaining_text, borderline_entities)

    try:
        import ollama
        response = ollama.chat(
            model=CONTEXT_GUARD_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content: str = response["message"]["content"]
    except ImportError:
        logger.warning(
            "[ContextGuard] ollama package not installed — skipping SLM stage"
        )
        return confirmed, uncertain
    except Exception as exc:
        logger.warning(
            f"[ContextGuard] Ollama unavailable or call failed: {exc}. "
            "Continuing without SLM detection."
        )
        return confirmed, uncertain

    # Parse JSON response
    try:
        # Strip any accidental markdown fences
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        parsed = json.loads(cleaned)
        detections = parsed.get("detected", [])
    except json.JSONDecodeError as exc:
        logger.warning(
            f"[ContextGuard] Failed to parse SLM JSON response: {exc}. "
            f"Raw content: {raw_content[:200]!r}"
        )
        return confirmed, uncertain

    # Convert parsed detections to DetectedEntity objects
    for item in detections:
        try:
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("type", "implicit_unknown")).strip()
            confidence = float(item.get("confidence", 0.0))
            # reason = item.get("reason", "")  # available for logging/taxonomy

            if not text:
                continue

            entity = DetectedEntity(
                text=text,
                start=0,   # SLM doesn't return character offsets
                end=len(text),
                type=entity_type,
                score=confidence,
                source="slm",
            )

            if confidence >= CONTEXT_GUARD_CONFIDENCE_THRESHOLD:
                confirmed.append(entity)
                logger.debug(
                    f"[ContextGuard] Confirmed: {text!r} ({entity_type}, {confidence:.2f})"
                )
            else:
                uncertain.append(entity)
                logger.debug(
                    f"[ContextGuard] Uncertain: {text!r} ({entity_type}, {confidence:.2f})"
                )

        except (KeyError, ValueError, TypeError) as exc:
            logger.debug(f"[ContextGuard] Skipping malformed detection item: {exc}")
            continue

    logger.info(
        f"[ContextGuard] confirmed={len(confirmed)}, uncertain={len(uncertain)}"
    )
    return confirmed, uncertain
