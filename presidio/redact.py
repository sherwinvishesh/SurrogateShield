"""
Apply [ENTITY_TYPE] redaction using Presidio detection results.
Produces the text that Presidio would send to an LLM.
"""

from __future__ import annotations
from presidio.detect import PresidioEntity


def redact(text: str, entities: list[PresidioEntity]) -> str:
    """
    Replace each detected span with [ENTITY_TYPE].

    Applies replacements from right to left (end → start) so that
    earlier offsets remain valid after each substitution.

    Example:
        "My name is John and SSN is 123-45-6789"
        → "My name is [PERSON] and SSN is [US_SSN]"
    """
    if not entities:
        return text

    # Sort descending by start so we replace from end to start
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
    result = text
    for ent in sorted_entities:
        label = f"[{ent.entity_type}]"
        result = result[: ent.start] + label + result[ent.end :]
    return result
