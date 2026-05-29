"""
SurrogateShield — Privacy-preserving PII proxy for LLMs.

Intercepts text before it reaches any LLM, replaces all PII with realistic
fake surrogates, and restores the real values in the LLM response.

Public API
──────────
    import SurrogateShield as ss

    ss.config(pii_off=["phone", "location"])
    sanitized = ss.mask(user_text)
    response  = llm.chat(sanitized)
    restored  = ss.unmask(response)
    ss.flush()
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ._state import cfg, session
from . import _display, _response_parser
from .core.detection import pipeline as _pipeline
from .core.detection import service_query as _service_query
from .core.reconstruction.resolve import ResolvePass as _ResolvePass

__version__ = "0.1.0"
__all__ = ["config", "scan", "pii_finder", "mask", "unmask", "flush"]


# ─────────────────────────────────────────────────────────────────────────────
# config()
# ─────────────────────────────────────────────────────────────────────────────

def config(
    detailed_view: bool = True,
    pii_mem: str = "temp",
    pii_off=None,
    service: bool = True,
    spacy_model: str = "en_core_web_lg",
    context_guard_enabled: bool = True,
    entity_trace_high_threshold: float = 0.85,
    entity_trace_low_threshold: float = 0.60,
    context_guard_threshold: float = 0.70,
    entity_trace_fallback_threshold: float = 0.65,
    fuzzy_threshold: int = 85,
) -> None:
    """
    Configure SurrogateShield.

    Args:
        detailed_view:                  Print detection/masking tables to stdout.
        pii_mem:                        "temp" for in-memory session (default), or
                                        a directory path for encrypted persistent storage.
        pii_off:                        PII types to detect but NOT replace.
                                        Accepts type names or aliases:
                                        "phone", "name", "location", "org", "email",
                                        "ssn", "dob", "address", "zip", "postcode",
                                        "credit_card", "ip_address", "api_key",
                                        "crypto", "bank", "license", "gender_indicator".
        service:                        Enable service-query detection (address fuzzing
                                        instead of full replacement for map queries).
        spacy_model:                    spaCy model name for named entity recognition.
        context_guard_enabled:          Enable the HuggingFace NER second-pass.
        entity_trace_high_threshold:    spaCy score ≥ this → confirmed entity.
        entity_trace_low_threshold:     spaCy score ≥ this → borderline entity.
        context_guard_threshold:        ContextGuard score ≥ this → confirmed.
        entity_trace_fallback_threshold: Promotion threshold when ContextGuard is off.
        fuzzy_threshold:                rapidfuzz partial_ratio threshold for unmask().

    Raises:
        ValueError: If pii_mem is not "temp" and the path does not exist or is
                    not a directory.
    """
    if pii_off is None:
        pii_off = []

    if pii_mem != "temp":
        if not os.path.isdir(pii_mem):
            raise ValueError(
                f"pii_mem path does not exist or is not a directory: {pii_mem!r}"
            )

    cfg.detailed_view = detailed_view
    cfg.pii_mem = pii_mem
    cfg.pii_off = list(pii_off)
    cfg.service = service
    cfg.spacy_model = spacy_model
    cfg.context_guard_enabled = context_guard_enabled
    cfg.entity_trace_high_threshold = entity_trace_high_threshold
    cfg.entity_trace_low_threshold = entity_trace_low_threshold
    cfg.context_guard_threshold = context_guard_threshold
    cfg.entity_trace_fallback_threshold = entity_trace_fallback_threshold
    cfg.fuzzy_threshold = fuzzy_threshold


# ─────────────────────────────────────────────────────────────────────────────
# scan()  /  pii_finder
# ─────────────────────────────────────────────────────────────────────────────

def scan(text: str) -> Dict[str, str]:
    """
    Detect all PII in *text* without modifying anything.

    Runs the full detection cascade (PatternScan → EntityTrace → ContextGuard)
    and returns every detected entity regardless of pii_off settings.
    Does NOT update the session shadow map.

    Args:
        text: Any string to scan for PII.

    Returns:
        Dict mapping detected_value → pii_type_string.
        Example: {"john@example.com": "email", "John Smith": "PERSON"}
    """
    confirmed, _ = _pipeline.run_cascade(
        text=text,
        skip_values=None,
        skip_location_entities=False,
        pii_off=None,  # scan is always comprehensive
        spacy_model=cfg.spacy_model,
        context_guard_enabled=cfg.context_guard_enabled,
        entity_trace_high_threshold=cfg.entity_trace_high_threshold,
        entity_trace_low_threshold=cfg.entity_trace_low_threshold,
        context_guard_threshold=cfg.context_guard_threshold,
        entity_trace_fallback_threshold=cfg.entity_trace_fallback_threshold,
    )

    if cfg.detailed_view:
        _display.show_scan_results(confirmed, cfg.pii_off)

    return {ent.text: ent.type for ent in confirmed}


# Alias
pii_finder = scan


# ─────────────────────────────────────────────────────────────────────────────
# mask()
# ─────────────────────────────────────────────────────────────────────────────

def mask(text: str) -> str:
    """
    Replace all PII in *text* with realistic fake surrogates.

    The original→surrogate mapping is stored in the session shadow map so
    that unmask() can restore the real values from the LLM response.

    Args:
        text: The text to sanitize before sending to an LLM.

    Returns:
        Sanitized text with PII replaced by surrogates.
    """
    skip_values = None
    skip_location_entities = False

    # Service-query detection: fuzz addresses, suppress location entities
    if cfg.service and _service_query.is_service_query(text):
        text, fuzz_map = _service_query.fuzz_addresses(text, verify=True)
        skip_location_entities = True
        skip_values = set(fuzz_map.values())

    # Run detection cascade
    confirmed, _ = _pipeline.run_cascade(
        text=text,
        skip_values=skip_values,
        skip_location_entities=skip_location_entities,
        pii_off=cfg.pii_off,
        spacy_model=cfg.spacy_model,
        context_guard_enabled=cfg.context_guard_enabled,
        entity_trace_high_threshold=cfg.entity_trace_high_threshold,
        entity_trace_low_threshold=cfg.entity_trace_low_threshold,
        context_guard_threshold=cfg.context_guard_threshold,
        entity_trace_fallback_threshold=cfg.entity_trace_fallback_threshold,
    )

    # Deduplicate
    confirmed = _pipeline.deduplicate(confirmed)

    if not confirmed:
        if cfg.detailed_view:
            _display.show_mask_results([], {})
        return text

    # Generate surrogates: {original_text → surrogate_text}
    surrogate_map = session.get_mimic().generate_all(confirmed)

    # Apply substitutions (longest match first to avoid substring collisions)
    sanitized = text
    for original, surrogate in sorted(surrogate_map.items(), key=lambda x: len(x[0]), reverse=True):
        sanitized = sanitized.replace(original, surrogate)

    # Store inverted map (surrogate → original) in session shadow map
    inverted = {v: k for k, v in surrogate_map.items()}
    session.get_shadow_map().update(inverted)

    if cfg.detailed_view:
        _display.show_mask_results(confirmed, surrogate_map)

    return sanitized


# ─────────────────────────────────────────────────────────────────────────────
# unmask()
# ─────────────────────────────────────────────────────────────────────────────

def unmask(response) -> str:
    """
    Restore original PII values in the LLM *response*.

    Extracts text from any major LLM SDK response object (Anthropic, OpenAI,
    Gemini) or accepts a plain string, then replaces surrogates with the
    originals stored in the session shadow map.

    Args:
        response: An LLM SDK response object or a plain string.

    Returns:
        Response text with surrogates replaced by the original PII values.
    """
    text = _response_parser.extract_text(response)
    shadow_map = session.get_shadow_map().get_all()

    resolver = _ResolvePass()
    restored = resolver.resolve(
        response_text=text,
        shadow_map=shadow_map,
        fuzzy_threshold=cfg.fuzzy_threshold,
    )

    if cfg.detailed_view:
        # Count how many surrogates were actually replaced
        replaced_count = sum(1 for s in shadow_map if s not in restored or s not in text)
        _display.show_unmask_results(len(shadow_map))

    return restored


# ─────────────────────────────────────────────────────────────────────────────
# flush()
# ─────────────────────────────────────────────────────────────────────────────

def flush() -> None:
    """
    Clear the session: discard all surrogate mappings and reset the session id.

    Call this after a conversation ends to ensure surrogate mappings from
    one session cannot bleed into the next.
    """
    session.reset()
    if cfg.detailed_view:
        _display.show_flush()
