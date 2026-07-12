"""
SurrogateShield — Privacy-preserving PII proxy for LLMs.

Intercepts text before it reaches any LLM, replaces all PII with realistic
fake surrogates, and restores the real values in the LLM response.

Public API
──────────
    import surrogateshield as shield

    shield.config(pii_off=["phone", "location"])
    sanitized = shield.mask(user_text)
    response  = llm.chat(sanitized)
    restored  = shield.unmask(response)
    shield.flush()
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ._state import cfg, session
from . import _display, _response_parser
from .core.detection import address_parser as _address_parser
from .core.detection import pipeline as _pipeline
from .core.detection import service_query as _service_query
from .core.entities import apply_entity_surrogates as _apply_entity_surrogates
from .core.reconstruction.resolve import ResolvePass as _ResolvePass

__version__ = "2.0.0"
__all__ = ["config", "scan", "pii_finder", "mask", "unmask", "flush"]


# ─────────────────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────────────────

_VALID_ADDRESS_MODES = ("shift", "replace", "auto")

# Concrete entity types plus the aliases accepted by pii_off.
_VALID_PII_OFF = {
    "email", "ssn", "phone_us", "phone_uk", "phone_intl", "address",
    "person", "credit_card", "dob", "ip_address", "zip_us", "postcode_uk",
    "api_key", "crypto", "us_bank_number", "us_driver_license",
    "gpe", "loc", "org", "fac", "gender_indicator", "implicit_location",
    # aliases (resolved in the detection pipeline)
    "phone", "postal_code", "zip", "postcode", "name", "names",
    "location", "facility", "bank", "license",
}


def _validate_config(**kwargs) -> None:
    """Raise ValueError with an actionable message on any invalid setting."""
    mode = kwargs["address_mode"]
    if mode not in _VALID_ADDRESS_MODES:
        raise ValueError(
            f"address_mode must be one of {_VALID_ADDRESS_MODES}, got {mode!r}"
        )

    shift_range = kwargs["address_shift_range"]
    if not isinstance(shift_range, int) or isinstance(shift_range, bool) or shift_range < 1:
        raise ValueError(
            f"address_shift_range must be an integer >= 1, got {shift_range!r}"
        )

    fuzzy = kwargs["fuzzy_threshold"]
    if not isinstance(fuzzy, (int, float)) or isinstance(fuzzy, bool) or not (0 <= fuzzy <= 100):
        raise ValueError(
            f"fuzzy_threshold must be a number in [0, 100], got {fuzzy!r}"
        )

    for name in (
        "entity_trace_high_threshold",
        "entity_trace_low_threshold",
        "context_guard_threshold",
        "entity_trace_fallback_threshold",
    ):
        value = kwargs[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be a number in [0.0, 1.0], got {value!r}")

    for name in ("spacy_model", "context_guard_model"):
        value = kwargs[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")

    if not isinstance(kwargs["context_guard_device"], int) or isinstance(
        kwargs["context_guard_device"], bool
    ):
        raise ValueError(
            f"context_guard_device must be an integer (-1 = CPU, >=0 = GPU id), "
            f"got {kwargs['context_guard_device']!r}"
        )

    for item in kwargs["pii_off"]:
        if not isinstance(item, str) or item.lower() not in _VALID_PII_OFF:
            raise ValueError(
                f"Unknown pii_off entry {item!r}. Valid entries: "
                f"{', '.join(sorted(_VALID_PII_OFF))}"
            )

    pii_mem = kwargs["pii_mem"]
    if pii_mem != "temp" and not os.path.isdir(pii_mem):
        raise ValueError(
            f"pii_mem path does not exist or is not a directory: {pii_mem!r}"
        )


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
    address_mode: str = "shift",
    address_shift_range: int = 1,
    verify_addresses: bool = False,
    context_guard_model: str = "dslim/distilbert-NER",
    context_guard_device: int = -1,
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
        service:                        Enable service-query detection (suppresses
                                        standalone city/state replacement for map
                                        queries; also drives address_mode="auto").
        spacy_model:                    spaCy model name for named entity recognition.
        context_guard_enabled:          Enable the HuggingFace NER second-pass.
        entity_trace_high_threshold:    spaCy score ≥ this → confirmed entity.
        entity_trace_low_threshold:     spaCy score ≥ this → borderline entity.
        context_guard_threshold:        ContextGuard score ≥ this → confirmed.
        entity_trace_fallback_threshold: Promotion threshold when ContextGuard is off.
        fuzzy_threshold:                rapidfuzz partial_ratio threshold for unmask().
        address_mode:                   How detected addresses are surrogated:
                                        "shift"   — house number shifted by up to
                                                    ±address_shift_range; street, city,
                                                    state, ZIP, and formatting preserved
                                                    byte-for-byte (default);
                                        "replace" — structure-preserving fake address
                                                    (every component faked, same shape);
                                        "auto"    — shift for service queries,
                                                    replace for everything else.
        address_shift_range:            Max house-number delta for shift mode (>= 1).
        verify_addresses:               Opt-in Nominatim existence check for detected
                                        addresses. Makes a NETWORK call per address —
                                        off by default.
        context_guard_model:            HuggingFace model for ContextGuard.
        context_guard_device:           Device for ContextGuard (-1 = CPU, >= 0 = GPU id).

    Raises:
        ValueError: On any invalid setting (unknown address_mode, threshold out
                    of range, unknown pii_off entry, bad pii_mem path, …).
    """
    if pii_off is None:
        pii_off = []

    _validate_config(
        pii_mem=pii_mem,
        pii_off=list(pii_off),
        spacy_model=spacy_model,
        entity_trace_high_threshold=entity_trace_high_threshold,
        entity_trace_low_threshold=entity_trace_low_threshold,
        context_guard_threshold=context_guard_threshold,
        entity_trace_fallback_threshold=entity_trace_fallback_threshold,
        fuzzy_threshold=fuzzy_threshold,
        address_mode=address_mode,
        address_shift_range=address_shift_range,
        context_guard_model=context_guard_model,
        context_guard_device=context_guard_device,
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
    cfg.address_mode = address_mode
    cfg.address_shift_range = address_shift_range
    cfg.verify_addresses = verify_addresses
    cfg.context_guard_model = context_guard_model
    cfg.context_guard_device = context_guard_device


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
        context_guard_model=cfg.context_guard_model,
        context_guard_device=cfg.context_guard_device,
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
    # Service-query detection: suppress standalone location entities and
    # resolve the effective address mode for this message.
    is_svc = cfg.service and _service_query.is_service_query(text)
    if cfg.address_mode == "auto":
        address_mode = "shift" if is_svc else "replace"
    else:
        address_mode = cfg.address_mode

    # Run detection cascade — addresses are detected as FULL single spans by
    # the canonical parser inside PatternScan.
    confirmed, _ = _pipeline.run_cascade(
        text=text,
        skip_values=None,
        skip_location_entities=is_svc,
        pii_off=cfg.pii_off,
        spacy_model=cfg.spacy_model,
        context_guard_enabled=cfg.context_guard_enabled,
        entity_trace_high_threshold=cfg.entity_trace_high_threshold,
        entity_trace_low_threshold=cfg.entity_trace_low_threshold,
        context_guard_threshold=cfg.context_guard_threshold,
        entity_trace_fallback_threshold=cfg.entity_trace_fallback_threshold,
        context_guard_model=cfg.context_guard_model,
        context_guard_device=cfg.context_guard_device,
    )

    # Deduplicate
    confirmed = _pipeline.deduplicate(confirmed)

    if not confirmed:
        if cfg.detailed_view:
            _display.show_mask_results([], {})
        return text

    # Opt-in address existence verification (network call — off by default)
    if cfg.verify_addresses:
        for ent in confirmed:
            if ent.type == "address" and getattr(ent, "parsed", None) is not None:
                _address_parser.verify_address_exists(ent.parsed)

    shadow = session.get_shadow_map()

    # Reuse surrogates for originals already seen in this session — O(1)
    # lookups via the shadow map's forward index (no per-call copy+invert).
    surrogate_map: Dict[str, str] = {}
    new_entities = []
    for ent in confirmed:
        key = ent.text.strip()
        existing = shadow.lookup_original(key)
        if existing is not None:
            surrogate_map[key] = existing
        else:
            new_entities.append(ent)

    if new_entities:
        # A new surrogate must never equal a real value from ANY prior turn.
        new_map = session.get_mimic().generate_all(
            new_entities,
            address_mode=address_mode,
            address_shift_range=cfg.address_shift_range,
            forbidden=set(shadow.originals()),
        )
        surrogate_map.update(new_map)
        # Store only the NEW mappings (surrogate → original)
        shadow.update({v: k for k, v in new_map.items()})

    # Span-safe substitution: splice at entity offsets, then word-boundary
    # pass for repeats — a surrogate can never corrupt another entity's span.
    sanitized = _apply_entity_surrogates(text, confirmed, surrogate_map)

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
