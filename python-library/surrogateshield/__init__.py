"""
SurrogateShield — Privacy-preserving PII proxy for LLMs.

Intercepts text before it reaches any LLM, replaces all PII with realistic
fake surrogates, and restores the real values in the LLM response.

Public API
──────────
    import surrogateshield as shield

    with shield.Session() as s:              # one per end-user conversation
        sanitized = s.mask(user_text)
        response  = llm.chat(sanitized)
        restored  = s.unmask(response)

    async with shield.Session() as s:        # asyncio: models run in a worker thread
        sanitized = await s.amask(user_text)
        restored  = await s.aunmask(await llm.achat(sanitized))

    # Module-level shortcuts use the current context's session:
    shield.config(pii_off=["phone"])
    shield.unmask(llm.chat(shield.mask(user_text)))
    shield.flush()
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Union

from ._state import Config, cfg
from .core.errors import DetectorUnavailable
from .core.storage.shadow_map import StorageError
from .session import (
    Detection,
    MaskResult,
    Session,
    current_session,
    use_session,
)

__version__ = "2.1.0"   # keep equal to pyproject.toml (tests/test_public_api.py)

__all__ = [
    "config", "scan", "pii_finder", "mask", "mask_result", "unmask", "forget", "flush",
    "Session", "Config", "Detection", "MaskResult", "current_session", "use_session",
    "DetectorUnavailable", "StorageError", "__version__",
]


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
    "iban", "vin", "mac_address", "passport", "id_number", "license_plate",
    "url", "handle", "credential", "age",
    # aliases (resolved in the detection pipeline)
    "phone", "postal_code", "zip", "postcode", "name", "names",
    "location", "facility", "bank", "license", "username", "password",
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

_UNSET = object()


def config(
    *,
    detailed_view=_UNSET,
    pii_mem=_UNSET,
    pii_off=_UNSET,
    service=_UNSET,
    spacy_model=_UNSET,
    context_guard_enabled=_UNSET,
    entity_trace_high_threshold=_UNSET,
    entity_trace_low_threshold=_UNSET,
    context_guard_threshold=_UNSET,
    entity_trace_fallback_threshold=_UNSET,
    fuzzy_threshold=_UNSET,
    address_mode=_UNSET,
    address_shift_range=_UNSET,
    verify_addresses=_UNSET,
    context_guard_model=_UNSET,
    context_guard_device=_UNSET,
) -> Config:
    """
    Change the settings that sessions use. Only the arguments you pass
    change; the rest keep their current values. Returns the settings.

    New sessions start from these settings, and the current context's
    session (if any) picks the change up immediately.

    Args:
        detailed_view:                  Print detection/masking tables to stdout.
                                        Shows ORIGINAL values — off by default.
        pii_mem:                        "temp" for in-memory session (default), or
                                        a directory path for encrypted persistent storage.
                                        Cannot change once the current session has
                                        mappings (use a new Session).
        pii_off:                        PII types to detect but NOT replace.
                                        Accepts type names or aliases:
                                        "phone", "name", "location", "org", "email",
                                        "ssn", "dob", "address", "zip", "postcode",
                                        "credit_card", "ip_address", "api_key",
                                        "crypto", "bank", "license", "gender_indicator".
                                        Reported in ``mask_result().unmasked``.
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
                                        "auto"    — shift for non-sensitive service
                                                    queries, replace for everything
                                                    else (default);
                                        "shift"   — house number shifted by up to
                                                    ±address_shift_range for EVERY
                                                    address; street, city, state, ZIP
                                                    and formatting are sent unchanged;
                                        "replace" — structure-preserving fake address
                                                    (every component faked, same shape).
        address_shift_range:            Max house-number delta for shift mode (>= 1).
        verify_addresses:               Deprecated, no effect. The Nominatim check sent
                                        each street to a third party and never changed
                                        the output (audit F3).
        context_guard_model:            HuggingFace model for ContextGuard.
        context_guard_device:           Device for ContextGuard (-1 = CPU, >= 0 = GPU id).

    Raises:
        ValueError:   On any invalid setting (unknown address_mode, threshold out
                      of range, unknown pii_off entry, bad pii_mem path, …).
        RuntimeError: pii_mem changed while the current session holds mappings.
    """
    changes = {k: v for k, v in locals().items() if v is not _UNSET}
    if changes.pop("verify_addresses", None) is not None:
        warnings.warn("config(verify_addresses=...) has no effect and will be removed "
                      "(audit F3: the Nominatim check never changed the output)",
                      DeprecationWarning, stacklevel=2)
    if "pii_off" in changes:
        changes["pii_off"] = list(changes["pii_off"] or [])
    merged = {f: getattr(cfg, f) for f in Config.__dataclass_fields__}
    merged.update(changes)
    _validate_config(**merged)

    current = current_session(create=False)
    if current is not None and "pii_mem" in changes and changes["pii_mem"] != current.config.pii_mem:
        if current.mappings:
            raise RuntimeError(
                "pii_mem cannot change while the current session holds mappings; "
                "call flush() first or create a new Session"
            )
        current.close()                   # the next call opens one with the new storage
        current = None

    for name, value in changes.items():
        setattr(cfg, name, value)
        if current is not None:
            setattr(current.config, name, value)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Module-level shortcuts — all act on current_session()
# ─────────────────────────────────────────────────────────────────────────────

def scan(text: str, *, as_dict: bool = False) -> Union[List[Detection], Dict[str, str]]:
    """
    Detect PII in *text* without modifying anything, with the same settings
    (service-query handling, pii_off) as :func:`mask`.

    Returns:
        A list of :class:`Detection` (text, type, start, end, score, source,
        masked). ``as_dict=True`` returns the old ``{text: type}`` mapping.

    Raises:
        DetectorUnavailable: a detection model is missing or failed (fail
            closed, audit I17).
    """
    detections = current_session().scan(text)
    if as_dict:
        return {d.text: d.type for d in detections}
    return detections


# Alias
pii_finder = scan


def mask(text: str) -> str:
    """
    Replace all PII in *text* with realistic fake surrogates.

    The original→surrogate mapping is stored in the current session so that
    :func:`unmask` can restore the real values from the LLM response.

    Raises:
        DetectorUnavailable: a detection model is missing or failed; the
            text is not masked (fail closed, audit I17).
        TypeError: *text* is not a str.
    """
    return current_session().mask(text)


def mask_result(text: str) -> MaskResult:
    """Like :func:`mask`, but also return the detections (including the
    ``pii_off`` ones sent verbatim) and the replacements made."""
    return current_session().mask_result(text)


def unmask(response) -> str:
    """
    Restore original PII values in the LLM *response* (a str, or an
    Anthropic / OpenAI / Gemini response object or dict).

    Raises:
        TypeError: *response* is None or has no text content.
    """
    return current_session().unmask(response)


def forget(original: str) -> int:
    """Erase every mapping for *original* from the current session."""
    return current_session().forget(original)


def flush() -> None:
    """
    End the current session: discard its mappings (and its persistent file).
    The next call in this context starts a new session.
    """
    current = current_session(create=False)
    if current is not None:
        current.close()
