"""
surrogateshield/_state.py — library settings.

``cfg`` holds the settings that new :class:`surrogateshield.Session` objects
start from. It holds no PII; per-conversation state is in ``session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Settings of a session. ``detailed_view`` prints original values to
    stdout, so it is off by default (audit I10)."""
    detailed_view: bool = False
    pii_mem: str = "temp"
    pii_off: List[str] = field(default_factory=list)
    service: bool = True
    spacy_model: str = "en_core_web_lg"
    context_guard_enabled: bool = True
    entity_trace_high_threshold: float = 0.85
    entity_trace_low_threshold: float = 0.60
    context_guard_threshold: float = 0.70
    entity_trace_fallback_threshold: float = 0.65
    fuzzy_threshold: int = 85
    # ── Address handling (v2) ────────────────────────────────────────────
    # "shift"   → house number ±address_shift_range, everything else kept
    # "replace" → structure-preserving fake address
    # "auto"    → shift for service queries, replace otherwise
    address_mode: str = "auto"
    address_shift_range: int = 1
    # Opt-in Nominatim existence check (network!). Never on the hot path
    # unless explicitly enabled.
    verify_addresses: bool = False
    # ── ContextGuard model (v2: configurable, was hard-coded) ────────────
    context_guard_model: str = "dslim/distilbert-NER"
    context_guard_device: int = -1


# The settings new sessions start from (surrogateshield.config() edits it).
# PII state lives only in Session objects (audit I9).
cfg = Config()
