"""
surrogateshield/_state.py — library settings.

``cfg`` holds the settings that new :class:`surrogateshield.Session` objects
start from. It holds no PII; per-conversation state is in ``session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Config:
    """Settings of a session. ``detailed_view`` prints original values to
    stdout, so it is off by default (audit I10)."""
    detailed_view: bool = False
    pii_mem: str = "temp"
    pii_off: List[str] = field(default_factory=list)
    service: bool = True
    spacy_model: str = "en_core_web_lg"
    context_guard_enabled: bool = False       # off in balanced: the tagger reads names (V3)
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
    # ── ContextGuard model (v2: configurable, was hard-coded) ────────────
    context_guard_model: str = "dslim/distilbert-NER"
    context_guard_device: int = -1
    # ── Detection config (V3 §3.6) ───────────────────────────────────────
    # A DetectionConfig (stages, models, thresholds, type routing, per-type
    # actions, gate). None: the preset or file the environment names
    # (SURROGATESHIELD_PRESET / SURROGATESHIELD_DETECTION_CONFIG), else
    # "balanced". The flat settings above that differ from their defaults
    # apply on top of it (see effective_detection).
    detection: Any = None


_FLAT_DETECTION = ("spacy_model", "context_guard_enabled", "entity_trace_high_threshold",
                   "entity_trace_low_threshold", "context_guard_threshold",
                   "entity_trace_fallback_threshold", "context_guard_model", "context_guard_device",
                   "address_mode", "address_shift_range", "service")


def effective_detection(c: "Config"):
    """The DetectionConfig a session with settings *c* runs: ``c.detection``
    (else the environment's, else ``balanced``) with every flat setting
    that differs from its default applied on top."""
    from .core.detection import config as dconfig
    base = c.detection
    if isinstance(base, str):
        base = dconfig.preset(base)
    elif isinstance(base, dict):
        base = dconfig.from_partial(base)
    if base is None:
        base = dconfig.from_env() if dconfig.env_selected() else dconfig.preset("balanced")
    defaults = Config()
    changed = {k: getattr(c, k) for k in _FLAT_DETECTION if getattr(c, k) != getattr(defaults, k)}
    return dconfig.from_settings(base, **changed) if changed else base


# The settings new sessions start from (surrogateshield.config() edits it).
# PII state lives only in Session objects (audit I9).
cfg = Config()
