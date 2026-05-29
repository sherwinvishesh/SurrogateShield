"""
surrogateshield/_state.py — Module-level singletons

Holds cfg (Config) and session (Session) as module-level singletons
so all public API calls share the same state within a Python process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class _Config:
    """Holds all library-wide configuration values."""
    detailed_view: bool = True
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


class _Session:
    """Holds per-session state: session id, shadow map, and mimic generator."""

    def __init__(self) -> None:
        self.id: str = str(uuid.uuid4())
        self._shadow_map = None
        self._mimic = None

    def reset(self) -> None:
        """Clear all session state and generate a new session id."""
        if self._shadow_map is not None:
            self._shadow_map.flush()
        self._shadow_map = None
        self._mimic = None
        self.id = str(uuid.uuid4())

    def get_mimic(self):
        """Return the MimicGen for this session, creating it if needed."""
        if self._mimic is None:
            from .core.generation.mimic import MimicGen
            self._mimic = MimicGen()
        return self._mimic

    def get_shadow_map(self):
        """Return the ShadowMap for this session, creating it if needed."""
        if self._shadow_map is None:
            from .core.storage.shadow_map import ShadowMap
            storage_dir = None if cfg.pii_mem == "temp" else cfg.pii_mem
            self._shadow_map = ShadowMap(self.id, storage_dir=storage_dir)
        return self._shadow_map


# Module-level singletons
cfg = _Config()
session = _Session()
