# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
settings_manager.py — Persistent user settings for SurrogateShield.

Settings are stored in <home>/settings.json ($SURROGATESHIELD_HOME, default
~/.surrogateshield; mode 0600, atomic write) and survive across sessions. Defaults are applied for any missing key.
"""

from __future__ import annotations

import json
from surrogateshield.core.storage.shadow_map import home, write_private

_SETTINGS_DIR = home()
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS: dict = {
    "llm_provider":        "claude",
    "detailed_view":       True,
    "presidio_comparison": False,
}


def load_settings() -> dict:
    """Return current settings merged with defaults."""
    if _SETTINGS_FILE.exists():
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_SETTINGS, **data}
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Persist settings to disk."""
    write_private(_SETTINGS_FILE, json.dumps(settings, indent=2).encode("utf-8"))
