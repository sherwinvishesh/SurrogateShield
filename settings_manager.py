"""
settings_manager.py — Persistent user settings for SurrogateShield.

Settings are stored in ~/.surrogateshield/settings.json and survive
across sessions. Defaults are applied for any missing key.
"""

from __future__ import annotations

import json
from pathlib import Path

_SETTINGS_DIR = Path.home() / ".surrogateshield"
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
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
