# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
settings_manager.py — Persistent user settings for SurrogateShield.

Settings are stored in <home>/settings.json ($SURROGATESHIELD_HOME, default
~/.surrogateshield; mode 0600, atomic write) and survive across sessions.

Every value is validated on load (audit I27): a missing key or an invalid
value falls back to its default with a warning, and a file that is not valid
JSON is moved aside to ``settings.json.corrupt-<ts>`` (never silently
overwritten) before the defaults are used.
"""

from __future__ import annotations

import json
import logging
import os
import time

from surrogateshield.core.storage.shadow_map import home, write_private

logger = logging.getLogger(__name__)

_SETTINGS_DIR = home()
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"

PROVIDERS = ("claude", "gemini", "chatgpt", "local")

DEFAULT_SETTINGS: dict = {
    "llm_provider":        "claude",
    "detailed_view":       True,
    "presidio_comparison": False,
}

_VALID = {
    "llm_provider":        lambda v: v in PROVIDERS,
    "detailed_view":       lambda v: isinstance(v, bool),
    "presidio_comparison": lambda v: isinstance(v, bool),
}


def validate(settings: dict) -> dict:
    """Return *settings* with every known key present and valid. Invalid
    values are replaced by their default (logged); unknown keys are kept."""
    out = {**DEFAULT_SETTINGS, **settings}
    for key, ok in _VALID.items():
        if not ok(out[key]):
            logger.warning("settings.json: invalid %s=%r, using %r",
                           key, out[key], DEFAULT_SETTINGS[key])
            out[key] = DEFAULT_SETTINGS[key]
    return out


def load_settings() -> dict:
    """Return the validated settings (defaults when there is no file)."""
    if not _SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (ValueError, UnicodeDecodeError) as exc:
        aside = _SETTINGS_FILE.with_name(f"{_SETTINGS_FILE.name}.corrupt-{int(time.time())}")
        os.replace(_SETTINGS_FILE, aside)
        logger.warning("settings.json could not be read (%s); moved to %s, using defaults",
                       exc, aside.name)
        return DEFAULT_SETTINGS.copy()
    return validate(data)


def save_settings(settings: dict) -> None:
    """Validate and persist *settings* (0600, atomic). Raises ValueError on an
    invalid value instead of writing it."""
    for key, ok in _VALID.items():
        if key in settings and not ok(settings[key]):
            raise ValueError(f"invalid setting {key}={settings[key]!r}")
    write_private(_SETTINGS_FILE, json.dumps(validate(settings), indent=2).encode("utf-8"))
