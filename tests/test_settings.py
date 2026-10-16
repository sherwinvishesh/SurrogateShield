"""Audit I27 / F3 — user settings are validated, written privately, and the
app reads the models it actually uses. Model-free, no network."""

import json
import logging
import stat

import pytest

import settings_manager as sm


@pytest.fixture
def store(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    monkeypatch.setattr(sm, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(sm, "_SETTINGS_FILE", f)
    return f


def test_I27_defaults_without_file(store):
    assert sm.load_settings() == sm.DEFAULT_SETTINGS


def test_I27_invalid_values_fall_back_with_warning(store, caplog):
    store.write_text(json.dumps({"llm_provider": "skynet", "detailed_view": "yes",
                                 "presidio_comparison": True, "extra": 1}))
    with caplog.at_level(logging.WARNING, logger="settings_manager"):
        s = sm.load_settings()
    assert s["llm_provider"] == "claude" and s["detailed_view"] is True
    assert s["presidio_comparison"] is True and s["extra"] == 1
    assert "llm_provider" in caplog.text and "detailed_view" in caplog.text


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", b"\xff\xfe"])
def test_I27_corrupt_file_moved_aside_not_overwritten(store, content, caplog):
    if isinstance(content, bytes):
        store.write_bytes(content)
    else:
        store.write_text(content)
    with caplog.at_level(logging.WARNING, logger="settings_manager"):
        assert sm.load_settings() == sm.DEFAULT_SETTINGS
    aside = list(store.parent.glob("settings.json.corrupt-*"))
    assert len(aside) == 1 and not store.exists()
    assert "moved to" in caplog.text


def test_I27_save_is_private_and_rejects_invalid(store):
    sm.save_settings({"llm_provider": "gemini"})
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert sm.load_settings()["llm_provider"] == "gemini"
    with pytest.raises(ValueError):
        sm.save_settings({"llm_provider": "skynet"})
    with pytest.raises(ValueError):
        sm.save_settings({"presidio_comparison": "on"})
    assert sm.load_settings()["llm_provider"] == "gemini"


def test_I27_providers_match_app_menu():
    import main
    assert tuple(slug for slug, _, _ in main._PROVIDERS) == sm.PROVIDERS


def test_I27_provider_test_uses_configured_models():
    import inspect
    import main
    src = inspect.getsource(main._test_provider)
    for literal in ("claude-haiku", "gemini-1.5", "gpt-4o", "llama3.2", "override=True"):
        assert literal not in src
    import config
    menu = {slug: desc for slug, _, desc in main._PROVIDERS}
    assert config.CLAUDE_MODEL in menu["claude"]
    assert config.GEMINI_MODEL in menu["gemini"]
    assert config.OPENAI_MODEL in menu["chatgpt"]


def test_I27_app_reads_no_contradicting_fallbacks():
    import inspect
    import main
    src = inspect.getsource(main)
    assert '"presidio_comparison", True' not in src
    assert '"detailed_view", False' not in src
