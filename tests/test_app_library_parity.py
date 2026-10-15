"""Pre-publish audit (Pass 6, item 5): the app's ``detection/``,
``generation/``, ``storage/`` and ``reconstruction/`` packages are aliases of
the library. The proof that nothing drifted: the app's ``Pipeline`` and the
library's ``Session`` mask the same twenty messages (the preset fixture of
``python-library/tests/test_detection_contract.py``) to the same text with
the same replacements, from the same seed, and restore the same answer.

Real detection models (the fast suite loads them elsewhere too); the
provider is a stub that echoes the sanitised text, so the restored answer
exercises the resolver on exactly the surrogates that were sent.
"""
from __future__ import annotations

import ast
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "python-library" / "tests" / "test_detection_contract.py"


def _messages() -> list:
    """The twenty-message fixture, read from the contract test without
    importing it (its module level reads the READMEs)."""
    tree = ast.parse(FIXTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MESSAGES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("MESSAGES not found in the contract test")


MESSAGES = _messages()
SEED = 20261015


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """A private home, no device key file, the app's settings as installed
    and a provider that echoes the sanitised text it was sent."""
    import config
    import surrogateshield.core.storage.shadow_map as store
    from chatbot import providers

    monkeypatch.setenv("SURROGATESHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SURROGATESHIELD_DETECTION", raising=False)
    monkeypatch.delenv("SURROGATESHIELD_DETECTION_CONFIG", raising=False)
    monkeypatch.setattr(store, "_secret_cache", {})
    monkeypatch.setattr(config, "SHADOWMAP_DIR", None)
    monkeypatch.setattr(config, "DEVICE_KEY_PATH", None)
    monkeypatch.setattr("chatbot.chat.load_settings", lambda: {"llm_provider": "claude"})
    monkeypatch.setattr("settings_manager.load_settings",
                        lambda: {"llm_provider": "claude", "detailed_view": False})
    sent = []

    def echo(payload, system):
        sent.append(payload[-1]["content"])
        return payload[-1]["content"]

    monkeypatch.setattr(providers, "build", lambda p: echo)
    return sent


def _app_turn(text: str, sent: list):
    import pipeline
    from chatbot.chat import ClaudeChat
    from util import Conversation

    chat = ClaudeChat(Conversation(id=uuid.uuid4().hex))
    p = pipeline.Pipeline(chat=chat, rag=None, seed=SEED)
    restored, confirmed, replacements = p.process_turn(text, interactive=False)
    return sent[-1], dict(replacements), restored


def _library_turn(text: str):
    from surrogateshield.session import Session

    with Session(seed=SEED) as s:
        result = s.mask_result(text)
        restored = s.unmask(result.text)
    return result.text, dict(result.replacements), restored


@pytest.mark.heavy      # the tagger's weights: cached locally, never on CI (HF offline)
@pytest.mark.parametrize("text", MESSAGES, ids=[f"msg{i:02d}" for i in range(len(MESSAGES))])
def test_app_and_library_mask_identically(app_env, text):
    app_text, app_map, app_restored = _app_turn(text, app_env)
    lib_text, lib_map, lib_restored = _library_turn(text)
    assert app_text == lib_text
    assert app_map == lib_map
    assert app_restored == lib_restored


def test_app_cascade_and_library_cascade_are_one_function():
    """The app's wrapper calls the library cascade with config.py's values;
    a value the library's balanced config already has is not a second copy
    that can drift (the identical-output test above is the check that it
    hasn't)."""
    import config
    from detection import logic
    from surrogateshield.core.detection import pipeline as lib

    assert logic._impl is lib
    assert logic.deduplicate is lib.deduplicate
    assert config.ADDRESS_MODE == lib.detection_config().address_mode
    assert config.ADDRESS_SHIFT_RANGE == lib.detection_config().address_shift_range
    assert config.SERVICE_QUERY_DETECTION_ENABLED == lib.detection_config().service_queries
