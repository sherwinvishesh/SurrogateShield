"""J10 — a key not scoped to a workspace needs the anthropic-workspace-id
header on every request. ANTHROPIC_WORKSPACE_ID supplies it to the
responder, the attacker and the Phase 8 driver alike."""

import pytest

from chatbot import providers

FAKE = "test-" + "placeholder"


def test_J10_no_workspace_means_no_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert providers.claude_client_kwargs() == {"api_key": FAKE}


def test_J10_workspace_id_becomes_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", " wrkspc_example ")
    kw = providers.claude_client_kwargs()
    assert kw["default_headers"] == {"anthropic-workspace-id": "wrkspc_example"}


def test_J10_client_sends_header(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_example")
    client = anthropic.Anthropic(**providers.claude_client_kwargs(), max_retries=0)
    assert client.default_headers.get("anthropic-workspace-id") == "wrkspc_example"


def test_J10_missing_key_still_fails_closed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        providers.claude_client_kwargs()
