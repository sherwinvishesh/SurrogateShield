"""Audit F2 / I20 / I21 — provider adapters: safe text extraction, one
SDK-agnostic retry loop with bounded retry-after, timeouts set and SDK retries
off, google-genai for Gemini, model ids from config. No network: every SDK
is a fake.
"""

import importlib
import sys
import types

import httpx
import pytest

from chatbot import providers as pv

NS = types.SimpleNamespace


# ── I20: text extraction ─────────────────────────────────────────────────────

def test_I20_claude_text_blocks_only():
    resp = NS(content=[NS(type="tool_use", id="x"), NS(type="text", text="Hi "),
                       NS(type="text", text="there")], stop_reason="end_turn")
    assert pv.text_of_claude(resp) == "Hi there"


@pytest.mark.parametrize("resp", [
    NS(content=[], stop_reason="refusal"),
    NS(content=[NS(type="tool_use", id="x")], stop_reason="tool_use"),
    NS(content=None, stop_reason=None),
])
def test_I20_claude_empty_raises(resp):
    with pytest.raises(pv.EmptyResponse):
        pv.text_of_claude(resp)


def test_I20_openai_none_content_raises():
    with pytest.raises(pv.EmptyResponse, match="content_filter"):
        pv.text_of_openai(NS(choices=[NS(message=NS(content=None), finish_reason="content_filter")]))
    with pytest.raises(pv.EmptyResponse):
        pv.text_of_openai(NS(choices=[]))
    assert pv.text_of_openai(NS(choices=[NS(message=NS(content="ok"), finish_reason="stop")])) == "ok"


def test_I20_gemini_blocked_and_safety():
    with pytest.raises(pv.EmptyResponse, match="SAFETY"):
        pv.text_of_gemini(NS(candidates=[], prompt_feedback=NS(block_reason=NS(name="SAFETY"))))
    with pytest.raises(pv.EmptyResponse, match="RECITATION"):
        pv.text_of_gemini(NS(candidates=[NS(finish_reason=NS(name="RECITATION"))], text=None))
    assert pv.text_of_gemini(NS(candidates=[NS(finish_reason=NS(name="STOP"))], text="ok")) == "ok"


def test_I20_ollama_object_and_dict():
    assert pv.text_of_ollama(NS(message=NS(content="ok"))) == "ok"
    assert pv.text_of_ollama({"message": {"content": "ok"}}) == "ok"
    with pytest.raises(pv.EmptyResponse):
        pv.text_of_ollama(NS(message=NS(content="")))


# ── I21: retry policy ────────────────────────────────────────────────────────

class StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = NS(status_code=status, headers=headers or {})


def flaky(*errors, result="done"):
    calls = []

    def adapter(payload, system):
        calls.append(payload)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result
    adapter.calls = calls
    return adapter


@pytest.mark.parametrize("err", [
    StatusError(429), StatusError(529), StatusError(500), StatusError(408),
    httpx.ConnectError("down"), httpx.ReadTimeout("slow"),
    type("APIConnectionError", (Exception,), {})("conn"),
])
def test_I21_transient_errors_are_retried_for_any_sdk(err):
    sleeps = []
    adapter = flaky(err)
    assert pv.complete(adapter, [], "s", provider="chatgpt", sleep=sleeps.append) == "done"
    assert len(adapter.calls) == 2 and sleeps == [pv.BASE_DELAY_S]


def test_I21_retry_after_is_honoured_and_capped():
    sleeps = []
    pv.complete(flaky(StatusError(429, {"retry-after": "3600"}),
                      StatusError(429, {"retry-after": "2"})),
                [], "s", sleep=sleeps.append)
    assert sleeps == [pv.MAX_DELAY_S, 2.0]


def test_I21_backoff_and_exhaustion():
    sleeps = []
    adapter = flaky(*[StatusError(503)] * pv.MAX_ATTEMPTS)
    with pytest.raises(pv.ProviderError, match="after 3 attempts"):
        pv.complete(adapter, [], "s", provider="gemini", sleep=sleeps.append)
    assert sleeps == [5.0, 10.0] and len(adapter.calls) == pv.MAX_ATTEMPTS


@pytest.mark.parametrize("provider, env", [("chatgpt", "OPENAI_API_KEY"),
                                           ("gemini", "GEMINI_API_KEY"),
                                           ("claude", "ANTHROPIC_API_KEY")])
def test_I21_auth_error_names_the_providers_key(provider, env):
    adapter = flaky(StatusError(401))
    with pytest.raises(pv.ProviderAuthError, match=env):
        pv.complete(adapter, [], "s", provider=provider, sleep=lambda _s: None)
    assert len(adapter.calls) == 1


def test_I21_client_errors_and_empty_replies_not_retried():
    for err in (StatusError(400), pv.EmptyResponse("refused"), ValueError("bad")):
        adapter = flaky(err)
        with pytest.raises(pv.ProviderError):
            pv.complete(adapter, [], "s", sleep=lambda _s: pytest.fail("slept"))
        assert len(adapter.calls) == 1


# ── clients: timeout set, SDK retries off ────────────────────────────────────

def test_I21_anthropic_client_timeout_and_no_sdk_retries(monkeypatch):
    import anthropic
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
            self.messages = NS(create=lambda **k: NS(
                content=[NS(type="text", text=f"{k['model']}|{k['system']}")], stop_reason="end_turn"))
    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    adapter = pv.build("claude")
    assert seen["timeout"] == pv.TIMEOUT_S and seen["max_retries"] == 0
    import config
    assert adapter([{"role": "user", "content": "x"}], "SYS") == f"{config.CLAUDE_MODEL}|SYS"


def test_I21_openai_client_timeout_and_no_sdk_retries(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
            self.chat = NS(completions=NS(create=lambda **k: NS(choices=[NS(
                message=NS(content=k["messages"][0]["content"]), finish_reason="stop")])))
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=Fake))
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    adapter = pv.build("chatgpt")
    assert seen["timeout"] == pv.TIMEOUT_S and seen["max_retries"] == 0
    assert adapter([{"role": "user", "content": "x"}], "SYS") == "SYS"


# ── F2: google-genai, model ids from config ──────────────────────────────────

def test_F2_gemini_uses_google_genai(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kw):
            seen["client"] = kw
            self.models = NS(generate_content=self._gen)

        def _gen(self, **kw):
            seen["call"] = kw
            return NS(candidates=[NS(finish_reason=NS(name="STOP"))], text="ok")

    gtypes = NS(HttpOptions=lambda **kw: ("http", kw),
                GenerateContentConfig=lambda **kw: ("cfg", kw))
    genai = NS(Client=Client, types=gtypes)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    adapter = pv.build("gemini")
    payload = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
               {"role": "user", "content": "c"}]
    assert adapter(payload, "SYS") == "ok"
    assert seen["client"]["http_options"] == ("http", {"timeout": 60000})
    assert [c["role"] for c in seen["call"]["contents"]] == ["user", "model", "user"]
    assert seen["call"]["config"][1]["system_instruction"] == "SYS"
    import config
    assert seen["call"]["model"] == config.GEMINI_MODEL


def test_F2_missing_key_and_package_are_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=lambda **kw: None))
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        pv.build("chatgpt")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(EnvironmentError, match="pip install openai"):
        pv.build("chatgpt")


def test_F2_model_ids_from_env(monkeypatch):
    import config
    monkeypatch.setenv("SURROGATESHIELD_GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("SURROGATESHIELD_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("SURROGATESHIELD_CLAUDE_MODEL", "claude-test")
    try:
        importlib.reload(config)
        assert (config.CLAUDE_MODEL, config.GEMINI_MODEL, config.OPENAI_MODEL) == (
            "claude-test", "gemini-test", "gpt-test")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
    assert config.GEMINI_MODEL != "gemini-1.5-flash"           # retired


def test_F2_app_has_no_deprecated_sdk_or_unsafe_parsing():
    import inspect
    import chatbot.chat as chat
    import main
    for src in (inspect.getsource(chat), inspect.getsource(main._test_provider)):
        assert "generativeai" not in src
        assert "content[0]" not in src and "choices[0]" not in src
