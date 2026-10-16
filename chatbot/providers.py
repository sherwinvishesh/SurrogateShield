"""
chatbot/providers.py — one adapter per LLM provider (audit F2, I20, I21).

Each adapter turns a ``[{role, content}]`` payload into reply text and owns
three things the old single function got wrong:

* **text extraction** — an empty, filtered or non-text reply raises
  :class:`EmptyResponse` instead of ``IndexError``/``AttributeError`` or
  returning ``None`` into the history (I20);
* **errors** — auth failures name the provider's own key variable; retries
  are decided by status code (408/409/429/5xx) and connection/timeout errors
  for every SDK, not only Anthropic's (I21);
* **clients** — built with a 60 s timeout and the SDK's own retries off, so
  one retry loop (:func:`complete`) owns the policy; ``retry-after`` is
  honoured up to 60 s.

Gemini uses ``google-genai`` (``google-generativeai`` is deprecated, F2).
Model ids come from ``config`` only.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import config
from util import get_logger

logger = get_logger(__name__)

TIMEOUT_S = 60.0
MAX_ATTEMPTS = 3
BASE_DELAY_S = 5.0
MAX_DELAY_S = 60.0
MAX_TOKENS = 4096

KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
           "chatgpt": "OPENAI_API_KEY"}


class ProviderError(RuntimeError):
    """A provider call failed for good (after retries, or not retryable)."""


class EmptyResponse(ProviderError):
    """The provider answered without any text (refusal, filter, tool call)."""


class ProviderAuthError(ProviderError):
    """The API key was rejected."""


Payload = List[Dict[str, str]]


# ─── text extraction ─────────────────────────────────────────────────────────

def _name(value) -> str:
    return str(getattr(value, "name", value) or "")


def text_of_claude(response) -> str:
    parts = [b.text for b in (getattr(response, "content", None) or [])
             if getattr(b, "type", "") == "text" and isinstance(getattr(b, "text", None), str)]
    text = "".join(parts)
    if not text.strip():
        raise EmptyResponse(f"Claude returned no text (stop_reason="
                            f"{getattr(response, 'stop_reason', None)})")
    return text


def text_of_openai(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise EmptyResponse("OpenAI returned no choices")
    content = getattr(choices[0].message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise EmptyResponse(f"OpenAI returned no text (finish_reason="
                            f"{getattr(choices[0], 'finish_reason', None)})")
    return content


def text_of_gemini(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise EmptyResponse(f"Gemini blocked the prompt "
                            f"({_name(getattr(feedback, 'block_reason', None)) or 'no candidates'})")
    reason = _name(getattr(candidates[0], "finish_reason", None))
    if reason not in ("", "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"):
        raise EmptyResponse(f"Gemini finish_reason={reason}")
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise EmptyResponse(f"Gemini returned no text (finish_reason={reason or None})")
    return text


def text_of_ollama(response) -> str:
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise EmptyResponse("Ollama returned no text")
    return content


# ─── error classification (SDK-agnostic) ─────────────────────────────────────

def _status(exc) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_auth_error(exc: BaseException) -> bool:
    return _status(exc) in (401, 403) or type(exc).__name__ in (
        "AuthenticationError", "PermissionDeniedError")


def is_retryable(exc: BaseException) -> bool:
    status = _status(exc)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:                                   # pragma: no cover
        pass
    names = {c.__name__ for c in type(exc).__mro__}
    return bool(names & {"APIConnectionError", "APITimeoutError", "ConnectionError",
                         "TimeoutError", "ServiceUnavailable", "ResourceExhausted",
                         "DeadlineExceeded"})


def retry_after(exc: BaseException) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    try:
        return min(max(float(raw), 0.0), MAX_DELAY_S) if raw is not None else None
    except (TypeError, ValueError):
        return None


# ─── adapters ────────────────────────────────────────────────────────────────

class Adapter:
    """``call(payload, system) -> text``; built by :func:`build`."""

    def __init__(self, provider: str, model: str, call: Callable[[Payload, str], str]):
        self.provider = provider
        self.model = model
        self._call = call

    def __call__(self, payload: Payload, system: str) -> str:
        return self._call(payload, system)


def _key(provider: str) -> str:
    env = KEY_ENV[provider]
    value = os.environ.get(env)
    if not value:
        raise EnvironmentError(f"{env} is not set. Add it to your .env file.")
    return value


WORKSPACE_ENV = "ANTHROPIC_WORKSPACE_ID"


def claude_client_kwargs() -> dict:
    """Keyword arguments for ``anthropic.Anthropic``. A key that is not
    scoped to a workspace needs the workspace ID on every request; set it
    with ``ANTHROPIC_WORKSPACE_ID``."""
    kwargs = {"api_key": _key("claude")}
    workspace = os.environ.get(WORKSPACE_ENV, "").strip()
    if workspace:
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace}
    return kwargs


def _missing(package: str) -> EnvironmentError:
    return EnvironmentError(f"{package} package not installed. Run: pip install {package}")


def build(provider: str) -> Adapter:
    """Construct the client for *provider* ("claude", "gemini", "chatgpt",
    "local"). Raises EnvironmentError for a missing key or package."""
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(**claude_client_kwargs(), timeout=TIMEOUT_S, max_retries=0)
        model = config.CLAUDE_MODEL

        def call(payload, system):
            return text_of_claude(client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system, messages=payload))
        return Adapter(provider, model, call)

    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise _missing("google-genai") from None
        client = genai.Client(api_key=_key("gemini"),
                              http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))
        model = config.GEMINI_MODEL

        def call(payload, system):
            contents = [{"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]} for m in payload]
            return text_of_gemini(client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system,
                                                   max_output_tokens=MAX_TOKENS)))
        return Adapter(provider, model, call)

    if provider == "chatgpt":
        try:
            import openai
        except ImportError:
            raise _missing("openai") from None
        client = openai.OpenAI(api_key=_key("chatgpt"), timeout=TIMEOUT_S, max_retries=0)
        model = config.OPENAI_MODEL

        def call(payload, system):
            return text_of_openai(client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": system}] + payload))
        return Adapter(provider, model, call)

    if provider == "local":
        try:
            import ollama
        except ImportError:
            raise _missing("ollama") from None
        client = ollama.Client(host=os.environ.get("LOCAL_LLM_HOST", config.LOCAL_LLM_HOST),
                               timeout=TIMEOUT_S)
        model = os.environ.get("LOCAL_LLM_MODEL", config.LOCAL_LLM_MODEL)

        def call(payload, system):
            return text_of_ollama(client.chat(
                model=model, messages=[{"role": "system", "content": system}] + payload))
        return Adapter(provider, model, call)

    raise EnvironmentError(f"Unknown LLM provider: {provider!r}")


# ─── the one retry loop ──────────────────────────────────────────────────────

def complete(adapter: Callable[[Payload, str], str], payload: Payload, system: str, *,
             provider: str = "", sleep: Callable[[float], None] = time.sleep) -> str:
    """Call *adapter* with retries for transient errors.

    Raises:
        ProviderAuthError: the key was rejected (names the key variable).
        EmptyResponse:     the provider answered without text (not retried).
        ProviderError:     a non-retryable error, or retries exhausted.
    """
    provider = provider or getattr(adapter, "provider", "")
    for attempt in range(MAX_ATTEMPTS):
        try:
            return adapter(payload, system)
        except EmptyResponse:
            raise
        except Exception as exc:
            if is_auth_error(exc):
                env = KEY_ENV.get(provider, "the API key")
                raise ProviderAuthError(f"{provider or 'provider'} rejected the key — "
                                        f"check {env}") from exc
            if not is_retryable(exc):
                raise ProviderError(f"{provider or 'provider'} call failed: {exc}") from exc
            if attempt == MAX_ATTEMPTS - 1:
                raise ProviderError(f"{provider or 'provider'} call failed after "
                                    f"{MAX_ATTEMPTS} attempts: {exc}") from exc
            delay = retry_after(exc)
            if delay is None:
                delay = min(BASE_DELAY_S * (2 ** attempt), MAX_DELAY_S)
            logger.warning(f"[providers] {provider} transient error ({type(exc).__name__}) — "
                           f"retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
            sleep(delay)
    raise AssertionError("unreachable")                    # pragma: no cover
