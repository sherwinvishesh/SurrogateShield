"""
surrogateshield/_response_parser.py — text of an LLM response (audit I10).

Accepts a str, or an Anthropic / OpenAI / Gemini response as an SDK object or
as a dict (``model_dump()`` / JSON). Anything else raises ``TypeError`` —
``unmask()`` must never turn ``None`` or an unknown object into its ``repr``.
"""

from __future__ import annotations


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_text(response) -> str:
    """Return the text content of *response*.

    * Anthropic: every ``text`` block of ``content``, concatenated (tool-use
      and thinking blocks are skipped).
    * OpenAI: ``choices[0].message.content`` (``None`` → ``""``, e.g. a
      tool-call-only reply).
    * Gemini: ``.text``.

    Raises:
        TypeError: *response* is None or no text could be found in it.
    """
    if isinstance(response, str):
        return response
    if response is None:
        raise TypeError("unmask() got None — pass the LLM response text or object")

    content = _get(response, "content")
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            kind = _get(block, "type")
            text = _get(block, "text")
            if isinstance(text, str) and kind in (None, "text"):
                parts.append(text)
        return "".join(parts)

    choices = _get(response, "choices")
    if isinstance(choices, (list, tuple)):
        if not choices:
            return ""
        message = _get(choices[0], "message")
        text = _get(message, "content") if message is not None else _get(choices[0], "text")
        if text is None:
            return ""
        if isinstance(text, str):
            return text

    text = _get(response, "text")
    if isinstance(text, str):
        return text

    raise TypeError(
        f"unmask() cannot find text in a {type(response).__name__}; pass a str or an "
        "Anthropic / OpenAI / Gemini response"
    )
