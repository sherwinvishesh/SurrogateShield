"""
surrogateshield/_response_parser.py — LLM response text extractor

Extracts plain text from any major LLM SDK response object.
"""

from __future__ import annotations


def extract_text(response) -> str:
    """
    Extract the text content from an LLM response object.

    Tries in order:
    1. Anthropic style: response.content[0].text
    2. OpenAI style:    response.choices[0].message.content
    3. Gemini style:    response.text  (has .text but no .choices)
    4. Fallback:        str(response)

    Args:
        response: Any LLM SDK response, or a plain string.

    Returns:
        The extracted text as a string.
    """
    if isinstance(response, str):
        return response

    # Anthropic: response.content is a list of content blocks
    try:
        if hasattr(response, "content") and isinstance(response.content, list):
            return response.content[0].text
    except (AttributeError, IndexError):
        pass

    # OpenAI: response.choices[0].message.content
    try:
        if hasattr(response, "choices"):
            return response.choices[0].message.content
    except (AttributeError, IndexError):
        pass

    # Gemini: response.text (but no .choices attribute)
    try:
        if hasattr(response, "text") and not hasattr(response, "choices"):
            return response.text
    except AttributeError:
        pass

    return str(response)
