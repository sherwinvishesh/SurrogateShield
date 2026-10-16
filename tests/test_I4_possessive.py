"""Audit I4 — possessives and clause punctuation are not part of a name.

Before: an NER span "Leo Tolstoy's" or "…收件人：李娜，邮箱：" was used as is, so
the clitic became part of the original (missing the public-figure list) and
a span ending in "：" glued its surrogate to the next word, which could then
never be restored.
"""

import pytest

from surrogateshield.core.detection.entity_trace import _clean_span
from surrogateshield.core.reconstruction.resolve import ResolvePass


@pytest.mark.parametrize("text, span, expected", [
    ("in Leo Tolstoy's novel", "Leo Tolstoy's", "Leo Tolstoy"),
    ("Mr. Jennings' car", "Jennings'", "Jennings"),
    ("Ask (Mia Lopez), please", "(Mia Lopez),", "Mia Lopez"),
    ("收件人：李娜，邮箱：x@y.com", "收件人：李娜，邮箱：", "收件人：李娜，邮箱"),
    ("Dr. Sun Inc.", "Sun Inc.", "Sun Inc."),
])
def test_I4_clean_span(text, span, expected):
    start = text.index(span)
    s, e = _clean_span(text, start, start + len(span))
    assert text[s:e] == expected


def test_I4_possessive_of_surrogate_restores():
    out = ResolvePass().resolve("Douglas Riley's file and Riley's desk.",
                                {"Douglas Riley": "Sarah Mitchell"})
    assert out == "Sarah Mitchell's file and Riley's desk."
