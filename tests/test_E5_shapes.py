"""Shape rows from the external diagnosis (campaign E5f): an age behind a
Markdown-emphasised, tight-colon or non-English label. Model-free; every
value is invented."""

import pytest

from surrogateshield.core.detection import pattern_scan


def _ages(text):
    return [text[e.start:e.end] for e in pattern_scan.scan(text) if e.type == "age"]


@pytest.mark.parametrize("text, ages", [
    ("**Age:** 34", ["Age:** 34"]),
    ("**Age**: 34", ["Age**: 34"]),
    ("*Age*: 52", ["Age*: 52"]),
    ("Age:34", ["Age:34"]),
    ("Age: 34", ["Age: 34"]),
    ("- **Age:** 41\n- **City:** Lima", ["Age:** 41"]),
    ("Edad: 34", ["Edad: 34"]),
    ("Idade: 41", ["Idade: 41"]),
    ("Âge : 29", ["Âge : 29"]),
])
def test_an_age_behind_an_emphasised_or_non_english_label(text, ages):
    assert _ages(text) == ages


@pytest.mark.parametrize("text", [
    "page: 34",
    "stage: 3",
    "storage: 34",
    "Age: 2024",
    "Age: 34%",
    "Age range: 25-34",
    "Coverage: 34",
    "**Weight:** 34",
    "Cache-Control: max-age=60",
    "max-age=3600",
])
def test_a_label_that_is_not_an_age(text):
    assert _ages(text) == []
