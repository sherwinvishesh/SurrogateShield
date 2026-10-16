"""Narrowings from the dev false-positive attribution (campaign E8): the
structural PERSON frames skip code lines, assistant greetings and an
uncapitalised "name" before an age; the "N ans" age pattern ends at a word
boundary. Model-free; names and texts are invented."""

import pytest

from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.detection import structural


def _persons(text):
    added, _removed = structural.detect(text, [])
    return sorted(e.text for e in added if e.type == "PERSON")


# ── speakers ─────────────────────────────────────────────────────────────────

def test_a_transcript_still_names_its_speakers():
    text = "Ifeoma: are you coming?\nTariq: yes, at six\nIfeoma: great\nTariq: see you"
    assert _persons(text) == ["Ifeoma", "Ifeoma", "Tariq", "Tariq"]


@pytest.mark.parametrize("text", [
    ".box {\n  top: 0;\n  opacity: 100%;\n}\n.other {\n  top: 0;\n  opacity: 80%;\n}",
    "margin-top: 5px;\nmargin-top: 8px;\ncolor: red;\ncolor: blue;",
    "config = {\n  retries: 3,\n  retries: 4,\n  timeout: 30,\n}",
    "URL: https://example.org/a\nURL: https://example.org/b\nSRC: www.example.org/c\nSRC: www.example.org/d",
])
def test_style_value_and_label_lines_are_not_speakers(text):
    assert _persons(text) == []


def test_a_speaker_who_posts_a_link_is_still_a_speaker():
    text = "Ifeoma: look at this\nTariq: https://example.org/x\nIfeoma: nice\nTariq: right?"
    assert "Tariq" in _persons(text)


# ── greetings ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, names", [
    ("Hi Ifeoma, can you check the draft?", ["Ifeoma"]),
    ("Hi McKenna, can you check the draft?", ["McKenna"]),
    ("Hello DeAndre, thanks for the note.", ["DeAndre"]),
    ("Hi ChatGPT, write me a poem.", []),
    ("Hi Chat, write me a poem.", []),
    ("Hello Bard, write me a poem.", []),
    ("Hi GPT4, write me a poem.", []),
    ("Hey LinkedIn, your checkout is broken.", []),
])
def test_a_greeting_names_a_person_not_an_assistant(text, names):
    assert _persons(text) == names


# ── name before an age ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, names", [
    ("Nadia tiene 34 años y vive en Lima.", ["Nadia"]),
    ("Nadia a 34 ans.", ["Nadia"]),
    ("Write a story: the hero is a 23 year old baker.", []),
    ("Someone is a 23 year old baker.", []),
    ("Our baker Nadia is 23 years old.", ["Nadia"]),
])
def test_the_name_before_an_age_is_capitalised(text, names):
    assert _persons(text) == names


# ── "N ans" ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, ages", [
    ("Elle a 20 ans.", ["20 ans"]),
    ("Il a 20ans.", ["20ans"]),
    ("photo by Ansel Adams,%20Ansel%20Adams", []),
    ("photo,%20Ansel Adams)", []),
    ("we run 20 ansible hosts", []),
])
def test_the_unit_ans_ends_the_word(text, ages):
    assert [text[e.start:e.end] for e in pattern_scan.scan(text) if e.type == "age"] == ages
