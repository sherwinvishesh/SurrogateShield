"""Audit I1, gate J2 — a model span that stops inside a word leaves the rest
of the name in the message ("Hyun" of "Hyun-woo Park", "n Adeyemi-" of
"Oluwaseun Adeyemi-Clarke"). pipeline._snap_to_words grows model spans to
whole words, drops outer quotes and a possessive 's, lets a one-word PERSON
take an adjacent name word or particles and a one-word ORG its company word.
Cases come from the dev2 split. Model-free.
"""

import pytest

from surrogateshield.core.detection.pipeline import _snap_to_words
from surrogateshield.core.entities import DetectedEntity


def snap(text, span, kind="PERSON", source="slm"):
    i = text.index(span)
    out = _snap_to_words([DetectedEntity(span, i, i + len(span), kind, 0.9, source)], text)[0]
    assert text[out.start:out.end] == out.text
    return out.text


@pytest.mark.parametrize("text, span, kind, want", [
    ("Cardholder: Hyun-woo Park, 2087 Larkspur Ln", "Hyun", "PERSON", "Hyun-woo Park"),
    ("Thanks,\nOluwaseun Adeyemi-Clarke\nSenior Analyst", "n Adeyemi-", "PERSON",
     "Oluwaseun Adeyemi-Clarke"),
    ("a letter to Jean-Luc d'Aubigné? Both sit", "Jean-Luc", "PERSON", "Jean-Luc d'Aubigné"),
    ("Is it 'Van der Merwe' or", "Merwe", "PERSON", "Van der Merwe"),
    ("my colleague is Pieter van der Merwe and", "Merwe", "PERSON", "van der Merwe"),
    ("Senior Analyst, Brightwater Logistics\nM: +234", "Brightwater", "ORG", "Brightwater Logistics"),
    ('u = User(name="Kalinda Whitehorse", email=', '"Kalinda Whitehorse"', "PERSON",
     "Kalinda Whitehorse"),
    ("currently at Endava in Cluj-Napoca, email", "Cluj", "GPE", "Cluj-Napoca"),
    ("Is it Thandiwe's or Thandiwes'", "Thandiwe's", "ORG", "Thandiwe"),
])
def test_I1_span_grows_to_the_whole_name(text, span, kind, want):
    assert snap(text, span, kind) == want


@pytest.mark.parametrize("text, span", [
    ("Hallo, ich wohne in der Lindenstraße 14", "Lindenstraße"),   # article, not a particle
    ("I met Anna Monday at noon", "Anna"),
    ("Tell Anna I said hi", "Anna"),
    ("my MacBook's screen cracked", "MacBook"),
])
def test_I1_span_does_not_grow_into_other_words(text, span):
    assert snap(text, span) == span


def test_I1_pattern_entities_are_left_alone():
    text = "call 0161 496 0812x"
    ent = DetectedEntity("0161 496 0812", 5, 18, "phone_uk", 1.0, "pattern")
    assert _snap_to_words([ent], text) == [ent]
