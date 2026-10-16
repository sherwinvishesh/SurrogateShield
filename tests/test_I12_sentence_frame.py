"""Audit I12, gate J2 — an English NER model run on German, Dutch, Hindi and
Portuguese text tags whole sentence frames as PERSON ("Mein Vermieter",
"Mera naam Rohit Bhardwaj hai"). relation_gate.trim_person cuts the span to
the name, drops a frame with no name in it, and keeps the whole span when it
cannot tell (fail closed). Cases come from the dev2 and synth dev splits.
Model-free.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def trim(span, text=None):
    text = text if text is not None else span
    start = text.index(span)
    ent = DetectedEntity(span, start, start + len(span), "PERSON", 0.88, "ner")
    out = rg.trim_person(ent, text)
    if out is not None:
        assert text[out.start:out.end] == out.text
    return None if out is None else out.text


@pytest.mark.parametrize("span, name", [
    ("Ich heiße Anna Müller", "Anna Müller"),
    ("ich bin Jörg Baumgartner", "Jörg Baumgartner"),
    ("ben Sanne Wijnberg", "Sanne Wijnberg"),
    ("Mera naam Rohit Bhardwaj hai", "Rohit Bhardwaj"),
    ("Kun je Anna Müller bellen", "Anna Müller"),
])
def test_I12_frame_is_cut_to_the_name(span, name):
    assert trim(span, f"Hallo, {span}. Danke") == name


@pytest.mark.parametrize("span", [
    "Mein Vermieter", "Meine Steuer", "Mijn BSN", "Mera Aadhaar", "Mein Name",
    "Kun je mijn bezwaarschrift aan de Belastingdienst",
])
def test_I12_frame_without_a_name_is_dropped(span):
    assert trim(span) is None


@pytest.mark.parametrize("span", [
    "Anna Müller", "Mia Hoang", "Ben Stiller", "Jean de la Fontaine", "Ahmed bin Rashid",
    "Ngozi Adeyemi + Chidi Adeyemi", "Anna and Ben Smith",
])
def test_I12_names_are_left_whole(span):
    assert trim(span) == span


def test_I12_lowercase_chat_is_left_to_is_junk():
    assert trim("hey maria Lopez") == "hey maria Lopez"


@pytest.mark.parametrize("span", ["karna zaroori hai", "mein rehta hoon", "como autônomo"])
def test_I12_lowercase_frame_is_junk(span):
    assert trim(span) == span
    assert rg.is_junk(DetectedEntity(span, 0, len(span), "PERSON", 0.88, "ner"), span)
