"""Audit I1 / I12, gate J2 — names the J2 dev2 split still leaked: a quoted
name read as code, an all-caps name hiding its title-case form, an e-mail
display name typed as a company, greeted names, kin + name and a dash
sign-off. Positives come from dev2 / synth dev; negatives are near misses.
Model-free.
"""

import pytest

from surrogateshield.core.detection import relation_gate, structural
from surrogateshield.core.entities import DetectedEntity


def ent(text, value, typ, source="model"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def test_I12_quoted_name_is_not_code():
    text = 'name="Kalinda Whitehorse"'
    assert not relation_gate.is_junk(ent(text, '"Kalinda Whitehorse"', "PERSON", "pattern"), text)
    assert relation_gate.is_junk(ent('call f(x) = y', 'f(x) = y', "PERSON"), 'call f(x) = y')


def test_I12_all_caps_name_does_not_hide_title_case():
    text = "JAVIER MONTOYA\nnotes about Javier Montoya. should I reply to Javier?"
    ents = [ent(text, "JAVIER MONTOYA", "PERSON"),
            DetectedEntity("Javier Montoya", text.index("Javier Montoya"),
                           text.index("Javier Montoya") + 14, "PERSON", 0.9, "model")]
    added, _ = structural._components(text, ents)
    assert any(e.text == "Javier" and e.start == text.rindex("Javier") for e in added)


def test_I12_email_display_name_is_a_person():
    text = "To: Ifeoma Chukwu <ifeoma@chukwustudio.com>\n\nHi Ifeoma,\nthanks"
    added, removed = structural._display_names(text, [ent(text, "Ifeoma Chukwu", "ORG")])
    assert [e.type for e in removed] == ["ORG"]
    assert ("PERSON", "Ifeoma Chukwu") in [(e.type, e.text) for e in added]
    assert ("PERSON", "Ifeoma") in [(e.type, e.text) for e in added]


@pytest.mark.parametrize("text", [
    "From: Acme Support <help@acme.com>", "Hi all,\nthe build is red", "Hello team, quick update",
])
def test_I8_display_name_near_misses(text):
    ents = [ent(text, "Acme Support", "ORG")] if "Acme" in text else []
    added, _ = structural._display_names(text, ents)
    assert added == []


@pytest.mark.parametrize("text, name", [
    ("can u help me write a bday msg for my sister ines, she turns 30 on friday", "ines"),
    ("Olá, a minha filha Beatriz tem 9 anos e anda na Escola", "Beatriz"),
    ("meine Tochter Lena ist 5", "Lena"),
    ("my daughter, Emma, is 4", "Emma"),
    ("the keys are under the matt, the alram code is 4091. -tash", "tash"),
])
def test_I12_kin_names_and_sign_offs(text, name):
    added, _ = structural._kin_names(text, [])
    assert name in [e.text for e in added]


@pytest.mark.parametrize("text", [
    "my sister is coming", "my son plays football, he loves it", "My wife Facebook-stalks me",
    "it's 5 minutes -ish", "My sister ines is here",  # lower-case name from a capitalising writer
    "-tash said hi\nthen more text",
])
def test_I8_kin_name_near_misses(text):
    added, _ = structural._kin_names(text, [])
    assert added == []
