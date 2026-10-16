"""Pass G / G2: a PERSON span grows over the rest of its name when no other
entity covers it: name particles ("Ó", "Mac", "van der", "de"), a mutated
Irish surname ("Ó hÉinniú"), a middle initial, and the other half of a field
whose whole value is "Surname, Given". A lowercase particle alone stays a
word. Pass R keeps a mutated surname; Pass S takes a name an e-mail spells
whole. Model-free; every name is made up.
"""

import pytest

from surrogateshield.core.detection import relation_gate, structural
from surrogateshield.core.detection.pipeline import (
    _complete_person_spans, _merge_adjacent_persons)
from surrogateshield.core.entities import DetectedEntity


def ent(text, value, typ="PERSON", source="ner"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.8, source)


def grown(text, *values, others=()):
    return [e.text for e in _complete_person_spans(
        [ent(text, v) for v in values], text, [ent(text, v, t) for v, t in others])]


@pytest.mark.parametrize("text, values, expected", [
    ('{"name": "Blažek, Samuel", "x": 1}', ["Samuel"], ["Blažek, Samuel"]),
    ("Name\nSimó, Estela\nPhone", ["Estela"], ["Simó, Estela"]),
    ("Regards,\nSuzanne Ó hÉinniú, Dublin", ["Suzanne Ó"], ["Suzanne Ó hÉinniú"]),
    ("I'm Judy Mac Giolla Bhuí and I", ["Giolla Bhuí"], ["Judy Mac Giolla Bhuí"]),
    ('{"n": "Ó Tomáis, Kieron"}', ["Tomáis", "Kieron"], ["Ó Tomáis", "Kieron"]),
    ("Liam Ó Tomáis wrote", ["Tomáis"], ["Liam Ó Tomáis"]),
    ("call Maria de Souza today", ["Maria"], ["Maria de Souza"]),
    ("call Maria de Souza today", ["Souza"], ["Maria de Souza"]),
    ("Ursula von der Leyen said", ["Leyen"], ["Ursula von der Leyen"]),
    ("I saw Van Halen live", ["Halen"], ["Van Halen"]),
    ("I am Suzana H. Peharda， and", ["Suzana H."], ["Suzana H. Peharda"]),
    ('{"c": "Vuković, Božo", "r": 1}', ["Vuković"], ["Vuković, Božo"]),
])
def test_span_grows_over_its_name(text, values, expected):
    assert grown(text, *values) == expected


@pytest.mark.parametrize("text, values", [
    ("le livre de Marie est là", ["Marie"]),          # "de" alone is a preposition
    ("de Souza called", ["Souza"]),
    ("go to la Maria place", ["Maria"]),
    ('"Thanks, Sarah"', ["Sarah"]),
    ('"Love, Sarah"', ["Sarah"]),
    ("Hello there Blažek, Samuel", ["Samuel"]),        # not a whole field value
    ("Wrote to Ana Silva about de facto rules", ["Ana Silva"]),
    ("I met Anna H. The rest", ["Anna H."]),
    ("Maria Lopez, Madrid\n", ["Maria Lopez"]),       # a name and its town
    ('"Smith, Inc"', ["Smith"]),
    ("Hello Vuković, Božo", ["Vuković"]),
])
def test_span_stays(text, values):
    assert grown(text, *values) == values


def test_span_does_not_grow_over_another_entity():
    text = '"Paris, Anna"'
    assert grown(text, "Anna", others=[("Paris", "GPE")]) == ["Anna"]


@pytest.mark.parametrize("text, values, expected", [
    ("Cauã I. Pacheco wrote", ["Cauã", "Pacheco"], ["Cauã I. Pacheco"]),
    ("Maria de Souza", ["Maria", "Souza"], ["Maria de Souza"]),
    ("Anna i. Bell", ["Anna", "Bell"], ["Anna", "Bell"]),
    ("Tom and Jerry", ["Tom", "Jerry"], ["Tom", "Jerry"]),
])
def test_merge_across_initial_or_particles(text, values, expected):
    merged = _merge_adjacent_persons([ent(text, v) for v in values], text)
    assert sorted(e.text for e in merged) == sorted(expected)


@pytest.mark.parametrize("text, name", [
    ("Regards, Suzanne Ó hÉinniú, Galway", "Suzanne Ó hÉinniú"),
    ("Uí nGallchóir Áine wrote", "Uí nGallchóir Áine"),
])
def test_frame_trim_keeps_a_mutated_surname(text, name):
    assert relation_gate.trim_person(ent(text, name), text).text == name


def local_part(text, *ents):
    added, removed = structural._local_part_names(text, list(ents))
    return [e.text for e in added], [e.text for e in removed]


def test_email_spelled_name_across_an_initial():
    text = "by Cauã I. Pacheco, caua.pacheco@mail.example"
    assert local_part(text, ent(text, "caua.pacheco@mail.example", "email", "pattern")) == (
        ["Cauã I. Pacheco"], [])


def test_email_spelled_name_replaces_a_model_piece():
    text = "Zoé Acuña | acuna.zoe@mail.example | DOB"
    mail = ent(text, "acuna.zoe@mail.example", "email", "pattern")
    assert local_part(text, ent(text, "Acuña", source="slm"), mail) == (["Zoé Acuña"], ["Acuña"])
    # a different type over the name is left to the gate and the resolver
    assert local_part(text, ent(text, "Zoé Acuña", "ORG", "slm"), mail) == ([], [])
