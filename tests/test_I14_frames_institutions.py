"""Audit I12 / I14, gate J2 — names, places and institutions the J2 dev2
split leaked: CJK and Hindi self-introductions and kin frames, named
schools / clinics / universities, the town between a UK street and its
postcode, and name particles carried by a propagated surname. Positives
come from dev2; negatives are near misses. Model-free.
"""

import pytest

from surrogateshield.core.detection import structural as S
from surrogateshield.core.entities import DetectedEntity


def got(rule, text, ents=()):
    added, _ = rule(text, list(ents))
    for e in added:
        assert text[e.start:e.end] == e.text
    return [(e.type, e.text) for e in added]


@pytest.mark.parametrize("text, expected", [
    ("はじめまして、佐々木 美咲です。札幌に住んでいます。", [("PERSON", "佐々木 美咲"), ("GPE", "札幌")]),
    ("请帮我把这段话翻译成英文：李梅的身份证号是110105199203157728，她在北京协和医院工作。",
     [("PERSON", "李梅"), ("ORG", "北京协和医院")]),
    ("मेरी माँ सुनीता देवी 67 साल की हैं। लखनऊ में अच्छे कार्डियोलॉजिस्ट कौन हैं?",
     [("PERSON", "सुनीता देवी"), ("GPE", "लखनऊ")]),
])
def test_I14_cjk_hindi_frames(text, expected):
    assert got(S._names_intl, text) == expected


@pytest.mark.parametrize("text", [
    "私は学生です。東京は好きです。", "मेरी माँ बीमार हैं", "लखनऊ में मौसम कैसा है?",
    "这是北京的天气。",
])
def test_I8_cjk_hindi_near_misses(text):
    assert got(S._names_intl, text) == []


@pytest.mark.parametrize("text, value", [
    ("Ma sage-femme à la Clinique Saint-Vincent de Besançon m'a dit", "Clinique Saint-Vincent de Besançon"),
    ("anda na Escola Básica de Alvalade, em Lisboa.", "Escola Básica de Alvalade"),
    ("how do i request my transcript from université laval? my matricule", "université laval"),
    ("im 41 and my doctor at riverside family clinic in dayton put me", "riverside family clinic"),
])
def test_I14_institutions(text, value):
    assert ("ORG", value) in got(S._institutions, text)


@pytest.mark.parametrize("text", [
    "i went to the local clinic today", "i go to the dermatology clinic",
    "i need a walk-in clinic near me", "The Hospital Is Closed",
    "How do I apply to université laval?",       # a capitalising writer: NER's job
])
def test_I8_institution_near_misses(text):
    assert got(S._institutions, text) == []


def test_I14_town_before_uk_postcode():
    text = "my grandma lives alone at 14 birchwood close, wokingham rg40 2hd. tips"
    assert ("address", "wokingham") in got(S._address_parts, text)


def test_I12_propagated_surname_keeps_its_particles():
    text = "Is it 'Van der Merwe' or 'van der Merwe'? My colleague is Pieter van der Merwe."
    i = text.index("Pieter")
    p = DetectedEntity("Pieter van der Merwe", i, i + 20, "PERSON", 0.9, "model")
    assert [v for _t, v in got(S._components, text, [p])] == ["Van der Merwe", "van der Merwe"]
