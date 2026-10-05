"""Gate J2 — names in frames NER misses or clips: another person's "name
is", titles of any language, greeting lines, display names before an
address, a clipped last word, overlapping PERSON spans, names spelled out
by a detected e-mail or handle, "contact X on <number>" and an initialled
payee. Model-free: entities are built by hand, as NER would emit them.
"""

import pytest

from surrogateshield.core.detection import pattern_scan, relation_gate as rg, structural
from surrogateshield.core.detection.pipeline import (
    _detect_structural_persons, _merge_adjacent_persons)
from surrogateshield.core.entities import DetectedEntity


def ent(text, value, typ="PERSON", source="ner"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.88, source)


def pass_e(text, *ents):
    return sorted(e.text for e in _detect_structural_persons(text, list(ents))[0])


def struct(text):
    ents = pattern_scan.scan(text)
    added, _ = structural.detect(text, ents)
    return sorted(e.text for e in added if e.type == "PERSON")


@pytest.mark.parametrize("text, name", [
    ("my maneger (her name is Krystal Okwuosa) said i cant", "Krystal Okwuosa"),
    ("HIS NAME IS KAIRAT ABENOV, BORN 09.09.2008", "KAIRAT ABENOV"),
    ("A professora Renata pediu os documentos", "Renata"),
    ("Need a fundraiser for Father Eamon Kirwan's mission trip", "Eamon Kirwan"),
    ("Ich schreibe an Frau Kessler wegen der Miete", "Kessler"),
    ("> Dear TAN HUI MIN,\n> We detected a sign-in", "TAN HUI MIN"),
    ("Dear Ms Okafor,\nThank you for", "Okafor"),
    ("3e88d41 initial pdf export — Sofía Paredes <sofia.p@ledgerloop.io>", "Sofía Paredes"),
])
def test_J2_name_frame_finds_the_name(text, name):
    assert name in pass_e(text)


@pytest.mark.parametrize("text", [
    "his name is irrelevant here",
    "Dear Hiring Manager,\nI am applying",
    "Dear Sir or Madam,\nI write",
    "Father Christmas came early",
    'From: "Chase Alerts" <alerts@chase.com>',
    "From: Account Security <noreply@bank.co>",
])
def test_J2_name_frame_leaves_non_names(text):
    assert pass_e(text) == []


def test_J2_clipped_name_takes_the_last_word_of_its_line():
    text = "- Employee: Abebe Girma Tesfaye\n- Role: Warehouse Supervisor"
    assert pass_e(text, ent(text, "Abebe Girma")) == ["Abebe Girma Tesfaye"]


@pytest.mark.parametrize("text, clipped, word", [
    ("I spoke to Anna Monday", "Anna", "Monday"),          # a weekday
    ("a call with Maria Tomorrow", "Maria", "Tomorrow"),   # a stop word
])
def test_J2_clipped_name_never_takes_a_common_word(text, clipped, word):
    assert not [n for n in pass_e(text, ent(text, clipped)) if word in n]


def test_J2_overlapping_person_spans_are_one_name():
    text = "a fundraiser for Father Eamon Kirwan's mission"
    merged = _merge_adjacent_persons(
        [ent(text, "Father Eamon", source="slm"), ent(text, "Eamon Kirwan")], text)
    assert [e.text for e in merged] == ["Father Eamon Kirwan"]
    trimmed = rg.trim_person(merged[0], text)
    assert trimmed.text == "Eamon Kirwan"


@pytest.mark.parametrize("text, span, name", [
    ("devo scrivere al mio medico, il dottor Bellandi dello Studio", "il dottor Bellandi", "Bellandi"),
    ("A professora Renata pediu os documentos", "professora Renata pediu", "Renata"),
])
def test_J2_title_and_trailing_verb_are_cut_from_a_model_name(text, span, name):
    assert rg.trim_person(ent(text, span), text).text == name


@pytest.mark.parametrize("text, name", [
    ("> To: hui.min.tan68@starhub-mail.sg\n> Dear Customer TAN HUI MIN", "TAN HUI MIN"),
    ("@farshad.m: can someone cover\n@deb.ostrowski: not again farshad", "farshad"),
    ("# YUKI TANAKA-OKONJO\n📧 yuki.tanaka.okonjo@kitamail.jp", "YUKI TANAKA-OKONJO"),
])
def test_J2_email_or_handle_spells_out_a_name(text, name):
    assert name in struct(text)


@pytest.mark.parametrize("text", [
    "ping me at hui.min@x.sg, back in 5 min",
    "`Database=Payroll;User Id=svc_payroll;` keeps timing out",
    "write to support.team@acme.com and the team will answer",
])
def test_J2_email_or_handle_words_that_are_not_names(text):
    assert struct(text) == []


def test_J2_contact_frame_and_initialled_payee():
    assert "Bríd" in struct("Contact Bríd on 086 422 7731.")
    assert struct("Contact Support on 0800 123 456") == []
    assert "M OKAFOR" in struct("2026-09-02,TRANSFER TO M OKAFOR,-250.00")
