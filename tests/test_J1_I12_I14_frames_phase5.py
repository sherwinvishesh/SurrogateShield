"""Gate J1 (synth dev key) with audit I12 / I14 — frames that carried every
unintended leak on the synthetic dev split: a Spanish "tiene N años y estudia
en <ORG>" line, the company line of a sign-off, an all-caps name above a
street or before a bracketed phone, "im <name> from <place>", a personnel
verb's object and subject, and "my name is X" when NER typed X as an ORG.
Two gate errors on the legacy key are here too: the author of the user's own
draft is not a public author, and "our <system> at <public company>" is an
employer. Model-free except where a test builds the entities by hand.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.detection import structural as st
from surrogateshield.core.detection.pipeline import _detect_structural_persons
from surrogateshield.core.entities import DetectedEntity


def ent(text, value, typ, source="pattern"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 1.0, source)


def added(text, ents=()):
    new, _ = st.detect(text, list(ents))
    for e in new:
        assert text[e.start:e.end] == e.text
    return [(e.type, e.text) for e in new]


# ── affiliation and age frames ───────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("Saoirse tiene 62 años y estudia en Quayside Accounting; mi teléfono es",
     [("PERSON", "Saoirse"), ("ORG", "Quayside Accounting")]),
    ("Kwame trabaja en Redfern Physiotherapy desde 2019", [("ORG", "Redfern Physiotherapy")]),
    ("je travaille chez Marlowe Design Studio", [("ORG", "Marlowe Design Studio")]),
    ("ich arbeite bei Kestrel Analytics", [("ORG", "Kestrel Analytics")]),
    ("Noa a 71 ans", [("PERSON", "Noa")]),
    ("Jonas is 41 years old", [("PERSON", "Jonas")]),
])
def test_J1_affiliation_and_age_frames(text, expected):
    got = added(text)
    for x in expected:
        assert x in got


@pytest.mark.parametrize("text", [
    "Mi hijo tiene 5 años",
    "El tiene 30 años",
    "The warranty is 2 years",
    "Python tiene 30 años de historia",
])
def test_J1_age_frame_needs_a_name(text):
    assert not any(t == "PERSON" for t, _ in added(text))


def test_J1_trailing_spanish_verb_is_cut_from_a_name():
    text = "Kwame tiene 25 años"
    e = DetectedEntity("Kwame tiene", 0, 11, "PERSON", 0.9, "ner")
    assert rg.trim_person(e, text).text == "Kwame"


# ── contact blocks ───────────────────────────────────────────────────────────

def test_J1_all_caps_name_above_a_street():
    text = "FOLAKE BANERJEE\n1827 Maple Street\nNaperville 64130"
    addr = ent(text, "1827 Maple Street\nNaperville 64130", "address")
    assert ("PERSON", "FOLAKE BANERJEE") in added(text, [addr])


def test_J1_label_line_above_a_street_is_not_a_name():
    text = "Shipping Address\n1827 Maple Street\nNaperville 64130"
    addr = ent(text, "1827 Maple Street\nNaperville 64130", "address")
    assert not any(t == "PERSON" for t, _ in added(text, [addr]))


def test_J1_all_caps_name_before_bracketed_phone():
    text = "URGENT: ULRIKE REYES (875.228.2393) DID NOT RECEIVE THE REFUND"
    ph = ent(text, "875.228.2393", "phone_us")
    assert ("PERSON", "ULRIKE REYES") in added(text, [ph])
    text2 = "CALL CUSTOMER SERVICE (800.555.0199) NOW"
    assert not any(t == "PERSON" for t, _ in added(text2, [ent(text2, "800.555.0199", "phone_us")]))


def test_J1_phone_brackets_come_in_pairs():
    assert [e.text for e in ps.scan("X (875.228.2393) DID")] == ["875.228.2393"]
    assert [e.text for e in ps.scan("call (683) 923-7935")] == ["(683) 923-7935"]


@pytest.mark.parametrize("org", ["Orbital Freight", "Hollis Middle School"])
def test_J1_signature_company_line(org):
    text = f"Thanks,\nAndrei Ruiz\n{org}\n+1-379-831-5710 | andrei@example.com"
    ents = [ent(text, "Andrei Ruiz", "PERSON", "ner"), ent(text, "+1-379-831-5710", "phone_us"),
            ent(text, "andrei@example.com", "email")]
    assert ("ORG", org) in added(text, ents)


def test_J1_signature_job_title_is_not_a_company():
    text = "Thanks,\nAndrei Ruiz\nSenior Analyst\nandrei@example.com"
    ents = [ent(text, "Andrei Ruiz", "PERSON", "ner"), ent(text, "andrei@example.com", "email")]
    assert not any(t == "ORG" for t, _ in added(text, ents))


# ── chat frames ──────────────────────────────────────────────────────────────

def test_J1_im_name_from_place():
    got = added("my gamertag is @x lol, add me. irl im bongani from moncton")
    assert ("PERSON", "bongani") in got
    for text in ("im back from work", "im fresh from the gym", "I'm calling from Leeds"):
        assert not any(t == "PERSON" for t, _ in added(text))


def test_J1_chat_abbreviations_are_not_names():
    assert rg.is_junk(DetectedEntity("irl", 0, 3, "PERSON", 0.9, "ner"), "irl im x")


def test_J1_personnel_verb_object_and_employer():
    text = "Can Lumen Credit Union fire Bilal Park for taking sick leave? He has worked there."
    got = added(text)
    assert ("PERSON", "Bilal Park") in got and ("ORG", "Lumen Credit Union") in got
    assert added("Can my boss fire me for this?") == []


def test_J1_my_name_is_retypes_an_org():
    text = "Translate to Spanish: 'My name is Paloma Achebe and I live in Bilbao.'"
    org = ent(text, "Paloma Achebe", "ORG", "ner")
    new, superseded = _detect_structural_persons(text, [org])
    assert [(e.type, e.text) for e in new] == [("PERSON", "Paloma Achebe")]
    assert superseded == [org]


# ── gate errors on the legacy key ────────────────────────────────────────────

def test_I12_author_of_own_draft_is_private():
    text = "Please review this professional introductory statement written by Olwethu Dlamini and fix it."
    p = ent(text, "Olwethu Dlamini", "PERSON", "ner")
    assert not rg.is_public_person(p, text)
    text2 = "Summarise a novel written by Haruki Murakami"
    assert rg.is_public_person(ent(text2, "Haruki Murakami", "PERSON", "ner"), text2)


def test_I12_our_system_at_public_company_is_employer():
    text = "why is our cloud infrastructure at Microsoft throwing errors"
    _, dropped = rg.gate(text, [ent(text, "Microsoft", "ORG", "ner")])
    assert dropped == []
    text2 = "our account at Chase was frozen"
    _, dropped = rg.gate(text2, [ent(text2, "Chase", "ORG", "ner")])
    assert [e.text for e in dropped] == ["Chase"]
