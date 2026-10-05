"""Audit I8 / I12, gate J2 — spurious replacements on the J2 dev2 split:
public brands reached through "my …" or an account number, capitalised
common nouns, degrees and job titles, member access in code, word-names
after "compared to", automated senders, the verb "handle", labelled
non-birth dates, item numbers in SSN shape and an English word read as a
German street. GUIDE.md lists all of these as keep. Model-free.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps, relation_gate as rg, structural
from surrogateshield.core.entities import DetectedEntity


def ent(text, value, typ, source="model"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def dropped(text, *spec, context=()):
    ents = [ent(text, v, t) for v, t in spec]
    _kept, gone = rg.gate(text, ents, context)
    return [e.text for e in gone]


def pat(text, value, typ):
    return ent(text, value, typ, "pattern")


@pytest.mark.parametrize("text, value, typ, ctx", [
    ("My actual Chase account is 7719-0024-55. Scam?", "Chase", "ORG", ("7719-0024-55", "id_number")),
    ("My Vodafone customer number is 300417762", "Vodafone", "ORG", ("300417762", "id_number")),
    ("give my sort code 40-11-62 to a buyer on Facebook Marketplace", "Facebook Marketplace", "ORG",
     ("40-11-62", "id_number")),
    ("my mum Rukhsana lost her phone (07983 446 120) somewhere in Heathrow T5", "Heathrow T5", "FAC",
     ("07983 446 120", "phone_intl")),
])
def test_I12_public_brand_with_personal_data_is_kept(text, value, typ, ctx):
    assert dropped(text, (value, typ), context=[pat(text, *ctx)]) == [value]


@pytest.mark.parametrize("text, value, typ", [
    ("Date of entry: 04/12/2019; Port of entry: SFO. Mr. Chen Jianguo", "Port", "GPE"),
    ("Passport C7HX91207, name Inês Carvalho Matos.", "Passport", "LOC"),
    ("tobias and I met at Uni in Graz", "Uni", "GPE"),
    ("Client: Bartholomew Ekwueme\nWants: a quote\nBudget: $25k", "Budget", "ORG"),
    ("Booking is under Nakamura Ren at Hotel Granvia Osaka, Room 1208.", "Room 1208", "FAC"),
    ("Ich bin Jörg. Mein Vermieter will die Kaution nicht zurückzahlen.", "Kaution", "GPE"),
])
def test_I8_common_nouns_in_anchored_messages_are_kept(text, value, typ):
    name = next((n for n in ("Nakamura Ren", "Bartholomew Ekwueme") if n in text), text.split()[0])
    anchor = pat(text, name, "PERSON")
    assert value in dropped(text, (value, typ), context=[anchor])


def test_I12_common_noun_tied_is_still_masked():
    assert dropped("I work at the Port of Tacoma", ("Port", "ORG")) == []


@pytest.mark.parametrize("text, value, typ", [
    ("Why does `user.getFirstName()` return null?", "user.getFirstName", "PERSON"),
    ("About me: Ana-Maria Popescu, 29, MSc from Politehnica", "MSc", "GPE"),
    ("Write a cover letter for a Data Analyst role at Spotify.", "Data Analyst", "ORG"),
    ("Got a text: 'Chase Alert: Verify $842.19 charge'", "Chase Alert", "PERSON"),
])
def test_I8_junk_phase5(text, value, typ):
    assert rg.is_junk(ent(text, value, typ), text)


@pytest.mark.parametrize("text, value", [
    ("Is Amber or Jade softer? And where does Ruby rank compared to Pearl?", "Pearl"),
    ("How do I bake Madeleine cookies?", "Madeleine"),
    ("लखनऊ में अच्छे कार्डियोलॉजिस्ट कौन हैं?", "अच्छे"),
])
def test_I8_word_names_phase5(text, value):
    assert rg.is_word_name(ent(text, value, "PERSON"), text)


def test_I12_hindi_known_name_is_not_a_word():
    text = "सुनीता को फोन करो"
    assert not rg.is_word_name(ent(text, "सुनीता", "PERSON"), text)


def test_I12_tea_names_are_public():
    text = "Is Earl Grey or Lady Grey better with milk?"
    assert rg.is_public_person(ent(text, "Earl Grey", "PERSON"), text)


@pytest.mark.parametrize("text", [
    "Which roses handle August heat best",
    "What day of the week was 14 June 1987? It's for a history quiz.",
    "Date of entry: 04/12/2019",
    "In the parts catalog, item 219-44-8812 is listed twice.",
])
def test_I8_pattern_near_misses_phase5(text):
    assert ps.scan(text) == []


@pytest.mark.parametrize("text, value", [
    ("my handle is Brightwater", "Brightwater"), ("my SSN is 219-44-8812", "219-44-8812"),
    ("I was born 14 June 1987", "14 June 1987"),
])
def test_I14_pattern_positives_kept(text, value):
    assert value in [e.text for e in ps.scan(text)]


def test_I8_season_is_not_a_german_street():
    added, _ = structural._streets("labs at Georgia Tech in Spring 2026. Lindenstraße 14", [])
    assert [e.text for e in added] == ["Lindenstraße 14"]
