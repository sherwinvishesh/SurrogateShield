"""Audit I8 / I13 — Pass S, structure- and frame-driven detection: lowercase
and non-English places after a residence cue, zh/hi name frames, surname
particles, intro prefixes, name components, CSV columns, chat speakers,
nicknames after a verb, payees and European streets. Model-free: the place
verifier is a stub.
"""

import re

import pytest

from surrogateshield.core.detection import structural as st
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def ent(text, value, typ="PERSON", source="ner"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def added(text, ents=(), verify=None, **kw):
    a, _ = st.detect(text, list(ents), place_verifier=verify, **kw)
    for e in a:
        assert text[e.start:e.end] == e.text
    return [(e.type, e.text) for e in a]


# ── places ───────────────────────────────────────────────────────────────────

def test_I8_lowercase_city_after_person_from():
    text = "hi im sarah mitchell from austin, my email is x"
    assert added(text, [ent(text, "sarah mitchell")]) == [("GPE", "austin")]


def test_I8_lowercase_place_uses_verifier_off_gazetteer():
    text = "she lives alone in flagstaff so i worry"
    assert added(text) == []                                   # no gazetteer hit
    calls = []

    def verify(sentence, s, e):
        calls.append(sentence[s:e])
        return sentence[s:e] == "flagstaff"
    assert added(text, verify=verify) == [("GPE", "flagstaff")]
    assert calls == ["flagstaff"]
    text = "i work night shifts at the warehouse in kent and"
    assert added(text, verify=lambda s, a, b: s[a:b] == "kent") == [("GPE", "kent")]


@pytest.mark.parametrize("text", [
    "how do i stop my iphone from autocorrecting 'ill' to 'I'll'",
    "she lives in fear of her landlord",
    "we moved to the suburbs",
    "data comes from the api",
])
def test_I8_residence_cue_negatives(text):
    assert added(text, verify=lambda *_: True) == []


@pytest.mark.parametrize("text, place", [
    ("Me llamo Carlos, vivo en Sevilla y mi correo", "Sevilla"),
    ("Olá, moro em Belo Horizonte há dez anos", "Belo Horizonte"),
    ("Ich wohne in Köln seit 2019", "Köln"),
    ("我叫王伟，电话是13812345678，住在上海市浦东新区。", "上海市浦东新区"),
    ("मेरा नाम राहुल वर्मा है और मैं पुणे में रहता हूँ।", "पुणे"),
])
def test_I13_foreign_residence_frames(text, place):
    assert ("GPE", place) in added(text)


def test_I13_service_query_mode_skips_places():
    assert added("vivo en Sevilla", skip_locations=True) == []


@pytest.mark.parametrize("text, name", [
    ("我叫王伟，电话是13812345678", "王伟"),
    ("मेरा नाम राहुल वर्मा है और", "राहुल वर्मा"),
])
def test_I13_zh_hi_name_frames(text, name):
    assert ("PERSON", name) in added(text)


# ── person spans ─────────────────────────────────────────────────────────────

def test_I13_intro_prefix_trimmed():
    text = "Bonjour, je m'appelle Élodie Marchand, j'habite"
    a, r = st.detect(text, [ent(text, "je m'appelle Élodie Marchand")])
    assert [(e.type, e.text) for e in a] == [("PERSON", "Élodie Marchand")]
    assert [e.text for e in r] == ["je m'appelle Élodie Marchand"]


def test_I13_particle_surname_merged():
    text = "Siobhan O'Brien and Jean-Luc van der Berg will attend."
    a, r = st.detect(text, [ent(text, "Jean-Luc"), ent(text, "der Berg", source="slm")])
    assert [e.text for e in a] == ["Jean-Luc van der Berg"]
    assert sorted(e.text for e in r) == ["Jean-Luc", "der Berg"]


def test_I8_name_components_propagate():
    text = ("From: Daniel Kowalczyk <d@x.com>\nTo: Hannah Reyes <h@x.com>\n\n"
            "Hannah, approved. Can you help me thank Daniel?")
    out = added(text, [ent(text, "Daniel Kowalczyk"), ent(text, "Hannah Reyes")])
    assert out == [("PERSON", "Daniel"), ("PERSON", "Hannah")] or sorted(out) == [
        ("PERSON", "Daniel"), ("PERSON", "Hannah")]


def test_I8_ambiguous_components_not_propagated():
    text = "Will Turner said the meeting will move. Will is late."
    assert added(text, [ent(text, "Will Turner")]) == []


# ── structure ────────────────────────────────────────────────────────────────

def test_I8_csv_columns_typed_and_retyped():
    text = ("first,last,email,phone,zip\n"
            "Marcus,Bell,mbell88@yahoo.com,602-555-0143,85004\n"
            "leah,okonkwo,leah.ok@outlook.com,480 555 0199,85251\n")
    email = ent(text, "mbell88@yahoo.com", "email", "pattern")
    bell = ent(text, "Bell", "GPE")
    a, r = st.detect(text, [email, bell])
    got = [(e.type, e.text) for e in a]
    assert ("PERSON", "Bell") in got and ("PERSON", "okonkwo") in got
    assert ("zip_us", "85251") in got and ("PERSON", "leah") in got
    assert r == [bell]
    assert not any(e.text == email.text for e in a)            # pattern span kept


def test_I8_form_labels_are_not_speakers_or_csv():
    text = "NAME: ROSA DELGADO\nDOB: 11/02/1987\nPHONE: (602) 555-0187\n"
    assert added(text) == []


def test_I8_chat_speakers():
    text = ("[10/02/26, 8:14 PM] tia: girl where r u\n"
            "[10/02/26, 8:15 PM] me: omw!!\n"
            "[10/02/26, 8:15 PM] tia: k im at table 12\n")
    assert added(text) == [("PERSON", "tia"), ("PERSON", "tia")]


def test_I8_nickname_after_verb():
    assert added("can you DM Kev the PDF before standup?") == [("PERSON", "Kev")]
    assert added("text Mom and ask Google Maps for the route") == []


def test_I8_payee_after_payment_verb():
    text = "2026-09-02,VENMO PAYMENT TO MARISOL ORTEGA,-120.00"
    assert added(text) == [("PERSON", "MARISOL ORTEGA")]
    assert added("ZELLE PAYMENT TO ACME CORP") == []


@pytest.mark.parametrize("text, value", [
    ("j'habite au 14 rue des Lilas à Nantes", "14 rue des Lilas"),
    ("meu endereço para Rua Augusta 1508, apto 52, São Paulo", "Rua Augusta 1508"),
    ("meu endereço para Rua Augusta 1508, apto 52, São Paulo", "apto 52"),
    ("Ich wohne in der Hauptstraße 5 in Köln", "Hauptstraße 5"),
])
def test_I13_european_streets(text, value):
    assert ("address", value) in added(text)


@pytest.mark.parametrize("seed", range(10))
def test_I13_european_street_surrogate_keeps_shape(seed):
    for value in ("14 rue des Lilas", "Rua Augusta 1508", "Hauptstraße 5"):
        out = MimicGen(seed=seed).generate(
            DetectedEntity(value, 0, len(value), "address", 1.0, "structural"),
            address_mode="replace")
        assert out != value
        assert re.sub(r"\d", "0", out).count("0") == re.sub(r"\d", "0", value).count("0")
        assert "Lilas" not in out and "Augusta" not in out and "Haupt" not in out


@pytest.mark.parametrize("text", [
    "my flat is located in postcode AB1 2CD",
    "she moved to zip 12345 last year",
    "we live in the north of the county",
])
def test_I8_field_nouns_are_not_places(text):
    assert added(text, verify=lambda *_: True) == []


def test_I13_wired_into_cascade_model_free():
    from surrogateshield.core.detection import pipeline
    text = ("first,last,zip\nleah,okonkwo,85251\n"
            "j'habite au 14 rue des Lilas. Ping Kev about it.\n"
            "the basement flat at Storgata 41B, Lillehammer")
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False)
    got = {(e.type, e.text, e.source) for e in conf}
    assert ("PERSON", "okonkwo", "structural") in got
    # a French layout is PatternScan's now (address_assembly); a weak
    # Nordic ending ("-gata") is still left to the structural pass
    assert ("address", "14 rue des Lilas", "pattern") in got
    assert ("address", "Storgata 41B", "structural") in got
    assert ("PERSON", "Kev", "structural") in got
    # service-query mode keeps place cues off
    conf, _ = pipeline.run_cascade("vivo en Sevilla", skip_location_entities=True,
                                   use_entity_trace=False, use_context_guard=False, use_tagger=False)
    assert not [e for e in conf if e.source == "structural"]
