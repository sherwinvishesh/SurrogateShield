"""V3 §3.5: an organisation's name read whole (detection/org_assembly.py,
Pass O) and the gate's ties for it (relation_gate.is_tied).

A firm list ("Quarrie, Ashdown and Pellow"), a legal form ("… GmbH & Co.
KG", "… S.L.N.E", "… e.V."), a family-firm tail ("… and Sons", "… e
Filhos") or a school's kind ("… Middle School") in a field, a work relation
or a sign-off is one ORG, and it replaces the pieces the model stages found
inside it. Model-free; every name here is invented.
"""

import time

import pytest

from surrogateshield.core.detection import org_assembly as O
from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


def _ent(text, value, typ, source="ner"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def _names(text, ents=(), opaque=()):
    added, _ = O.assemble(text, list(ents), opaque)
    return [a.text for a in added]


POSITIVE = [
    # a field for one
    ("Name: Tamsin Orr\nEmployer: Quarrie, Ashdown and Pellow\nPhone: x", "Quarrie, Ashdown and Pellow"),
    ('{"name": "Tamsin Orr", "employer": "Hollen, Brisco and Vane"}', "Hollen, Brisco and Vane"),
    ("Organisation/Employer: Pendle Community College\nAddress: x", "Pendle Community College"),
    ("School: Ranpur Middle School\nGrade: 7", "Ranpur Middle School"),
    ("School/Employer: Arrieta\nWeb: x", "Arrieta"),
    ("| Age: 41 | Org: Penhale and Sons | IP: x", "Penhale and Sons"),
    ("Company name: Ostrander & Klee\n", "Ostrander & Klee"),
    # a work relation
    ("I work at Quarrie, Ashdown and Pellow, my boss is strict", "Quarrie, Ashdown and Pellow"),
    ("I'm Dorian Faye, working at Restrepo e Filhos. Can you help", "Restrepo e Filhos"),
    ("I'm a paralegal at Wilson Sonsini Goodrich & Rosati in town", "Wilson Sonsini Goodrich & Rosati"),
    ("my employer Quarrie and Pellow sent it", "Quarrie and Pellow"),
    # a sign-off
    ("Thanks a lot.\n\n— Tamsin Orr, Hollen, Brisco and Vane", "Hollen, Brisco and Vane"),
    ("Thanks a lot.\n\n— Tamsin Orr | Brack Veltmann GmbH & Co. KG | +49 30", "Brack Veltmann GmbH & Co. KG"),
    ("See you.\n\n— Dorian Faye, Ortez Institute of Technology", "Ortez Institute of Technology"),
    ("Kind regards,\nTamsin Orr\nBlane, Corrow and Hesk\n", "Blane, Corrow and Hesk"),
    ("Cheers,\nPorrin, Mackay and Dunne\n", "Porrin, Mackay and Dunne"),
    ("ok\n\n— van Oordtwijck NV | +31 6 1234", "van Oordtwijck NV"),
    ("Tamsin Orr | Harbel Quist e.V. | +49 30", "Harbel Quist e.V."),
    # the legal form alone
    ("I bought it from Roach Fenning PLC last week", "Roach Fenning PLC"),
    ("Sent by Larue Oduya y asociados S.A. yesterday", "Larue Oduya y asociados S.A."),
    ("Invoice from Casa Merindo S.L.N.E attached", "Casa Merindo S.L.N.E"),
    ("ask Tolland and Sons about it", "Tolland and Sons"),
]


@pytest.mark.parametrize("text,name", POSITIVE)
def test_a_name_is_read_whole(text, name):
    assert name in _names(text)


NEGATIVE = [
    "Compare Microsoft, Apple and Nvidia for me",
    "Tom, Dick and Harry went home",
    "I've lived in London, Paris and Berlin",
    "I live in Reno NV and love it",
    "I'm working at Christmas this year",
    "Employer: Self-employed",
    "Employer: None",
    "Best,\nMike and Sarah",
    "Thanks.\n\n— Tamsin Orr, Seattle",
    "I work at home on Fridays",
    "We met Ostrander and Klee at the fair",
    "Smith AG",
    # people listed after a person, in prose or a citation
    "Thanks to my friends - Maria, Tomas and Natalia.",
    "as shown before (Lund, Okafor-Bell, & Ruiz, 2019, p. 34) the effect holds",
    "We invited Ana Bell, Tom Reyes, and Lia Moreno Castellanos to the panel last week",
    "I work with Tom Reyes and Lia Moreno on this",
]


@pytest.mark.parametrize("text", NEGATIVE)
def test_no_organisation_without_evidence(text):
    assert _names(text) == []


def test_a_sign_off_name_stays_out_of_the_firm():
    assert _names("— Tamsin Orr, Hollen Brisco Ltd") == ["Hollen Brisco Ltd"]


def test_pieces_are_replaced_and_an_edge_crossing_span_is_taken_in():
    text = "Employer: Quarrie, Ashdown and Pellow Group\nPhone: x"
    pieces = [_ent(text, "Quarrie", "PERSON"), _ent(text, "Ashdown", "GPE"),
              _ent(text, "Pellow Group", "ORG", "slm")]
    added, superseded = O.assemble(text, pieces)
    assert [a.text for a in added] == ["Quarrie, Ashdown and Pellow Group"]
    assert added[0].type == "ORG" and added[0].source == O.SOURCE
    assert set(map(id, superseded)) == set(map(id, pieces))


def test_an_unconfirmed_model_span_on_the_whole_name_is_taken_over():
    # a ContextGuard span it was unsure of is never replaced unattended; the
    # name in its slot is
    text = "I work at Ostrander Klee Ltd now"
    unsure = _ent(text, "Ostrander Klee Ltd", "ORG", "slm")
    added, superseded = O.assemble(text, [unsure])
    assert [a.text for a in added] == ["Ostrander Klee Ltd"] and superseded == [unsure]


def test_a_bare_name_after_a_work_cue_needs_a_model_span():
    text = "I'm Ana Bell, working at Fenlon-McBreen. Call me"
    assert _names(text) == []
    assert _names(text, [_ent(text, "Fenlon-McBreen", "ORG", "slm")]) == ["Fenlon-McBreen"]


def test_nothing_inside_a_pattern_entity_or_a_url():
    text = "Employer: Quarrie and Pellow"
    pat = _ent(text, "Quarrie and Pellow", "address", "pattern")
    assert _names(text, [pat]) == []
    assert _names(text, opaque=[(10, len(text))]) == []


def test_a_person_link_before_the_firm():
    text = "Ana Bell from Quarrie, Ashdown and Pellow wrote back"
    assert _names(text) == []
    assert _names(text, [_ent(text, "Ana Bell", "PERSON")]) == ["Quarrie, Ashdown and Pellow"]


@pytest.mark.parametrize("text,person,org", [
    ("Ana Bell, Quarrie, Ashdown and Pellow\nana@x.org", "Ana Bell", "Quarrie, Ashdown and Pellow"),
    ("Ana Bell with Ostrander and Klee asked for the deck", "Ana Bell", "Ostrander and Klee"),
    ("Thanks to my friends - Maria, Tomas and Natalia.", "Maria", None),
    ("as shown before (Lund, Okafor-Bell, & Ruiz, 2019, p. 34) the effect holds", "Lund", None),
    ("We invited Ana Bell, Tom Reyes, and Lia Moreno Castellanos to the panel", "Ana Bell", None),
    ("My two kids and I went with our neighbours and the dog to the coast for a long weekend, "
     "Ana Bell, Quarrie and Pellow came too", "Ana Bell", None),
])
def test_a_person_then_a_comma(text, person, org):
    assert _names(text, [_ent(text, person, "PERSON")]) == ([org] if org else [])


@pytest.mark.parametrize("name,form", [
    ("Seifert Trubin e.V.", True), ("Freitas e Filhos", True), ("Roach PLC", True),
    ("Casa Merindo S.L.N.E", True), ("van Oordtwijck NV", False), ("PLC", False), ("Quarrie", False)])
def test_has_form(name, form):
    assert O.has_form(name) is form


@pytest.mark.parametrize("text,name", [
    ("Company name: Ostrander Klee\n", "Ostrander Klee"),
    ("I was hired by Ostrander Klee last year", "Ostrander Klee"),
    ("Thanks.\n\n— Tamsin Orr | Ostrander Klee | +1 555", "Ostrander Klee"),
])
def test_the_gate_ties_an_org_in_its_slot(text, name):
    org = _ent(text, name, "ORG")
    kept, dropped = rg.gate(text, [org])
    assert kept == [org] and dropped == []


def test_a_field_key_under_a_person_is_no_employer():
    assert O.slot("Client: Ana Bell\nBudget: $25k", 17, end=23) is None


def test_the_gate_keeps_a_name_with_its_legal_form():
    text = "the Harbel Quist e.V. newsletter"
    org = _ent(text, "Harbel Quist e.V.", "ORG", O.SOURCE)
    assert rg.gate(text, [org])[0] == [org]


@pytest.mark.parametrize("name,kept", [
    ("Quarrie, Ashdown and Pellow", (",", " and ")),
    ("Brack Veltmann GmbH & Co. KG", ("GmbH & Co. KG",)),
    ("Casa Merindo S.L.N.E", ("S.L.N.E",)),
    ("Harbel Quist e.V.", ("e.V.",)),
    ("Tolland and Sons", ("and Sons",)),
    ("Larue Oduya y asociados S.A.", ("y asociados S.A.",)),
])
def test_the_surrogate_keeps_the_form_and_changes_the_names(name, kept):
    out = MimicGen(11)._gen_institution(name)
    for k in kept:
        assert k in out
    words = {w.strip(",.") for w in name.split()} - {w.strip(",.") for k in kept for w in k.split()}
    assert not words & {w.strip(",.") for w in out.split()}


# a "Name:" field holds a person whatever type the model gave the value; the
# form's employer list, now one ORG, no longer anchors the message by chance
@pytest.mark.parametrize("text,typ,person", [
    ("Name: Odile Varga\nCity: Brest\nEmployer: Hollen, Brisco and Vane", "LOC", True),
    ("| Age: 41 | Full name: Odile Varga | IP: x", "GPE", True),
    ("- Name: Odile Varga", "LOC", True),
    ("Company name: Odile Varga\n", "ORG", False),
    ("The product name: Odile Varga", "LOC", False),
])
def test_a_name_field_value_is_a_person(text, typ, person):
    from surrogateshield.core.detection.pipeline import _detect_structural_persons
    model = _ent(text, "Odile Varga", typ, "slm")
    new, superseded = _detect_structural_persons(text, [model])
    assert ([e.text for e in new] == ["Odile Varga"] and superseded == [model]) is person


@pytest.mark.parametrize("text", [
    "A" + " A," * 3000,
    "Employer: " + "Ab-" * 3000,
    "Quarrie " * 1500 + "Ltd",
    "& " * 4000,
    "Thanks,\n" + "Ab\n" * 2000,
    "I work at " + "Ab and " * 1500,
    "Employer/" * 1000 + ": Quarrie",
], ids=["commas", "hyphens", "repeats", "ampersands", "closing", "work-and", "slashes"])
def test_adversarial_text_stays_fast(text):
    t0 = time.perf_counter()
    O.assemble(text, [])
    for i in range(0, len(text), max(1, len(text) // 100)):
        O.slot(text, i)
    assert time.perf_counter() - t0 < 0.25
