"""Campaign phase 2, N4: an untied place or organisation is kept for an
identifier near it, not for one anywhere in the message. Model-free; the
names, places and numbers are invented."""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def ent(text, span, kind, source="ner", score=0.85):
    i = text.index(span)
    return DetectedEntity(span, i, i + len(span), kind, score, source)


ESSAY = ("The castle above Harrogate was rebuilt twice, the second time after the fire of "
         "1740, and the gardens were laid out a century later by a committee that argued "
         "for years about the fountain. The parish records say little about who paid for "
         "any of it, and the one surviving ledger stops in 1802.")
assert len(ESSAY) > rg._ANCHOR_REACH


def gate(text, place, *context, vouched=()):
    reasons = {}
    kept, dropped = rg.gate(text, [place], list(context), reasons=reasons, vouched=list(vouched))
    return [e.text for e in kept], [reasons[id(e)] for e in dropped]


def test_a_name_at_the_end_of_an_essay_does_not_make_its_places_the_writer_s():
    text = ESSAY + "\n\nThanks for reading,\nAna Ruiz"
    place = ent(text, "Harrogate", "GPE")
    assert gate(text, place, ent(text, "Ana Ruiz", "PERSON")) == ([], ["untied"])


def test_a_name_in_the_same_paragraph_still_does():
    text = "Ana Ruiz wrote a short piece on the castle above Harrogate, and " + ESSAY[18:]
    place = ent(text, "Harrogate", "GPE")
    assert gate(text, place, ent(text, "Ana Ruiz", "PERSON")) == (["Harrogate"], [])


def test_a_name_in_the_next_lines_still_does():
    text = "Harrogate, Tuesday\n\nAna Ruiz"
    place = ent(text, "Harrogate", "GPE")
    assert gate(text, place, ent(text, "Ana Ruiz", "PERSON")) == (["Harrogate"], [])


def test_a_name_a_paragraph_away_does_not():
    text = ("Harrogate, Tuesday\n\nThe committee met again and argued for most of the "
            "afternoon about the fountain, the ledger and the parish.  Ana Ruiz")
    place = ent(text, "Harrogate", "GPE")
    assert gate(text, place, ent(text, "Ana Ruiz", "PERSON")) == ([], ["untied"])


def test_a_phone_number_near_the_place_keeps_it_even_across_a_blank_line():
    text = "The castle above Harrogate was rebuilt twice.\n\nCall me on 0161 496 0123."
    place = ent(text, "Harrogate", "GPE")
    phone = ent(text, "0161 496 0123", "phone_uk", source="pattern", score=1.0)
    assert gate(text, place, phone) == (["Harrogate"], [])


def test_a_phone_number_far_from_the_place_does_not():
    text = ESSAY + "\n\nCall me on 0161 496 0123."
    place = ent(text, "Harrogate", "GPE")
    phone = ent(text, "0161 496 0123", "phone_uk", source="pattern", score=1.0)
    assert gate(text, place, phone) == ([], ["untied"])


def test_an_organisation_is_held_to_the_same_reach():
    text = ESSAY.replace("castle above Harrogate", "Pennine Trust's castle") + "\n\n0161 496 0123"
    org = ent(text, "Pennine Trust", "ORG", source="pii_tagger", score=0.8)
    phone = ent(text, "0161 496 0123", "phone_uk", source="pattern", score=1.0)
    assert gate(text, org, phone) == ([], ["untied"])


def test_a_vouched_place_needs_no_anchor():
    text = ESSAY
    place = ent(text, "Harrogate", "GPE", source="pii_tagger", score=0.99)
    assert gate(text, place, vouched=[place]) == (["Harrogate"], [])
    assert gate(text, place) == ([], ["untied"])


@pytest.mark.parametrize("gap, near", [
    ("x" * rg._ANCHOR_REACH, True),
    ("x" * (rg._ANCHOR_REACH + 1), False),
    ("x" * (rg._NAME_REACH_ACROSS - 2) + "\n\n", True),
    ("\n\n" + "x" * (rg._NAME_REACH_ACROSS - 1), False),
])
def test_the_reach_is_measured_in_characters(gap, near):
    text = "Harrogate" + gap + "Ana Ruiz"
    assert rg._anchor_near(ent(text, "Harrogate", "GPE"), text, [ent(text, "Ana Ruiz", "PERSON")]) is near
