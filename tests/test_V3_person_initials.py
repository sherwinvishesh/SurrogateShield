"""Initials in a name (campaign E5b): a surname the tagger reads alone after
initials ("A.A. " + "Rowe"), and an initial read as its own PERSON with its
period left in the gap ("J" + ". " + "Rowe"), come out as one name.
Model-free; names and texts are invented."""

import pytest

from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity


def _ent(text, value, typ="PERSON", source="pii_tagger"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


@pytest.mark.parametrize("text, pieces, whole", [
    ("J. Rowe signed it", ["J", "Rowe"], "J. Rowe"),
    ("A.A. Rowe, B. Lane, C.C. Vance", ["A.A", "Rowe"], "A.A. Rowe"),
    ("Cauã I. Pacheco signed it", ["Cauã", "Pacheco"], "Cauã I. Pacheco"),       # unchanged
])
def test_an_initial_and_the_surname_merge(text, pieces, whole):
    ents = [_ent(text, p) for p in pieces]
    out = pipeline._merge_adjacent_persons(ents, text)
    assert [e.text for e in out] == [whole]


@pytest.mark.parametrize("text, pieces", [
    ("I met Ana. Rowe came later", ["Ana", "Rowe"]),          # a sentence ends between them
    ("plan A. Rowe will present", ["plan A", "Rowe"]),         # not an initial
])
def test_a_sentence_boundary_is_not_an_initial(text, pieces):
    ents = [_ent(text, p) for p in pieces]
    out = pipeline._merge_adjacent_persons(ents, text)
    assert sorted(e.text for e in out) == sorted(pieces)


@pytest.mark.parametrize("text, found, whole", [
    ("A.A. Rowe, B. Lane", "Rowe", "A.A. Rowe"),
    ("J. K. Rowe wrote it", "Rowe", "J. K. Rowe"),
    ("by R.L. Vance\n", "Vance", "R.L. Vance"),
])
def test_a_surname_grows_back_over_its_initials(text, found, whole):
    out = pipeline._complete_person_spans([_ent(text, found)], text)
    assert [e.text for e in out] == [whole]


@pytest.mark.parametrize("text, found", [
    ("Dr. Rowe will see you", "Rowe"),              # a title, not an initial
    ("the item is a. Rowe said so", "Rowe"),        # lower-case letter
    ("N.A.S.A. Rowe", "Rowe"),                      # more than three initials
    ("vitamin D.Rowe", "Rowe"),                     # no space before the surname
])
def test_what_is_not_initials_is_left(text, found):
    out = pipeline._complete_person_spans([_ent(text, found)], text)
    assert [e.text for e in out] == [found]


def test_initials_taken_by_another_entity_are_not_claimed_twice():
    text = "A.A. Rowe, B. Lane"
    other = _ent(text, "A.A.", typ="ORG", source="pattern")
    out = pipeline._complete_person_spans([_ent(text, "Rowe")], text, others=[other])
    assert [e.text for e in out] == ["Rowe"]


def test_both_passes_read_initials_and_surname_as_one_name():
    text = "Cc: A.A. Rowe, J. Lane and the rest of the team."
    found = [_ent(text, "Rowe"), _ent(text, "J"), _ent(text, "Lane")]
    # the two passes in the order run_cascade applies them after the model stages
    merged = pipeline._merge_adjacent_persons(found, text)
    grown = pipeline._complete_person_spans(merged, text)
    assert sorted(e.text for e in grown) == ["A.A. Rowe", "J. Lane"]
