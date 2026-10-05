"""Audit I1 / I12, gate J2 — model spans that cannot be edited.

A ContextGuard span that runs across a line break comes back with the break
normalised to a space; its text no longer matches the message, so no edit was
planned and the name went out unmasked (dev2 "University of Leicester").
A lower-case clause tagged PERSON on non-English text is not a name.
Model-free.
"""

from surrogateshield.core.detection import pipeline, relation_gate
from surrogateshield.core.entities import DetectedEntity, plan_substitutions


def test_I1_model_span_across_a_line_break_is_reanchored():
    text = "Dept. of Microbiology, University of Leicester\nORCID 0000-0002-8814-3307"
    s = text.index("University")
    ent = DetectedEntity("University of Leicester ORCID", s, s + len("University of Leicester\nORCID"),
                         "ORG", 0.8, "slm")
    (out,) = pipeline._anchor_to_lines([ent], text)
    assert out.text == "University of Leicester" and text[out.start:out.end] == out.text


def test_I12_lowercase_clause_is_not_a_person():
    clause = DetectedEntity("je veux vérifier la clé", 0, 23, "PERSON", 0.88, "ner")
    assert relation_gate.is_junk(clause)
    for name in ("maria de la cruz", "jean du plessis", "ahmed bin rashid", "brendon"):
        assert not relation_gate.is_junk(DetectedEntity(name, 0, len(name), "PERSON", 0.9, "ner"))


def test_I1_reanchored_span_gets_an_edit():
    text = "Fellow | Dept. of Microbiology, University of Leicester\nORCID 0000"
    s = text.index("University")
    raw = DetectedEntity("University of Leicester ORCID", s, s + 29, "ORG", 0.8, "slm")
    assert plan_substitutions(text, [raw], {raw.text: "X"}) == []      # the old leak
    (ent,) = pipeline._anchor_to_lines([raw], text)
    assert plan_substitutions(text, [ent], {ent.text: "X"}) == [
        (ent.start, ent.end, "University of Leicester", "X")]


def test_I1_pattern_spans_pass_through_unchanged():
    text = "call 555-0101\nnow"
    ent = DetectedEntity("555-0101", 5, 13, "phone_us", 1.0, "pattern")
    assert pipeline._anchor_to_lines([ent], text) == [ent]
