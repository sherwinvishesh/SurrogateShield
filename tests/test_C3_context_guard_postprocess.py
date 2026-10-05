"""Audit C3 — ContextGuard's handling of NER output, model-free: a fake NER
returns the dicts a HuggingFace pipeline with aggregation would. Every entity
it emits must satisfy text[start:end] == entity.text, including when the
model's word is not in the text ([UNK] for a character outside the vocab).
"""

import pytest

from surrogateshield.core.detection import context_guard as cg
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.errors import DetectorUnavailable


def run(monkeypatch, text, results, **kw):
    monkeypatch.setattr(cg, "_get_ner", lambda *_a, **_k: (lambda _t: results))
    confirmed, uncertain = cg.guard(text, [], **kw)
    for e in confirmed + uncertain:
        assert text.replace("█", " ")[e.start:e.end] == e.text, e
    return [(e.type, e.text) for e in confirmed], [(e.type, e.text) for e in uncertain]


def r(word, start, end, label="PER", score=0.95):
    return {"entity_group": label, "word": word, "start": start, "end": end, "score": score}


def test_C3_word_found_is_confirmed_by_score(monkeypatch):
    t = "ask Priya Nair about it"
    assert run(monkeypatch, t, [r("Priya Nair", 4, 14)]) == ([("PERSON", "Priya Nair")], [])
    assert run(monkeypatch, t, [r("Priya Nair", 4, 14, score=0.5)]) == ([], [("PERSON", "Priya Nair")])


def test_C3_fragment_expands_to_whole_words(monkeypatch):
    t = "email p@x.com ask Zuberi Nkosi now"
    s = t.index("ri Nkosi")
    assert run(monkeypatch, t, [r("ri Nkosi", s, s + 8)])[0] == [("PERSON", "Zuberi Nkosi")]


def test_C3_fragment_of_a_label_word_is_dropped(monkeypatch):
    t = "Client: see the notes"
    assert run(monkeypatch, t, [r("lient", 1, 6, "ORG")]) == ([], [])


@pytest.mark.parametrize("word, label", [
    ("Al", "PER"),                 # too short
    ("Mobile", "ORG"),             # blocklisted label word
    ("Dana\nSmith", "PER"),        # spans a line break
    ("Tuesday", "MISC"),           # label not kept
])
def test_C3_rejected_outputs(monkeypatch, word, label):
    t = f"x {word} y"
    assert run(monkeypatch, t, [r(word, 2, 2 + len(word), label)]) == ([], [])


def test_C3_word_not_in_text_uses_the_model_offsets(monkeypatch):
    # "Ǯofi Mensah": the model emits "[UNK]ofi Mensah", which is nowhere in the
    # text; the entity used to keep that word with the offsets of the real name
    t = "call Ǯofi Mensah today"
    s = t.index("Ǯofi")
    conf, _ = run(monkeypatch, t, [r("[UNK]ofi Mensah", s, s + 11)])
    assert conf == [("PERSON", "Ǯofi Mensah")]


@pytest.mark.parametrize("text, word, want", [
    # the model joins the lines with a space, so the word is not in the text
    ("From Bartholomew Ekwueme\nCell 555-0100", "Bartholomew Ekwueme Cell", "Bartholomew Ekwueme"),
    ("Lecturer, University of Leicester\nORCID 0000", "University of Leicester ORCID",
     "University of Leicester"),
])
def test_C3_span_across_a_line_break_keeps_its_main_line(monkeypatch, text, word, want):
    # a regression of the fix above dropped these, and the name went out
    s = text.index(want.split()[0])
    conf, _ = run(monkeypatch, text, [r(word, s, s + len(word))])
    assert [t for _, t in conf] == [want]


def test_C3_offsets_outside_the_text_are_dropped(monkeypatch):
    assert run(monkeypatch, "short", [r("Nobody Here", 40, 51)]) == ([], [])


def test_C3_placeholders_keep_offsets(monkeypatch):
    t = "██████ met Dana Whitfield"
    s = t.index("Dana")
    assert run(monkeypatch, t, [r("Dana Whitfield", s, s + 14)])[0] == [("PERSON", "Dana Whitfield")]


def test_C3_borderline_and_disabled(monkeypatch):
    hi = DetectedEntity("Tempe", 0, 5, "GPE", 0.85, "spacy")
    lo = DetectedEntity("Lake", 6, 10, "LOC", 0.5, "spacy")
    monkeypatch.setattr(cg, "_get_ner", lambda *_a, **_k: pytest.fail("model called"))
    assert cg.guard("Tempe Lake", [hi, lo], enabled=False) == ([hi], [lo])
    assert cg.guard("█████", [], enabled=True) == ([], [])          # nothing left to read


def test_C3_model_error_fails_closed(monkeypatch):
    def ner(_t):
        raise RuntimeError("Already borrowed")
    monkeypatch.setattr(cg, "_get_ner", lambda *_a, **_k: ner)
    with pytest.raises(DetectorUnavailable):
        cg.guard("Dana Whitfield", [])
