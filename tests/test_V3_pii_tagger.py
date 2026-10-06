"""V3 §3.3 — the PIITagger's decoder, its training labels and its training data.

Model-free: decoding runs on hand-made token probabilities, label alignment on
hand-made offsets, and the data builder on a small seeded sample.
"""

import random

import pytest

from bench.realdata import identities as ids
from bench.tagger import data as D
from bench.tagger import evaluate as E
from bench.tagger import train as TR
from surrogateshield.core.detection import pii_tagger as T
from surrogateshield.core.errors import DetectorUnavailable


def _onehot(label, p=0.9):
    probs = [(1 - p) / (len(T.LABELS) - 1)] * len(T.LABELS)
    probs[T.LABELS.index(label)] = p
    return probs


def _tokens(text, labelled):
    """``labelled``: ``[(substring, label)]`` in order; each becomes one token."""
    out, at = [], 0
    for piece, label in labelled:
        s = text.index(piece, at)
        out.append((s, s + len(piece), _onehot(label)))
        at = s + len(piece)
    return out


def _found(cands, text):
    return [(text[c.start:c.end], c.type) for c in cands]


# ── decode ───────────────────────────────────────────────────────────────────

def test_bio_spans_and_scores():
    text = "I am Ana Ruiz, 34."
    toks = _tokens(text, [("I", "O"), ("am", "O"), ("Ana", "B-PERSON"), ("Ruiz,", "I-PERSON"), ("34.", "B-AGE")])
    got = T.decode(text, toks, T.LABELS)
    assert _found(got, text) == [("Ana Ruiz", "PERSON"), ("34", "AGE")]
    assert all(0.89 < c.score <= 1.0 for c in got)


def test_an_i_of_another_type_starts_a_new_value():
    text = "Ana Acme"
    toks = _tokens(text, [("Ana", "B-PERSON"), ("Acme", "I-ORG")])
    assert _found(T.decode(text, toks, T.LABELS), text) == [("Ana", "PERSON"), ("Acme", "ORG")]


def test_an_i_after_o_starts_a_new_value():
    text = "and Ruiz"
    toks = _tokens(text, [("and", "O"), ("Ruiz", "I-PERSON")])
    assert _found(T.decode(text, toks, T.LABELS), text) == [("Ruiz", "PERSON")]


def test_address_parts_join_into_one_address():
    text = "Send it to 12 Rue Lafayette, Lyon 69001 please"
    toks = _tokens(text, [("Send", "O"), ("it", "O"), ("to", "O"), ("12", "B-STREET"), ("Rue", "I-STREET"),
                          ("Lafayette,", "I-STREET"), ("Lyon", "B-CITY"), ("69001", "B-POSTCODE"), ("please", "O")])
    assert _found(T.decode(text, toks, T.LABELS), text) == [("12 Rue Lafayette, Lyon 69001", "ADDRESS")]


def test_a_lone_city_is_a_location_and_a_lone_region_is_dropped():
    text = "I live in Lyon, Rhône and work in Ontario"
    toks = _tokens(text, [("Lyon,", "B-CITY"), ("Rhône", "B-REGION"), ("Ontario", "B-REGION")])
    assert _found(T.decode(text, toks, T.LABELS), text) == [("Lyon", "LOCATION")]


def test_parts_with_words_between_stay_apart():
    text = "12 Main St in Springfield"
    toks = _tokens(text, [("12", "B-STREET"), ("Main", "I-STREET"), ("St", "I-STREET"), ("Springfield", "B-CITY")])
    assert _found(T.decode(text, toks, T.LABELS), text) == [("12 Main St", "ADDRESS"), ("Springfield", "LOCATION")]


def test_merge_address_off_drops_the_parts():
    text = "12 Main St"
    toks = _tokens(text, [("12", "B-STREET"), ("Main", "I-STREET"), ("St", "I-STREET")])
    assert T.decode(text, toks, T.LABELS, merge_address=False) == []


def test_address_score_is_length_weighted():
    text = "12 Main St, Lyon"
    toks = [(0, 2, _onehot("B-STREET", .9)), (3, 7, _onehot("I-STREET", .9)), (8, 11, _onehot("I-STREET", .9)),
            (12, 16, _onehot("B-CITY", .6))]
    (c,) = T.decode(text, toks, T.LABELS)
    assert c.type == "ADDRESS" and c.score == pytest.approx((3 * .9 + .6) / 4, abs=0.01)


# ── _trim ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, tag, want", [
    ("Ana Ruiz.", "PERSON", "Ana Ruiz"),
    ("Acme S.A.R.L.", "ORG", "Acme S.A.R.L."),
    ("Acme Ltd.", "ORG", "Acme Ltd."),
    ("Ruiz Jr.", "PERSON", "Ruiz Jr."),
    ('"Ana Ruiz"', "PERSON", "Ana Ruiz"),
    ("**ana@example.org**", "EMAIL", "ana@example.org"),
    ("(555) 010-2030)", "PHONE", "(555) 010-2030"),
    ("(ana_r", "HANDLE", "ana_r"),
    ("hunter2!?", "CREDENTIAL", "hunter2!?"),
    ("*pa55*word", "CREDENTIAL", "*pa55*word"),
    ("  Lyon,\n", "LOCATION", "Lyon"),
    ("'Ana", "PERSON", "Ana"),
])
def test_trim(text, tag, want):
    s, e = T._trim(text, 0, len(text), tag)
    assert text[s:e] == want


def test_trim_of_nothing_is_empty():
    s, e = T._trim("  .,  ", 0, 6, "PERSON")
    assert e <= s


# ── training labels ──────────────────────────────────────────────────────────

def _labels(text, spans, offsets):
    return [TR.IGNORE if i == TR.IGNORE else T.LABELS[i]
            for i in TR.token_labels(text, spans, offsets, TR.char_owner(text, spans))]


def test_token_labels_bio():
    text = "Hi Ana Ruiz!"
    spans = [[3, 11, "PERSON"]]
    offsets = [(0, 0), (0, 2), (2, 6), (6, 11), (11, 12), (0, 0)]
    assert _labels(text, spans, offsets) == [TR.IGNORE, "O", "B-PERSON", "I-PERSON", "O", TR.IGNORE]


def test_a_line_break_token_inside_a_value_continues_it():
    text = "Acme\nCorp"
    spans = [[0, 9, "ORG"]]
    assert _labels(text, spans, [(0, 4), (4, 5), (5, 9)]) == ["B-ORG", "I-ORG", "I-ORG"]


def test_a_line_break_token_before_a_value_is_outside():
    text = "Hi\nAna"
    spans = [[3, 6, "PERSON"]]
    assert _labels(text, spans, [(0, 2), (2, 3), (3, 6)]) == ["O", "O", "B-PERSON"]


def test_a_character_split_over_two_tokens_has_one_b():
    text = "Óscar"
    spans = [[0, 5, "PERSON"]]
    assert _labels(text, spans, [(0, 1), (0, 1), (1, 5)]) == ["B-PERSON", "I-PERSON", "I-PERSON"]


def test_adjacent_values_each_start_with_b():
    text = "Ana Ruiz"
    spans = [[0, 3, "PERSON"], [4, 8, "PERSON"]]
    assert _labels(text, spans, [(0, 3), (3, 8)]) == ["B-PERSON", "B-PERSON"]


def test_labels_agree_between_builder_trainer_and_product():
    assert D.TAGS == T.TAGS and D.LABELS == T.LABELS
    assert set(TR.LABEL2ID) == set(T.LABELS)


# ── training data ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample():
    return D.build(250, seed=4242)


def test_build_is_deterministic(sample):
    assert D.data_hash(D.build(250, seed=4242)) == D.data_hash(sample)
    assert D.data_hash(D.build(250, seed=4243)) != D.data_hash(sample)


def test_spans_are_clean_and_in_bounds(sample):
    for r in sample:
        prev = 0
        for s, e, t in r["spans"]:
            v = r["text"][s:e]
            assert 0 <= prev <= s < e <= len(r["text"]) and t in D.TAGS, (r["id"], s, e, t)
            assert v == v.strip(), (r["id"], t)
            prev = e


def test_values_are_train_pool_only(sample):
    for r in sample:
        assert ids.in_pool(r["meta"]["pool_tokens"], "train"), r["id"]


def test_no_held_out_format_and_no_provider_token_shape(sample):
    for r in sample:
        assert not any(tuple(v) in D.HELD_OUT for v in r["meta"]["values"]), r["id"]
        assert not ids.FORBIDDEN.search(r["text"]), r["id"]


def test_the_sample_has_negatives_and_values(sample):
    assert any(not r["spans"] for r in sample) and sum(bool(r["spans"]) for r in sample) > 150


# ── the stage and the evaluator ──────────────────────────────────────────────

def test_the_stage_needs_a_model():
    from surrogateshield.core.detection.config import Stage
    with pytest.raises(DetectorUnavailable):
        T.PIITagger(Stage("pii_tagger", enabled=True))


@pytest.mark.heavy                      # imports torch and transformers
def test_a_missing_model_is_unavailable(tmp_path):
    T._models.clear()
    with pytest.raises(DetectorUnavailable):
        T.get_model(str(tmp_path / "nowhere"), None, "cpu")


def test_edits_at_thresholds():
    cands = [[0, 3, "PERSON", .55], [4, 8, "AGE", .95], [9, 12, "ORG", .45]]
    assert [e[2] for e in E.edits_at(cands, .5)] == ["PERSON", "AGE"]
    assert [e[2] for e in E.edits_at(cands, {"PERSON": .6, "*": .4})] == ["AGE", "ORG"]
    assert E.edits_at(cands, .5, types={"AGE"}) == [[4, 8, "AGE", "[AGE]"]]


def test_splits_keep_a_source_conversation_together():
    from bench.tagger import splits as S
    a = {"id": "x-inj", "source_id": "conv-17"}
    b = {"id": "conv-17"}
    assert S.half("oasst1", a) == S.half("oasst1", b)
    halves = {S.half("oasst1", {"id": f"c{i}"}) for i in range(64)}
    assert halves == set(S.HALVES)


def test_the_tagger_is_the_builtin_pii_tagger_detector():
    from surrogateshield.core.detection import plugins
    from surrogateshield.core.detection.config import Stage
    with pytest.raises(DetectorUnavailable, match="needs a model"):
        plugins.get_detector(Stage("pii_tagger", enabled=True))
    assert "pii_tagger" not in plugins.registered()


def test_a_registered_pii_tagger_replaces_the_builtin():
    from surrogateshield.core.detection import plugins
    from surrogateshield.core.detection.config import Stage

    class Fixed:
        def detect(self, text, view):
            return [plugins.Candidate(0, 3, "PERSON", 0.9)]

    plugins.register_detector("pii_tagger", lambda stage: Fixed())
    try:
        assert plugins.get_detector(Stage("pii_tagger", enabled=True)).detect("Ana", None)[0].type == "PERSON"
    finally:
        plugins.unregister_detector("pii_tagger")


def test_an_extra_config_merges_onto_the_tagger_stage():
    from pathlib import Path
    extra = {"type_sources": {"PHONE": ["pattern_scan", "pii_tagger"]},
             "detectors": [{"name": "pii_tagger", "thresholds": {"PHONE": 0.9}, "options": {"window": 128}},
                           {"name": "context_guard", "enabled": False}]}
    cfg = E.tagger_config(Path("m"), "cpu", {"PERSON": 0.5}, extra)
    tagger, other = cfg["detectors"]
    assert tagger["thresholds"] == {"PERSON": 0.5, "PHONE": 0.9}
    assert tagger["options"] == {"device": "cpu", "window": 128} and tagger["enabled"] is True
    assert other == {"name": "context_guard", "enabled": False}
    assert cfg["type_sources"] == {"PHONE": ["pattern_scan", "pii_tagger"]}
