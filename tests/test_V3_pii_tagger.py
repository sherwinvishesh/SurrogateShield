"""V3 §3.3 — the PIITagger's decoder, its training labels and its training data.

Model-free: decoding runs on hand-made token probabilities, label alignment on
hand-made offsets, and the data builder on a small seeded sample.
"""

import random

import pytest

from bench.realdata import identities as ids
from bench.tagger import accept as A
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


def test_a_local_model_is_found_by_name_and_pinned_by_its_weights(tmp_path, monkeypatch):
    import hashlib
    folder = tmp_path / "pii-tagger-x"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"weights")
    pin = "sha256:" + hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setenv(T.MODELS_ENV, str(tmp_path))
    assert T.local_dir("pii-tagger-x") == str(folder) and T.local_dir(str(folder)) == str(folder)
    assert T.local_dir("org/hub-model") is None and T.local_dir("not-there") is None
    T.check_pin(str(folder), pin)
    T.check_pin(str(folder), None)
    with pytest.raises(DetectorUnavailable, match="not the pinned"):
        T.check_pin(str(folder), "sha256:" + "0" * 64)
    with pytest.raises(DetectorUnavailable, match="sha256:<64 hex>"):
        T.check_pin(str(folder), "4b419818330868dff6a60ad3e6b1c730f8b8c0c6")
    with pytest.raises(DetectorUnavailable, match="no weights file"):
        T.check_pin(str(tmp_path), pin)


def test_a_named_model_that_is_not_installed_fails_closed_before_torch(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv(T.MODELS_ENV, str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)              # importing it would raise
    with pytest.raises(DetectorUnavailable, match="no model folder 'pii-tagger-x'.*'classic'"):
        T.TaggerModel("pii-tagger-x", "sha256:" + "0" * 64)
    (tmp_path / "pii-tagger-x").mkdir()
    (tmp_path / "pii-tagger-x" / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(DetectorUnavailable, match="not the pinned"):
        T.TaggerModel("pii-tagger-x", "sha256:" + "0" * 64)


def test_install_copies_a_model_under_its_name_and_checks_the_pin(tmp_path, monkeypatch):
    import hashlib
    from bench.tagger import install as I
    from surrogateshield.core.detection import config as C
    src, root = tmp_path / "run", tmp_path / "models"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "config.json").write_text("{}")
    (src / "train.log").write_text("")
    (src / "checkpoint-10").mkdir()
    with pytest.raises(DetectorUnavailable, match="not the pinned"):
        I.install(src, root=root)                                # the default name needs the pinned weights
    dest = I.install(src, "pii-tagger-x", root=root)
    assert dest == root / "pii-tagger-x"
    assert sorted(p.name for p in dest.iterdir()) == ["config.json", "model.safetensors"]
    assert I.install(src, "pii-tagger-x", root=root) == dest     # the same weights: kept
    (src / "model.safetensors").write_bytes(b"other")
    with pytest.raises(SystemExit, match="holds other weights"):
        I.install(src, "pii-tagger-x", root=root)
    monkeypatch.setattr(C, "PII_TAGGER_REVISION", "sha256:" + hashlib.sha256(b"other").hexdigest())
    assert I.install(src, root=root) == root / C.PII_TAGGER_MODEL
    assert not list(root.glob("*.part"))


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


def test_as_configured_runs_the_benchmark_config_with_its_pinned_weights(tmp_path):
    from surrogateshield.core.detection import config as C
    with pytest.raises(SystemExit, match="needs --ss"):
        E.main(["--model", str(tmp_path), "--as-configured", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="takes no --extra"):
        E.main(["--model", str(tmp_path), "--ss", "--as-configured", "--extra", "x.json",
                "--out", str(tmp_path / "o.json")])
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(SystemExit, match="does not hold the weights"):
        E.with_ss(tmp_path, "dev", "dev", {}, {}, "cpu", True, as_configured=True)
    assert C.from_partial({}, C.benchmark()).config_hash() == C.benchmark().config_hash()


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


def test_acceptance_per_type_pooled_and_spurious():
    def arm(leaked, spurious, n=100):
        return {"leaked_by_type": leaked, "values_by_type": {"ORG": n, "URL": n},
                "leak": {"rate": sum(leaked.values()) / (2 * n)}, "natural_spurious": spurious}
    arms = {"ss": arm({"ORG": 1}, 50), "gliner_pii": arm({"ORG": 5, "URL": 1}, 500),
            "gliner_pii_tuned": arm({"ORG": 4, "URL": 0}, 600)}
    ok = E.acceptance({**arms, "ss_tagger": arm({"ORG": 4, "URL": 2}, 40)})
    assert ok["ok"] and ok["by_type"]["ORG"]["bound"] == 0.04 and ok["by_type"]["URL"]["bound"] == 0.02
    bad = E.acceptance({**arms, "ss_tagger": arm({"ORG": 5, "URL": 3}, 60)})
    assert not bad["ok"] and bad["checks"] == {"types": False, "pooled": True, "spurious": False}
    assert [t for t, v in bad["by_type"].items() if not v["ok"]] == ["ORG", "URL"]


def test_acceptance_pools_the_splits_before_the_bounds():
    """One leaked ORG in 40 fails 0.02 on a split; pooled with a clean split of 60 it passes."""
    def arm(org, values, spurious):
        return {"leaked_by_type": {"ORG": org}, "values_by_type": {"ORG": values},
                "leak": {"k": org, "n": values, "rate": org / values}, "natural_spurious": spurious,
                "natural_edits": spurious}
    def split(org, values):
        return {"ss_tagger": arm(org, values, 10), "ss": arm(org, values, 20),
                "gliner_pii": arm(0, values, 90), "gliner_pii_tuned": arm(0, values, 99)}
    assert not E.acceptance(split(1, 40))["ok"]
    pooled = A.pool([split(1, 40), split(0, 60)])
    assert pooled["ss_tagger"]["values_by_type"] == {"ORG": 100} and pooled["ss"]["natural_spurious"] == 40
    assert pooled["ss_tagger"]["leak"]["rate"] == 0.01 and E.acceptance(pooled)["ok"]


@pytest.mark.heavy                      # loads the shipped tagger (local weights only; skipped when not installed)
def test_the_shipped_tagger_loads_with_its_pinned_weights_and_finds_pii():
    from surrogateshield.core.detection import config as C
    stage = next(s for s in C.benchmark().detectors if s.name == "pii_tagger")
    folder = T.local_dir(stage.model)
    if not folder:
        pytest.skip(f"{stage.model} is not installed under {T.models_dir()}")
    T.check_pin(folder, stage.revision)
    text = "Hi, I'm Priya Raman, write to priya.raman@example.org; I live at 42 Wren Lane, Leeds LS6 2AB and I'm 34."
    found = {(text[c.start:c.end], c.type) for c in T.PIITagger(stage).detect(text)}
    assert {("Priya Raman", "PERSON"), ("priya.raman@example.org", "EMAIL"),
            ("42 Wren Lane, Leeds LS6 2AB", "ADDRESS"), ("34", "AGE")} <= found
