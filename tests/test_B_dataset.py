"""Audit B1–B6, gate J11: the generated evaluation set and its lint."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiment"))

import lint_key  # noqa: E402
import make_dataset  # noqa: E402

import eval_metrics as em  # noqa: E402

SPLITS = ("dev", "test")


@pytest.fixture(scope="module")
def committed():
    return {s: json.loads((ROOT / f"experiment/synth_{s}_key.json").read_text(encoding="utf-8"))
            for s in SPLITS}


@pytest.fixture(scope="module")
def small():
    return make_dataset.generate(200, seed=3, strict=False)


def _lint(data):
    return lint_key.lint_generated({f"{k}.json": v for k, v in data.items()})


def test_J11_committed_set_is_reproducible(tmp_path):
    manifest = json.loads((ROOT / "experiment/synth_manifest.json").read_text())
    data = make_dataset.generate(manifest["n"], manifest["seed"])
    make_dataset.write(tmp_path, data, {k: manifest[k] for k in ("script", "version", "seed", "n")})
    for name in ("synth_dev_key.json", "synth_test_key.json", "synth_dev.json",
                 "synth_test.json", "synth_manifest.json"):
        assert (tmp_path / name).read_bytes() == (ROOT / "experiment" / name).read_bytes(), name


def test_J11_deterministic_under_seed():
    assert make_dataset.generate(120, 5, strict=False) == make_dataset.generate(120, 5, strict=False)
    assert make_dataset.generate(120, 5, strict=False) != make_dataset.generate(120, 6, strict=False)


def test_J11_committed_set_lints_clean(committed):
    assert _lint(committed) == []


def test_J11_runner_input_matches_key(committed):
    for s in SPLITS:
        inputs = json.loads((ROOT / f"experiment/synth_{s}.json").read_text(encoding="utf-8"))
        assert [r["input"] for r in inputs] == [r["Question"] for r in committed[s]]


def test_B1_identity_pool_never_reused(committed):
    ids = [r["identity_id"] for s in SPLITS for r in committed[s] if r["identity_id"] is not None]
    assert len(ids) == len(set(ids))
    names = [sp["value"].casefold() for s in SPLITS for r in committed[s] for sp in r["Spans"]
             if sp["type"] == "name" and " " in sp["value"]]
    first_row = {}
    for s in SPLITS:
        for r in committed[s]:
            for sp in r["Spans"]:
                if sp["type"] == "name" and " " in sp["value"]:
                    assert first_row.setdefault(sp["value"].casefold(), r["id"]) == r["id"]
    assert len(set(names)) > 500


def test_B1_lint_flags_reuse(small):
    bad = copy.deepcopy(small)
    donor = next(r for r in bad["dev"] if any(sp["type"] == "email" for sp in r["Spans"]))
    email = next(sp["value"] for sp in donor["Spans"] if sp["type"] == "email")
    row = next(r for r in bad["test"] if r["Spans"])
    text = row["Question"] + " " + email
    row["Question"] = text
    row["Spans"].append({"start": len(text) - len(email), "end": len(text), "type": "email",
                         "value": email})
    row["Answer-Key"] = make_dataset._key(row["Spans"])
    assert any(" B1 email" in f for f in _lint(bad))


def test_B2_key_values_in_text_at_offsets(committed):
    for s in SPLITS:
        for r in committed[s]:
            gold, missing = em.entry_gold(r["Question"], r)
            assert missing == []
            assert all(r["Question"][g.start:g.end] == g.value for g in gold)


def test_B2_evaluator_rejects_a_shifted_offset(small):
    row = copy.deepcopy(next(r for r in small["dev"] if r["Spans"]))
    row["Spans"][0]["start"] += 1
    with pytest.raises(ValueError):
        em.entry_gold(row["Question"], row)
    assert any(" B2 " in f for f in lint_key.lint_generated({"x": [row]}))


def test_B3_labels_known_and_key_agrees(committed):
    for s in SPLITS:
        for r in committed[s]:
            assert all(em.gold_type(sp["type"]) != "other" for sp in r["Spans"])
            assert set(em.key_values(r["Answer-Key"])) == {
                (em.gold_type(sp["type"]), sp["value"]) for sp in r["Spans"]}


def test_B4_lint_flags_unannotated_pii(small):
    bad = copy.deepcopy(small)
    row = next(r for r in bad["dev"] if r["Spans"])
    row["Question"] += " Also SSN 219-09-9999."
    assert any("B4 unannotated ssn" in f for f in _lint(bad))


def test_B4_every_occurrence_annotated(committed):
    for s in SPLITS:
        for r in committed[s]:
            for sp in r["Spans"]:
                assert make_dataset._only_at_spans(r["Question"], sp["value"], r["Spans"])


def test_B6_negatives_split_and_languages(committed):
    for s in SPLITS:
        rows = committed[s]
        neg = [r for r in rows if not r["Spans"]]
        assert len(neg) / len(rows) >= 0.20
        assert any(r["lang"] != "en" for r in neg)
        assert all(not r["Answer-Key"] and r["identity_id"] is None for r in neg)
        assert sum(1 for r in neg if r["Keep"]) >= len(neg) / 2       # numeric / public distractors
    dev_t = {r["template"] for r in committed["dev"]}
    test_t = {r["template"] for r in committed["test"]}
    assert not dev_t & test_t
    assert len(committed["test"]) / (len(committed["dev"]) + len(committed["test"])) == 0.2


def test_B6_every_type_in_both_splits(committed):
    want = set(make_dataset.SPAN_LABEL.values())
    for s in SPLITS:
        assert {sp["type"] for r in committed[s] for sp in r["Spans"]} == want


def test_D2_pronouns_follow_identity_gender(committed):
    for s in SPLITS:
        for r in committed[s]:
            if r["gender"] == "f" and r["lang"] == "en":
                assert not lint_key._PRONOUNS["f"].search(
                    make_dataset.re.sub("|".join(make_dataset.re.escape(sp["value"]) for sp in r["Spans"]),
                                        " ", r["Question"]))


def test_B_lint_legacy_key_reports_known_problems(tmp_path):
    legacy = [
        {"Question": "Call 480-555-0101 now", "Answer-Key": {"phone": "480-555-0199"}},
        {"Question": "my ssn is 219-09-9999", "Answer-Key": {}},
        {"Question": "hi phone", "Answer-Key": {"name": "phone"}},
    ]
    findings = lint_key.lint_legacy("k", legacy)
    assert any("B2 phone='480-555-0199'" in f for f in findings)
    assert any("B4 ssn-shaped" in f for f in findings)
    assert any("B3 label word 'phone'" in f for f in findings)


def test_B3_lint_flags_org_value_that_names_no_organisation():
    legacy = [
        {"Question": "Draft a note for an internal company channel", "Answer-Key": {"org": "internal"}},
        {"Question": "I work at Kestrel Analytics", "Answer-Key": {"org": "Kestrel Analytics"}},
    ]
    findings = [f for f in lint_key.lint_legacy("k", legacy) if "names no organisation" in f]
    assert findings == ["k:0: B3 ORG value 'internal' names no organisation"]
