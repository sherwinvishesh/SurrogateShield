"""Gate J2 corpus hygiene (audit B-series, I1/I8/I12/I14): both splits lint
clean, ids are unique across splits, and the shape the gate relies on holds.
Reads only annotations and counts; no model, no network."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("realworld", ROOT / "bench" / "realworld.py")
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)


@pytest.mark.parametrize("split, minimum", [("dev", 100), ("dev2", 200), ("dev3", 200),
                                            ("dev4", 200), ("test", 200)])
def test_J2_corpus_lints_clean(split, minimum):
    records = rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")
    assert rw.lint(records) == []
    s = rw.stats(records)
    assert s["messages"] >= minimum
    assert s["negative_share"] >= 0.20


def test_J2_ids_unique_across_splits():
    ids = [r["id"] for split in rw.SPLITS
           for r in rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("rid, ok", [("rw-dev-0001", True), ("rw-dev3-0200", True), ("rw-test-0001", True),
                                     ("rd-oasst1-dev-0001", True), ("rd-wildchat-test-0250", True),
                                     ("rw-dev5-0001", False), ("rd-oasst1-dev2-0001", False),
                                     ("rd-Oasst1-dev-0001", False), ("rd-oasst1-test-001", False), ("x-0001", False)])
def test_J2_lint_accepts_the_real_data_ids_and_nothing_else(rid, ok):
    rec = {"id": rid, "text": "hi", "category": "qa", "lang": "en", "service_query": False}
    assert (rw.lint([rec]) == []) is ok


def test_J2_cjk_values_are_matched():
    assert rw.occurrences("请联系王伟，电话13812345678", "王伟") == [(3, 5)]


def test_J2_no_message_is_in_two_splits():
    seen = {}
    for split in rw.SPLITS:
        for r in rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl"):
            key = " ".join(r["text"].lower().split())
            assert seen.setdefault(key, split) == split, (r["id"], seen[key])


@pytest.mark.parametrize("split", ["test", "dev4"])
def test_J2_show_refuses_held_out_splits(split, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["realworld.py", "--split", split, "--show"])
    with pytest.raises(SystemExit) as e:
        rw.main()
    assert e.value.code == 2 and "not inspected" in capsys.readouterr().err


def test_J2_run_scores_any_system_with_the_same_code():
    """bench/compare.py scores Presidio through run(prepare=…): an empty edit
    list leaks every non-policy protect value and makes no spurious edit."""
    from types import SimpleNamespace
    s = rw.run("dev", show=False, prepare=lambda text, i: SimpleNamespace(edits=[], sanitized=text))
    assert s["leaked"] == s["protect_values"] - s["policy"] and s["leak_rate"] == 1.0
    assert s["edits"] == s["spurious"] == 0
    assert s["negatives_untouched"] == s["negatives"]
