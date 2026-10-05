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


@pytest.mark.parametrize("split, minimum", [("dev", 100), ("dev2", 200), ("test", 200)])
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


def test_J2_cjk_values_are_matched():
    assert rw.occurrences("请联系王伟，电话13812345678", "王伟") == [(3, 5)]


def test_J2_dev2_shares_no_message_with_test():
    test = {r["text"] for r in rw.load(ROOT / "bench" / "realworld" / "test.jsonl")}
    assert not any(r["text"] in test for r in rw.load(ROOT / "bench" / "realworld" / "dev2.jsonl"))


def test_J2_run_scores_any_system_with_the_same_code():
    """bench/compare.py scores Presidio through run(prepare=…): an empty edit
    list leaks every non-policy protect value and makes no spurious edit."""
    from types import SimpleNamespace
    s = rw.run("dev", show=False, prepare=lambda text, i: SimpleNamespace(edits=[], sanitized=text))
    assert s["leaked"] == s["protect_values"] - s["policy"] and s["leak_rate"] == 1.0
    assert s["edits"] == s["spurious"] == 0
    assert s["negatives_untouched"] == s["negatives"]
