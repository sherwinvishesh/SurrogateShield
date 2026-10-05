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


@pytest.mark.parametrize("split, minimum", [("dev", 100), ("test", 200)])
def test_J2_corpus_lints_clean(split, minimum):
    records = rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")
    assert rw.lint(records) == []
    s = rw.stats(records)
    assert s["messages"] >= minimum
    assert s["negative_share"] >= 0.20


def test_J2_ids_unique_across_splits():
    ids = [r["id"] for split in ("dev", "test")
           for r in rw.load(ROOT / "bench" / "realworld" / f"{split}.jsonl")]
    assert len(ids) == len(set(ids))


def test_J2_cjk_values_are_matched():
    assert rw.occurrences("请联系王伟，电话13812345678", "王伟") == [(3, 5)]
