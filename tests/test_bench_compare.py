"""bench/compare.py — SurrogateShield and default-config Presidio on the same
messages (audit A4, A12; Phase 5 entry point). Needs spaCy: heavy."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.heavy


def _load():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "bench" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_synth_scores_both_systems_on_the_same_rows():
    pytest.importorskip("presidio_analyzer")
    r = _load().score_synth("dev", limit=40)
    a = r["any_type"]
    assert r["messages"] == a["ss"]["questions"] == a["prs"]["questions"] == 40
    assert a["ss"]["micro"]["gold"] == a["prs"]["micro"]["gold"] == r["gold_spans"]
    # rnr only adds predictions: recall with rnr is never below protection
    assert a["ss_rnr"]["micro"]["recall"] >= a["ss"]["micro"]["recall"]
    assert r["presidio"]["score_threshold"] == 0.4
    for c in r["recall_by_gold_type"].values():
        assert 0 <= c["ss"] <= c["ss_rnr"] <= c["gold"] and c["presidio"] <= c["gold"]
