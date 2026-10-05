"""Shared evaluation metrics (eval_metrics.py) — audit A1–A4, A12, N1/N2.

Model-free: every test builds spans by hand.
"""

import pytest

import eval_metrics as em
from offline_eval import _pred_spans


def S(start, end, typ, value=""):
    return em.Span(start, end, typ, value)


# ── A1: recognised-not-replaced is not protection ────────────────────────────

def test_A1_rnr_not_credited():
    spans = [
        {"text": "Tempe", "start": 10, "end": 15, "type": "GPE", "replaced": False,
         "reason": "topical_geo_filtered"},
        {"text": "Ann", "start": 0, "end": 3, "type": "PERSON", "replaced": True},
    ]
    gold = [S(0, 3, "PERSON"), S(10, 15, "GPE")]

    prot, recog = em.Tally(), em.Tally()
    prot.add(gold, _pred_spans(spans, replaced_only=True))
    recog.add(gold, _pred_spans(spans, replaced_only=False))

    assert prot.micro()["recall"] == 0.5          # Tempe went out verbatim
    assert recog.micro()["recall"] == 1.0         # but it was recognised
    assert prot.per_type()["GPE"]["fn"] == 1


# ── A2: empty denominators score zero, never one ─────────────────────────────

def test_A2_empty_sets_score_zero():
    assert em.prf(0, 0, 0, 0) == (0.0, 0.0, 0.0)
    t = em.Tally()
    t.add([], [S(0, 3, "PERSON")])                # negative question, one FP
    micro = t.micro()
    assert micro["precision"] == 0.0 and micro["recall"] == 0.0 and micro["f1"] == 0.0
    assert t.negatives()["negative_questions_with_any_prediction"] == 1


def test_A2_macro_ignores_types_absent_from_both():
    t = em.Tally()
    t.add([S(0, 3, "PERSON")], [S(0, 3, "PERSON")])
    t.by_type["email"]  # touched but empty → must not enter the macro average
    assert t.macro() == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_types": 1}


# ── A3: span overlap, exact boundary reported separately ─────────────────────

def test_A3_overlap_matching():
    gold = [S(0, 22, "address", "123 Main St, Tempe, AZ")]
    pred = [S(0, 11, "address", "123 Main St")]
    g_hit, p_hit = em.match(gold, pred)
    assert g_hit == [True] and p_hit == [True]

    t = em.Tally()
    t.add(gold, pred)
    assert t.micro()["tp"] == 1
    assert t.micro()["exact_boundary_matches"] == 0


def test_A3_adjacent_spans_do_not_overlap():
    assert not S(0, 3, "x").overlaps(S(3, 6, "x"))


def test_A3_whole_token_occurrences():
    assert em.find_occurrences("Annual report by Ann", "Ann") == [(17, 20)]
    assert em.find_occurrences("call ann today", "Ann") == [(5, 8)]


# ── A4: one type universe, applied symmetrically ─────────────────────────────

def test_A4_symmetric_types():
    assert em.ss_shared_type("GPE") == em.presidio_shared_type("LOCATION", "Tempe") == em.GEO
    assert em.ss_shared_type("address") == em.GEO
    assert em.presidio_shared_type("DATE_TIME", "1990-04-22") == "dob"
    assert em.presidio_shared_type("DATE_TIME", "next Tuesday") is None
    assert em.ss_shared_type("ORG") is None
    assert em.normalize_type("phone_uk") == "phone"
    assert em.normalize_type("zip_us") == em.normalize_type("postcode_uk") == "postal_code"


def test_A4_drop_neutral_neither_credits_nor_penalises():
    pred = [S(0, 5, em.GEO), S(10, 15, "PERSON")]
    neutral = [S(10, 15, "ORG")]                  # outside the shared universe
    assert em.drop_neutral(pred, neutral) == [S(0, 5, em.GEO)]


# ── N1/N2: gold spans from the key ───────────────────────────────────────────

def test_N1_gold_spans_report_missing_values():
    spans, missing = em.gold_spans("Ann lives in Tempe", {"name": "Ann", "gpe": ["Tempe", "Mesa"]})
    assert {(s.start, s.end, s.type) for s in spans} == {(0, 3, "PERSON"), (13, 18, "GPE")}
    assert missing == [("GPE", "Mesa")]          # canonical type


# ── A12: statistics ──────────────────────────────────────────────────────────

def test_A12_bootstrap_ci():
    diffs = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.09, 0.11]
    lo, hi = em.bootstrap_ci(diffs, n_boot=2000, seed=1)
    assert lo <= sum(diffs) / len(diffs) <= hi
    assert em.bootstrap_ci(diffs, n_boot=2000, seed=1) == (lo, hi)   # seeded


def test_A12_paired_stats_sample_std_and_p_display():
    a = [0.9, 0.8, 0.85, 0.95]
    b = [0.7, 0.75, 0.6, 0.8]
    out = em.paired_stats(a, b)
    assert out["available"] is True
    assert out["ss_std"] == pytest.approx(0.0645, abs=1e-4)          # ddof=1
    assert out["mean_diff"] == pytest.approx(0.1625, abs=1e-4)
    assert out["ci_low"] <= out["mean_diff"] <= out["ci_high"]
    assert em.format_p(0.0) == "< 1e-300"
    assert em.paired_stats([1.0], [0.5])["available"] is False
