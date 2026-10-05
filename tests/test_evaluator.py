"""Answers-file evaluator (evaluator.py) — audit A6, A7, A9, A12, I32, J9.

Model-free: every answers file is written by hand into a temp directory.
"""

import json

import pytest

import eval_metrics as em
from evaluator import EVAL_FIELDS, parse_key_entry, run_evaluation

ALL = {k: True for k, _ in EVAL_FIELDS}

Q0 = "Ann Lee lives at 12 Elm St, Tempe, AZ."
Q1 = "Email bob@x.com about the Annual report."
Q2 = "What is the weather like today?"


def _span(q, text, typ, replaced=True, reason=None):
    s = q.index(text)
    row = {"text": text, "start": s, "end": s + len(text), "type": typ, "replaced": replaced}
    if reason:
        row["reason"] = reason
    return row


def _write(tmp_path, answers, keys=None, questions=(Q0, Q1, Q2)):
    keys = keys or [
        {"Question": Q0, "Answer-Key": {"name": "Ann Lee", "address": "12 Elm St, Tempe, AZ"}},
        {"Question": Q1, "Answer-Key": {"email": "bob@x.com"}},
        {"Question": Q2, "Answer-Key": {}},
    ]
    (tmp_path / "q.json").write_text(json.dumps([{"input": q} for q in questions]))
    (tmp_path / "a.json").write_text(json.dumps(answers))
    (tmp_path / "k.json").write_text(json.dumps(keys))


def _eval(tmp_path, fields=ALL, **kw):
    return run_evaluation("q.json", "a.json", "k.json", fields, experiment_dir=tmp_path, **kw)


def _good_rows():
    r0 = {
        "question": Q0,
        "pii_spans": [_span(Q0, "Ann Lee", "PERSON"), _span(Q0, "12 Elm St, Tempe, AZ", "address")],
        "surrogate_map": {"Ann Lee": "Zoe Park", "12 Elm St, Tempe, AZ": "9 Oak Ave, Mesa, AZ"},
        "sanitized_input": "Zoe Park lives at 9 Oak Ave, Mesa, AZ.",
        "address_mode": "replace",
        "llm_response": "Hello Zoe Park.",
        "final_output": "Hello Ann Lee.",
    }
    r1 = {
        "question": Q1,
        "pii_spans": [_span(Q1, "bob@x.com", "email")],
        "surrogate_map": {"bob@x.com": "kim@y.org"},
        "sanitized_input": "Email kim@y.org about the Annual report.",
        "address_mode": "replace",
        "llm_response": "Sent to kim@y.org.",
        "final_output": "Sent to bob@x.com.",
    }
    r2 = {"question": Q2, "pii_spans": [], "surrogate_map": {}, "sanitized_input": Q2,
          "address_mode": "replace", "llm_response": "Sunny.", "final_output": "Sunny."}
    return [r0, r1, r2]


# ── I32: error rows counted and excluded; nothing swallowed ──────────────────

def test_I32_error_rows_excluded(tmp_path):
    rows = _good_rows()
    rows[1] = {"question": Q1, "error": "provider down", "error_type": "ConnectionError"}
    _write(tmp_path, rows)
    seen = []
    r = _eval(tmp_path, progress_cb=lambda i, n, s: seen.append(s))
    assert seen == ["ok", "error", "ok"]
    assert r["error_rows"] == 1 and r["rows_scored"] == 2
    assert r["error_types"] == {"ConnectionError": 1}
    assert r["answers"]["answer_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["answers"]["answer_rate_excluding_errors"] == 1.0
    # the email gold of the error row is in no denominator
    assert "email" not in r["detection"]["protection"]["per_type"]
    assert r["sanitization"]["gold_values_in_input"] == 2   # Q0 only


def test_I32_numeric_key_value_scored():
    flat, typed = parse_key_entry({"zip": 85281, "name": "Ann"})
    assert flat == ["85281", "Ann"] and typed == {"postal_code": ["85281"], "PERSON": ["Ann"]}


def test_I32_misaligned_files_raise(tmp_path):
    rows = _good_rows()
    rows[0]["question"] = "a different question"
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="row 0"):
        _eval(tmp_path)
    _write(tmp_path, _good_rows()[:2])
    with pytest.raises(ValueError, match="length mismatch"):
        _eval(tmp_path)


def test_I32_malformed_row_raises_with_index(tmp_path):
    rows = _good_rows()
    rows[2]["pii_spans"] = [{"text": "x"}]            # no offsets
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="row 2"):
        _eval(tmp_path)


# ── A1/A2 through the evaluator: protection is the headline ──────────────────

def test_A1_evaluator_protection_vs_recognition(tmp_path):
    rows = _good_rows()
    # Tempe recognised but deliberately left in place, address not replaced
    rows[0]["pii_spans"] = [_span(Q0, "Ann Lee", "PERSON"),
                            _span(Q0, "Tempe", "GPE", replaced=False, reason="topical_geo_filtered")]
    rows[0]["surrogate_map"] = {"Ann Lee": "Zoe Park"}
    rows[0]["sanitized_input"] = "Zoe Park lives at 12 Elm St, Tempe, AZ."
    _write(tmp_path, rows)
    det = _eval(tmp_path)["detection"]
    assert det["headline"] == "protection"
    assert det["protection"]["per_type"]["address"]["fn"] == 1
    assert det["recognition"]["per_type"]["address"]["tp"] == 1   # Tempe overlaps it
    neg = det["protection"]["negatives"]
    assert neg["negative_questions"] == 1 and neg["negative_questions_with_any_prediction"] == 0


# ── A6: every verbatim gold value counts; deliberate vs unintended ───────────

def test_A6_sanitization_counts_deliberate_and_unintended(tmp_path):
    rows = _good_rows()
    rows[0]["pii_spans"] = [_span(Q0, "Ann Lee", "PERSON"),
                            _span(Q0, "12 Elm St, Tempe, AZ", "address")]
    rows[0]["surrogate_map"] = {"Ann Lee": "Zoe Park", "12 Elm St, Tempe, AZ": "13 Elm St, Tempe, AZ"}
    rows[0]["sanitized_input"] = "Zoe Park lives at 13 Elm St, Tempe, AZ."
    rows[0]["address_mode"] = "shift"
    rows[1]["sanitized_input"] = Q1                      # email went out unreplaced
    keys = [
        {"Question": Q0, "Answer-Key": {"name": "Ann Lee", "address": "12 Elm St, Tempe, AZ",
                                        "gpe": ["Tempe", "AZ"]}},
        {"Question": Q1, "Answer-Key": {"email": "bob@x.com"}},
        {"Question": Q2, "Answer-Key": {}},
    ]
    _write(tmp_path, rows, keys)
    s = _eval(tmp_path)["sanitization"]
    assert s["gold_values_in_input"] == 5
    assert s["leaked_values"] == 3                      # Tempe, AZ, bob@x.com
    assert s["deliberate_leaks"] == 2
    assert s["deliberate_by_reason"] == {"service_query_address_shift": 2}
    assert s["unintended_leaks"] == 1
    assert s["unintended_examples"][0]["value"] == "bob@x.com"
    assert s["leak_rate"] == pytest.approx(3 / 5)
    assert s["shift_mismatches"] == 0


# ── A7: the stored final_output is scored, not a re-implementation ───────────

def test_A7_scores_final_output(tmp_path):
    rows = _good_rows()
    rows[0]["llm_response"] = "Hello Zoe Park. Zoe is at 9 Oak Ave, Mesa, AZ."
    rows[0]["final_output"] = "Hello Ann Lee. Zoe is at 9 Oak Ave, Mesa, AZ."     # left one
    rows[1]["surrogate_map"] = {"bob@x.com": "Ann"}
    rows[1]["llm_response"] = "Annual numbers sent to Ann."
    rows[1]["final_output"] = "bob@x.comual numbers sent to bob@x.com."          # collateral
    _write(tmp_path, rows)
    r = _eval(tmp_path)["restoration"]
    assert r["rows_recomputed_with_resolvepass"] == 0
    assert r["surrogates_left"] == 1
    assert r["examples"]["surrogates_left"][0] == {"row": 0, "surrogate": "9 Oak Ave, Mesa, AZ"}
    assert r["collateral_edits"] >= 1 and r["rows_with_collateral_edit"] == 1
    assert r["over_restored_values"] == 0


def test_A7_score_restoration_unit():
    smap = {"Ann": "Zoe", "Ann Lee": "Victoria Mitchell"}
    clean = em.score_restoration("Hi Victoria, Zoe says hi.", "Hi Ann, Ann says hi.", smap)
    assert clean == {"surrogates_left": [], "over_restored": [], "collateral_edits": []}
    # an original appearing with no source in the answer is over-restored
    over = em.score_restoration("Hello there.", "Hello Ann.", smap)
    assert over["over_restored"] == [{"original": "Ann", "extra": 1}]
    # a surrogate component used alone is not over-restoration
    part = em.score_restoration("Hi Victoria.", "Hi Ann.", {"Ann": "Victoria Mitchell"})
    assert part["over_restored"] == [] and part["collateral_edits"] == []


# ── A4/A12: Presidio comparison on one universe, same rows, CI ───────────────

def test_A4_presidio_comparison_shared_universe(tmp_path):
    rows = _good_rows()
    for r in rows:
        r["presidio_found_piis"] = []
    rows[0]["presidio_found_piis"] = [
        {"value": "Ann Lee", "type": "PERSON", "start": 0, "end": 7, "score": 0.85},
        {"value": "Tempe", "type": "LOCATION", "start": 27, "end": 32, "score": 0.85},
    ]
    rows[1]["presidio_found_piis"] = [
        {"value": "bob@x.com", "type": "EMAIL_ADDRESS", "start": 6, "end": 15, "score": 1.0},
        {"value": "today", "type": "DATE_TIME", "start": 0, "end": 5, "score": 0.85},
    ]
    keys = [
        {"Question": Q0, "Answer-Key": {"name": "Ann Lee", "address": "12 Elm St, Tempe, AZ",
                                        "gpe": "Tempe", "zip": "AZ"}},
        {"Question": Q1, "Answer-Key": {"email": "bob@x.com"}},
        {"Question": Q2, "Answer-Key": {}},
    ]
    _write(tmp_path, rows, keys)
    pc = _eval(tmp_path)["presidio_comparison"]
    assert pc["available"] and pc["rows"] == 3 and pc["data_status"] == "full"
    # shared gold: PERSON, GEO(Tempe), email; SS's address span competes as GEO
    for arm in ("ss", "presidio"):
        mi = pc["untyped"][arm]["micro"]
        assert (mi["gold"], mi["tp"]) == (3, 3), arm
    assert pc["presidio_only_counts"] == {"DATE_TIME (not date-shaped)": 1}
    out = pc["outside_universe"]
    assert out["address"]["gold"] == 1 and out["address"]["presidio_found"] == 1  # by overlap
    assert "reason" in out["address"]
    bs = pc["bootstrap_untyped_micro_f1_ss_minus_presidio"]
    assert bs["available"] and bs["n_questions"] == 3


def test_A4_presidio_comparison_unavailable_without_data(tmp_path):
    _write(tmp_path, _good_rows())
    pc = _eval(tmp_path)["presidio_comparison"]
    assert pc["available"] is False and pc["rows_without_presidio_data"] == 3


def test_A12_bootstrap_micro_f1_diff_identical_is_zero():
    rows = [(1, 1, 1, 1), (0, 1, 0, 1), (2, 2, 1, 2)]
    out = em.bootstrap_micro_f1_diff(rows, rows, n_boot=200)
    assert out["diff"] == 0.0 and out["ci_low"] == 0.0 and out["ci_high"] == 0.0


# ── A8: BERTScore excludes identical-text rows, flags raw scores ─────────────

def test_A8_bertscore_pairs_and_rescale_flag(tmp_path):
    rows = _good_rows()
    for i, r in enumerate(rows):
        r["presidio_sanitized_input"] = "[PERSON] text"
        r["bertscore_ss"] = {"precision": 0.9, "recall": 0.9, "f1": 0.9 - i / 10, "rescaled": True}
        r["bertscore_presidio"] = {"precision": 0.5, "recall": 0.5, "f1": 0.5 + i / 20, "rescaled": True}
    _write(tmp_path, rows)
    b = _eval(tmp_path)["bertscore"]
    assert b["rescaled"] is True and "warning" not in b
    inp = b["input_fidelity"]
    assert inp["excluded_identical_text"] == 1           # row 2 sent its question unchanged
    assert inp["paired"]["n_paired"] == 2
    assert inp["ss"]["n"] == 3
    rows[0]["bertscore_ss"]["rescaled"] = False
    _write(tmp_path, rows)
    assert _eval(tmp_path)["bertscore"]["rescaled"] == "mixed"


# ── A9: no post-hoc ablation ─────────────────────────────────────────────────

def test_A9_ablation_points_to_cascade_rerun(tmp_path):
    _write(tmp_path, _good_rows())
    ab = _eval(tmp_path)["ablation"]
    assert ab["available"] is False and "--ablation" in ab["command"]


def test_A9_signed_delta_in_ablation_report(capsys):
    from offline_eval import _print_ablation
    _print_ablation({"key_file": "k", "questions": 2, "configurations": {
        "pattern_only": {"label": "PatternScan only",
                         "micro": {"precision": 1.0, "recall": 0.5, "f1": 0.6667},
                         "delta_micro_f1_vs_full": -0.2, "bootstrap_vs_full": {"available": False}},
        "full": {"label": "Full", "micro": {"precision": 1.0, "recall": 0.75, "f1": 0.8667}},
    }})
    assert "-0.2000" in capsys.readouterr().out


# ── Legacy answers files: spans located, labelled as such ────────────────────

def test_legacy_rows_without_pii_spans_are_located(tmp_path):
    rows = _good_rows()
    for r in rows:
        del r["pii_spans"]
    rows[0]["pii_detail"] = {"Ann Lee": {"type": "PERSON"}}
    _write(tmp_path, rows)
    r = _eval(tmp_path)
    assert r["span_source"] == {"located": 3}
    assert r["detection"]["protection"]["per_type"]["PERSON"]["tp"] == 1


def test_run_meta_attached(tmp_path):
    _write(tmp_path, _good_rows())
    (tmp_path / "a.meta.json").write_text(json.dumps({"base_seed": 7, "presidio": None}))
    assert _eval(tmp_path)["run_meta"]["base_seed"] == 7


# ── A11/I23: the report only prints what the result dict holds ───────────────

def test_A11_no_hardcoded_claims(tmp_path):
    from rich.console import Console

    from eval_report import render

    rows = _good_rows()
    rows[1]["sanitized_input"] = Q1                       # a real unintended leak
    for r in rows:
        r["presidio_found_piis"] = []
    _write(tmp_path, rows)
    result = _eval(tmp_path)
    con = Console(record=True, width=200)
    render(result, con)
    out = con.export_text()
    for phrase in ("fully clean", "Cannot Detect", "proves", "0.0ms", "Key Findings",
                   "outperforms", "superior"):
        assert phrase not in out, phrase
    s = result["sanitization"]
    line = next(ln.strip(" │") for ln in out.splitlines() if ln.strip(" │").startswith("unintended "))
    assert line.split()[1] == str(s["unintended_leaks"]) == "1"
    assert "bob@x.com" in out                            # the leak is shown, not hidden
    assert "n/a" in out                                  # missing run metadata / bertscore


def test_A11_main_eval_screen_has_no_result_strings():
    src = open("main.py", encoding="utf-8").read()
    start = src.index("def _run_evaluation")
    end = src.index("# ─── Attacker Experiment")
    body = src[start:end]
    for phrase in ("0.0ms", "fully clean", "Presidio Cannot Detect", "Key Findings"):
        assert phrase not in body, phrase
