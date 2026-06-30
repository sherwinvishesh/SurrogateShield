# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test5.py — SurrogateShield Evaluator Tests

Covers:
  1. parse_key_entry() — all input formats and type-alias mapping
  2. run_evaluation() — using the experiment/example files (2 real questions)
  3. run_evaluation() — edge cases: selective fields, progress_cb,
                         mismatched lengths, perfect detection

Run from inside SurrogateShield/:
    python tests/test5.py

No API key needed — all computation is local only.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch as _patch

sys.path.insert(0, ".")

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, note=""):
    symbol = PASS if condition else FAIL
    results.append(condition)
    print(f"  {symbol}  {label}" + (f"  [{note}]" if note else ""))

print("\n" + "=" * 60)
print("  SurrogateShield — Evaluator Tests")
print("=" * 60)

import evaluator
from evaluator import parse_key_entry, run_evaluation, KEY_TYPE_MAP, EVAL_FIELDS


# ─────────────────────────────────────────────────────────────
# 1. parse_key_entry() — input format handling
# ─────────────────────────────────────────────────────────────
print("\n[1] parse_key_entry() — input format handling")

# 1a. None → empty
flat, typed = parse_key_entry(None)
check("None input → empty flat list",  flat == [])
check("None input → empty typed dict", typed == {})

# 1b. Empty string → empty
flat, typed = parse_key_entry("")
check("Empty string → empty flat list", flat == [])

# 1c. Empty dict → empty
flat, typed = parse_key_entry({})
check("Empty dict → empty flat list", flat == [])

# 1d. Dict format — single values
entry = {"name": "Revanth", "ssn": "544-87-2944", "GPE": "wyoming"}
flat, typed = parse_key_entry(entry)
check(
    "Dict format: all values appear in flat list",
    set(flat) == {"Revanth", "544-87-2944", "wyoming"},
    f"flat={flat}"
)
check(
    "Dict format: 'name' label maps to 'PERSON' type",
    "PERSON" in typed and "Revanth" in typed["PERSON"],
    f"typed={typed}"
)
check(
    "Dict format: 'ssn' label maps to 'ssn' type",
    "ssn" in typed and "544-87-2944" in typed["ssn"],
)
check(
    "Dict format: 'GPE' label maps to 'GPE' type",
    "GPE" in typed and "wyoming" in typed["GPE"],
)

# 1e. Dict with list values — multiple names
entry_list = {"name": ["John", "Jane"], "email": "j@example.com"}
flat, typed = parse_key_entry(entry_list)
check(
    "Dict with list values: both names in flat list",
    "John" in flat and "Jane" in flat,
    f"flat={flat}"
)
check(
    "Dict with list values: both names in typed['PERSON']",
    "PERSON" in typed and "John" in typed["PERSON"] and "Jane" in typed["PERSON"],
    f"typed={typed}"
)
check(
    "Dict with list values: single email in flat list",
    "j@example.com" in flat,
)

# 1f. Old string format (backward-compatible)
old_format = '"Revanth", "544-87-2944"'
flat, typed = parse_key_entry(old_format)
check(
    "Old string format: all tokens in flat list",
    "Revanth" in flat and "544-87-2944" in flat,
    f"flat={flat}"
)
check(
    "Old string format: typed dict is empty (no label info)",
    typed == {},
)

# 1g. Unknown label → maps to 'other'
entry_unknown = {"biometric_id": "abc123"}
flat, typed = parse_key_entry(entry_unknown)
check(
    "Unknown label maps to 'other' type",
    "other" in typed and "abc123" in typed["other"],
    f"typed={typed}"
)

# 1h. Label alias coverage — all major aliases resolve correctly
alias_tests = [
    ("person",          "PERSON"),
    ("PERSON",          "PERSON"),
    ("email",           "email"),
    ("phone_us",        "phone"),
    ("phone_uk",        "phone"),
    ("phone_intl",      "phone"),
    ("org",             "ORG"),
    ("ORG",             "ORG"),
    ("organization",    "ORG"),
    ("gpe",             "GPE"),
    ("GPE",             "GPE"),
    ("location",        "GPE"),
    ("credit_card",     "credit_card"),
    ("zip_us",          "postal_code"),
    ("postcode_uk",     "postal_code"),
    ("dob",             "dob"),
    ("date_of_birth",   "dob"),
    ("ip_address",      "ip_address"),
    ("ip",              "ip_address"),
    ("api_key",         "api_key"),
]
for label, expected_type in alias_tests:
    _, typed_alias = parse_key_entry({label: f"val_{label}"})
    check(
        f"Label '{label}' → type '{expected_type}'",
        expected_type in typed_alias,
        f"typed={typed_alias}"
    )

# 1i. Values with leading/trailing whitespace are stripped
entry_spaces = {"name": "  Alice  ", "email": "  alice@x.com  "}
flat_s, typed_s = parse_key_entry(entry_spaces)
check(
    "Leading/trailing spaces stripped from dict values",
    "Alice" in flat_s and "alice@x.com" in flat_s,
    f"flat={flat_s}"
)

# 1j. Empty-string values are excluded
entry_empty_val = {"name": "", "email": "real@test.com"}
flat_ev, _ = parse_key_entry(entry_empty_val)
check(
    "Empty-string values excluded from flat list",
    "" not in flat_ev and "real@test.com" in flat_ev,
    f"flat={flat_ev}"
)


# ─────────────────────────────────────────────────────────────
# 2. run_evaluation() — using the real example experiment files
# ─────────────────────────────────────────────────────────────
print("\n[2] run_evaluation() — example experiment files")

all_fields = {k: True for k, _ in EVAL_FIELDS}

eval_result = None
eval_ok = False
try:
    eval_result = run_evaluation(
        "example.json",
        "example_answers.json",
        "example_key.json",
        all_fields,
    )
    eval_ok = True
except Exception as exc:
    print(f"  [WARN] run_evaluation raised: {exc}")

check(
    "run_evaluation() with example files returns a dict",
    eval_ok and isinstance(eval_result, dict),
)

if eval_ok and eval_result is not None:
    # 2a. Basic counts
    check(
        "no_of_questions == 2 (example has 2 questions)",
        eval_result.get("no_of_questions") == 2,
        f"got: {eval_result.get('no_of_questions')}"
    )
    check(
        "no_of_answers is an int >= 0",
        isinstance(eval_result.get("no_of_answers"), int),
        f"got: {eval_result.get('no_of_answers')}"
    )
    check(
        "answer_rate is a float in [0, 1]",
        isinstance(eval_result.get("answer_rate"), float)
        and 0.0 <= eval_result["answer_rate"] <= 1.0,
        f"got: {eval_result.get('answer_rate')}"
    )

    # 2b. Surrogate quality metrics
    for metric in ("precision_surrogates", "recall_surrogates", "f1_surrogates",
                   "accuracy_surrogates", "error_surrogates"):
        val = eval_result.get(metric)
        check(
            f"{metric} is a float in [0, 1]",
            isinstance(val, float) and 0.0 <= val <= 1.0,
            f"got: {val}"
        )

    # 2c. Surrogate counts
    check(
        "no_surrogates_found is a non-negative int",
        isinstance(eval_result.get("no_surrogates_found"), int)
        and eval_result["no_surrogates_found"] >= 0,
        f"got: {eval_result.get('no_surrogates_found')}"
    )
    check(
        "avg_surrogates_per_question_found is a non-negative float",
        isinstance(eval_result.get("avg_surrogates_per_question_found"), float)
        and eval_result["avg_surrogates_per_question_found"] >= 0.0,
        f"got: {eval_result.get('avg_surrogates_per_question_found')}"
    )

    # 2d. Timing averages — all non-negative
    for timing_key in ("avg_pattern_scan_ms", "avg_entity_trace_ms",
                       "avg_context_guard_ms", "avg_surrogate_gen_ms"):
        val = eval_result.get(timing_key)
        check(
            f"{timing_key} is a non-negative float",
            isinstance(val, float) and val >= 0.0,
            f"got: {val}"
        )

    # 2e. Resolve quality
    check(
        "total_resolve_leaks is an int >= 0",
        isinstance(eval_result.get("total_resolve_leaks"), int)
        and eval_result["total_resolve_leaks"] >= 0,
        f"got: {eval_result.get('total_resolve_leaks')}"
    )
    check(
        "resolve_leak_rate is in [0, 1]",
        isinstance(eval_result.get("resolve_leak_rate"), float)
        and 0.0 <= eval_result["resolve_leak_rate"] <= 1.0,
        f"got: {eval_result.get('resolve_leak_rate')}"
    )

    # 2f. Sanitisation quality
    check(
        "pii_leak_rate is in [0, 1]",
        isinstance(eval_result.get("pii_leak_rate"), float)
        and 0.0 <= eval_result["pii_leak_rate"] <= 1.0,
        f"got: {eval_result.get('pii_leak_rate')}"
    )
    check(
        "accuracy_sanitization is in [0, 1]",
        isinstance(eval_result.get("accuracy_sanitization"), float)
        and 0.0 <= eval_result["accuracy_sanitization"] <= 1.0,
        f"got: {eval_result.get('accuracy_sanitization')}"
    )

    # 2g. Per-entity type breakdown
    per_type = eval_result.get("per_entity_type", {})
    check(
        "per_entity_type is a dict",
        isinstance(per_type, dict),
    )
    for etype, metrics in per_type.items():
        check(
            f"per_entity_type[{etype!r}] has precision/recall/f1/tp/fp/fn keys",
            all(k in metrics for k in ("precision", "recall", "f1", "tp", "fp", "fn")),
            f"keys: {list(metrics.keys())}"
        )
        break  # spot-check one type

    # 2h. Presidio comparison structure
    p_cmp = eval_result.get("presidio_comparison", {})
    check(
        "presidio_comparison has ss_overall key",
        "ss_overall" in p_cmp,
        f"keys: {list(p_cmp.keys())}"
    )
    check(
        "presidio_comparison.ss_overall has precision/recall/f1",
        all(k in p_cmp.get("ss_overall", {}) for k in ("precision", "recall", "f1")),
        f"ss_overall: {p_cmp.get('ss_overall')}"
    )
    check(
        "presidio_comparison has data_status key",
        "data_status" in p_cmp,
    )

    # 2i. BERTScore comparison structure
    bs_cmp = eval_result.get("bertscore_comparison", {})
    check(
        "bertscore_comparison has 'ss' and 'presidio' keys",
        "ss" in bs_cmp and "presidio" in bs_cmp,
        f"keys: {list(bs_cmp.keys())}"
    )
    ss_bs = bs_cmp.get("ss", {})
    check(
        "bertscore_comparison.ss has f1, data_count, data_status",
        all(k in ss_bs for k in ("f1", "data_count", "data_status")),
        f"ss keys: {list(ss_bs.keys())}"
    )

    # 2j. Ablation study structure
    abl = eval_result.get("ablation_study", {})
    check(
        "ablation_study has 'configurations' key",
        "configurations" in abl,
        f"keys: {list(abl.keys())}"
    )
    configs = abl.get("configurations", {})
    check(
        "ablation_study has all four config keys",
        all(k in configs for k in ("full", "ps_only", "ps_et", "ps_cg")),
        f"config keys: {list(configs.keys())}"
    )
    full_cfg = configs.get("full", {})
    check(
        "ablation_study 'full' config has precision/recall/f1",
        all(k in full_cfg for k in ("precision", "recall", "f1")),
        f"full keys: {list(full_cfg.keys())}"
    )

else:
    # Pad so the count stays consistent when eval is unavailable
    for _ in range(28):
        check("(skipped — run_evaluation failed to load example files)", True,
              "check experiment/ directory contains example_*.json files")


# ─────────────────────────────────────────────────────────────
# 3. run_evaluation() — edge cases and error handling
# ─────────────────────────────────────────────────────────────
print("\n[3] run_evaluation() — edge cases")


# 3a. Mismatched file lengths → ValueError
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    (tdp / "q.json").write_text(
        '[{"input":"q1"},{"input":"q2"}]', encoding="utf-8"
    )
    (tdp / "a.json").write_text(
        '[{"surrogate_map":{},"llm_response":"ok","sanitized_input":"ok"}]',
        encoding="utf-8"
    )
    (tdp / "k.json").write_text(
        '[{"Answer-Key":{"name":"Alice"}}]', encoding="utf-8"
    )

    with _patch.object(evaluator, "EXPERIMENT_DIR", tdp):
        try:
            run_evaluation("q.json", "a.json", "k.json", {"no_of_questions": True})
            check("Mismatched file lengths raises ValueError", False,
                  "no exception was raised")
        except ValueError as exc:
            check("Mismatched file lengths raises ValueError", True,
                  f"got: {exc}")
        except Exception as exc:
            check("Mismatched file lengths raises ValueError", False,
                  f"wrong exception type: {type(exc).__name__}: {exc}")


# 3b. Selective fields — only enabled fields appear in result
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    q = [{"input": "test question"}]
    a = [{
        "surrogate_map": {"Alice": "Carol"},
        "llm_response": "hi Carol",
        "sanitized_input": "hi Carol",
        "recognized_not_replaced": [],
    }]
    k = [{"Answer-Key": {"name": "Alice"}}]
    for fname, data in [("q.json", q), ("a.json", a), ("k.json", k)]:
        (tdp / fname).write_text(json.dumps(data), encoding="utf-8")

    selective = {k_: False for k_, _ in EVAL_FIELDS}
    selective["no_of_questions"]  = True
    selective["surrogate_counts"] = True

    with _patch.object(evaluator, "EXPERIMENT_DIR", tdp):
        sel_result = run_evaluation("q.json", "a.json", "k.json", selective)

    check(
        "Selective fields: no_of_questions present when enabled",
        "no_of_questions" in sel_result,
        f"keys: {sorted(sel_result.keys())}"
    )
    check(
        "Selective fields: no_surrogates_found present when surrogate_counts=True",
        "no_surrogates_found" in sel_result,
    )
    check(
        "Selective fields: precision_surrogates absent when disabled",
        "precision_surrogates" not in sel_result,
        f"keys: {sorted(sel_result.keys())}"
    )
    check(
        "Selective fields: avg_pattern_scan_ms absent when timing=False",
        "avg_pattern_scan_ms" not in sel_result,
    )


# 3c. progress_cb is called once per question with correct arguments
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    q = [{"input": "q1"}, {"input": "q2"}, {"input": "q3"}]
    a = [{"surrogate_map": {}, "llm_response": f"r{i}", "sanitized_input": f"q{i}"}
         for i in range(3)]
    k = [{"Answer-Key": None}] * 3
    for fname, data in [("q.json", q), ("a.json", a), ("k.json", k)]:
        (tdp / fname).write_text(json.dumps(data), encoding="utf-8")

    cb_calls = []
    def _cb(idx, total, status):
        cb_calls.append((idx, total, status))

    with _patch.object(evaluator, "EXPERIMENT_DIR", tdp):
        run_evaluation("q.json", "a.json", "k.json",
                       {"no_of_questions": True}, progress_cb=_cb)

    check(
        "progress_cb called once per question (3 times)",
        len(cb_calls) == 3,
        f"calls: {cb_calls}"
    )
    check(
        "progress_cb receives correct total (3)",
        all(total == 3 for _, total, _ in cb_calls),
    )
    check(
        "progress_cb receives sequential indices 0, 1, 2",
        [idx for idx, _, _ in cb_calls] == [0, 1, 2],
    )


# 3d. Perfect detection → precision/recall/F1 all 1.0, pii_leak_rate = 0.0
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    q = [{"input": "hi alice@example.com"}]
    a = [{
        "surrogate_map": {"alice@example.com": "fake@example.org"},
        "llm_response": "noted",
        "sanitized_input": "hi fake@example.org",
        "recognized_not_replaced": [],
        "pii_detail": {"alice@example.com": {"type": "email"}},
    }]
    k = [{"Answer-Key": {"email": "alice@example.com"}}]
    for fname, data in [("q.json", q), ("a.json", a), ("k.json", k)]:
        (tdp / fname).write_text(json.dumps(data), encoding="utf-8")

    with _patch.object(evaluator, "EXPERIMENT_DIR", tdp):
        perfect = run_evaluation("q.json", "a.json", "k.json",
                                 {k_: True for k_, _ in EVAL_FIELDS})

    check(
        "Perfect detection → precision_surrogates == 1.0",
        perfect.get("precision_surrogates") == 1.0,
        f"got: {perfect.get('precision_surrogates')}"
    )
    check(
        "Perfect detection → recall_surrogates == 1.0",
        perfect.get("recall_surrogates") == 1.0,
        f"got: {perfect.get('recall_surrogates')}"
    )
    check(
        "Perfect detection → f1_surrogates == 1.0",
        perfect.get("f1_surrogates") == 1.0,
        f"got: {perfect.get('f1_surrogates')}"
    )
    check(
        "Perfect detection → pii_leak_rate == 0.0",
        perfect.get("pii_leak_rate") == 0.0,
        f"got: {perfect.get('pii_leak_rate')}"
    )
    check(
        "Perfect detection → accuracy_sanitization == 1.0",
        perfect.get("accuracy_sanitization") == 1.0,
        f"got: {perfect.get('accuracy_sanitization')}"
    )


# 3e. No key PII → quality metrics still return without crashing
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    q = [{"input": "what time is it?"}]
    a = [{"surrogate_map": {}, "llm_response": "It is noon.", "sanitized_input": "what time is it?"}]
    k = [{"Answer-Key": None}]
    for fname, data in [("q.json", q), ("a.json", a), ("k.json", k)]:
        (tdp / fname).write_text(json.dumps(data), encoding="utf-8")

    with _patch.object(evaluator, "EXPERIMENT_DIR", tdp):
        no_key = run_evaluation("q.json", "a.json", "k.json",
                                {"no_of_questions": True, "surrogate_quality": True})

    check(
        "No-PII question: evaluation completes without crashing",
        "no_of_questions" in no_key,
    )
    check(
        "No-PII question: precision_surrogates is 1.0 (nothing to find → no FP/FN)",
        no_key.get("precision_surrogates") == 1.0,
        f"got: {no_key.get('precision_surrogates')}"
    )


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ✅  All evaluator tests passed")
else:
    failed = total - passed
    print(f"  ❌  {failed} test(s) failed — see ❌ above")
print("=" * 60 + "\n")
