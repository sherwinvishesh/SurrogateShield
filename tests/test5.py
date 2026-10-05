# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test5.py — SurrogateShield Evaluator Tests

Covers:
  1. parse_key_entry() — all input formats and type-alias mapping
  2. run_evaluation() — length mismatch raises (the rest lives in
     tests/test_evaluator.py)

Run from inside SurrogateShield/:
    python tests/test5.py

No API key needed — all computation is local only.
"""

import sys
import json
import tempfile
from pathlib import Path

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
# 2 and 3 retired (audit A3/A6/A7/A9/A12/I32).
# They asserted the old result schema, including "nothing to find →
# precision 1.0", which the audit identified as a bug (A3). The evaluator's
# behaviour is now covered by tests/test_evaluator.py and
# tests/test_eval_metrics.py (pytest, model-free).
# ─────────────────────────────────────────────────────────────
print("\n[2] run_evaluation() — see tests/test_evaluator.py")
with tempfile.TemporaryDirectory() as _d:
    _d = Path(_d)
    (_d / "q.json").write_text(json.dumps([{"input": "a"}, {"input": "b"}]))
    (_d / "a.json").write_text(json.dumps([{"question": "a"}]))
    (_d / "k.json").write_text(json.dumps([{"Question": "a", "Answer-Key": {}},
                                          {"Question": "b", "Answer-Key": {}}]))
    try:
        run_evaluation("q.json", "a.json", "k.json", {}, experiment_dir=_d)
        check("Mismatched file lengths raises ValueError", False, "no exception")
    except ValueError:
        check("Mismatched file lengths raises ValueError", True)


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
