# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test6.py — SurrogateShield Attacker Experiment Tests

Covers:
  1. _build_types_list()          — readable bullet-list formatting
  2. _types_from_pii_detail()     — SS type normalization
  3. _types_from_presidio_found() — Presidio type mapping
  4. score_recovery()             — exact-match PII scoring + address separation
  5. run_attacker_call()          — API call, JSON parsing, fence stripping, fallbacks
  6. compute_analysis()           — aggregation, by_type, rates, address exclusion
  7. run_experiment()             — end-to-end with mocked API client

Run from inside SurrogateShield/:
    python tests/test6.py

No real API key needed — all API calls are mocked.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch as _patch

sys.path.insert(0, ".")

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, note=""):
    symbol = PASS if condition else FAIL
    results.append(condition)
    print(f"  {symbol}  {label}" + (f"  [{note}]" if note else ""))

print("\n" + "=" * 60)
print("  SurrogateShield — Attacker Experiment Tests")
print("=" * 60)

import attacker
from attacker import (
    _build_types_list,
    _types_from_pii_detail,
    _types_from_presidio_found,
    score_recovery,
    run_attacker_call,
    compute_analysis,
    run_experiment,
    ALL_PII_TYPES,
    EXPERIMENT_DIR,
)


# ─────────────────────────────────────────────────────────────
# 1. _build_types_list()
# ─────────────────────────────────────────────────────────────
print("\n[1] _build_types_list() — readable type formatting")

# 1a. Empty list → fallback sentinel string
result = _build_types_list([])
check(
    "Empty list → fallback 'unknown type' string",
    "unknown" in result and result.startswith("  •"),
    f"got: {result!r}"
)

# 1b. PERSON → readable label "name (PERSON)"
result = _build_types_list(["PERSON"])
check(
    "PERSON → 'name (PERSON)' label",
    "name (PERSON)" in result,
    f"got: {result!r}"
)

# 1c. email → "email address"
result = _build_types_list(["email"])
check(
    "email → 'email address' label",
    "email address" in result,
    f"got: {result!r}"
)

# 1d. ssn → "Social Security Number (SSN)"
result = _build_types_list(["ssn"])
check(
    "ssn → SSN label",
    "Social Security Number" in result,
    f"got: {result!r}"
)

# 1e. Unknown type → raw type string used as label
result = _build_types_list(["my_custom_type"])
check(
    "Unknown type → raw type string in bullet",
    "my_custom_type" in result,
    f"got: {result!r}"
)

# 1f. Multiple types → one bullet per type
result = _build_types_list(["PERSON", "email", "ssn"])
lines = [ln for ln in result.split("\n") if ln.strip()]
check(
    "Multiple types → one bullet line per type (3 lines)",
    len(lines) == 3,
    f"lines: {lines}"
)
check(
    "All three bullets present in output",
    all(lbl in result for lbl in ("name (PERSON)", "email address", "Social Security")),
    f"got: {result!r}"
)


# ─────────────────────────────────────────────────────────────
# 2. _types_from_pii_detail()
# ─────────────────────────────────────────────────────────────
print("\n[2] _types_from_pii_detail() — SS type extraction")

# 2a. Empty dict → empty list
check(
    "Empty pii_detail → empty type list",
    _types_from_pii_detail({}) == [],
)

# 2b. Known types pass through unchanged
detail_known = {
    "Alice":        {"type": "PERSON", "score": 0.9, "source": "ner"},
    "123-45-6789":  {"type": "ssn",    "score": 1.0, "source": "pattern"},
}
types = _types_from_pii_detail(detail_known)
check(
    "PERSON and ssn types extracted",
    set(types) == {"PERSON", "ssn"},
    f"got: {types}"
)

# 2c. phone_us normalizes to phone
detail_phone = {"480-555-1234": {"type": "phone_us", "score": 1.0, "source": "pattern"}}
types = _types_from_pii_detail(detail_phone)
check(
    "phone_us normalizes to 'phone'",
    types == ["phone"],
    f"got: {types}"
)

# 2d. zip_us normalizes to postal_code
detail_zip = {"12345": {"type": "zip_us", "score": 1.0, "source": "pattern"}}
types = _types_from_pii_detail(detail_zip)
check(
    "zip_us normalizes to 'postal_code'",
    types == ["postal_code"],
    f"got: {types}"
)

# 2e. Duplicate types deduplicated
detail_dup = {
    "Alice":   {"type": "PERSON", "score": 0.9},
    "Bob":     {"type": "PERSON", "score": 0.8},
    "revanth@gmail.com": {"type": "email", "score": 1.0},
}
types = _types_from_pii_detail(detail_dup)
check(
    "Duplicate PERSON type deduplicated to one entry",
    types.count("PERSON") == 1,
    f"got: {types}"
)
check(
    "Two distinct types (PERSON, email) both present",
    set(types) == {"PERSON", "email"},
    f"got: {types}"
)

# 2f. Non-dict value in pii_detail handled gracefully (no crash)
detail_bad = {"weird_entry": "not_a_dict"}
try:
    types_bad = _types_from_pii_detail(detail_bad)
    check(
        "Non-dict pii_detail value handled without exception",
        True,
        f"got: {types_bad}"
    )
except Exception as exc:
    check("Non-dict pii_detail value handled without exception", False,
          f"raised: {exc}")


# ─────────────────────────────────────────────────────────────
# 3. _types_from_presidio_found()
# ─────────────────────────────────────────────────────────────
print("\n[3] _types_from_presidio_found() — Presidio type mapping")

# 3a. Empty list → empty list
check(
    "Empty presidio_found → empty type list",
    _types_from_presidio_found([]) == [],
)

# 3b. PERSON maps to PERSON
check(
    "PERSON entity type maps to PERSON",
    _types_from_presidio_found([{"value": "Alice", "type": "PERSON"}]) == ["PERSON"],
)

# 3c. EMAIL_ADDRESS maps to email
check(
    "EMAIL_ADDRESS entity type maps to 'email'",
    _types_from_presidio_found([{"value": "a@b.com", "type": "EMAIL_ADDRESS"}]) == ["email"],
)

# 3d. US_SSN maps to ssn
check(
    "US_SSN entity type maps to 'ssn'",
    _types_from_presidio_found([{"value": "123-45-6789", "type": "US_SSN"}]) == ["ssn"],
)

# 3e. Duplicate Presidio types deduplicated
found_dup = [
    {"value": "Alice", "type": "PERSON", "score": 0.9},
    {"value": "Bob",   "type": "PERSON", "score": 0.8},
    {"value": "a@b.com", "type": "EMAIL_ADDRESS", "score": 0.9},
]
types_dup = _types_from_presidio_found(found_dup)
check(
    "Duplicate PERSON deduplicated; EMAIL_ADDRESS maps to email",
    types_dup.count("PERSON") == 1 and "email" in types_dup,
    f"got: {types_dup}"
)


# ─────────────────────────────────────────────────────────────
# 4. score_recovery()
# ─────────────────────────────────────────────────────────────
print("\n[4] score_recovery() — exact-match PII scoring")

ORIG_SET = {"alice", "123-45-6789", "bob@example.com"}

# 4a. Empty recovery_attempts → 0 recovered
score = score_recovery({"recovery_attempts": []}, ORIG_SET)
check(
    "Empty recovery_attempts → recovered_count == 0",
    score["recovered_count"] == 0,
)

# 4b. null guessed_original → not counted
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": None, "confidence": 0.0, "method": "none"},
    ]},
    ORIG_SET,
)
check(
    "null guessed_original → not counted as recovered",
    score["recovered_count"] == 0,
)

# 4c. Empty-string guessed_original → not counted
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": "", "confidence": 0.0, "method": "none"},
    ]},
    ORIG_SET,
)
check(
    "Empty-string guessed_original → not counted as recovered",
    score["recovered_count"] == 0,
)

# 4d. Exact match → counted as recovered
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": "Alice", "confidence": 0.9, "method": "inference"},
    ]},
    ORIG_SET,
)
check(
    "Exact match on 'Alice' → recovered_count == 1",
    score["recovered_count"] == 1,
    f"recovered: {score['recovered']}"
)

# 4e. Case-insensitive match ("ALICE" matches "alice" in original set)
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": "ALICE", "confidence": 0.9, "method": "inference"},
    ]},
    ORIG_SET,
)
check(
    "Case-insensitive match: 'ALICE' matches 'alice' in original set",
    score["recovered_count"] == 1,
)

# 4f. Address type tracked separately via exclude_types
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "123 Fake St", "pii_type": "address",
         "guessed_original": "alice", "confidence": 0.7, "method": "proximity"},
    ]},
    ORIG_SET,
    exclude_types={"address"},
)
check(
    "Address-type recovery counted in address_recovered_count",
    score["address_recovered_count"] == 1,
    f"addr: {score['address_recovered_count']}"
)
check(
    "Address-type recovery NOT counted in non_address_recovered_count",
    score["non_address_recovered_count"] == 0,
    f"non_addr: {score['non_address_recovered_count']}"
)

# 4g. Wrong guess → not recovered
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": "Eve", "confidence": 0.3, "method": "guess"},
    ]},
    ORIG_SET,
)
check(
    "Wrong guess ('Eve' not in original set) → recovered_count == 0",
    score["recovered_count"] == 0,
)

# 4h. Mixed: one match + one miss → recovered_count == 1
score = score_recovery(
    {"recovery_attempts": [
        {"surrogate_seen": "Carol",  "pii_type": "PERSON",
         "guessed_original": "Alice",  "confidence": 0.9, "method": "inference"},
        {"surrogate_seen": "Fake@co", "pii_type": "email",
         "guessed_original": "Eve@nowhere.com", "confidence": 0.3, "method": "guess"},
    ]},
    ORIG_SET,
)
check(
    "Mixed: one exact match, one miss → recovered_count == 1",
    score["recovered_count"] == 1,
    f"recovered: {score['recovered']}"
)


# ─────────────────────────────────────────────────────────────
# 5. run_attacker_call() — with mock Anthropic client
# ─────────────────────────────────────────────────────────────
print("\n[5] run_attacker_call() — JSON parsing and fallbacks")

def _make_mock_client(response_text: str):
    """Return a mock Anthropic client whose messages.create returns response_text."""
    mock_resp    = MagicMock()
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_resp.content = [mock_content]
    mock_client  = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    return mock_client

# 5a. Valid JSON response → parsed dict returned
valid_response = json.dumps({
    "recovery_attempts": [
        {"surrogate_seen": "Carol", "pii_type": "PERSON",
         "guessed_original": None, "confidence": 0.0, "method": "none"}
    ],
    "overall_assessment": "No recovery possible.",
})
result = run_attacker_call("hi Carol", "  • name (PERSON)", _make_mock_client(valid_response))
check(
    "Valid JSON response → dict with 'recovery_attempts' key",
    isinstance(result, dict) and "recovery_attempts" in result,
    f"keys: {list(result.keys())}"
)
check(
    "Valid JSON response → overall_assessment present",
    "overall_assessment" in result,
)
check(
    "Valid JSON response → recovery_attempts is a list",
    isinstance(result.get("recovery_attempts"), list),
)

# 5b. Response wrapped in ```json fences → stripped and parsed
fenced_response = (
    "```json\n"
    + json.dumps({"recovery_attempts": [], "overall_assessment": "nothing"})
    + "\n```"
)
result = run_attacker_call("test", "  • email address", _make_mock_client(fenced_response))
check(
    "```json fenced response → parsed correctly",
    isinstance(result, dict) and "recovery_attempts" in result,
    f"keys: {list(result.keys())}"
)

# 5c. Response with text preamble → JSON extracted via first/last {}
preamble_response = (
    "Sure, here is my analysis:\n"
    + json.dumps({"recovery_attempts": [], "overall_assessment": "preamble test"})
    + "\nThat concludes my analysis."
)
result = run_attacker_call("test", "  • email address", _make_mock_client(preamble_response))
check(
    "Response with preamble → JSON extracted and parsed",
    isinstance(result, dict) and result.get("overall_assessment") == "preamble test",
    f"result: {result}"
)

# 5d. Completely unparseable response → error dict with _error key
unparseable_response = "This is not JSON at all. No braces here!"
result = run_attacker_call("test", "  • email", _make_mock_client(unparseable_response))
check(
    "Unparseable response → dict with '_error' key",
    isinstance(result, dict) and "_error" in result,
    f"keys: {list(result.keys())}"
)
check(
    "Unparseable response → recovery_attempts is empty list",
    result.get("recovery_attempts") == [],
    f"recovery_attempts: {result.get('recovery_attempts')}"
)

# 5e. API exception → error dict with _error key
def _make_failing_client():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API timeout")
    return mock_client

result = run_attacker_call("test", "  • email", _make_failing_client())
check(
    "API exception → error dict returned (no crash)",
    isinstance(result, dict) and "_error" in result,
    f"_error: {result.get('_error')}"
)


# ─────────────────────────────────────────────────────────────
# 6. compute_analysis()
# ─────────────────────────────────────────────────────────────
print("\n[6] compute_analysis() — aggregation and per-type breakdown")

# 6a. Empty results → all zeros
analysis = compute_analysis([], "test_answers.json")
check(
    "Empty results → total_questions == 0",
    analysis.get("total_questions") == 0,
    f"got: {analysis.get('total_questions')}"
)
check(
    "Empty results → ss.total_recovered == 0",
    analysis.get("ss", {}).get("total_recovered") == 0,
)
check(
    "Empty results → ss.recovery_rate == 0.0",
    analysis.get("ss", {}).get("recovery_rate") == 0.0,
)
check(
    "Empty results → 'address_note' key present",
    "address_note" in analysis,
)
check(
    "Empty results → source_file stored correctly",
    analysis.get("source_file") == "test_answers.json",
)

# 6b. Single available entry, zero recovery → rate == 0.0
single_entry = [{
    "question_index":    0,
    "question_preview":  "Test question",
    "pii_types_targeted": ["PERSON", "ssn"],
    "original_pii_count": 2,
    "ss": {
        "available":                   True,
        "total_targeted":              2,
        "recovered_count":             0,
        "address_recovered_count":     0,
        "non_address_recovered_count": 0,
        "recovery_rate":               0.0,
        "recovered_values":            [],
        "attacker_response":           {"recovery_attempts": [], "overall_assessment": "none"},
        "error":                       None,
    },
    "presidio": {
        "available":                   True,
        "total_targeted":              2,
        "recovered_count":             0,
        "address_recovered_count":     0,
        "non_address_recovered_count": 0,
        "recovery_rate":               0.0,
        "recovered_values":            [],
        "attacker_response":           {"recovery_attempts": [], "overall_assessment": "none"},
        "error":                       None,
    },
}]

analysis = compute_analysis(single_entry, "test.json")
check(
    "Single entry: total_questions == 1",
    analysis["total_questions"] == 1,
)
check(
    "Single entry, no recovery: ss.recovery_rate == 0.0",
    analysis["ss"]["recovery_rate"] == 0.0,
    f"got: {analysis['ss']['recovery_rate']}"
)
check(
    "Single entry, no recovery: ss.total_targeted == 2",
    analysis["ss"]["total_targeted"] == 2,
    f"got: {analysis['ss']['total_targeted']}"
)

# 6c. by_type: PERSON and ssn should appear as targeted
by_type_ss = analysis["ss"]["by_type"]
check(
    "by_type PERSON targeted == 1 (appeared in one entry's pii_types_targeted)",
    by_type_ss["PERSON"]["targeted"] == 1,
    f"PERSON: {by_type_ss.get('PERSON')}"
)
check(
    "by_type ssn targeted == 1",
    by_type_ss["ssn"]["targeted"] == 1,
    f"ssn: {by_type_ss.get('ssn')}"
)
check(
    "by_type email targeted == 0 (not present in this entry)",
    by_type_ss["email"]["targeted"] == 0,
    f"email: {by_type_ss.get('email')}"
)

# 6d. Unavailable entry does not contribute to totals
mixed_entries = [
    single_entry[0],
    {
        "question_index":    1,
        "question_preview":  "No PII question",
        "pii_types_targeted": [],
        "original_pii_count": 0,
        "ss": {
            "available":                   False,
            "total_targeted":              0,
            "recovered_count":             0,
            "address_recovered_count":     0,
            "non_address_recovered_count": 0,
            "recovery_rate":               0.0,
            "recovered_values":            [],
            "attacker_response":           None,
            "error":                       None,
        },
        "presidio": {
            "available":                   False,
            "total_targeted":              0,
            "recovered_count":             0,
            "address_recovered_count":     0,
            "non_address_recovered_count": 0,
            "recovery_rate":               0.0,
            "recovered_values":            [],
            "attacker_response":           None,
            "error":                       None,
        },
    },
]
analysis2 = compute_analysis(mixed_entries, "test.json")
check(
    "Unavailable entry: questions_available == 1 (not 2)",
    analysis2["ss"]["questions_available"] == 1,
    f"got: {analysis2['ss']['questions_available']}"
)
check(
    "Unavailable entry: total_targeted still == 2 (only from available entry)",
    analysis2["ss"]["total_targeted"] == 2,
    f"got: {analysis2['ss']['total_targeted']}"
)

# 6e. ALL_PII_TYPES all present in by_type
check(
    "by_type contains all ALL_PII_TYPES keys",
    all(t in by_type_ss for t in ALL_PII_TYPES),
    f"missing: {[t for t in ALL_PII_TYPES if t not in by_type_ss]}"
)


# ─────────────────────────────────────────────────────────────
# 7. run_experiment() — end-to-end with mocked API
# ─────────────────────────────────────────────────────────────
print("\n[7] run_experiment() — end-to-end with mocked API")

# Synthetic answers file data
SYNTHETIC_ANSWERS = [
    {
        "question": "My name is Alice and my SSN is 123-45-6789.",
        "sanitized_input": "My name is Carol Smith and my SSN is 987-65-4321.",
        "surrogate_map": {
            "Alice":       "Carol Smith",
            "123-45-6789": "987-65-4321",
        },
        "pii_detail": {
            "Alice":       {"type": "PERSON", "score": 0.95, "source": "ner"},
            "123-45-6789": {"type": "ssn",    "score": 1.0,  "source": "pattern"},
        },
        "presidio_sanitized_input": "My name is [PERSON] and my SSN is [US_SSN].",
        "presidio_found_piis": [
            {"value": "Alice",       "type": "PERSON", "score": 0.85},
            {"value": "123-45-6789", "type": "US_SSN", "score": 0.85},
        ],
    },
    {
        "question": "Send to bob@example.com please.",
        "sanitized_input": "Send to fake@surrogate.org please.",
        "surrogate_map": {"bob@example.com": "fake@surrogate.org"},
        "pii_detail": {
            "bob@example.com": {"type": "email", "score": 1.0, "source": "pattern"},
        },
        "presidio_sanitized_input": "Send to [EMAIL_ADDRESS] please.",
        "presidio_found_piis": [
            {"value": "bob@example.com", "type": "EMAIL_ADDRESS", "score": 0.85},
        ],
    },
]

# Attacker always returns no recovery
NO_RECOVERY_RESPONSE = {
    "recovery_attempts": [
        {"surrogate_seen": "Carol Smith", "pii_type": "PERSON",
         "guessed_original": None, "confidence": 0.0, "method": "indeterminate"},
    ],
    "overall_assessment": "The surrogate values provide no recoverable signal.",
}

def _run_patched_experiment(answers_data, out_dir, answers_filename="test_answers.json",
                             atk_response=None):
    """Helper: write answers to temp dir, patch API, run experiment, return paths."""
    if atk_response is None:
        atk_response = NO_RECOVERY_RESPONSE

    ans_path = out_dir / answers_filename
    ans_path.write_text(json.dumps(answers_data, ensure_ascii=False), encoding="utf-8")

    os.environ["ANTHROPIC_API_KEY"] = "test-fake-key-does-not-call-real-api"

    import anthropic as _anthropic
    with _patch.object(_anthropic, "Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        with _patch("attacker.run_attacker_call", return_value=atk_response):
            with _patch.object(attacker, "EXPERIMENT_DIR", out_dir):
                result_path = run_experiment(answers_filename)

    stem          = Path(answers_filename).stem
    analysis_path = out_dir / f"{stem}_Attacker_Experiment_Analysis.json"
    results_path  = out_dir / f"{stem}_Attacker_Experiment.json"
    return results_path, analysis_path


# 7a. Output file is created
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    res_path, ana_path = _run_patched_experiment(SYNTHETIC_ANSWERS, tdp)

    check(
        "_Attacker_Experiment.json output file created",
        res_path.exists(),
        f"path: {res_path}"
    )
    check(
        "_Attacker_Experiment_Analysis.json analysis file created",
        ana_path.exists(),
        f"path: {ana_path}"
    )

    per_q = json.loads(res_path.read_text(encoding="utf-8"))
    analysis = json.loads(ana_path.read_text(encoding="utf-8"))

    # 7b. Results list has one entry per question
    check(
        "Results list has 2 entries (one per question)",
        len(per_q) == 2,
        f"got: {len(per_q)}"
    )

    # 7c. Each entry has required top-level keys
    required_keys = {"question_index", "question_preview", "pii_types_targeted",
                     "original_pii_count", "ss", "presidio"}
    for idx, entry in enumerate(per_q):
        missing = required_keys - set(entry.keys())
        check(
            f"Entry {idx} has all required top-level keys",
            len(missing) == 0,
            f"missing: {missing}"
        )

    # 7d. ss side has all required keys
    ss_keys = {"available", "total_targeted", "recovered_count", "address_recovered_count",
               "non_address_recovered_count", "recovery_rate", "recovered_values",
               "attacker_response", "error"}
    for idx, entry in enumerate(per_q):
        missing = ss_keys - set(entry.get("ss", {}).keys())
        check(
            f"Entry {idx} ss side has all required keys",
            len(missing) == 0,
            f"missing: {missing}"
        )

    # 7e. First entry: ss.available == True, total_targeted == 2 (Alice + SSN)
    ss0 = per_q[0]["ss"]
    check(
        "Entry 0: ss.available == True",
        ss0["available"] is True,
    )
    check(
        "Entry 0: ss.total_targeted == 2 (Alice + 123-45-6789)",
        ss0["total_targeted"] == 2,
        f"got: {ss0['total_targeted']}"
    )

    # 7f. No recovery response → recovered_count == 0
    check(
        "No-recovery attacker response → recovered_count == 0",
        ss0["recovered_count"] == 0,
        f"got: {ss0['recovered_count']}"
    )
    check(
        "No-recovery attacker response → recovery_rate == 0.0",
        ss0["recovery_rate"] == 0.0,
        f"got: {ss0['recovery_rate']}"
    )

    # 7g. Analysis has correct top-level structure
    check(
        "Analysis has 'ss', 'presidio', 'total_questions', 'address_note', 'source_file'",
        all(k in analysis for k in ("ss", "presidio", "total_questions",
                                    "address_note", "source_file")),
        f"keys: {list(analysis.keys())}"
    )
    check(
        "Analysis total_questions == 2",
        analysis["total_questions"] == 2,
        f"got: {analysis['total_questions']}"
    )

    # 7h. Analysis ss.recovery_rate == 0.0 (no recovery)
    check(
        "Analysis: ss.recovery_rate == 0.0",
        analysis["ss"]["recovery_rate"] == 0.0,
        f"got: {analysis['ss']['recovery_rate']}"
    )

    # 7i. pii_types_targeted correctly populated from pii_detail types
    check(
        "Entry 0: pii_types_targeted contains 'PERSON' and 'ssn'",
        set(per_q[0]["pii_types_targeted"]) == {"PERSON", "ssn"},
        f"got: {per_q[0]['pii_types_targeted']}"
    )
    check(
        "Entry 1: pii_types_targeted contains 'email' (phone_us → phone normalized)",
        "email" in per_q[1]["pii_types_targeted"],
        f"got: {per_q[1]['pii_types_targeted']}"
    )


# 7j. Resume logic: existing results are not re-processed
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)

    # Write only the first answer pre-processed (simulates a prior partial run)
    stem = "test_answers"
    prior = [{
        "question_index":    0,
        "question_preview":  "Pre-existing entry",
        "pii_types_targeted": ["PERSON"],
        "original_pii_count": 1,
        "ss": {
            "available": True, "total_targeted": 1, "recovered_count": 0,
            "address_recovered_count": 0, "non_address_recovered_count": 0,
            "recovery_rate": 0.0, "recovered_values": [],
            "attacker_response": {"recovery_attempts": [], "overall_assessment": "prior"},
            "error": None,
        },
        "presidio": {
            "available": False, "total_targeted": 0, "recovered_count": 0,
            "address_recovered_count": 0, "non_address_recovered_count": 0,
            "recovery_rate": 0.0, "recovered_values": [],
            "attacker_response": None, "error": None,
        },
    }]
    (tdp / f"{stem}_Attacker_Experiment.json").write_text(
        json.dumps(prior), encoding="utf-8"
    )

    call_count = {"n": 0}
    def _counting_attacker(*args, **kwargs):
        call_count["n"] += 1
        return NO_RECOVERY_RESPONSE

    ans_path = tdp / f"{stem}.json"
    ans_path.write_text(json.dumps(SYNTHETIC_ANSWERS), encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = "test-fake-key"

    import anthropic as _anthropic
    with _patch.object(_anthropic, "Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        with _patch("attacker.run_attacker_call", side_effect=_counting_attacker):
            with _patch.object(attacker, "EXPERIMENT_DIR", tdp):
                run_experiment(f"{stem}.json")

    final_results = json.loads(
        (tdp / f"{stem}_Attacker_Experiment.json").read_text(encoding="utf-8")
    )
    check(
        "Resume: final results list has 2 entries total (1 prior + 1 new)",
        len(final_results) == 2,
        f"got: {len(final_results)}"
    )
    # SYNTHETIC_ANSWERS[1] has sanitized_input + presidio → 2 calls expected
    check(
        "Resume: run_attacker_call called only for unprocessed questions (2 calls, not 4)",
        call_count["n"] == 2,
        f"got: {call_count['n']} calls"
    )

    # Progress callback fires correctly
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    cb_events = []
    def _cb(i, total, preview, status, elapsed):
        cb_events.append((i, total, status))

    res_path, _ = _run_patched_experiment(SYNTHETIC_ANSWERS, tdp, "cb_test.json")
    with _patch.object(attacker, "EXPERIMENT_DIR", tdp):
        pass  # already ran; just verify the events we'd collect

    # Re-run to capture events (temp dir already has results, will resume — re-create)
    (tdp / "cb_test_Attacker_Experiment.json").unlink(missing_ok=True)
    (tdp / "cb_test_Attacker_Experiment_Analysis.json").unlink(missing_ok=True)

    ans_path2 = tdp / "cb_test.json"
    ans_path2.write_text(json.dumps(SYNTHETIC_ANSWERS), encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = "test-fake-key"

    import anthropic as _anthropic
    with _patch.object(_anthropic, "Anthropic") as mock_cls2:
        mock_cls2.return_value = MagicMock()
        with _patch("attacker.run_attacker_call", return_value=NO_RECOVERY_RESPONSE):
            with _patch.object(attacker, "EXPERIMENT_DIR", tdp):
                run_experiment("cb_test.json", progress_cb=_cb)

    question_events = [(i, t, s) for i, t, s in cb_events if s != "done"]
    done_events     = [(i, t, s) for i, t, s in cb_events if s == "done"]

    check(
        "progress_cb: 4 question events (2 questions × running+ok)",
        len(question_events) == 4,
        f"events: {question_events}"
    )
    check(
        "progress_cb: exactly 1 'done' event fired at completion",
        len(done_events) == 1,
        f"done events: {done_events}"
    )
    check(
        "progress_cb: 'done' event has i == total (2)",
        done_events[0][0] == 2 and done_events[0][1] == 2,
        f"done event: {done_events[0]}"
    )


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ✅  All attacker experiment tests passed")
else:
    failed = total - passed
    print(f"  ❌  {failed} test(s) failed — see ❌ above")
print("=" * 60 + "\n")
