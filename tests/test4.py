"""
test4.py — SurrogateShield Quasi-Identifier, Settings & Pipeline Tests

Covers:
  1. QuasiIdentifier score()     — combination detection and risk levels
  2. QuasiIdentifier format_warning() — human-readable output
  3. Settings Manager            — load/save round-trips, defaults, error handling
  4. Pipeline anonymise_text()   — PII sanitisation without an API call

Run from inside SurrogateShield/:
    python tests/test4.py

No API key needed — all tests are local only.
"""

import sys
sys.path.insert(0, ".")

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, note=""):
    symbol = PASS if condition else FAIL
    results.append(condition)
    print(f"  {symbol}  {label}" + (f"  [{note}]" if note else ""))

print("\n" + "=" * 60)
print("  SurrogateShield — Quasi-ID, Settings & Pipeline Tests")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. QUASI-IDENTIFIER SCORER — score()
# ─────────────────────────────────────────────────────────────
print("\n[1] QuasiIdentifier score()")

from detection.quasi_identifier import score, format_warning
from util import DetectedEntity


def make_ent(text, etype, score_val=1.0):
    return DetectedEntity(text, 0, len(text), etype, score_val)


# 1a. Name + SSN → high-risk, all fields matched
ents_name_ssn = [
    make_ent("John Doe", "PERSON"),
    make_ent("123-45-6789", "ssn"),
]
matches_name_ssn = score(ents_name_ssn)
check(
    "Name + SSN triggers high-risk quasi-ID match",
    any(m.combination_name == "Name + SSN" for m in matches_name_ssn),
    f"matches: {[m.combination_name for m in matches_name_ssn]}"
)
check(
    "Name + SSN match has risk_level='high'",
    any(m.risk_level == "high" and m.combination_name == "Name + SSN"
        for m in matches_name_ssn),
)
check(
    "Name + SSN all_fields_matched=True",
    any(m.all_fields_matched and m.combination_name == "Name + SSN"
        for m in matches_name_ssn),
)

# 1b. Full Sweeney triple: ZIP + DOB + Gender → all_fields_matched=True
ents_sweeney_full = [
    make_ent("90210", "zip_us"),
    make_ent("03/14/1990", "dob"),
    make_ent("Female", "gender_indicator"),
]
matches_sweeney_full = score(ents_sweeney_full)
sweeney_match = next(
    (m for m in matches_sweeney_full if m.combination_name == "ZIP + DOB + Gender"),
    None,
)
check(
    "Full Sweeney triple (ZIP + DOB + Gender) triggers combo",
    sweeney_match is not None,
    f"matches: {[m.combination_name for m in matches_sweeney_full]}"
)
check(
    "Full Sweeney triple is all_fields_matched=True",
    sweeney_match is not None and sweeney_match.all_fields_matched,
)
check(
    "Full Sweeney triple risk_level='high'",
    sweeney_match is not None and sweeney_match.risk_level == "high",
)

# 1c. Partial Sweeney: ZIP + DOB only (2 of 3 fields) → still triggers, all_fields_matched=False
ents_sweeney_partial = [
    make_ent("90210", "zip_us"),
    make_ent("03/14/1990", "dob"),
]
matches_sweeney_partial = score(ents_sweeney_partial)
sweeney_partial = next(
    (m for m in matches_sweeney_partial if m.combination_name == "ZIP + DOB + Gender"),
    None,
)
check(
    "Partial Sweeney (ZIP + DOB only) still triggers combo (required_count=2)",
    sweeney_partial is not None,
    f"matches: {[m.combination_name for m in matches_sweeney_partial]}"
)
check(
    "Partial Sweeney all_fields_matched=False",
    sweeney_partial is not None and not sweeney_partial.all_fields_matched,
)

# 1d. Name + DOB → high-risk
ents_name_dob = [
    make_ent("Alice Smith", "PERSON"),
    make_ent("January 1 1985", "dob"),
]
check(
    "Name + DOB triggers high-risk combo",
    any(m.combination_name == "Name + DOB" and m.risk_level == "high"
        for m in score(ents_name_dob)),
)

# 1e. Name + Employer + City → medium-risk, all 3 fields
ents_triple = [
    make_ent("Bob Jones", "PERSON"),
    make_ent("Acme Corp", "ORG"),
    make_ent("Phoenix", "GPE"),
]
matches_triple = score(ents_triple)
check(
    "Name + Employer + City triggers medium-risk combo",
    any(m.combination_name == "Name + Employer + City" and m.risk_level == "medium"
        for m in matches_triple),
    f"matches: {[m.combination_name for m in matches_triple]}"
)
check(
    "Name + Employer + City all_fields_matched=True",
    any(m.combination_name == "Name + Employer + City" and m.all_fields_matched
        for m in matches_triple),
)

# 1f. Email + Location → medium-risk
ents_email_loc = [
    make_ent("user@example.com", "email"),
    make_ent("Chicago", "GPE"),
]
check(
    "Email + Location triggers medium-risk combo",
    any(m.combination_name == "Email + Location" and m.risk_level == "medium"
        for m in score(ents_email_loc)),
)

# 1g. No quasi-ID fields → empty result
ents_clean = [make_ent("hello world", "NORP")]
check(
    "No quasi-ID fields → empty match list",
    len(score(ents_clean)) == 0,
)

# 1h. Empty entity list → empty result
check(
    "Empty entity list → empty match list",
    len(score([])) == 0,
)

# 1i. Multiple combos fire simultaneously
ents_multi = [
    make_ent("Jane Doe", "PERSON"),
    make_ent("555-12-3456", "ssn"),
    make_ent("03/14/1990", "dob"),
    make_ent("+1-555-123-4567", "phone_us"),
]
matches_multi = score(ents_multi)
check(
    "Multiple combos fire when several quasi-ID fields present",
    len(matches_multi) >= 2,
    f"combos: {[m.combination_name for m in matches_multi]}"
)

# 1j. matched_entities list is populated
name_ssn_match = next(
    (m for m in matches_name_ssn if m.combination_name == "Name + SSN"),
    None,
)
check(
    "QuasiIdMatch.matched_entities is a non-empty list",
    name_ssn_match is not None and len(name_ssn_match.matched_entities) >= 2,
    f"entities: {[e.text for e in (name_ssn_match.matched_entities if name_ssn_match else [])]}"
)

# 1k. matched_fields list only contains fields that were present
check(
    "QuasiIdMatch.matched_fields only contains present field types",
    name_ssn_match is not None and set(name_ssn_match.matched_fields) <= {"PERSON", "ssn"},
    f"fields: {name_ssn_match.matched_fields if name_ssn_match else None}"
)


# ─────────────────────────────────────────────────────────────
# 2. QUASI-IDENTIFIER format_warning()
# ─────────────────────────────────────────────────────────────
print("\n[2] QuasiIdentifier format_warning()")

# 2a. Full match uses full reference (identity theft / fraud language)
warning_full = format_warning(matches_name_ssn)
check(
    "format_warning returns a non-empty string",
    isinstance(warning_full, str) and len(warning_full) > 0,
)
check(
    "format_warning contains 'Name' label for PERSON field",
    "Name" in warning_full,
    f"warning: {warning_full[:120]!r}"
)
check(
    "format_warning contains 'SSN' label",
    "SSN" in warning_full,
)
check(
    "format_warning contains reference text (identity theft / fraud)",
    "identity theft" in warning_full.lower() or "fraud" in warning_full.lower(),
    f"warning: {warning_full!r}"
)

# 2b. Partial match: partial_reference used (not the full claim)
warning_partial = format_warning(matches_sweeney_partial)
check(
    "format_warning non-empty for partial Sweeney match",
    isinstance(warning_partial, str) and len(warning_partial) > 0,
)

# 2c. Multiple combos → multiple lines in output
warning_multi = format_warning(matches_multi)
check(
    "format_warning has multiple lines for multiple combos",
    warning_multi.count("\n") >= 1,
    f"line count: {warning_multi.count(chr(10))}"
)

# 2d. Empty match list → empty string
check(
    "format_warning returns empty string for no matches",
    format_warning([]) == "",
)

# 2e. Quasi-identifier warning for Name+Employer+City uses correct labels
warning_triple = format_warning(matches_triple)
check(
    "format_warning for Name+Employer+City contains 'Employer' label",
    "Employer" in warning_triple,
    f"warning: {warning_triple!r}"
)


# ─────────────────────────────────────────────────────────────
# 3. SETTINGS MANAGER
# ─────────────────────────────────────────────────────────────
print("\n[3] Settings Manager (settings_manager.py)")

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import settings_manager
from settings_manager import load_settings, save_settings, DEFAULT_SETTINGS


with tempfile.TemporaryDirectory() as tmpdir:
    tmp_dir = Path(tmpdir)
    tmp_file = tmp_dir / "settings.json"

    # 3a. No file → all default keys present
    with patch.object(settings_manager, "_SETTINGS_FILE", tmp_file):
        s = load_settings()
    check(
        "load_settings() with no file returns default llm_provider",
        s.get("llm_provider") == DEFAULT_SETTINGS["llm_provider"],
        f"got: {s.get('llm_provider')!r}"
    )
    check(
        "load_settings() with no file returns all default keys",
        all(k in s for k in DEFAULT_SETTINGS),
        f"missing: {[k for k in DEFAULT_SETTINGS if k not in s]}"
    )

    # 3b. save + reload round-trip
    new_settings = {**DEFAULT_SETTINGS, "llm_provider": "gemini", "detailed_view": False}
    with patch.object(settings_manager, "_SETTINGS_DIR", tmp_dir), \
         patch.object(settings_manager, "_SETTINGS_FILE", tmp_file):
        save_settings(new_settings)
        reloaded = load_settings()

    check(
        "save_settings → load_settings round-trips llm_provider",
        reloaded.get("llm_provider") == "gemini",
        f"got: {reloaded.get('llm_provider')!r}"
    )
    check(
        "save_settings → load_settings round-trips detailed_view=False",
        reloaded.get("detailed_view") is False,
        f"got: {reloaded.get('detailed_view')!r}"
    )

    # 3c. Partial saved file → missing keys filled with defaults
    tmp_file.write_text(json.dumps({"llm_provider": "chatgpt"}), encoding="utf-8")
    with patch.object(settings_manager, "_SETTINGS_FILE", tmp_file):
        merged = load_settings()
    check(
        "Partial settings file: explicit key preserved over default",
        merged.get("llm_provider") == "chatgpt",
        f"got: {merged.get('llm_provider')!r}"
    )
    check(
        "Partial settings file: missing keys filled with defaults",
        "detailed_view" in merged and merged["detailed_view"] == DEFAULT_SETTINGS["detailed_view"],
        f"merged: {merged}"
    )

    # 3d. Corrupted JSON → graceful fallback to defaults
    tmp_file.write_text("{not valid json", encoding="utf-8")
    with patch.object(settings_manager, "_SETTINGS_FILE", tmp_file):
        fallback = load_settings()
    check(
        "Corrupted settings.json → graceful fallback returns default llm_provider",
        fallback.get("llm_provider") == DEFAULT_SETTINGS["llm_provider"],
        f"got: {fallback.get('llm_provider')!r}"
    )
    check(
        "Corrupted settings.json → all default keys still present",
        all(k in fallback for k in DEFAULT_SETTINGS),
    )

    # 3e. Saved file is valid JSON on disk
    clean_settings = {**DEFAULT_SETTINGS, "llm_provider": "local"}
    with patch.object(settings_manager, "_SETTINGS_DIR", tmp_dir), \
         patch.object(settings_manager, "_SETTINGS_FILE", tmp_file):
        save_settings(clean_settings)
    raw = tmp_file.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        check("save_settings writes valid JSON to disk", parsed.get("llm_provider") == "local")
    except json.JSONDecodeError:
        check("save_settings writes valid JSON to disk", False, f"raw: {raw!r}")


# ─────────────────────────────────────────────────────────────
# 4. PIPELINE anonymise_text()
# ─────────────────────────────────────────────────────────────
print("\n[4] Pipeline.anonymise_text()")

from pipeline import anonymise_text
from generation.logic import MimicGen

# 4a. Text with clear pattern PII → sanitised, map populated
pii_text = "Contact me at sarah@example.com, SSN 123-45-6789"
sanitised, smap = anonymise_text(pii_text)

check(
    "Real email not in sanitised text",
    "sarah@example.com" not in sanitised,
    f"sanitised: {sanitised!r}"
)
check(
    "Real SSN not in sanitised text",
    "123-45-6789" not in sanitised,
    f"sanitised: {sanitised!r}"
)
check(
    "Surrogate map non-empty for PII text",
    len(smap) >= 1,
    f"map keys: {list(smap.keys())}"
)
check(
    "All surrogate map keys and values are strings",
    all(isinstance(k, str) and isinstance(v, str) for k, v in smap.items()),
)
check(
    "Sanitised text still has content",
    isinstance(sanitised, str) and len(sanitised) > 10,
)

# 4b. Clean text → unchanged, empty surrogate map
clean_text = "The quick brown fox jumps over the lazy dog."
sanitised_clean, smap_clean = anonymise_text(clean_text)
check(
    "Clean text passes through unchanged",
    sanitised_clean == clean_text,
    f"got: {sanitised_clean!r}"
)
check(
    "Clean text produces empty surrogate map",
    len(smap_clean) == 0,
    f"map: {smap_clean}"
)

# 4c. Shared MimicGen produces no duplicate surrogates across two calls
shared_mimic = MimicGen()
_, smap1 = anonymise_text("Email me at alice@test.com", mimic=shared_mimic)
_, smap2 = anonymise_text("Contact bob@test.com", mimic=shared_mimic)
all_surrogates = list(smap1.values()) + list(smap2.values())
check(
    "Shared MimicGen produces no duplicate surrogates across two calls",
    len(all_surrogates) == len(set(all_surrogates)),
    f"surrogates: {all_surrogates}"
)

# 4d. Return types are always correct regardless of input
for test_input in ["", "hello", "email me at x@y.com"]:
    s_out, m_out = anonymise_text(test_input)
    ok = isinstance(s_out, str) and isinstance(m_out, dict)
    check(
        f"anonymise_text always returns (str, dict) for {test_input!r}",
        ok,
        f"types: {type(s_out).__name__}, {type(m_out).__name__}"
    )

# 4e. Credit card PII detected and replaced
cc_text = "My card number is 4532015112830366"
sanitised_cc, smap_cc = anonymise_text(cc_text)
check(
    "Credit card number not in sanitised text",
    "4532015112830366" not in sanitised_cc,
    f"sanitised: {sanitised_cc!r}"
)

# 4f. UK phone PII detected and replaced
phone_text = "Call me at +44 7911 123456 please"
sanitised_ph, smap_ph = anonymise_text(phone_text)
check(
    "UK phone not in sanitised text",
    "+44 7911 123456" not in sanitised_ph,
    f"sanitised: {sanitised_ph!r}"
)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ✅  All quasi-ID, settings, and pipeline tests passed")
else:
    failed = total - passed
    print(f"  ❌  {failed} test(s) failed — see ❌ above")
print("=" * 60 + "\n")
