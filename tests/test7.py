# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test7.py — SurrogateShield Python Library Tests

Tests the standalone python-library/ package (surrogateshield) which is
completely independent from the main application code.

Covers:
  1.  Library entities      — DetectedEntity, mask_spans, remove_span_overlap
  2.  ShadowMap memory mode — update / get_all / flush with no disk I/O
  3.  ShadowMap persistent  — AES-256-GCM encrypt / decrypt round-trip
  4.  Response parser       — extract_text() for all four response formats
  5.  State singletons      — cfg defaults, session reset, lazy initialisation
  6.  pipeline pii_off      — alias resolution and type-exclusion filtering
  7.  pipeline thresholds   — threshold parameters wired through correctly
  8.  Library ResolvePass   — fuzzy_threshold param, simplified interface
  9.  config() validation   — defaults, pii_mem path guard
  10. scan() / pii_finder   — returns dict, comprehensive regardless of pii_off
  11. mask() round-trip     — real PII absent from sanitised text
  12. unmask() formats      — accepts str, Anthropic, OpenAI, Gemini objects
  13. flush()               — session reset, new id, shadow map cleared

Run from inside SurrogateShield/:
    python tests/test7.py

No API key needed — all tests are local only.
"""

import re
import sys
import os
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, "./python-library")

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, note=""):
    symbol = PASS if condition else FAIL
    results.append(condition)
    print(f"  {symbol}  {label}" + (f"  [{note}]" if note else ""))

print("\n" + "=" * 60)
print("  SurrogateShield — Python Library Tests")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. LIBRARY ENTITIES
# ─────────────────────────────────────────────────────────────
print("\n[1] Library entities (surrogateshield.core.entities)")

from surrogateshield.core.entities import (
    DetectedEntity,
    mask_spans,
    remove_span_overlap,
)

# DetectedEntity construction and defaults
ent_a = DetectedEntity(text="alice@example.com", start=0, end=17, type="email")
check("DetectedEntity score defaults to 1.0", ent_a.score == 1.0)
check("DetectedEntity source defaults to 'pattern'", ent_a.source == "pattern")
check("DetectedEntity text stored correctly", ent_a.text == "alice@example.com")

# overlaps()
ent_b = DetectedEntity("Smith", 5, 10, "PERSON")
ent_c = DetectedEntity("Jones", 12, 17, "PERSON")
ent_d = DetectedEntity("over", 7, 14, "ORG")
check("Non-overlapping spans: overlaps() returns False", not ent_b.overlaps(ent_c))
check("Overlapping spans: overlaps() returns True",      ent_d.overlaps(ent_b))
check("overlaps() is symmetric",                         ent_b.overlaps(ent_d) == ent_d.overlaps(ent_b))

# mask_spans()
text = "Hi John Smith, your email is john@test.com"
ent_name  = DetectedEntity("John Smith", 3, 13, "PERSON")
ent_email = DetectedEntity("john@test.com", 29, 42, "email")
masked = mask_spans(text, [ent_name, ent_email])
check("mask_spans replaces name span with placeholder",  "John Smith" not in masked)
check("mask_spans replaces email span with placeholder", "john@test.com" not in masked)
check("mask_spans preserves non-PII text",              "Hi" in masked and "your email is" in masked)
check("mask_spans on empty entities returns original",  mask_spans(text, []) == text)
check("mask_spans default placeholder is block char",   "█" in masked)

# remove_span_overlap()
existing = [DetectedEntity("x@x.com", 0, 7, "email")]
candidate_overlap    = DetectedEntity("x@x.com", 0, 7, "email")
candidate_no_overlap = DetectedEntity("John",   10, 14, "PERSON")
check("remove_span_overlap returns True for overlapping candidate",      remove_span_overlap(candidate_overlap, existing))
check("remove_span_overlap returns False for non-overlapping candidate", not remove_span_overlap(candidate_no_overlap, existing))
check("remove_span_overlap with empty existing always returns False",    not remove_span_overlap(candidate_overlap, []))


# ─────────────────────────────────────────────────────────────
# 2. SHADOWMAP — MEMORY MODE
# ─────────────────────────────────────────────────────────────
print("\n[2] ShadowMap — memory mode (no disk I/O)")

from surrogateshield.core.storage.shadow_map import ShadowMap

sm_mem = ShadowMap(session_id="lib-test-mem-001", storage_dir=None)

check("New memory ShadowMap starts empty",    len(sm_mem) == 0)
check("get_all() on empty map returns {}",    sm_mem.get_all() == {})

sm_mem.update({"FakeName": "RealName", "fake@mail.com": "real@mail.com"})
check("update() adds both entries",           len(sm_mem) == 2)
check("get_all() contains correct mapping",   sm_mem.get_all()["FakeName"] == "RealName")
check("get_all() returns a copy (not ref)",   sm_mem.get_all() is not sm_mem.get_all())

sm_mem.update({"FakeName": "UpdatedName"})
check("update() overwrites existing key",     sm_mem.get_all()["FakeName"] == "UpdatedName")

sm_mem.flush()
check("flush() clears in-memory dict",        len(sm_mem) == 0)
check("get_all() empty after flush()",        sm_mem.get_all() == {})

# Verify no files were written to disk
check("Memory mode creates no disk files",
      not any(f.endswith(".shadowmap") or f.endswith(".key")
              for f in os.listdir(".") if os.path.isfile(f)))


# ─────────────────────────────────────────────────────────────
# 3. SHADOWMAP — PERSISTENT MODE
# ─────────────────────────────────────────────────────────────
print("\n[3] ShadowMap — persistent mode (AES-256-GCM)")

with tempfile.TemporaryDirectory() as tmpdir:
    session_id = "lib-test-persist-001"

    sm1 = ShadowMap(session_id=session_id, storage_dir=tmpdir)
    sm1.update({"SurrogatePerson": "RealPerson"})
    sm1.update({"surrogate@fake.com": "real@real.com"})

    # Files must exist after update()
    map_path = os.path.join(tmpdir, f"{session_id}.shadowmap")
    key_path = os.path.join(tmpdir, f"{session_id}.key")
    check("Persistent mode creates .shadowmap file", os.path.exists(map_path))
    check("Persistent mode creates .key file",       os.path.exists(key_path))

    # File must be binary (encrypted), not plain JSON
    with open(map_path, "rb") as f:
        raw_bytes = f.read()
    check("Shadowmap file is not plain JSON (encrypted)",
          b"SurrogatePerson" not in raw_bytes)

    # Key file must be exactly 32 bytes, owner-only permissions
    key_stat = os.stat(key_path)
    check("Session key file is 32 bytes", key_stat.st_size == 32)
    check("Session key file has 0o600 permissions",
          oct(key_stat.st_mode)[-3:] == "600")

    # Round-trip: load from disk into fresh ShadowMap instance
    sm2 = ShadowMap(session_id=session_id, storage_dir=tmpdir)
    check("Persistent ShadowMap: person surrogate round-trips correctly",
          sm2.get_all().get("SurrogatePerson") == "RealPerson")
    check("Persistent ShadowMap: email surrogate round-trips correctly",
          sm2.get_all().get("surrogate@fake.com") == "real@real.com")
    check("Persistent ShadowMap: loaded entry count correct",
          len(sm2) == 2)

    # flush() must delete both files
    sm2.flush()
    check("flush() clears in-memory mappings",      len(sm2) == 0)
    check("flush() deletes .shadowmap file",         not os.path.exists(map_path))
    check("flush() deletes .key file",               not os.path.exists(key_path))

    # A fresh load after delete starts empty (graceful)
    sm3 = ShadowMap(session_id=session_id, storage_dir=tmpdir)
    check("New ShadowMap after flush starts empty", len(sm3) == 0)

    # Different session IDs produce different ciphertext (keys differ)
    sm_a = ShadowMap(session_id="lib-key-test-aaa", storage_dir=tmpdir)
    sm_b = ShadowMap(session_id="lib-key-test-bbb", storage_dir=tmpdir)
    sm_a.update({"same_surrogate": "same_original"})
    sm_b.update({"same_surrogate": "same_original"})

    raw_a = open(os.path.join(tmpdir, "lib-key-test-aaa.shadowmap"), "rb").read()
    raw_b = open(os.path.join(tmpdir, "lib-key-test-bbb.shadowmap"), "rb").read()
    check("Different session IDs produce different ciphertext",
          raw_a != raw_b)
    sm_a.flush()
    sm_b.flush()


# ─────────────────────────────────────────────────────────────
# 4. RESPONSE PARSER — extract_text()
# ─────────────────────────────────────────────────────────────
print("\n[4] Response parser — extract_text()")

from surrogateshield._response_parser import extract_text

# Plain string — returned as-is
check("Plain string returned unchanged", extract_text("hello world") == "hello world")
check("Empty string returned unchanged", extract_text("") == "")

# Anthropic style: .content[0].text
class _AnthropicItem:
    text = "Claude says hello"
class _AnthropicResponse:
    content = [_AnthropicItem()]
check("Anthropic-style response: content[0].text extracted",
      extract_text(_AnthropicResponse()) == "Claude says hello")

# Anthropic with empty content list → falls through gracefully
class _AnthropicEmpty:
    content = []
result_empty_content = extract_text(_AnthropicEmpty())
check("Anthropic-style with empty content does not crash",
      isinstance(result_empty_content, str))

# OpenAI style: .choices[0].message.content
class _OAIMessage:
    content = "GPT says hello"
class _OAIChoice:
    message = _OAIMessage()
class _OAIResponse:
    choices = [_OAIChoice()]
check("OpenAI-style response: choices[0].message.content extracted",
      extract_text(_OAIResponse()) == "GPT says hello")

# Gemini style: .text (no .choices attribute)
class _GeminiResponse:
    text = "Gemini says hello"
    # No .choices — must NOT match OpenAI path
check("Gemini-style response: .text extracted",
      extract_text(_GeminiResponse()) == "Gemini says hello")

# Fallback: unknown object with __str__
class _UnknownResponse:
    def __str__(self):
        return "unknown response text"
check("Unknown response type falls back to str()",
      extract_text(_UnknownResponse()) == "unknown response text")

# Anthropic takes precedence over Gemini (both have .text via content block)
# Verify Anthropic path fires when .content is a list
class _AmbiguousResponse:
    class _block:
        text = "via anthropic path"
    content = [_block()]
    text = "via gemini path"
check("Anthropic path takes precedence over Gemini when .content is a list",
      extract_text(_AmbiguousResponse()) == "via anthropic path")


# ─────────────────────────────────────────────────────────────
# 5. STATE SINGLETONS — cfg and session
# ─────────────────────────────────────────────────────────────
print("\n[5] State singletons (_state.py)")

from surrogateshield._state import cfg, session, _Config, _Session

# cfg defaults
check("cfg.detailed_view defaults to True",              cfg.detailed_view is True)
check("cfg.pii_mem defaults to 'temp'",                  cfg.pii_mem == "temp")
check("cfg.pii_off defaults to []",                      cfg.pii_off == [])
check("cfg.service defaults to True",                    cfg.service is True)
check("cfg.spacy_model defaults to 'en_core_web_lg'",    cfg.spacy_model == "en_core_web_lg")
check("cfg.context_guard_enabled defaults to True",      cfg.context_guard_enabled is True)
check("cfg.entity_trace_high_threshold defaults 0.85",   cfg.entity_trace_high_threshold == 0.85)
check("cfg.entity_trace_low_threshold defaults 0.60",    cfg.entity_trace_low_threshold == 0.60)
check("cfg.context_guard_threshold defaults to 0.70",    cfg.context_guard_threshold == 0.70)
check("cfg.entity_trace_fallback_threshold defaults 0.65", cfg.entity_trace_fallback_threshold == 0.65)
check("cfg.fuzzy_threshold defaults to 85",              cfg.fuzzy_threshold == 85)

# session defaults
import uuid
check("session.id is a valid UUID4",
      len(session.id) == 36 and session.id.count("-") == 4)

# session.reset() generates a new id
old_id = session.id
session.reset()
check("session.reset() generates a new session ID", session.id != old_id)
check("New session ID is also a valid UUID4",
      len(session.id) == 36 and session.id.count("-") == 4)

# get_mimic() lazy initialisation
mimic_first  = session.get_mimic()
mimic_second = session.get_mimic()
check("get_mimic() returns same instance on repeated calls", mimic_first is mimic_second)

# After reset(), get_mimic() returns a NEW instance
session.reset()
mimic_after_reset = session.get_mimic()
check("get_mimic() returns new instance after reset()", mimic_after_reset is not mimic_first)

# get_shadow_map() lazy initialisation (memory mode since cfg.pii_mem='temp')
sm_first  = session.get_shadow_map()
sm_second = session.get_shadow_map()
check("get_shadow_map() returns same instance on repeated calls", sm_first is sm_second)

# reset() clears the shadow map data
sm_first.update({"fake": "real"})
session.reset()
sm_new = session.get_shadow_map()
check("Shadow map is fresh (empty) after reset()", sm_new.get_all() == {})
check("get_shadow_map() after reset returns new instance", sm_new is not sm_first)


# ─────────────────────────────────────────────────────────────
# 6. PIPELINE pii_off FILTERING
# ─────────────────────────────────────────────────────────────
print("\n[6] pipeline.py — pii_off filtering and alias resolution")

from surrogateshield.core.detection.pipeline import run_cascade, deduplicate

# Baseline: phone number is detected without pii_off
phone_text = "Call me at +1-555-867-5309 please"
confirmed_no_off, _ = run_cascade(phone_text, context_guard_enabled=False)
phone_detected = any(e.type in {"phone_us", "phone_uk", "phone_intl"} for e in confirmed_no_off)
check("phone_us detected without pii_off",       phone_detected)

# pii_off=["phone"] suppresses phone_us/phone_uk/phone_intl via alias
confirmed_phone_off, _ = run_cascade(phone_text, pii_off=["phone"], context_guard_enabled=False)
check("pii_off=['phone'] removes phone_us from confirmed",
      not any(e.type in {"phone_us", "phone_uk", "phone_intl"} for e in confirmed_phone_off))

# pii_off=["phone"] does not suppress email in the same message
mixed_text = "Email: bob@corp.com, phone: +1-555-867-5309"
confirmed_mixed, _ = run_cascade(mixed_text, pii_off=["phone"], context_guard_enabled=False)
check("pii_off=['phone'] still detects email",
      any(e.type == "email" for e in confirmed_mixed))
check("pii_off=['phone'] still suppresses phone_us",
      not any(e.type == "phone_us" for e in confirmed_mixed))

# pii_off with direct type string (not alias)
confirmed_email_off, _ = run_cascade(
    "Contact bob@corp.com", pii_off=["email"], context_guard_enabled=False
)
check("pii_off=['email'] direct type string suppresses email",
      not any(e.type == "email" for e in confirmed_email_off))

# pii_off=["zip"] alias → suppresses zip_us
zip_text = "My ZIP code is 90210"
confirmed_zip_off, _ = run_cascade(zip_text, pii_off=["zip"], context_guard_enabled=False)
check("pii_off=['zip'] alias suppresses zip_us",
      not any(e.type == "zip_us" for e in confirmed_zip_off))

# pii_off=["postal_code"] alias → suppresses both zip_us and postcode_uk
pc_text = "Postcode SW1A 1AA and ZIP 90210"
confirmed_pc_off, _ = run_cascade(pc_text, pii_off=["postal_code"], context_guard_enabled=False)
check("pii_off=['postal_code'] suppresses zip_us",
      not any(e.type == "zip_us" for e in confirmed_pc_off))
check("pii_off=['postal_code'] suppresses postcode_uk",
      not any(e.type == "postcode_uk" for e in confirmed_pc_off))

# _qi_matches and _skipped_entities attributes preserved after pii_off filtering
confirmed_attr, _ = run_cascade(
    "zip 90210, born 03/14/1990", pii_off=[], context_guard_enabled=False
)
check("_qi_matches attribute present on result",        hasattr(confirmed_attr, "_qi_matches"))
check("_skipped_entities attribute present on result",  hasattr(confirmed_attr, "_skipped_entities"))

confirmed_attr_off, _ = run_cascade(
    "zip 90210, born 03/14/1990", pii_off=["zip"], context_guard_enabled=False
)
check("_qi_matches preserved after pii_off filtering",
      hasattr(confirmed_attr_off, "_qi_matches"))
check("_skipped_entities preserved after pii_off filtering",
      hasattr(confirmed_attr_off, "_skipped_entities"))


# ─────────────────────────────────────────────────────────────
# 7. PIPELINE THRESHOLD PARAMETERS
# ─────────────────────────────────────────────────────────────
print("\n[7] pipeline.py — threshold parameters wired through correctly")

from unittest.mock import patch
import surrogateshield.core.detection.entity_trace as _et_mod
import surrogateshield.core.detection.context_guard as _cg_mod

# Build a fake entity for the mock to return
_fake_ent = DetectedEntity("Alice", 3, 8, "PERSON", score=0.75, source="ner")

# Mock entity_trace.trace to return a borderline entity at score=0.75
def _mock_trace(text, existing_entities=None, spacy_model="en_core_web_lg",
                high_threshold=0.85, low_threshold=0.60):
    # 0.75 is above 0.60 (low) but below 0.85 (high) → borderline
    if 0.75 >= high_threshold:
        return [_fake_ent], []
    elif 0.75 >= low_threshold:
        return [], [_fake_ent]
    else:
        return [], []

with patch.object(_et_mod, "trace", side_effect=_mock_trace):
    # With default thresholds (high=0.85, low=0.60): score 0.75 → borderline
    confirmed_default, _ = run_cascade(
        "Hi Alice",
        context_guard_enabled=False,
        entity_trace_high_threshold=0.85,
        entity_trace_low_threshold=0.60,
        entity_trace_fallback_threshold=0.80,  # 0.75 < 0.80 → not promoted
    )
    check("Borderline entity (0.75) below fallback threshold (0.80) not promoted",
          not any(e.text == "Alice" for e in confirmed_default))

    confirmed_low_fallback, _ = run_cascade(
        "Hi Alice",
        context_guard_enabled=False,
        entity_trace_high_threshold=0.85,
        entity_trace_low_threshold=0.60,
        entity_trace_fallback_threshold=0.70,  # 0.75 > 0.70 → promoted
    )
    check("Borderline entity (0.75) above fallback threshold (0.70) is promoted",
          any(e.text == "Alice" for e in confirmed_low_fallback))

    # With lower high_threshold: score 0.75 becomes confirmed directly
    confirmed_low_high, _ = run_cascade(
        "Hi Alice",
        context_guard_enabled=False,
        entity_trace_high_threshold=0.70,   # 0.75 >= 0.70 → confirmed immediately
        entity_trace_low_threshold=0.50,
        entity_trace_fallback_threshold=0.65,
    )
    check("Score 0.75 confirmed when high_threshold lowered to 0.70",
          any(e.text == "Alice" for e in confirmed_low_high))

# Verify context_guard_enabled=False path skips guard() entirely
guard_called = []
original_guard = _cg_mod.guard
def _tracking_guard(*args, **kwargs):
    guard_called.append(True)
    return original_guard(*args, **kwargs)

with patch.object(_cg_mod, "guard", side_effect=_tracking_guard):
    run_cascade("test text", context_guard_enabled=False)
check("context_guard_enabled=False skips guard() call",  len(guard_called) == 0)

with patch.object(_cg_mod, "guard", side_effect=_tracking_guard):
    run_cascade("test text", context_guard_enabled=True)
check("context_guard_enabled=True calls guard() exactly once", len(guard_called) == 1)


# ─────────────────────────────────────────────────────────────
# 8. LIBRARY RESOLVEPASS
# ─────────────────────────────────────────────────────────────
print("\n[8] Library ResolvePass (surrogateshield.core.reconstruction.resolve)")

from surrogateshield.core.reconstruction.resolve import ResolvePass

rp = ResolvePass()

# Basic exact replacement
shadow = {"Marcus Ellison": "Ahmed Al-Rashidi", "fake@mail.com": "real@mail.com"}
response = "Hello Marcus Ellison, we reached you at fake@mail.com."
restored = rp.resolve(response, shadow)
check("Exact hit: real name restored",            "Ahmed Al-Rashidi" in restored)
check("Exact hit: real email restored",           "real@mail.com" in restored)
check("Exact hit: surrogate name removed",        "Marcus Ellison" not in restored)
check("Exact hit: surrogate email removed",       "fake@mail.com" not in restored)

# Empty shadow map
check("Empty shadow map returns text unchanged",
      rp.resolve("no surrogates here", {}) == "no surrogates here")

# fuzzy_threshold parameter wired through
# Create a response where the surrogate is slightly misspelled
rp_fuzzy = ResolvePass()
shadow_fuzzy = {"Ellison Marcus": "Ahmed Al-Rashidi"}
response_typo = "Hello Ellison Marcus, welcome back."
restored_fuzzy_85 = rp_fuzzy.resolve(response_typo, shadow_fuzzy, fuzzy_threshold=85)
check("Exact match with properly cased surrogate works",
      "Ahmed Al-Rashidi" in restored_fuzzy_85)

# Very high threshold should suppress fuzzy matches
rp_strict = ResolvePass()
shadow_strict = {"XyzAbcDefinitely NotPresent": "RealPerson"}
response_absent = "This response has completely different content."
restored_strict = rp_strict.resolve(response_absent, shadow_strict, fuzzy_threshold=100)
check("fuzzy_threshold=100 does not restore absent surrogate",
      "RealPerson" not in restored_strict)

# Library ResolvePass does NOT have a 'failures' attribute (simplified version)
check("Library ResolvePass has no .failures attribute",
      not hasattr(rp, "failures"))
check("Library ResolvePass has no get_failure_summary() method",
      not hasattr(rp, "get_failure_summary"))

# Component matching: multi-word surrogate where model used only first name
rp_comp = ResolvePass()
shadow_comp = {"Rachel Torres": "Sarah Mitchell"}
response_comp = "Thanks for your message, Rachel. We'll get back to you."
restored_comp = rp_comp.resolve(response_comp, shadow_comp)
check("Component match: 'Rachel' (partial surrogate) resolved to 'Sarah'",
      "Sarah" in restored_comp)


# ─────────────────────────────────────────────────────────────
# 9. PUBLIC API — config() VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n[9] Public API — config() validation")

import surrogateshield as ss

# Reset to known state
ss.flush()
ss.config(detailed_view=False)  # silence output for all remaining tests

# Default values applied
ss.config()
check("config() detailed_view default is True",             ss._state.cfg.detailed_view is True)
ss.config(detailed_view=False)
check("config() detailed_view=False applied correctly",     ss._state.cfg.detailed_view is False)

ss.config(fuzzy_threshold=75)
check("config() fuzzy_threshold=75 applied correctly",      ss._state.cfg.fuzzy_threshold == 75)
ss.config(fuzzy_threshold=85)   # restore

ss.config(pii_off=["phone", "location"])
check("config() pii_off=['phone','location'] stored",       ss._state.cfg.pii_off == ["phone", "location"])
ss.config(pii_off=None)
check("config() pii_off=None stored as empty list",         ss._state.cfg.pii_off == [])

ss.config(spacy_model="en_core_web_sm")
check("config() spacy_model updated",                       ss._state.cfg.spacy_model == "en_core_web_sm")
ss.config(spacy_model="en_core_web_lg")   # restore

ss.config(service=False)
check("config() service=False applied",                     ss._state.cfg.service is False)
ss.config(service=True)   # restore

# pii_mem="temp" is always valid
try:
    ss.config(pii_mem="temp")
    check("config() pii_mem='temp' does not raise",         True)
except Exception:
    check("config() pii_mem='temp' does not raise",         False)

# pii_mem with non-existent path raises ValueError
try:
    ss.config(pii_mem="/this/path/does/not/exist/xyzzy")
    check("config() non-existent pii_mem path raises ValueError", False)
except ValueError:
    check("config() non-existent pii_mem path raises ValueError", True)

# pii_mem with a valid existing directory succeeds
with tempfile.TemporaryDirectory() as tmpdir:
    try:
        ss.config(pii_mem=tmpdir)
        check("config() valid directory path accepted", True)
    except Exception as exc:
        check("config() valid directory path accepted", False, str(exc))
    finally:
        ss.config(pii_mem="temp")   # restore

# pii_mem with a file path (not a directory) raises ValueError
with tempfile.NamedTemporaryFile() as tmpf:
    try:
        ss.config(pii_mem=tmpf.name)
        check("config() file path (not directory) raises ValueError", False)
    except ValueError:
        check("config() file path (not directory) raises ValueError", True)


# ─────────────────────────────────────────────────────────────
# 10. PUBLIC API — scan() / pii_finder
# ─────────────────────────────────────────────────────────────
print("\n[10] Public API — scan() / pii_finder")

ss.config(detailed_view=False)
ss.flush()

# scan() return type
result = ss.scan("Contact alice@corp.com, SSN 123-45-6789.")
check("scan() returns a dict",                  isinstance(result, dict))
check("scan() dict: email value detected",      "alice@corp.com" in result)
check("scan() dict: SSN value detected",        "123-45-6789" in result)
check("scan() dict: email mapped to 'email'",   result.get("alice@corp.com") == "email")
check("scan() dict: SSN mapped to 'ssn'",       result.get("123-45-6789") == "ssn")

# scan() is comprehensive: ignores pii_off
ss.config(pii_off=["email"])
result_with_off = ss.scan("Contact alice@corp.com, SSN 123-45-6789.")
check("scan() detects email even when pii_off=['email']",
      "alice@corp.com" in result_with_off)
ss.config(pii_off=[])   # restore

# scan() with no PII returns empty dict
empty_result = ss.scan("The weather is nice today.")
check("scan() returns empty dict for PII-free text",   empty_result == {})

# pii_finder is the same function object
check("pii_finder is an alias for scan()",
      ss.pii_finder is ss.scan)

# pii_finder produces same result as scan()
pii_finder_result = ss.pii_finder("Call me at +1-555-123-4567")
scan_result       = ss.scan("Call me at +1-555-123-4567")
check("pii_finder() returns same result as scan()",   pii_finder_result == scan_result)

# scan() does not update the session shadow map
ss.flush()
shadow_before = ss._state.session.get_shadow_map().get_all()
ss.scan("My email is scan@test.com")
shadow_after = ss._state.session.get_shadow_map().get_all()
check("scan() does not update the session shadow map",
      shadow_before == shadow_after)


# ─────────────────────────────────────────────────────────────
# 11. PUBLIC API — mask() ROUND-TRIP
# ─────────────────────────────────────────────────────────────
print("\n[11] Public API — mask() round-trip")

ss.config(detailed_view=False)
ss.flush()

# Baseline: pattern PII is reliably sanitised (no NER needed)
msg = "Hi, my email is carol@example.com, SSN 123-45-6789, and card 4532015112830366."
sanitized = ss.mask(msg)

check("Real email absent from sanitised text",      "carol@example.com" not in sanitized)
check("Real SSN absent from sanitised text",        "123-45-6789" not in sanitized)
check("Real credit card absent from sanitised text","4532015112830366" not in sanitized)
check("Sanitised text still has content",           len(sanitized) > 20)
check("Sanitised text retains non-PII words",
      "Hi" in sanitized and "my" in sanitized)

# Shadow map was populated after mask()
shadow = ss._state.session.get_shadow_map().get_all()
check("Shadow map non-empty after mask()",          len(shadow) > 0)
check("Shadow map values are the original PII",
      "carol@example.com" in shadow.values() or
      "123-45-6789" in shadow.values())

# Calling mask() twice with same PII reuses same surrogates (session-level MimicGen)
sanitized2 = ss.mask(msg)
check("Second mask() call produces same surrogates for same real values",
      sanitized == sanitized2)

# pii_off respected in mask()
ss.flush()
ss.config(pii_off=["email"])
sanitized_off = ss.mask("Email bob@test.com, SSN 987-65-4321")
check("mask() with pii_off=['email'] keeps real email",
      "bob@test.com" in sanitized_off)
check("mask() with pii_off=['email'] still replaces SSN",
      "987-65-4321" not in sanitized_off)
ss.config(pii_off=[])   # restore

# Clean text passes through unchanged
ss.flush()
clean_text = "The quick brown fox jumps over the lazy dog."
sanitized_clean = ss.mask(clean_text)
check("Clean text unchanged by mask()", sanitized_clean == clean_text)


# ─────────────────────────────────────────────────────────────
# 12. PUBLIC API — unmask() FORMAT SUPPORT
# ─────────────────────────────────────────────────────────────
print("\n[12] Public API — unmask() format support")

ss.config(detailed_view=False)
ss.flush()

# Populate shadow map via mask()
msg = "My email is diana@work.com and my phone is +1-555-999-1234."
sanitized = ss.mask(msg)

# Identify the surrogate values used
shadow_map = ss._state.session.get_shadow_map().get_all()
email_surrogate = next((k for k, v in shadow_map.items() if v == "diana@work.com"), None)
phone_surrogate = next((k for k, v in shadow_map.items() if v == "+1-555-999-1234"), None)

# Build a simulated LLM response that echoes the surrogates
if email_surrogate and phone_surrogate:
    fake_reply = f"I've noted {email_surrogate} and {phone_surrogate}."
elif email_surrogate:
    fake_reply = f"I've noted {email_surrogate}."
else:
    fake_reply = sanitized   # fallback in case detection missed something

# unmask() with plain string
restored_str = ss.unmask(fake_reply)
check("unmask() with plain string: real email restored",
      "diana@work.com" in restored_str)

# unmask() with Anthropic-style response object
class _MockAnthropic:
    class _item:
        pass
    _item.text = fake_reply
    content = [_item()]
restored_anthropic = ss.unmask(_MockAnthropic())
check("unmask() with Anthropic response object: real email restored",
      "diana@work.com" in restored_anthropic)

# unmask() with OpenAI-style response object
class _MockOAI:
    class _msg:
        pass
    _msg.content = fake_reply
    class _choice:
        pass
    _choice.message = _msg()
    choices = [_choice()]
restored_oai = ss.unmask(_MockOAI())
check("unmask() with OpenAI response object: real email restored",
      "diana@work.com" in restored_oai)

# unmask() with Gemini-style response object
class _MockGemini:
    text = fake_reply
restored_gemini = ss.unmask(_MockGemini())
check("unmask() with Gemini response object: real email restored",
      "diana@work.com" in restored_gemini)

# unmask() on text with no surrogates returns text unchanged
ss.flush()
ss.mask("fresh start xyz")  # populate map with something unrelated
plain_reply = "There is nothing sensitive here."
restored_plain = ss.unmask(plain_reply)
check("unmask() on surrogate-free text returns it unchanged",
      restored_plain == plain_reply)


# ─────────────────────────────────────────────────────────────
# 13. PUBLIC API — flush()
# ─────────────────────────────────────────────────────────────
print("\n[13] Public API — flush()")

ss.config(detailed_view=False)

# Populate session state
ss.flush()
ss.mask("My email is test@test.com and SSN 111-22-3333")
id_before = ss._state.session.id
map_before = ss._state.session.get_shadow_map().get_all()

check("Shadow map non-empty before flush()",    len(map_before) > 0)
check("Session ID exists before flush()",       len(id_before) > 0)

ss.flush()

id_after  = ss._state.session.id
map_after = ss._state.session.get_shadow_map().get_all()

check("flush() generates new session ID",       id_after != id_before)
check("flush() clears the shadow map",          map_after == {})
check("New session ID is a valid UUID4",
      len(id_after) == 36 and id_after.count("-") == 4)

# After flush, unmask() can no longer resolve old surrogates
# (shadow map is empty, so surrogate strings pass through unchanged)
old_surrogate = list(map_before.keys())[0] if map_before else "nonexistent_surrogate"
response_with_old = f"The answer involves {old_surrogate}."
restored_after_flush = ss.unmask(response_with_old)
check("After flush, old surrogates are no longer resolved",
      # The surrogate stays in the text because the map is empty
      old_surrogate in restored_after_flush)

# Multiple flush() calls in a row are safe
try:
    ss.flush()
    ss.flush()
    ss.flush()
    check("Multiple consecutive flush() calls do not crash", True)
except Exception as exc:
    check("Multiple consecutive flush() calls do not crash", False, str(exc))


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ✅  All python-library tests passed")
else:
    failed = total - passed
    print(f"  ❌  {failed} test(s) failed — see ❌ above")
print("=" * 60 + "\n")
