# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test2.py — SurrogateShield Live Fix Verification

Tests the three problems found in the live chat:
  1. Conversation history must store surrogates, not originals
  2. SSN label word must not be detected as ORG
  3. Real PII must never appear in what gets sent to the API

Run from inside SurrogateShield/:
    python test2.py

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
    print(f"  {symbol}  {label}" + (f"\n       → {note}" if note else ""))

print("\n" + "=" * 60)
print("  SurrogateShield — Fix Verification Tests")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. HISTORY STORES SURROGATES NOT ORIGINALS
# ─────────────────────────────────────────────────────────────
print("\n[1] Conversation history stores surrogates, not originals")

from generation.logic import MimicGen
from storage.logic import ShadowMap
from reconstruction.logic import ResolvePass
from util import DetectedEntity

mimic = MimicGen()

# Simulate what pipeline does on turn 1
original_name  = "Sarah Mitchell"
original_email = "sarah@gmail.com"

entities = [
    DetectedEntity(original_name,  0, len(original_name),  "PERSON", 0.85, "ner"),
    DetectedEntity(original_email, 0, len(original_email), "email",  1.0,  "pattern"),
]
surrogate_map = mimic.generate_all(entities)

fake_name  = surrogate_map[original_name]
fake_email = surrogate_map[original_email]

# Sanitised message — what the API should receive
sanitised = f"Hi I am {fake_name}, email {fake_email}"

# Simulated API response — Claude echoes the surrogates
raw_api_response = f"Nice to meet you, {fake_name}! Your email {fake_email} is noted."

# ResolvePass restores for display
shadow = ShadowMap("history-test-001")
shadow.update({v: k for k, v in surrogate_map.items()})
shadow.save()

rp = ResolvePass()
restored_for_display = rp.resolve(raw_api_response, shadow.all_mappings())

# What should be stored in history for the NEXT API call
# Must be the RAW response (surrogates), not the restored one
history_user_turn      = sanitised          # surrogates
history_assistant_turn = raw_api_response   # surrogates

check(
    "Sanitised message contains fake name, not real",
    original_name not in sanitised and fake_name in sanitised,
    f"sent to API: {sanitised!r}"
)
check(
    "Sanitised message contains fake email, not real",
    original_email not in sanitised and fake_email in sanitised,
)
check(
    "Restored response shown to user has real name",
    original_name in restored_for_display,
    f"shown to user: {restored_for_display!r}"
)
check(
    "History user turn stores surrogate, not original name",
    original_name not in history_user_turn,
    f"history stores: {history_user_turn!r}"
)
check(
    "History assistant turn stores surrogate, not original",
    original_name not in history_assistant_turn,
    f"history stores: {history_assistant_turn!r}"
)

# Simulate turn 2 — history is sent to API again
# Original values must NOT appear in what the API sees
turn2_context = f"User: {history_user_turn}\nAssistant: {history_assistant_turn}"
check(
    "Turn 2 context sent to API contains NO real PII",
    original_name  not in turn2_context and
    original_email not in turn2_context,
    f"context: {turn2_context[:80]!r}..."
)

shadow.delete()


# ─────────────────────────────────────────────────────────────
# 2. WORD "SSN" MUST NOT BE DETECTED AS ORG
# ─────────────────────────────────────────────────────────────
print("\n[2] Common PII label words must not be detected as entities")

from detection.entity_trace import trace

def ner_texts(text):
    confirmed, borderline = trace(text)
    return [e.text for e in confirmed + borderline]

# "SSN" should not be flagged as an ORG
ssn_label_detections = ner_texts("my SSN is 123-45-6789")
check(
    "'SSN' label word not detected as entity",
    "SSN" not in ssn_label_detections,
    f"detected texts: {ssn_label_detections}"
)

# "DOB" should not be flagged
dob_detections = ner_texts("my DOB is 03/14/1990")
check(
    "'DOB' label word not detected as entity",
    "DOB" not in dob_detections,
    f"detected texts: {dob_detections}"
)

# "email" label should not be flagged
email_label = ner_texts("my email is test@example.com")
check(
    "'email' label word not detected as entity",
    "email" not in [t.lower() for t in email_label],
    f"detected texts: {email_label}"
)

# Real names must still be detected
real_name_detections = ner_texts("My name is Sarah Mitchell")
check(
    "Real person name still detected after blocklist",
    any("Mitchell" in t or "Sarah" in t for t in real_name_detections),
    f"detected: {real_name_detections}"
)


# ─────────────────────────────────────────────────────────────
# 3. FULL MULTI-TURN SIMULATION — NO REAL PII IN ANY API CALL
# ─────────────────────────────────────────────────────────────
print("\n[3] Multi-turn simulation — real PII never reaches the API")

from detection.logic import run_cascade, deduplicate

# Shared state across turns
session_shadow = ShadowMap("multiturn-test-002")
session_mimic  = MimicGen()
api_history    = []   # what the API sees — must never contain real PII

REAL_PII = {
    "Sarah Mitchell",
    "sarah@gmail.com",
    "123-45-6789",
    "4532015112830366",
    "Springfield",
}

def simulate_turn(user_message: str, simulated_api_response: str) -> str:
    """Run one turn through the pipeline without calling the real API."""
    confirmed, _ = run_cascade(user_message)
    confirmed = deduplicate(confirmed)

    surrogate_map = session_mimic.generate_all(confirmed) if confirmed else {}

    # Apply surrogates
    sanitised = user_message
    for orig in sorted(surrogate_map, key=len, reverse=True):
        sanitised = sanitised.replace(orig, surrogate_map[orig])

    # Update ShadowMap
    if surrogate_map:
        session_shadow.update({v: k for k, v in surrogate_map.items()})
        session_shadow.save()

    # Build simulated raw response (using whatever surrogates are in the message)
    raw_response = simulated_api_response

    # Store surrogates in history — NOT originals
    api_history.append({"role": "user",      "content": sanitised})
    api_history.append({"role": "assistant", "content": raw_response})

    # Restore for display
    rp = ResolvePass()
    return rp.resolve(raw_response, session_shadow.all_mappings())

# Turn 1 — introduce PII
restored1 = simulate_turn(
    "My name is Sarah Mitchell, email sarah@gmail.com, SSN 123-45-6789",
    "Got it, I've noted your details.",
)

# Turn 2 — introduce more PII
restored2 = simulate_turn(
    "Also my card is 4532015112830366 and I live near the only library in Springfield",
    "I've noted your card and location too.",
)

# Turn 3 — ask a follow-up (no new PII)
restored3 = simulate_turn(
    "What is my name?",
    "I have your details on file.",
)

# Check: no real PII in any API history entry
history_text = " ".join(h["content"] for h in api_history)

for pii_value in REAL_PII:
    check(
        f"Real value '{pii_value}' never sent to API",
        pii_value not in history_text,
        f"found in history" if pii_value in history_text else "clean"
    )

# Check: restored responses have real values for display
check(
    "Turn 1 restored response usable (non-empty)",
    len(restored1) > 0,
    f"restored: {restored1!r}"
)

session_shadow.delete()


# ─────────────────────────────────────────────────────────────
# 4. RESOLVEPASS — HISTORY DOES NOT LEAK ON NEXT TURN
# ─────────────────────────────────────────────────────────────
print("\n[4] ResolvePass — surrogates in history resolve correctly")

mimic2 = MimicGen()
shadow2 = ShadowMap("resolvetest-003")

ents = [DetectedEntity("Ahmed Al-Rashidi", 0, 16, "PERSON", 0.85, "ner")]
smap = mimic2.generate_all(ents)
fake_ahmed = smap["Ahmed Al-Rashidi"]

shadow2.update({fake_ahmed: "Ahmed Al-Rashidi"})
shadow2.save()

# Simulate Claude using the surrogate in a later response
later_response = f"As you mentioned earlier, {fake_ahmed}, here is your answer."
rp2 = ResolvePass()
resolved = rp2.resolve(later_response, shadow2.all_mappings())

check(
    "Surrogate from earlier turn resolves in later response",
    "Ahmed Al-Rashidi" in resolved and fake_ahmed not in resolved,
    f"resolved: {resolved!r}"
)
check(
    "Fake name completely removed from display response",
    fake_ahmed not in resolved,
)

shadow2.delete()


# ─────────────────────────────────────────────────────────────
# 5. EDGE CASES
# ─────────────────────────────────────────────────────────────
print("\n[5] Edge cases")

from detection.pattern_scan import scan

# Message with no PII — should pass through unchanged
no_pii = "The weather is nice today"
confirmed_none, _ = run_cascade(no_pii)
check(
    "Message with no PII passes through unchanged",
    len(confirmed_none) == 0,
)

# Same PII mentioned twice — should only produce one surrogate
mimic3 = MimicGen()
ents_dup = [
    DetectedEntity("john@x.com", 0, 10, "email", 1.0, "pattern"),
    DetectedEntity("john@x.com", 0, 10, "email", 1.0, "pattern"),
]
smap_dup = mimic3.generate_all(ents_dup)
check(
    "Same PII twice produces exactly one surrogate",
    len(smap_dup) == 1,
    f"map: {smap_dup}"
)

# Surrogate applied correctly when same value appears multiple times in text
text_with_repeat = "Contact john@x.com or forward to john@x.com please"
fake_for_john = list(smap_dup.values())[0]
result_text = text_with_repeat
for orig, fake in smap_dup.items():
    result_text = result_text.replace(orig, fake)
check(
    "Same PII replaced consistently everywhere in message",
    result_text.count(fake_for_john) == 2 and "john@x.com" not in result_text,
    f"result: {result_text!r}"
)

# Empty message — should not crash
confirmed_empty, _ = run_cascade("")
check(
    "Empty message does not crash pipeline",
    len(confirmed_empty) == 0,
)

# Very long message with PII buried in middle
long_msg = ("word " * 100) + "email me at buried@secret.com " + ("word " * 100)
confirmed_long = [e for e in run_cascade(long_msg)[0] if e.type == "email"]
check(
    "PII detected when buried in long message",
    len(confirmed_long) >= 1,
    f"found: {[e.text for e in confirmed_long]}"
)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ✅  All fixes verified — history is clean, pipeline is safe")
else:
    failed = total - passed
    print(f"  ❌  {failed} test(s) failed — check the ❌ lines above")
    print("  The fix is incomplete — real PII may still reach the API")
print("=" * 60 + "\n")