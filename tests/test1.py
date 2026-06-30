# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
test1.py — SurrogateShield test suite
Run from inside the SurrogateShield/ directory:
    python tests/test1.py
"""

import sys
sys.path.insert(0, ".")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
results = []

def check(label, condition, note=""):
    symbol = PASS if condition else FAIL
    results.append(condition)
    print(f"  {symbol}  {label}" + (f"  [{note}]" if note else ""))

print("\n" + "="*60)
print("  SurrogateShield — Test Suite")
print("="*60)


# ─────────────────────────────────────────────────────────────
# 1. PATTERN SCAN
# ─────────────────────────────────────────────────────────────
print("\n[1] PatternScan")

from detection.pattern_scan import scan

def has(text, expected_type):
    return any(e.type == expected_type for e in scan(text))

def empty(text):
    return len(scan(text)) == 0

check("Email detected",              has("email me at test@example.com", "email"))
check("SSN with dashes",             has("SSN is 123-45-6789", "ssn"))
check("SSN no dashes",               has("SSN 123456789", "ssn"))
check("Credit card Luhn valid",      has("card 4532015112830366", "credit_card"))
check("Credit card Luhn invalid → rejected", empty("bad card 1234567890123456"))
check("US phone",                    has("call +1-555-867-5309", "phone_us"))
check("UK phone",                    has("call +44 7911 123456", "phone_uk"))
check("IPv4 address",                has("server 192.168.1.100", "ip_address"))
check("API key sk-",                 has("key: sk-abcdefghijklmnopqrstuvwxyz123456", "api_key"))
check("GitHub token ghp_",          has("Token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "api_key"))
check("US ZIP code",                 has("zip 90210", "zip_us"))
check("UK postcode",                 has("I live in SW1A 1AA", "postcode_uk"))
check("DOB MM/DD/YYYY",              has("born 03/14/1990", "dob"))
check("DOB written month",           has("Birthday is March 14 1990", "dob"))
check("No PII → nothing detected",   empty("The weather is nice today"))
check("All PatternScan scores = 1.0",
      all(e.score == 1.0 for e in scan("test@example.com and 123-45-6789")))


# ─────────────────────────────────────────────────────────────
# 2. ENTITY TRACE (spaCy)
# ─────────────────────────────────────────────────────────────
print("\n[2] EntityTrace (spaCy en_core_web_lg)")

from detection.entity_trace import trace

def ner_confirmed_types(text):
    confirmed, _ = trace(text)
    return [e.type for e in confirmed]

def ner_finds(text, etype):
    confirmed, _ = trace(text)
    return any(e.type == etype for e in confirmed)

check("Detects PERSON",   ner_finds("My name is Sarah Mitchell", "PERSON"))
check("Detects ORG",      ner_finds("I work at Google", "ORG"))
check("Detects GPE city", ner_finds("Ahmed lives in New York", "GPE"))
check("Returns two lists", isinstance(trace("hello")[0], list))
check("No overlap with PatternScan",
      all(e.type != "email"
          for e in trace("sarah@gmail.com")[0]))


# ─────────────────────────────────────────────────────────────
# 3. SENTINEL LAYER CASCADE
# ─────────────────────────────────────────────────────────────
print("\n[3] SentinelLayer cascade")

from detection.logic import run_cascade, deduplicate

def cascade_types(text):
    confirmed, _ = run_cascade(text)
    return [e.type for e in confirmed]

def cascade_count(text):
    confirmed, _ = run_cascade(text)
    return len(confirmed)

check("Email + SSN detected",
      cascade_count("email ahmed@gmail.com, SSN 123-45-6789") >= 2)

check("Name + email together",
      cascade_count("I am Sarah Mitchell, email sarah@gmail.com") >= 2)

check("Card + city",
      cascade_count("card 4532015112830366 in New York") >= 1)

check("No PII → 0 entities",
      cascade_count("The weather is lovely today") == 0)

check("Deduplicate keeps highest score", (lambda: (
    e := deduplicate([
        __import__('util').DetectedEntity("x@x.com", 0, 7, "email", 1.0),
        __import__('util').DetectedEntity("x@x.com", 0, 7, "email", 0.8),
    ]),
    len(e) == 1 and e[0].score == 1.0
)[-1])())


# ─────────────────────────────────────────────────────────────
# 4. MIMICGEN
# ─────────────────────────────────────────────────────────────
print("\n[4] MimicGen surrogate generation")

from generation.logic import MimicGen
from util import DetectedEntity

m = MimicGen()

def surrogate(text, etype):
    return m.generate(DetectedEntity(text, 0, len(text), etype, 1.0))

email_s    = surrogate("ahmed@gmail.com", "email")
ssn_s      = surrogate("123-45-6789", "ssn")
person_s   = surrogate("Ahmed Al-Rashidi", "PERSON")
cc_s       = surrogate("4532015112830366", "credit_card")
phone_us_s = surrogate("+1-555-867-5309", "phone_us")
phone_uk_s = surrogate("+44 7911 123456", "phone_uk")

check("Email surrogate has @",         "@" in email_s)
check("SSN surrogate format",
      __import__('re').match(r"\d{3}-\d{2}-\d{4}", ssn_s) is not None)
check("Person surrogate is a string",  len(person_s) > 0)
check("Credit card is digits only",    cc_s.replace(" ", "").replace("-", "").isdigit())
check("US phone starts with +1",       phone_us_s.startswith("+1"))
check("UK phone starts with +44",      phone_uk_s.startswith("+44"))

# Collision resistance
collisions_m = MimicGen()
emails = [
    collisions_m.generate(DetectedEntity(f"user{i}@test.com", 0, 15, "email", 1.0))
    for i in range(50)
]
check("50 email surrogates all unique", len(set(emails)) == 50)

# generate_all deduplicates
ents = [
    DetectedEntity("john@x.com", 0, 10, "email", 1.0),
    DetectedEntity("john@x.com", 0, 10, "email", 1.0),
]
mapping = MimicGen().generate_all(ents)
check("generate_all deduplicates same text", len(mapping) == 1)


# ─────────────────────────────────────────────────────────────
# 5. SHADOWMAP
# ─────────────────────────────────────────────────────────────
print("\n[5] ShadowMap encrypt / decrypt")

from storage.logic import ShadowMap

sm = ShadowMap("runtest-conv-abc")
try:
    sm.add("FakeName", "RealName")
    sm.add("fake@mail.com", "real@mail.com")
    sm.save()

    sm2 = ShadowMap("runtest-conv-abc")
    check("Name round-trips through disk",  sm2.get("FakeName") == "RealName")
    check("Email round-trips through disk", sm2.get("fake@mail.com") == "real@mail.com")
    check("len() correct",                  len(sm2) == 2)
    check("Missing key returns None",       sm2.get("nothere") is None)

    sm2.delete()
    sm3 = ShadowMap("runtest-conv-abc")
    check("After delete → empty mapping",   len(sm3) == 0)

    # Missing file is graceful (no crash)
    try:
        ShadowMap("conv-that-never-existed-xyz999")
        check("Missing .shadowmap file is graceful", True)
    except Exception:
        check("Missing .shadowmap file is graceful", False)

finally:
    # Always clean up test files, even if an assertion above failed
    sm.delete()
    ShadowMap("runtest-conv-abc").delete()


# ─────────────────────────────────────────────────────────────
# 6. RESOLVEPASS
# ─────────────────────────────────────────────────────────────
print("\n[6] ResolvePass swap-back")

from reconstruction.logic import ResolvePass

rp = ResolvePass()

shadow = {
    "Marcus Ellison":  "Ahmed Al-Rashidi",
    "d.lee@yahoo.com": "ahmed@gmail.com",
}
response = "Hello Marcus Ellison, your email d.lee@yahoo.com has been verified."
restored = rp.resolve(response, shadow)

check("Real name restored",           "Ahmed Al-Rashidi" in restored)
check("Real email restored",          "ahmed@gmail.com" in restored)
check("Fake name removed",            "Marcus Ellison" not in restored)
check("Empty shadow map → unchanged", ResolvePass().resolve("no surrogates", {}) == "no surrogates")

# Failure log
rp2 = ResolvePass()
rp2.resolve("some response text", {"NotInResponse": "original"})
summary = rp2.get_failure_summary()
check("Exact miss logged correctly",  summary["exact_miss"] >= 1)


# ─────────────────────────────────────────────────────────────
# 7. FULL PIPELINE ROUND-TRIP (no API)
# ─────────────────────────────────────────────────────────────
print("\n[7] Full pipeline round-trip (no API call)")

from detection.logic import run_cascade, deduplicate
from generation.logic import MimicGen
from storage.logic import ShadowMap
from reconstruction.logic import ResolvePass

msg = "Hi, I am Sarah Mitchell, email sarah@gmail.com, SSN 123-45-6789."

confirmed, _ = run_cascade(msg)
confirmed     = deduplicate(confirmed)
mimic         = MimicGen()
surrogate_map = mimic.generate_all(confirmed)

# Apply surrogates
sanitised = msg
for orig in sorted(surrogate_map, key=len, reverse=True):
    sanitised = sanitised.replace(orig, surrogate_map[orig])

check("Original PII not in sanitised message",
      "sarah@gmail.com" not in sanitised and "123-45-6789" not in sanitised)
check("Sanitised message still has content",
      len(sanitised) > 10)

# Simulate Claude echoing the surrogates back
name_fake  = surrogate_map.get("Sarah Mitchell", "[name]")
email_fake = surrogate_map.get("sarah@gmail.com", "[email]")
fake_response = f"Nice to meet you, {name_fake}! Your email {email_fake} is noted."

# Restore
shadow_inv = {v: k for k, v in surrogate_map.items()}
sm_rt      = ShadowMap("roundtrip-test-999")
try:
    sm_rt.update(shadow_inv)
    sm_rt.save()

    rp_rt   = ResolvePass()
    restored = rp_rt.resolve(fake_response, sm_rt.all_mappings())

    check("Real name restored in response",  "Sarah Mitchell" in restored)
    check("Real email restored in response", "sarah@gmail.com" in restored)
    check("Fake name not in final response", name_fake not in restored)
finally:
    sm_rt.delete()
    ShadowMap("roundtrip-test-999").delete()

print("\nDetails:")
print(f"  Original  : {msg}")
print(f"  Sanitised : {sanitised}")
print(f"  API saw   : {fake_response}")
print(f"  Restored  : {restored}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print("\n" + "="*60)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  All tests passed — ready to run: python main.py chat")
else:
    print(f"  {total - passed} test(s) failed — see ❌ above")
print("="*60 + "\n")