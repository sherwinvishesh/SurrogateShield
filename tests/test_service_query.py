"""Service-query classification, sensitive overrides, deprecated wrapper —
both trees."""

import warnings

import pytest

from surrogateshield.core.detection import service_query as lib_sq
from detection import service_query as root_sq


@pytest.fixture(params=["library", "root"])
def sq(request):
    return lib_sq if request.param == "library" else root_sq


# ── is_service_query ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "any good restaurants near 1126 E Apache Blvd, Tempe, AZ?",
    "find me a coffee shop around downtown",
    "what's the weather in Phoenix?",
    "directions to 500 Main St please",
    "nearest gas station close to me",
    "is there a pharmacy near Mill Ave?",
    "best places to eat in Tempe",
    "check if there is parking near the stadium",
])
def test_service_queries_detected(sq, text):
    assert sq.is_service_query(text)


@pytest.mark.parametrize("text", [
    "my billing address is 789 Crescent Row, Tempe, AZ",
    "here is my SSN 123-45-6789",
    "update my profile with the new phone number",
    "the meeting is at 3pm tomorrow",
])
def test_non_service_queries(sq, text):
    assert not sq.is_service_query(text)


# ── Sensitive-topic override ──────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "therapy clinics near 789 Crescent Row, Tempe, AZ",
    "find an immigration lawyer near me",
    "rehab centers around Phoenix",
    "domestic violence shelter close to downtown",
    "HIV testing sites near Mill Ave",
])
def test_sensitive_topics_override_service_classification(sq, text):
    assert sq.is_sensitive_topic(text)
    assert not sq.is_service_query(text)


def test_non_sensitive_text(sq):
    assert not sq.is_sensitive_topic("best pizza near campus")


# ── Precompiled patterns (perf fix regression guard) ──────────────────────────

def test_patterns_are_precompiled(sq):
    import re
    assert all(isinstance(p, re.Pattern) for p in sq._SERVICE_PATTERNS)
    assert all(isinstance(p, re.Pattern) for p in sq._SENSITIVE_OVERRIDES)


# ── Deprecated fuzz_addresses wrapper ─────────────────────────────────────────

def test_fuzz_addresses_deprecated_but_functional(sq):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fuzzed, mapping = sq.fuzz_addresses(
            "meet me at 42 Baker Street, London"
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert mapping, "wrapper must still fuzz addresses"
    original, shifted = next(iter(mapping.items()))
    assert original == "42 Baker Street, London"
    assert shifted in ("41 Baker Street, London", "43 Baker Street, London")
    assert shifted in fuzzed and original not in fuzzed


def test_fuzz_addresses_no_address_is_noop(sq):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fuzzed, mapping = sq.fuzz_addresses("nothing to see here")
    assert fuzzed == "nothing to see here"
    assert mapping == {}
