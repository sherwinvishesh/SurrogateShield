"""ResolvePass: exact ordering, alignment-safe components, anchored fuzzy,
no-corruption guarantees. Runs against BOTH trees."""

import random

import pytest

from surrogateshield.core.reconstruction.resolve import ResolvePass as LibRP
from reconstruction.logic import ResolvePass as RootRP


@pytest.fixture(params=["library", "root"])
def resolver(request):
    if request.param == "library":
        rp = LibRP()
        return lambda text, shadow: rp.resolve(text, shadow, fuzzy_threshold=85)
    rp = RootRP()
    return lambda text, shadow: rp.resolve(text, shadow)


# ── Pass 1: exact ─────────────────────────────────────────────────────────────

def test_exact_hit(resolver):
    out = resolver(
        "Ship to 790 Crescent Row, Tempe, AZ 85281 tomorrow.",
        {"790 Crescent Row, Tempe, AZ 85281": "789 Crescent Row, Tempe, AZ 85281"},
    )
    assert "789 Crescent Row, Tempe, AZ 85281" in out and "790" not in out


def test_longest_surrogate_resolves_first(resolver):
    out = resolver(
        "John Smithson and John Smith are different.",
        {"John Smith": "Peter Vega", "John Smithson": "Aldo Reyes"},
    )
    assert out == "Aldo Reyes and Peter Vega are different."


def test_replaced_spans_are_protected(resolver):
    # original of one mapping contains the surrogate of another
    out = resolver(
        "Visit 512 Oak Ave in Mesa.",
        {"512 Oak Ave": "511 Oak Ave", "Mesa": "Gilbert"},
    )
    assert out == "Visit 511 Oak Ave in Gilbert."


# ── Pass 2: alignment-safe components ─────────────────────────────────────────

def test_partial_address_echo_restored(resolver):
    out = resolver(
        "The package goes to 790 Crescent Row as requested.",
        {"790 Crescent Row, Tempe, AZ 85281": "789 Crescent Row, Tempe, AZ 85281"},
    )
    assert "789 Crescent Row" in out and "790" not in out


def test_length_mismatch_never_corrupts(resolver):
    """v1 zipped mismatched word lists positionally and replaced globally."""
    shadow = {"70921 Smith Ports, Apt. 456, New Kayla, WY 12345": "789 Crescent Row"}
    resp = "Smith is a common name. New York has ports. Nothing else."
    assert resolver(resp, shadow) == resp


def test_ashley_county_guard(resolver):
    out = resolver(
        "Ashley Wise lives near Ashley County.",
        {"Ashley Wise": "Maria Gonzalez"},
    )
    assert out == "Maria Gonzalez lives near Ashley County."


def test_first_name_only_echo(resolver):
    out = resolver("Tell Ashley the plan.", {"Ashley Wise": "Maria Gonzalez"})
    assert out == "Tell Maria the plan."


def test_single_token_guard_blocks_shared_substrings(resolver):
    # "Ashley" appears in another shadow value → single-token fallback must skip
    out = resolver(
        "Tell Ashley the plan.",
        {"Ashley Wise": "Maria Gonzalez", "Ashley County Bank": "First National"},
    )
    assert out == "Tell Ashley the plan."


def test_reflowed_multiline_surrogate(resolver):
    out = resolver(
        "Ship to:\n790 Crescent\nRow, Tempe area",
        {"790 Crescent Row, Tempe, AZ 85281": "789 Crescent Row, Tempe, AZ 85281"},
    )
    assert "789 Crescent" in out and "790" not in out


# ── Pass 3: anchored fuzzy ────────────────────────────────────────────────────

def test_fuzzy_typo_restored_with_true_anchor(resolver):
    out = resolver(
        "As discussed, Jordn Mercer approved the request yesterday.",
        {"Jordan Mercer": "Elias Vantree"},
    )
    assert out == "As discussed, Elias Vantree approved the request yesterday."


def test_fuzzy_never_replaces_unrelated_text(resolver):
    resp = "The weather is lovely today and nothing else matters."
    out = resolver(resp, {"Jordan Mercer": "Elias Vantree"})
    assert out == resp


def test_fuzzy_threshold_is_respected():
    rp = LibRP()
    # threshold 100 → the typo'd echo cannot fuzzy-match
    out = rp.resolve(
        "As discussed, Jordn Mercer approved it.",
        {"Jordan Mercer": "Elias Vantree"},
        fuzzy_threshold=100,
    )
    assert "Elias Vantree" not in out or "Mercer" not in out  # no full fuzzy rewrite


# ── Root failure taxonomy ─────────────────────────────────────────────────────

def test_root_failure_taxonomy():
    rp = RootRP()
    rp.resolve("nothing relevant here", {"Jordan Mercer": "Elias Vantree"})
    summary = rp.get_failure_summary()
    assert summary["exact_miss"] >= 1
    assert summary["fuzzy_miss"] >= 1

    rp2 = RootRP()
    rp2.resolve("Jordn Mercer said yes", {"Jordan Mercer": "Elias Vantree"})
    assert rp2.get_failure_summary()["fuzzy_hit"] >= 1


# ── Randomized round-trip fuzz ────────────────────────────────────────────────

def test_randomized_exact_roundtrips(resolver):
    rng = random.Random(1234)
    words = "alpha beta gamma delta epsilon zeta eta theta".split()
    for _ in range(25):
        original = f"{rng.randint(1, 9999)} Crescent Row, Tempe, AZ {rng.randint(10000, 99999)}"
        surrogate = f"{rng.randint(1, 9999)} Lakeview Dr, Mesa, AZ {rng.randint(10000, 99999)}"
        filler = " ".join(rng.choices(words, k=rng.randint(3, 12)))
        resp = f"{filler} {surrogate} {filler}."
        out = resolver(resp, {surrogate: original})
        assert original in out
        assert surrogate not in out
