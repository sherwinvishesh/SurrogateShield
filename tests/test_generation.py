"""Surrogate generation: replace-mode structure preservation, uniqueness,
type-consistent formats."""

import re

import pytest

from surrogateshield.core.detection import address_parser as ap
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen, _aba_check


def _entity(text, etype="address"):
    parsed = ap.parse(text) if etype == "address" else None
    return DetectedEntity(text=text, start=0, end=len(text), type=etype, parsed=parsed)


# ── Replace mode: structure-preserving, one unit ──────────────────────────────

def test_replace_preserves_structure_full_address():
    gen = MimicGen(seed=7)
    original = "789 Crescent Row, Tempe, AZ 85281"
    fake = gen._gen_address(_entity(original), mode="replace")
    assert fake != original
    # same shape: 3-digit number, one street word, same suffix, city, state, zip
    assert re.fullmatch(r"\d{3} [A-Z][A-Za-z'\-]+ Row, [A-Za-z .'\-]+, [A-Z]{2} \d{5}", fake), fake
    # no component of the original survives
    assert "789" not in fake and "Crescent" not in fake
    assert "Tempe" not in fake and "85281" not in fake


def test_replace_keeps_unit_designator_fakes_number():
    gen = MimicGen(seed=9)
    fake = gen._gen_address(_entity("500 Main St Apt 4B, Tempe, AZ 85281"), mode="replace")
    assert "Apt" in fake
    assert "Main" not in fake
    assert ", " in fake  # separators survive


def test_replace_zip4_shape_kept():
    gen = MimicGen(seed=11)
    fake = gen._gen_address(_entity("55 Birch Ln, Salem, OR 97301-1234"), mode="replace")
    assert re.search(r"\d{5}-\d{4}$", fake), fake
    assert "97301-1234" not in fake


def test_replace_full_state_name_stays_full():
    gen = MimicGen(seed=13)
    fake = gen._gen_address(_entity("88 Pine St, Portland, Oregon 97205"), mode="replace")
    assert not re.search(r", [A-Z]{2} \d{5}$", fake), fake  # not abbreviated


def test_replace_never_concatenates_extra_components():
    """v1 replaced a street line with a FULL Faker address (own apt/city/zip)."""
    gen = MimicGen(seed=15)
    fake = gen._gen_address(_entity("789 Crescent Row"), mode="replace")
    # same shape: number + street name + suffix, nothing more
    assert re.fullmatch(r"\d{3} [A-Z][A-Za-z'\-]+ Row", fake), fake


# ── Uniqueness ────────────────────────────────────────────────────────────────

def test_surrogates_unique_within_session():
    gen = MimicGen(seed=21)
    seen = set()
    for i in range(50):
        s = gen.generate(_entity(f"user{i}@example.com", "email"))
        assert s not in seen
        seen.add(s)


def test_generate_all_dedupes_repeated_entity():
    gen = MimicGen(seed=23)
    e1 = _entity("789 Crescent Row, Tempe, AZ 85281")
    e2 = _entity("789 Crescent Row, Tempe, AZ 85281")
    mapping = gen.generate_all([e1, e2], address_mode="shift")
    assert len(mapping) == 1


# ── Type-consistent formats ───────────────────────────────────────────────────

def test_ssn_format():
    s = MimicGen().generate(_entity("123-45-6789", "ssn"))
    assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", s)


def test_phone_us_format():
    s = MimicGen().generate(_entity("480-555-1234", "phone_us"))
    assert re.fullmatch(r"\+1-\d{3}-\d{3}-\d{4}", s)


def test_generated_routing_number_passes_aba_checksum():
    for _ in range(5):
        s = MimicGen().generate(_entity("021000021", "us_bank_number"))
        assert _aba_check(s), s


def test_crypto_surrogate_is_base58():
    s = MimicGen().generate(_entity("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "crypto"))
    assert s[0] == "1"
    assert not set(s) & set("0OIl")  # base58 excludes these


def test_gender_surrogate_stays_readable():
    s = MimicGen().generate(_entity("she/her", "gender_indicator"))
    assert s in {
        "male", "female", "non-binary",
        "he/him", "she/her", "they/them",
        "gender: male", "gender: female", "sex: male", "sex: female",
    }
