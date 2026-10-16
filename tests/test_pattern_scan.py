"""PatternScan: full-span address claiming, other entity types, skip_values,
and a performance guard."""

import time

import pytest

from surrogateshield.core.detection import pattern_scan


def _types(entities):
    return {e.type: e.text for e in entities}


# ── Address claims the FULL span before other patterns run ────────────────────

def test_address_claims_zip_and_components():
    ents = pattern_scan.scan("ship to 789 Crescent Row, Tempe, AZ 85281 asap")
    by_type = _types(ents)
    assert by_type["address"] == "789 Crescent Row, Tempe, AZ 85281"
    assert "zip_us" not in by_type, "ZIP inside an address must not split out"


def test_address_carries_parsed_payload():
    ents = pattern_scan.scan("ship to 789 Crescent Row, Tempe, AZ 85281 asap")
    addr = next(e for e in ents if e.type == "address")
    assert addr.parsed is not None
    assert addr.parsed.city == "Tempe"


def test_standalone_zip_still_detected():
    ents = pattern_scan.scan("my zip is 85281 thanks")
    assert _types(ents).get("zip_us") == "85281"


def test_address_and_ssn_coexist():
    ents = pattern_scan.scan("SSN 123-45-6789, address 500 Main St Apt 4B, Tempe, AZ 85281")
    by_type = _types(ents)
    assert by_type["ssn"] == "123-45-6789"
    assert by_type["address"] == "500 Main St Apt 4B, Tempe, AZ 85281"


def test_unit_number_does_not_leak():
    """v1 leaked apt/unit numbers because the span stopped at the suffix."""
    ents = pattern_scan.scan("I live at 500 Main St Apt 4B ok")
    addr = next(e for e in ents if e.type == "address")
    assert "Apt 4B" in addr.text


# ── Other entity types unchanged ──────────────────────────────────────────────

@pytest.mark.parametrize("text, expected_type, expected_value", [
    ("email me at jane.doe@example.com", "email", "jane.doe@example.com"),
    ("call 480-555-1234", "phone_us", "480-555-1234"),
    ("card: 4111 1111 1111 1111", "credit_card", "4111 1111 1111 1111"),
    ("server at 192.168.1.100", "ip_address", "192.168.1.100"),
    ("key sk-abcdefghijklmnopqrstuvwxyz123456", "api_key",
     "sk-abcdefghijklmnopqrstuvwxyz123456"),
    ("born 03/15/1990", "dob", "03/15/1990"),
    ("routing 021000021", "us_bank_number", "021000021"),
])
def test_other_types(text, expected_type, expected_value):
    assert _types(pattern_scan.scan(text)).get(expected_type) == expected_value


def test_luhn_invalid_card_rejected():
    ents = pattern_scan.scan("number 1234 5678 9012 3456")
    assert "credit_card" not in _types(ents)


@pytest.mark.parametrize("value", [
    "+49 30 8842 6610",   # 2-digit city code (v1 miss)
    "+48 22 123 4567",
    "+234 1 234 5678",    # 1-digit group
    "+81 90 1234 5678",
    "+55 11 91234-5678",
    "+7 495 374 8120",    # 1-digit country code (v1 split into phone_us)
    "+91 98765 43210",
])
def test_intl_phone_formats(value):
    assert _types(pattern_scan.scan(f"call {value} now")).get("phone_intl") == value


@pytest.mark.parametrize("text", [
    "+3 4 5 is math",
    "a +2 modifier on the roll",
    "at +1-480-555-1234",       # US number, not intl
    "ring +44 1632 960854",     # UK number, not intl
])
def test_intl_phone_negatives(text):
    assert "phone_intl" not in _types(pattern_scan.scan(text))


def test_gender_identifies_as():
    ents = pattern_scan.scan("Bjorn identifies as non-binary, update the record")
    assert _types(ents).get("gender_indicator") == "identifies as non-binary"


# ── skip_values (surrogates quoted back must not be re-detected) ──────────────

def test_skip_values_exact():
    ents = pattern_scan.scan(
        "confirm 790 Crescent Row, Tempe, AZ 85281 please",
        skip_values={"790 Crescent Row, Tempe, AZ 85281"},
    )
    assert "address" not in _types(ents)


def test_skip_values_substring():
    # detected span is a fragment of a longer skip value
    ents = pattern_scan.scan(
        "confirm 790 Crescent Row please",
        skip_values={"790 Crescent Row, Tempe, AZ 85281"},
    )
    assert "address" not in _types(ents)


# ── Performance guard ─────────────────────────────────────────────────────────

def test_scan_10kb_under_threshold():
    filler = (
        "The quarterly report shows strong growth across all regions. "
        "Customers gave us 5 stars and shipping takes 3-5 business days. "
    )
    text = filler * 80  # ~10 KB
    text += " Contact: 789 Crescent Row, Tempe, AZ 85281, jane@example.com."
    start = time.perf_counter()
    ents = pattern_scan.scan(text)
    elapsed = time.perf_counter() - start
    assert _types(ents).get("address") == "789 Crescent Row, Tempe, AZ 85281"
    assert elapsed < 0.5, f"scan of 10KB took {elapsed:.3f}s"
