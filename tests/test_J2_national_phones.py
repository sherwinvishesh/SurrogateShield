"""Gate J2 — national-format phone numbers with a trunk 0 and no country
code or label (UK mobile 5-3-3, AU, FR pairs, DE, NL) went out unmasked: the
UK pattern took only 4-6 / 3-3-4 / 2-4-4 groupings. Found on the blind dev3
split; the leak count on dev4 (scored, never inspected) fell from 5 to 1.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


@pytest.mark.parametrize("number", [
    "07412 680 335",       # UK mobile
    "0412 773 519",        # AU mobile
    "06 47 21 93 58",      # FR
    "0176 4482 1903",      # DE mobile
    "06-41982277",         # NL
    "02 96 39 71 48",      # FR landline
])
def test_J2_national_phone_is_found_whole(number):
    found = [(e.type, e.text) for e in ps.scan(f"Call me on {number} after six.")]
    assert any(t.startswith("phone") and v == number for t, v in found), found


@pytest.mark.parametrize("text", [
    "dated 01 02 2023",                                   # a date: 8 digits
    "order 0012 3456",                                    # too short
    "release v0.12.3 2024",                               # version
    "sécurité sociale 2 84 07 75 112 058 43",             # tail of a longer number
    "at 12/03/2024 10:02",
])
def test_J2_trunk_zero_rule_does_not_claim_other_numbers(text):
    assert not any(e.type.startswith("phone") for e in ps.scan(text))
