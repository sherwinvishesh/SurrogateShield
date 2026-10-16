"""Shift-mode surrogate generation: number-only change, byte-exact formatting,
edge cases, collision handling, determinism."""

import random
import re

import pytest

from surrogateshield.core.detection import address_parser as ap
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen, shift_house_number


def _entity(text):
    parsed = ap.parse(text)
    assert parsed is not None
    return DetectedEntity(
        text=parsed.full_text, start=parsed.start, end=parsed.end,
        type="address", parsed=parsed,
    )


# ── Only the house number changes; formatting is byte-exact ───────────────────

@pytest.mark.parametrize("address", [
    "789 Crescent Row, Tempe, AZ 85281",
    "500 Main St Apt 4B, Tempe, AZ 85281",
    "1126 E Apache Blvd, Tempe, AZ",
    "55 Birch Ln, Salem, OR 97301-1234",
    "123 Main St\nTempe, AZ 85281",
    "1600 Pennsylvania Ave NW, Washington, DC 20500",
])
def test_shift_changes_only_the_number(address):
    parsed = ap.parse(address)
    for seed in range(10):
        shifted = shift_house_number(parsed, rng=random.Random(seed))
        assert shifted is not None
        assert shifted != address
        # exactly the number token differs; the rest is byte-identical
        rs, re_ = parsed.house_number_relative_span
        numeric = re.match(r"\d+", parsed.house_number).group()
        assert shifted[:rs] == address[:rs]
        new_numeric = re.match(r"\d+", shifted[rs:]).group()
        assert abs(int(new_numeric) - int(numeric)) == 1
        assert shifted[rs + len(new_numeric):] == address[rs + len(numeric):]


def test_shift_never_reformats_commas_or_newlines():
    address = "123 Main St\nTempe, AZ 85281"
    shifted = shift_house_number(ap.parse(address), rng=random.Random(0))
    assert "\n" in shifted
    assert shifted.count(",") == address.count(",")


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_house_number_one_always_goes_up():
    parsed = ap.parse("1 Infinite Loop, Cupertino, CA 95014")
    for seed in range(25):
        shifted = shift_house_number(parsed, rng=random.Random(seed))
        assert shifted == "2 Infinite Loop, Cupertino, CA 95014"


def test_leading_zeros_preserved():
    parsed = ap.parse("0123 Elm Ave")
    for seed in range(10):
        shifted = shift_house_number(parsed, rng=random.Random(seed))
        assert shifted in ("0122 Elm Ave", "0124 Elm Ave")


def test_alpha_suffix_rides_along():
    parsed = ap.parse("123A Main St")
    shifted = shift_house_number(parsed, rng=random.Random(1))
    assert shifted in ("122A Main St", "124A Main St")


def test_po_box_number_shifts():
    parsed = ap.parse("PO Box 1234, Flagstaff, AZ 86001")
    shifted = shift_house_number(parsed, rng=random.Random(2))
    assert shifted in (
        "PO Box 1233, Flagstaff, AZ 86001",
        "PO Box 1235, Flagstaff, AZ 86001",
    )


def test_shift_range_respected():
    parsed = ap.parse("500 Oak Ave")
    seen = set()
    for seed in range(60):
        shifted = shift_house_number(parsed, shift_range=3, rng=random.Random(seed))
        n = int(re.match(r"\d+", shifted).group())
        seen.add(n - 500)
    assert seen <= {-3, -2, -1, 1, 2, 3}
    assert len(seen) > 1  # actually random, not constant


# ── Collision handling ────────────────────────────────────────────────────────

def test_collision_with_real_neighbor_addresses():
    parsed = ap.parse("789 Crescent Row")
    forbidden = frozenset({"788 Crescent Row", "790 Crescent Row"})
    for seed in range(10):
        shifted = shift_house_number(parsed, forbidden=forbidden, rng=random.Random(seed))
        assert shifted is not None
        assert shifted not in forbidden
        assert shifted != "789 Crescent Row"


def test_collision_exhaustion_returns_none():
    parsed = ap.parse("789 Crescent Row")
    # forbid every candidate the widening can reach (±1 … ±9)
    forbidden = frozenset(
        f"{789 + d} Crescent Row" for d in range(-9, 10)
    )
    assert shift_house_number(parsed, forbidden=forbidden, rng=random.Random(0)) is None


def test_gen_address_falls_back_to_replace_on_exhaustion():
    gen = MimicGen(seed=5)
    ent = _entity("789 Crescent Row")
    forbidden = frozenset(f"{789 + d} Crescent Row" for d in range(-9, 10))
    surrogate = gen._gen_address(ent, mode="shift", forbidden=forbidden)
    assert surrogate not in forbidden
    assert surrogate != ent.text
    assert surrogate.endswith("Row")  # structure-preserving replace fallback


def test_generate_all_dodges_other_real_address_in_message():
    # "789 X" and "790 X" both present: the shift for one must not equal the other
    gen = MimicGen(seed=3)
    e1 = _entity("789 Crescent Row")
    e2 = _entity("790 Crescent Row")
    mapping = gen.generate_all([e1, e2], address_mode="shift")
    assert mapping["789 Crescent Row"] not in ("789 Crescent Row", "790 Crescent Row")
    assert mapping["790 Crescent Row"] not in ("789 Crescent Row", "790 Crescent Row")
    assert mapping["789 Crescent Row"] != mapping["790 Crescent Row"]


# ── Determinism ───────────────────────────────────────────────────────────────

def test_seeded_generation_is_deterministic():
    ent = _entity("789 Crescent Row, Tempe, AZ 85281")
    a = MimicGen(seed=42).generate(ent, address_mode="shift")
    b = MimicGen(seed=42).generate(ent, address_mode="shift")
    assert a == b

    c = MimicGen(seed=42).generate(ent, address_mode="replace")
    d = MimicGen(seed=42).generate(ent, address_mode="replace")
    assert c == d


def test_parsed_fallback_when_entity_has_no_parse_payload():
    # dedup or NER paths may drop .parsed — generation re-parses from text
    ent = DetectedEntity(
        text="789 Crescent Row, Tempe, AZ 85281", start=0, end=33, type="address",
    )
    surrogate = MimicGen(seed=1).generate(ent, address_mode="shift")
    assert surrogate.endswith("Crescent Row, Tempe, AZ 85281")
    assert surrogate != ent.text
