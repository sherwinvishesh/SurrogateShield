"""Component-extraction tests for the canonical address parser."""

import pytest

from surrogateshield.core.detection import address_parser as ap


def _parse_one(text):
    results = ap.find_addresses(text)
    assert results, f"no address found in {text!r}"
    return results[0]


# ── Full component extraction ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, house, street, suffix, unit, city, state, zip_code",
    [
        ("500 Main St Apt 4B, Tempe, AZ 85281",
         "500", "Main", "St", "Apt 4B", "Tempe", "AZ", "85281"),
        ("789 Crescent Row, Tempe, AZ 85281",
         "789", "Crescent", "Row", None, "Tempe", "AZ", "85281"),
        ("12 Oak Rd, Suite 200, Mesa, AZ",
         "12", "Oak", "Rd", "Suite 200", "Mesa", "AZ", None),
        ("55 Birch Ln, Salem, OR 97301-1234",
         "55", "Birch", "Ln", None, "Salem", "OR", "97301-1234"),
        ("88 Pine St, Portland, Oregon 97205",
         "88", "Pine", "St", None, "Portland", "Oregon", "97205"),
        ("123 5th Avenue, New York, NY 10003",
         "123", "5th", "Avenue", None, "New York", "NY", "10003"),
        ("9 Elm Dr Unit 7",
         "9", "Elm", "Dr", "Unit 7", None, None, None),
        ("4400 N Scottsdale Rd # 9, Scottsdale, AZ",
         "4400", "Scottsdale", "Rd", "# 9", "Scottsdale", "AZ", None),
        ("123 Main St Tempe AZ 85281",
         "123", "Main", "St", None, "Tempe", "AZ", "85281"),
        ("871 Martin Luther King Jr Blvd, Atlanta, GA",
         "871", "Martin Luther King Jr", "Blvd", None, "Atlanta", "GA", None),
    ],
)
def test_components(text, house, street, suffix, unit, city, state, zip_code):
    p = _parse_one(text)
    assert p.house_number == house
    assert p.street_name == street
    assert p.suffix == suffix
    assert p.unit == unit
    assert p.city == city
    assert p.state == state
    assert p.zip_code == zip_code


# ── Directionals ──────────────────────────────────────────────────────────────

def test_pre_directional():
    p = _parse_one("1126 E Apache Blvd, Tempe, AZ")
    assert p.pre_directional == "E"
    assert p.street_name == "Apache"


def test_post_directional():
    p = _parse_one("1600 Pennsylvania Ave NW, Washington, DC 20500")
    assert p.post_directional == "NW"
    assert p.city == "Washington"
    assert p.state == "DC"


def test_full_word_directional():
    p = _parse_one("1428 North Alpine Way, Flagstaff, AZ")
    assert p.pre_directional == "North"
    assert p.street_name == "Alpine"
    assert p.suffix == "Way"


# ── PO boxes ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, box_number", [
    ("PO Box 1234, Flagstaff, AZ 86001", "1234"),
    ("P.O. Box 98", "98"),
    ("po box 4521, Tucson, AZ", "4521"),
    ("Post Office Box 77, Yuma, AZ 85364", "77"),
    ("POB 55", "55"),
])
def test_po_box(text, box_number):
    p = _parse_one(text)
    assert p.is_po_box
    assert p.house_number == box_number


# ── House-number span (splice anchor for shift mode) ──────────────────────────

def test_house_number_spans():
    text = "I live at 789 Crescent Row, Tempe, AZ 85281."
    p = _parse_one(text)
    s, e = p.house_number_span
    assert text[s:e] == "789"
    rs, re_ = p.house_number_relative_span
    assert p.full_text[rs:re_] == "789"


def test_house_number_alpha_suffix():
    p = _parse_one("123A Main St")
    assert p.house_number == "123A"
    rs, re_ = p.house_number_relative_span
    assert p.full_text[rs:re_] == "123A"


def test_house_number_leading_zeros():
    p = _parse_one("0123 Elm Ave")
    assert p.house_number == "0123"


# ── Multi-line ────────────────────────────────────────────────────────────────

def test_multiline_address_is_one_span():
    text = "Shipping label:\n123 Main St\nTempe, AZ 85281\nUSA"
    p = _parse_one(text)
    assert p.full_text == "123 Main St\nTempe, AZ 85281"
    assert p.city == "Tempe"
    assert p.zip_code == "85281"


# ── State handling ────────────────────────────────────────────────────────────

def test_lowercase_two_letter_state_not_matched():
    # "or" / "in" / "me" as English words must never be treated as states.
    p = _parse_one("go to 123 Main St or take a left")
    assert p.full_text == "123 Main St"
    assert p.state is None


def test_uppercase_state_without_zip_needs_delimiter():
    # "...Main St OR take" — uppercase OR not comma-delimited, no ZIP → trimmed.
    p = _parse_one("go to 123 Main St OR take a left")
    assert p.full_text == "123 Main St"


def test_uppercase_state_comma_delimited_accepted():
    p = _parse_one("go to 123 Main St, OR 97201")
    assert p.state == "OR"
    assert p.zip_code == "97201"


# ── Sentence-final period stays outside the span ──────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("I live at 7 Main St.", "7 Main St"),
    ("They moved to 10 Sunset Cove.", "10 Sunset Cove"),
    ("Ship to 44 Oak Ave, Tempe.", "44 Oak Ave, Tempe"),
])
def test_sentence_final_period_excluded(text, expected):
    assert _parse_one(text).full_text == expected


def test_mid_sentence_abbreviation_period_kept():
    p = _parse_one("Ship to 123 Main St., Tempe, AZ 85281 today")
    assert p.full_text == "123 Main St., Tempe, AZ 85281"


# ── parse() convenience ───────────────────────────────────────────────────────

def test_parse_returns_none_for_non_address():
    assert ap.parse("no address here at all") is None


def test_parse_standalone_fragment():
    p = ap.parse("6720 Palm Dr, Phoenix, AZ")
    assert p is not None
    assert p.full_text == "6720 Palm Dr, Phoenix, AZ"


# ── Multiple addresses in one text ────────────────────────────────────────────

def test_multiple_addresses_non_overlapping_sorted():
    text = "From 12 Oak Rd, Mesa, AZ to 789 Crescent Row, Tempe, AZ please."
    results = ap.find_addresses(text)
    assert [a.full_text for a in results] == [
        "12 Oak Rd, Mesa, AZ",
        "789 Crescent Row, Tempe, AZ",
    ]
    assert results[0].end <= results[1].start
