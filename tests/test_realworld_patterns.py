"""
Real-world detection regression suite (model-free).

Distilled from an adversarial benchmark of chat messages, support tickets,
medical intake forms, finance docs, tech logs, and negative controls.
Every case here encodes a bug class that was actually observed and fixed:
span collisions, checksum gaps, context-gating, case-degenerate text.

PatternScan and the structural passes are pure regex/logic — no models —
so this whole file runs in milliseconds.
"""

import pytest

from detection import pattern_scan
from detection.logic import (
    _detect_structural_persons,
    _filter_implausible_orgs,
    _merge_adjacent_persons,
)
from util import DetectedEntity


def types_of(text):
    return [(e.type, e.text) for e in pattern_scan.scan(text)]


def found(text, wanted_type, wanted_value):
    return any(t == wanted_type and wanted_value in v for t, v in types_of(text))


def nothing_found(text):
    return types_of(text) == []


# ─────────────────────────────────────────────────────────────────────────────
# Payment cards — generalized Luhn (13-19 digits, Amex/Diners/Discover)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("number", [
    "4111 1111 1111 1111",       # Visa spaced
    "5555-5555-5555-4444",       # MC dashed
    "378282246310005",           # Amex bare 15
    "6011 1111 1111 1117",       # Discover spaced
    "6759649826438453",          # Maestro bare
])
def test_card_formats(number):
    assert found(f"my card {number} was charged", "credit_card", number.split()[0])


def test_luhn_invalid_card_needs_card_context():
    # Luhn-invalid but the text SAYS it's a card → masked anyway
    assert found("payment card is Visa 5412751234123412 ok", "credit_card",
                 "5412751234123412")
    # Luhn-invalid with no card framing → not a card
    assert not any(t == "credit_card"
                   for t, _ in types_of("ref 5412751234123412 attached"))


# ─────────────────────────────────────────────────────────────────────────────
# Checksummed identifiers — IBAN, VIN, routing
# ─────────────────────────────────────────────────────────────────────────────

def test_iban_compact_and_spaced():
    assert found("wire from DE89370400440532013000 arrived", "iban",
                 "DE89370400440532013000")
    assert found("refund to GB82 WEST 1234 5698 7654 32 please", "iban", "GB82")


def test_iban_checksum_rejects_lookalikes():
    assert not any(t == "iban" for t, _ in
                   types_of("board ref DE00370400440532013000 for review"))


def test_vin_check_digit():
    assert found("stolen car VIN 1HGBH41JXMN109186 reported", "vin",
                 "1HGBH41JXMN109186")
    assert not any(t == "vin" for t, _ in
                   types_of("code 1HGBH41JXMN109187 is not a vin"))


def test_routing_requires_bank_context():
    assert found("routing number 021000021 for deposit", "us_bank_number",
                 "021000021")
    assert not any(t == "us_bank_number" for t, _ in
                   types_of("lot 021000021 sold at auction"))


# ─────────────────────────────────────────────────────────────────────────────
# Phones — boundary guards, extensions, 1- prefix, six-group intl
# ─────────────────────────────────────────────────────────────────────────────

def test_phone_never_claims_inside_eth_address():
    ents = types_of("sent to 0x52908400098527886E0F7030069857D2E4169EE7 today")
    assert ("crypto", "0x52908400098527886E0F7030069857D2E4169EE7") in ents
    assert not any(t.startswith("phone") for t, _ in ents)


def test_phone_extension_and_one_prefix():
    assert found("call (480) 555-0142 ext. 214 now", "phone_us", "ext. 214")
    assert found("call 1-800-555-0175 today", "phone_us", "1-800-555-0175")


def test_phone_intl_six_groups():
    assert found("Paris office +33 1 42 68 53 00 open", "phone_intl",
                 "+33 1 42 68 53 00")


def test_phone_rejects_po_numbers_and_hashes():
    assert nothing_found("purchase order PO-4805550123 cleared")
    assert nothing_found("artifact hash 4805550123abcdef pushed")


# ─────────────────────────────────────────────────────────────────────────────
# SSN / ZIP context gating
# ─────────────────────────────────────────────────────────────────────────────

def test_bare_ssn_needs_context():
    assert found("my ssn is 536904399 ok", "ssn", "536904399")
    assert nothing_found("invoice 123456789 was paid")


def test_formatted_ssn_unconditional():
    assert found("SSN 536-90-4399 on file", "ssn", "536-90-4399")
    assert found("SSN 536 90 4399 on file", "ssn", "536 90 4399")


@pytest.mark.parametrize("text,expected", [
    ("zip code 85281 identifies most people", True),
    ("within the 60611 area", True),
    ("resides inside 90210 for benefits", True),
    ("living within 30303 whose dob is", True),
    ("port 55234 held all quarter", False),
    ("budget cap 85000 USD approved", False),
    ("Final score 98765 fans strong", False),
    ("SKU 55555-4444 sold out", False),
])
def test_zip_context_gating(text, expected):
    got = any(t == "zip_us" for t, _ in types_of(text))
    assert got == expected, text


# ─────────────────────────────────────────────────────────────────────────────
# IPv6 / MAC / secrets
# ─────────────────────────────────────────────────────────────────────────────

def test_ipv6_and_mac():
    ents = types_of("gw fe80::1ff:fe23:4567:890a MAC 00:1B:44:11:3A:B7.")
    assert ("ip_address", "fe80::1ff:fe23:4567:890a") in ents
    assert ("mac_address", "00:1B:44:11:3A:B7") in ents


def test_clock_time_is_not_ipv6():
    assert nothing_found("the meeting is at 12:30:45 sharp")


def test_snake_case_secret_identifier():
    assert found("Bearer token secret_auth_token_9988776655_admin failed",
                 "api_key", "secret_auth_token_9988776655_admin")
    assert nothing_found("use snake_case_names for readability")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-gated IDs — DL, passport, MRN, insurance, KTN, Aadhaar, plates
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,ty,value", [
    ("driver's license number is D08954142", "us_driver_license", "D08954142"),
    ("Florida license G645-201-88-123-0 held", "us_driver_license",
     "G645-201-88-123-0"),
    ("passport number is 512346789", "passport", "512346789"),
    ("MRN: 4471982 attached", "id_number", "4471982"),
    ("MRN: 88-123-99 ready", "id_number", "88-123-99"),
    ("Insurance ID: XQV882736401 active", "id_number", "XQV882736401"),
    ("Medicare ID: 1EG4-TE5-MK73 on file", "id_number", "1EG4-TE5-MK73"),
    ("KTN is TT10234756 for precheck", "id_number", "TT10234756"),
    ("USCIS number A-088-471-556 lost", "id_number", "A-088-471-556"),
    ("Aadhaar number is 2314 5678 9012 here", "id_number", "2314 5678 9012"),
    ("patient ID OKW-2214 to view", "id_number", "OKW-2214"),
    ("Arizona plate CDL4821 stolen", "license_plate", "CDL4821"),
])
def test_gated_ids(text, ty, value):
    assert found(text, ty, value), text


def test_account_number_is_protected():
    # 10-digit account numbers are claimed by the earlier phone pattern —
    # the requirement is that the VALUE is masked, whatever the label
    assert any("8837221904" in v
               for _, v in types_of("account number 8837221904 for deposit"))


def test_gated_ids_need_digits():
    assert nothing_found("the license AGREEMENT was signed")
    assert nothing_found("price tag is 45000 dollars")  # plates need letters


# ─────────────────────────────────────────────────────────────────────────────
# Dates — short years, ordinals
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,value", [
    ("born 03/04/85 in ohio", "03/04/85"),
    ("date of birth March 3rd, 1946.", "March 3rd, 1946"),
    ("DOB: 04 Jul 1988", "04 Jul 1988"),
    ("DOB: 1988-07-22", "1988-07-22"),
])
def test_dob_formats(text, value):
    assert found(text, "dob", value), text


def test_versions_are_not_dates():
    assert nothing_found("the v2.5.1 release shipped")
    assert nothing_found("released 2.5.24 build yesterday")


# ─────────────────────────────────────────────────────────────────────────────
# Pass E — structural persons (case-degenerate text), precision probes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("this is deshawn washington from apt 4b", "deshawn washington"),
    ("my name is sarah jane mcallister", "sarah jane mcallister"),
    ("name: sarah jane mcallister\ndob: x", "sarah jane mcallister"),
    ("mother's maiden name: kowalski", "kowalski"),
    ("sarah oconnor here, waiting on results", "sarah oconnor"),
    ("review with mr. thompson moved to 3pm", "thompson"),
    ("auth log: user asmith connected from host", "asmith"),
    ("Customer: ROBERT J. DELGADO\nCallback: x", "ROBERT J. DELGADO"),
    ("APPLICANT: WASHINGTON, DESHAWN M\nPHONE: x", "WASHINGTON, DESHAWN M"),
    ("intake: Name - Sunita Rathod, Email - s@x.co", "Sunita Rathod"),
])
def test_structural_person_recall(text, expected):
    ents, _ = _detect_structural_persons(text, [])
    assert expected in [e.text for e in ents], text


@pytest.mark.parametrize("text", [
    "Customer: enterprise plan renewal for Q3",
    "Contact: sales department for pricing",
    "this is getting out of hand",
    "this is absolutely unacceptable service",
    "im heading home now",
    "i'm very disappointed with the delivery",
    "user permissions were updated by admin",
    "a user who declared a gender preference",
    "failed login attempts from the same address",
    "Beneficiary: trust fund distribution pending",
    "Emergency contact: front desk staff",
    "name: tbd\nphone: tbd",
    "domain name: example.com is parked",
    "enhance its professional impact for the MBA",
])
def test_structural_person_precision(text):
    ents, _ = _detect_structural_persons(text, [])
    assert ents == [], f"FP {[e.text for e in ents]} in {text!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Pass F / Pass G — org plausibility and person merging
# ─────────────────────────────────────────────────────────────────────────────

def _org(text_value, start, text, source="ner"):
    return DetectedEntity(text=text_value, start=start,
                          end=start + len(text_value), type="ORG",
                          score=0.9, source=source)


@pytest.mark.parametrize("value,keep", [
    ("Microsoft", True),           # brand orgs ARE maskable PII here
    ("Visa", True),
    ("NHS", True),
    ("Meridian Capital Group", True),
    ("the national insurance board", True),   # suffix word beats lowercase
    ("phoenix program", False),    # lowercase junk
    ("B22", False),                # gate/seat code
])
def test_org_plausibility(value, keep):
    text = f"note about {value} today"
    ent = _org(value, 11, text)
    kept = _filter_implausible_orgs([ent], text)
    assert bool(kept) == keep, value


def test_org_standard_numbers_dropped():
    text = "passed SOC 2 and ISO 27001 audits"
    orgs = [_org("SOC", 7, text), _org("ISO 27001", 17, text)]
    assert _filter_implausible_orgs(orgs, text) == []


def test_adjacent_person_merge():
    text = "for Priya Nambiar: born"
    a = DetectedEntity(text="Priya", start=4, end=9, type="PERSON",
                       score=0.9, source="slm")
    b = DetectedEntity(text="Nambiar", start=10, end=17, type="PERSON",
                       score=0.88, source="ner")
    merged = _merge_adjacent_persons([a, b], text)
    assert [e.text for e in merged] == ["Priya Nambiar"]


# ─────────────────────────────────────────────────────────────────────────────
# Address parser — case-degenerate real-world formats
# ─────────────────────────────────────────────────────────────────────────────

from detection import address_parser


@pytest.mark.parametrize("text,expected", [
    ("yo im moving to 2214 birchwood ln, madison wi 53711 next month",
     "2214 birchwood ln, madison wi 53711"),
    ("at 1190 e apache blvd tempe az 85281, package never came",
     "1190 e apache blvd tempe az 85281"),
    ("ship to 88 Marine Parade, Flat 3, Brighton", "88 Marine Parade"),
    ("ADDRESS: 1190 E APACHE BLVD APT 4B TEMPE AZ 85281",
     "1190 E APACHE BLVD APT 4B TEMPE AZ 85281"),
])
def test_address_case_variants(text, expected):
    got = [a.full_text for a in address_parser.find_addresses(text)]
    assert expected in got, f"{text!r} → {got}"


@pytest.mark.parametrize("text", [
    "5 mile run this morning",
    "we waited 3-5 business days on Oak St for the permit",
    "The office at 123 Main St, Springfield is beautiful in spring",
])
def test_address_negatives_hold(text):
    got = [a.full_text for a in address_parser.find_addresses(text)]
    for g in got:
        assert "Springfield" not in g and "mile run" not in g \
            and "business days" not in g, f"{text!r} → {got}"
