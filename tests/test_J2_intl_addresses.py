"""Gate J2 — street, unit and postcode forms outside the US/UK: number-first
Romance streets, Dutch / Nordic / Finnish compounds, floor and door after a
street, Indian house numbers and PINs, bracketed and labelled postcodes, and
Chinese addresses — each with a surrogate of the same shape.
Model-free: pattern scan and structural rules only.
"""

import pytest

from surrogateshield.core.detection import pattern_scan, structural
from surrogateshield.core.detection.pipeline import _detect_structural_persons
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation import places
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    ents = pattern_scan.scan(text)
    added, _ = structural.detect(text, ents)
    return {(e.text, e.type) for e in list(ents) + added}


def addresses(text):
    return {t for t, typ in found(text) if typ in ("address", "zip_us")}


@pytest.mark.parametrize("text, want", [
    ("address is 12 Rua do Sol, Lagos", {"12 Rua do Sol"}),
    # whole addresses where a layout reads them (address_assembly), parts elsewhere
    ("Kerkstraat 14, 8861 AJ Harlingen", {"Kerkstraat 14, 8861 AJ Harlingen"}),
    ("the basement flat at Storgata 41B, Lillehammer", {"Storgata 41B"}),
    ("asunnossa Mannerheimintie 12, Helsinki", {"Mannerheimintie 12"}),
    ("vivo en Calle Mayor 17, 3º B, Ponferrada (24401).", {"Calle Mayor 17, 3º B, Ponferrada (24401)"}),
    ("Moro na Travessa do Carmo 8, 2.º Esq., em Tomar", {"Travessa do Carmo 8", "2.º Esq."}),
    ("in der Wohnung Lindenallee 3a, 2. OG links, ist", {"Lindenallee 3a", "2. OG links"}),
    ('{"address": "Bv. San Juan 1120, Piso 4 Dpto B"}', {"Bv. San Juan 1120", "Piso 4", "Dpto B"}),
    ("I moved out of Unit 3C at 77 Harbor View Terrace on August 31",
     {"Unit 3C", "77 Harbor View Terrace"}),
])
def test_J2_european_street_unit_and_postcode(text, want):
    assert want <= addresses(text)


def test_J2_indian_house_number_and_pin_after_locality():
    got = addresses("Address – H.No. 3-118, Gandhi Nagar, Almora 263601.")
    assert {"H.No. 3-118, Gandhi Nagar, Almora 263601"} <= got


@pytest.mark.parametrize("text, value", [
    ('{"city": "Córdoba", "zip": "X5000"}', "X5000"),
    ("PLZ 79098 Freiburg", "79098"),
    ("CAP 20121, Milano", "20121"),
    ("código postal: 1100-148", "1100-148"),
    ("postal code K1A 0B1", "K1A 0B1"),
])
def test_J2_labelled_postcode_any_country(text, value):
    assert value in addresses(text)


def test_J2_pin_code_is_masked_either_way():
    # an Indian PIN code or a card PIN: the credential rule claims it
    assert ("263601", "credential") in found("pincode - 263601")


@pytest.mark.parametrize("text", [
    "Gate 12 boards at 3pm; Room 4B",
    "we have 2 OG cables in stock",
    "I scored 3º in the race",
    "Version 2 of the CAP theorem; the CPU has 8 cores",
    "my PIN 4821 stopped working",
    "Spring 2024 schedule, String 3 of the guitar",
])
def test_J2_street_and_postcode_rules_leave_ordinary_numbers(text):
    assert not addresses(text)


def test_J2_chinese_city_district_and_road():
    got = found("告诉他我们公寓（深圳南山区桃园路88号5栋1203）的热水器坏了")
    assert {("深圳", "GPE"), ("南山区", "GPE"), ("桃园路88号5栋1203", "address")} <= got
    assert ("成都", "GPE") in found("在成都高新区一家小公司做前端")


@pytest.mark.parametrize("text", ["北京烤鸭很好吃", "上海是一个大城市", "我在上海工作"])
def test_J2_chinese_city_alone_is_not_an_address(text):
    assert not {t for t, typ in found(text) if typ in ("GPE", "address")}


def test_J2_structural_person_never_claims_a_street():
    text = "vivo en Calle Mayor 17, 3º B"
    mayor = DetectedEntity("Mayor", 15, 20, "PERSON", 0.8, "ner")
    new, _ = _detect_structural_persons(text, [mayor])
    assert not [e for e in new if "Calle" in e.text]


def gen(text, typ="address", seed=3):
    return MimicGen(seed=seed).generate(DetectedEntity(text, 0, len(text), typ, 0.95, "pattern"),
                                        address_mode="replace")


@pytest.mark.parametrize("text, shape", [
    ("Dpto B", r"Dpto [A-Z]"),
    ("3º B", r"\dº [A-Z]"),
    ("2. OG links", r"\d\. OG links"),
    ("H.No. 3-118", r"H\.No\. \d-\d{3}"),
    ("Kerkstraat 14", r"[A-Z]\w+straat \d\d"),
    ("Storgata 41B", r"[A-ZÆØÅ]\w+gata \d\dB"),
    ("8861 AJ Harlingen", r"\d{4} [A-Z]{2} \S+"),
    ("桃园路88号5栋1203", r"[一-鿿]{2}路\d\d号\d栋\d{4}"),
])
def test_J2_unit_and_street_surrogates_keep_their_shape(text, shape):
    import re
    out = gen(text)
    assert out != text and re.fullmatch(shape, out), out
    assert "Kerk" not in out and "桃园" not in out and "AJ" not in out


def test_J2_chinese_place_surrogate_is_a_chinese_place():
    pick = lambda xs: xs[0]
    assert places.real_place("南山区", "", pick, lambda c: False).endswith("区")
    city = places.real_place("深圳", "", pick, lambda c: False)
    assert city != "深圳" and all("一" <= ch <= "鿿" for ch in city)
