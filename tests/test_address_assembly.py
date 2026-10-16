"""V3 §3.5: international address layouts (detection/address_assembly.py)
and their part-by-part surrogates (MimicGen._replace_parts).

Whole addresses in the layouts of the benchmark generator's twelve countries
(street-first and number-first, postcode before or after the town, region
codes, flat / unit prefixes, Mexican exterior + interior numbers, Irish
addresses without a postcode), in a sentence, on a form line, as a JSON
value and in a signature; never prose. Model-free. Every address is fake:
drawn from bench/realdata/identities.py (seeded Faker, both pool halves) or
written here.
"""

import random
import re
import time

import pytest

from bench.realdata import identities as I
from surrogateshield.core.detection import address_assembly as aa
from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen
from tests.test_address_negatives import NEGATIVES


def _drawn(country, n=8):
    out = []
    for seed in range(n):
        for pool in (None, "train", "eval"):
            c = I.Ctx(random.Random(f"asm-{country}-{pool}-{seed}"), shift=False, pool=pool)
            c.country, c.city = country, None
            out.append(I.address(c)[0])
    return out


CONTEXTS = (
    "Please send it to {} by Friday.",
    "Name: Jo\nAddress: {}\nPhone: none",
    '{{"address": "{}", "note": "x"}}',
    "Thanks,\nJo Reyes\n{}\n",
    "{}",
)


@pytest.mark.parametrize("country", sorted(I.COUNTRIES))
def test_every_generator_layout_is_one_whole_address(country):
    for value in _drawn(country):
        for ctx in CONTEXTS:
            text = ctx.format(value)
            found = [e.text for e in pattern_scan.scan(text) if e.type == "address"]
            assert value in found, (text, found)


@pytest.mark.parametrize("text, value, country, roles", [
    ("Ship to Kambsstraße 2-8, 90146 Augsburg please", "Kambsstraße 2-8, 90146 Augsburg", "DE",
     ["name", "house", "postcode", "city"]),
    ("633 James Green Apt. 500, Edmonton, AB L7K 7J3", "633 James Green Apt. 500, Edmonton, AB L7K 7J3",
     "CA", ["house", "name", "unit", "city", "region", "postcode"]),
    ("Rua das Flores 12, Recife - PE, 50030-230.", "Rua das Flores 12, Recife - PE, 50030-230", "BR",
     ["type", "name", "house", "city", "region", "postcode"]),
    ("vivo en Calle Mayor 17, 3º B, Ponferrada (24401).", "Calle Mayor 17, 3º B, Ponferrada (24401)", "ES",
     ["type", "name", "house", "unit", "city", "postcode"]),
    ("j'habite 12 rue de Rivoli, 75001 Paris", "12 rue de Rivoli, 75001 Paris", "FR",
     ["house", "type", "name", "postcode", "city"]),
    ("Flat 3, 35 Hennigan Street, Waterford by Friday", "Flat 3, 35 Hennigan Street, Waterford", None,
     ["pre", "house", "name", "city"]),
    ("4 Old Church Lane, Leeds LS1 4AP", "4 Old Church Lane, Leeds LS1 4AP", "GB",
     ["house", "name", "city", "postcode"]),
    ("Retorno Sur Adame 907 743, 13205-7920 Aguascalientes", "Retorno Sur Adame 907 743, 13205-7920 Aguascalientes",
     "MX", ["type", "name", "house", "postcode", "city"]),
    ("Address – H.No. 3-118, Gandhi Nagar, Almora 263601.", "H.No. 3-118, Gandhi Nagar, Almora 263601", "IN",
     ["pre", "name", "city", "postcode"]),
    ("Hauptstraße 5\n10115 Berlin\nGermany", "Hauptstraße 5\n10115 Berlin", "DE",
     ["name", "house", "postcode", "city"]),
    ("Send it to 4 Strand Road, Tralee, Co. Kerry, V92 X2E1.", "4 Strand Road, Tralee, Co. Kerry, V92 X2E1",
     "IE", ["house", "name", "city", "region", "postcode"]),
    ("4 Strand Road, Co. Kerry V92 X2E1", "4 Strand Road, Co. Kerry V92 X2E1", "IE",
     ["house", "name", "region", "postcode"]),
    ("12 Main Street, Dublin 8, D08 VF8H\nIreland", "12 Main Street, Dublin 8, D08 VF8H", "IE",
     ["house", "name", "city", "postcode"]),
])
def test_layouts_are_read_part_by_part(text, value, country, roles):
    (p,) = aa.find(text)
    assert p.full_text == value and text[p.start:p.end] == value and p.country == country
    assert [r for r, _s, _e in p.parts] == roles
    ends = [(s, e) for _r, s, e in p.parts]
    assert all(0 <= s < e <= len(value) for s, e in ends)
    assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))          # in order, no overlap
    assert all(value[s:e] == value[s:e].strip() for s, e in ends)


def test_parse_spans_the_whole_fragment_or_nothing():
    assert aa.parse("  Kerkstraat 14, 8861 AJ Harlingen ").full_text == "Kerkstraat 14, 8861 AJ Harlingen"
    assert aa.parse("see Kerkstraat 14, 8861 AJ Harlingen") is None


PROSE = [
    "via Slack 3 times",
    "In 2019, Thompson Road was closed, Smith said.",
    "Dr. R. Smith 3 times a week",
    "Spring 2024 schedule, String 3 of the guitar",
    "Kanye West 5, Chicago IL 60601",
    "Building 7 Collapse, New York, NY",
    "please confirm 790 tickets, Tempe office",
    "Windows 10, 2019 Edition",
    "Room 12, 101 Dalmatians",
    "Chapter 3, 150 Pages",
    "Page 12, 2024 Report",
    "I sent it via FedEx 2 days ago",
    "Gate 12 boards at 3pm; Room 4B",
    "we have 2 OG cables in stock",
    "I scored 3º in the race",
    "Version 2 of the CAP theorem; the CPU has 8 cores",
    "Plaza Hotel 5 stars, Monday",
    "Keyring 2 and Spring 3 are sold out",
    "Weg 5 is wrong",
    "I farm 12 acres near Boise",
    "The Beatles played 4 nights in Hamburg",
]


@pytest.mark.parametrize("text", NEGATIVES + PROSE)
def test_no_layout_reads_prose_or_numbers(text):
    assert aa.find(text) == []


def test_a_unit_word_needs_its_own_word():
    # "rm" (room) inside "confirm" is no unit: the address starts at the house
    (p,) = aa.find("confirm 790 Crescent Row, Tempe, AZ 85281 please")
    assert p.full_text == "790 Crescent Row, Tempe, AZ 85281"


def test_a_postcode_is_never_split_by_the_town():
    (p,) = aa.find("4 Old Church Lane, Leeds LS1 4AP")
    assert dict((r, p.full_text[s:e]) for r, s, e in p.parts)["postcode"] == "LS1 4AP"


@pytest.mark.parametrize("text", [
    "Aa " * 3000,
    "1 " + "Aa " * 3000,
    "12 " + "Bob " * 2000 + ",",
    "Flat 1, " * 500,
    "rue " + "de " * 2000 + "X",
    "Kambsstraße 2-8, " * 400,
    "1 Ab Cd Ef Gh, Ij Kl Mn Op Qr St, " * 300,
])
def test_adversarial_text_stays_fast(text):
    t0 = time.perf_counter()
    aa.find(text)
    assert time.perf_counter() - t0 < 0.25


# ── surrogates ───────────────────────────────────────────────────────────────

_KEEP = re.compile(r"(?i)^(?:de|des|du|la|le|da|do|dos|das|del|della|di|von|der|of|the|and"
                   r"|n|s|e|w|north|south|east|west|nagar)$")


def _replace(value, seed):
    p = aa.parse(value)
    g = MimicGen(seed)
    g._context = f"my address is {value}"
    return p, g._gen_address(DetectedEntity(value, 0, len(value), "address", parsed=p), mode="replace")


@pytest.mark.parametrize("country", sorted(I.COUNTRIES))
def test_surrogate_keeps_the_layout_and_leaks_no_part(country):
    for k, value in enumerate(_drawn(country, n=3)):
        p, out = _replace(value, k)
        assert out != value
        q = aa.parse(out)
        assert q is not None, (value, out)
        assert [r for r, _s, _e in q.parts] == [r for r, _s, _e in p.parts], (value, out)
        assert q.country == p.country, (value, out)
        part = {r: value[s:e] for r, s, e in p.parts}
        new = {r: out[s:e] for r, s, e in q.parts}
        for role in ("city", "postcode"):
            if role in part:
                assert not re.search(rf"(?<!\w){re.escape(part[role])}(?!\w)", out), (value, out)
        if "type" in part:
            assert new["type"] == part["type"]                       # "Calle" stays "Calle"
        words = re.findall(r"[^\W\d_]{3,}", part["name"])
        street_type = words[-1] if p.country in (None, "US", "CA", "GB", "AU", "IE", "IN") else None
        for w in words:
            if w != street_type and not _KEEP.match(w) and not re.search(
                    r"(?i)(?:" + "|".join(re.escape(e.rstrip(".")) for e in aa._CMP_STRONG + aa._CMP_WEAK) + r")$", w):
                assert not re.search(rf"(?<!\w){re.escape(w)}(?!\w)", out), (w, value, out)


@pytest.mark.parametrize("value, shape", [
    ("Kambsstraße 2-8, 90146 Augsburg", r"[A-ZÄÖÜ]\w+straße \d-\d, \d{5} [\w ]+"),
    ("Kerkstraat 14, 8861 AJ Harlingen", r"[A-Z]\w+straat \d\d, \d{4} [A-Z]{2} [\w ]+"),
    ("4 Old Church Lane, Leeds LS1 4AP", r"\d [\w ]+ Lane, [\w ]+ [A-Z]{2}\d [\d][A-Z]{2}"),
    ("633 James Green Apt. 500, Edmonton, AB L7K 7J3",
     r"\d{3} [\w' ]+ Green Apt\. \d{3}, [\w ]+, [A-Z]{2} [A-Z]\d[A-Z] \d[A-Z]\d"),
    ("Calle Mayor 17, 3º B, Ponferrada (24401)", r"Calle \w+ \d\d, \dº [A-Z], [\w ]+ \(\d{5}\)"),
    ("Hauptstr. 5, D-10115 Berlin", r"[A-Z]\w+str\. [1-9], D-\d{5} [\w ]+"),
    ("22 Rue de l'Église, 67000 Strasbourg", r"\d\d Rue de l'[AEIOUÉÈÊÂÔÎÛ]\w+, \d{5} [\w ]+"),
    ("4 Strand Road, Tralee, Co. Kerry, V92 X2E1",
     r"\d [A-Z]\w+ Road, [\w ]+, Co\. (?!Kerry)[A-Z]\w+, [ACDEFHKNPRTVWXY]\d\d [0-9ACDEFHKNPRTVWXY]{4}"),
])
def test_surrogate_shapes(value, shape):
    for seed in range(3):
        _p, out = _replace(value, seed)
        assert re.fullmatch(shape, out), out


def test_region_stays_in_its_country():
    for seed in range(6):
        _p, out = _replace("15 Kings Road, Hobart TAS 7000", seed)
        assert re.search(r" (?:NSW|VIC|QLD|WA|SA|NT|ACT) \d{4}$", out), out
        _p, out = _replace("Rua das Flores 12, Recife - PE, 50030-230", seed)
        assert re.search(r" - (?!PE)[A-Z]{2}, \d{5}-\d{3}$", out), out


def test_an_irish_address_stays_irish():
    # no generator layout has a county or an Eircode; real Irish mail often does
    value = "12 Main Street, Dublin 8, D08 VF8H"
    for seed in range(4):
        p, out = _replace(value, seed)
        q = aa.parse(out)
        assert q is not None and q.country == "IE", out
        new = {r: out[s:e] for r, s, e in q.parts}
        assert "Dublin" not in out and new["postcode"] != "D08 VF8H", out


@pytest.mark.parametrize("value, street", [
    ("5 North Street, Tempe, AZ 85281", "North Street"),
    ("084 West Light, San Antonio, TX 60547", "West Light"),
    ("77 East Ave, Tempe, AZ 85281", "East Ave"),
])
def test_a_direction_that_is_the_name_is_redrawn(value, street):
    # the US parser reads "North" as a directional and finds no street name;
    # the layouts read it as the name: either way the street must change
    from surrogateshield.core.detection import address_parser
    for seed in range(4):
        g = MimicGen(seed)
        g._context = "x"
        assert street not in g._replace_address(address_parser.parse(value))
        _p, out = _replace(value, seed)
        assert street not in out, out
    _p, out = _replace("12 West Elm Street, Austin, TX 78701", 1)
    assert out.split()[1] == "West" and out.split()[3] == "Street,"       # a direction before a name stays


def test_an_entity_without_a_parse_is_read_whole():
    # the US parser's first hit would be a piece ("35 Hennigan Street"); the
    # surrogate must replace the whole value, flat and town included
    value = "Flat 3, 35 Hennigan Street, Waterford"
    out = MimicGen(5)._gen_address(DetectedEntity(value, 0, len(value), "address"), mode="replace")
    assert out.startswith("Flat ") and "Hennigan" not in out and "Waterford" not in out
