"""Audit I14, gate J2 — address parts left in the message on the dev2 split:
postcode-towns before or after a street ("79098 Freiburg", "Hamilton 3204"),
bracketed Dutch postcodes, unit numbers ("flat 4,"), number-after streets
confirmed by a postcode-town ("Laugavegur 52, 101 Reykjavík"), lower-case
Italian and British streets. Negatives are headings and product names in the
same shape. Model-free (pattern scan + structural pass).
"""

import pytest

from surrogateshield.core.detection import pattern_scan, structural


def addresses(text):
    ents = pattern_scan.scan(text)
    added, _ = structural.detect(text, ents)
    out = [e for e in list(ents) + added if e.type == "address"]
    for e in out:
        assert text[e.start:e.end] == e.text
    return sorted((e.start, e.text) for e in out)


def covered(text):
    return [t for _s, t in addresses(text)]


@pytest.mark.parametrize("text, parts", [
    ("abito a Modena in via Emilia Est 211. Il mio codice", ["via Emilia Est 211"]),
    ("lives alone at 14 birchwood close, wokingham rg40 2hd. any tips", ["14 birchwood close", "wokingham"]),
    ("Her address is Laugavegur 52, 101 Reykjavík.", ["Laugavegur 52", "101 Reykjavík"]),
    ("Shipping name was Ludmila Horvatová, 17 Hlavná, 040 01 Košice.",
     ["17 Hlavná", "040 01 Košice"]),
    ("flat 4, 70 Cowley Road\nim home after 6", ["flat 4", "70 Cowley Road"]),
    # one address now: a German layout (address_assembly), not two parts
    ("wohne in der Lindenstraße 14, 79098 Freiburg. Mein", ["Lindenstraße 14, 79098 Freiburg"]),
    ("Current address: 9 Kowhai Grove, Hamilton 3204\nEmployer:", ["9 Kowhai Grove", "Hamilton 3204"]),
    ("ik ben Sanne Wijnberg uit Zwolle (8011 PK). Kun je", ["8011 PK"]),
])
def test_I14_address_parts_found(text, parts):
    assert covered(text) == parts


@pytest.mark.parametrize("text", [
    "I sent it via FedEx 2 days ago",
    "Page 12, 2024 Report",
    "Windows 10, 2019 Edition",
    "Room 12, 101 Dalmatians",
    "Chapter 3, 150 Pages",
    "my address is in the Version 3, 2024 Edition doc",
    "we met in 1500 BC Rome (1500 BC)",
    "Laugavegur 52, 101 Reykjavík is a nice street",   # no address cue
])
def test_I8_address_part_near_misses(text):
    assert covered(text) == []
