"""Cue rules added after the devlarge attribution (V3 §3.1): a bare ID label,
a quoted or named password, a bare "key:" with a generated value, and a site
named after someone in the message. Every value here is made up. Model-free:
pattern_scan only.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


def found(text):
    return [(e.type, e.text) for e in ps.scan(text)]


def urls(text):
    return [(text[s:e], p) for s, e, p in ps.find_urls(text)]


@pytest.mark.parametrize("text, value", [
    ("Hi, I'm new here (ID: EMP-7712345, DOB below)", "EMP-7712345"),
    ("Name: Tova Lind\nID: U012345\nPhone: none", "U012345"),
    ("ID: ID 112233", "ID 112233"),
    ("a student at the college (ID 123456), living nearby", "123456"),
    ("my staff ID is S12345678, thanks", "S12345678"),
    ("licence Q01234567, IBAN to follow", "Q01234567"),
    ("Licence: F1234567 | Passport: none", "F1234567"),
    ("insurance HMO-123456789.", "HMO-123456789"),
    ("保险单号 INS-1234567。", "INS-1234567"),
])
def test_bare_id_label(text, value):
    assert ("id_number", value) in found(text)


@pytest.mark.parametrize("text", [
    "order ID 1234567 has not shipped",
    "process ID 4412345 crashed again",
    "released under the MIT license 2024 edition",
    "under the Apache License 2.0",
    "insurance costs 1200000 dollars a year",
    "ID 3 and ID 4 are swapped",
])
def test_bare_id_label_leaves(text):
    assert not [v for t, v in found(text) if t == "id_number"]


@pytest.mark.parametrize("text, value", [
    ('{"name": "x", "password": "Cobalt%raven123"}', "Cobalt%raven123"),
    ("my password for the sim session is Willow%stone25.", "Willow%stone25"),
    ("(password for registry: Maple$frost026)", "Maple$frost026"),
    ("Use Amber*grove835 as the access password in the form", "Amber*grove835"),
    ("mail | key: ab12cd34-ef56gh78-ij90kl12", "ab12cd34-ef56gh78-ij90kl12"),
    ("phone, key 1a2b3c4d-5e6f7g8h-9i0j1k2l).", "1a2b3c4d-5e6f7g8h-9i0j1k2l"),
])
def test_named_secret(text, value):
    assert ("credential", value) in found(text)


@pytest.mark.parametrize("text", [
    "reset my password for Gmail",
    "the password for my account is too weak",
    "cache key: session12345abc",
    "the key: value pairs come next",
    "the key is in C major",
    "Use Python3 as the language",
])
def test_named_secret_leaves(text):
    assert not [v for t, v in found(text) if t == "credential"]


@pytest.mark.parametrize("text, url", [
    ("I'm Ana Kovač (https://anakovac.dev) and I need help", "https://anakovac.dev"),
    ("Kerr, Ó Bríain, Liam\nsite https://liamobriain.dev", "https://liamobriain.dev"),
    ("Name: Rita M. Solano\nWebsite: https://ritasolano.dev", "https://ritasolano.dev"),
    ("Thanks\nritasolano21@mail.example\nhttps://ritasolano.dev", "https://ritasolano.dev"),
    ("by Jane-Doe at https://jane-doe.co.uk/blog", "https://jane-doe.co.uk/blog"),
])
def test_site_named_after_someone_in_the_message(text, url):
    assert (url, True) in urls(text)


@pytest.mark.parametrize("text, url", [
    ("Docs at https://web.dev say otherwise", "https://web.dev"),
    ("visit https://jane-doe.com today", "https://jane-doe.com"),     # the URL alone names no one
    ("I like a reactjs app: https://reactjs.org", "https://reactjs.org"),
    ("see https://go.dev/doc for Go", "https://go.dev/doc"),
])
def test_site_not_named_after_anyone_stays_opaque(text, url):
    assert (url, False) in urls(text)
