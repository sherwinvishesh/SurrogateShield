"""Gate J2 / audit I1 — URLs that identify a person: a seller page on a
marketplace, a share link with a document path, a family or vanity domain,
and "my personal <service> at <url>". Public pages stay opaque (kept)."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


def personal(text):
    return [(text[s:e], p) for s, e, p in ps.find_urls(text)]


@pytest.mark.parametrize("text,url", [
    ("Shop: etsy.com/shop/KnotsByNaledi, IG @knotsbynaledi", "etsy.com/shop/KnotsByNaledi"),
    ("my personal nextcloud at https://cloud.okeke-family.ng is down", "https://cloud.okeke-family.ng"),
    ("summarise https://www.dropbox.com/scl/fi/k2m9x7/Contract_Olamide_Bankole_signed.pdf?dl=0",
     "https://www.dropbox.com/scl/fi/k2m9x7/Contract_Olamide_Bankole_signed.pdf?dl=0"),
    ("www.analuisakr.photo | bookings open", "www.analuisakr.photo"),
    ("tip me at ko-fi.com/lunarkoi", "ko-fi.com/lunarkoi"),
    ("store: https://knotsbynaledi.myshopify.com", "https://knotsbynaledi.myshopify.com"),
    ("photos at thejonesfamily.com", "thejonesfamily.com"),
])
def test_J2_I1_personal_url(text, url):
    assert (url, True) in personal(text)


@pytest.mark.parametrize("text,url", [
    ("see https://www.apple.com/shop/buy-iphone", "https://www.apple.com/shop/buy-iphone"),
    ("docs at https://docs.python.org/3/library/re.html", "https://docs.python.org/3/library/re.html"),
    ("download from https://www.dropbox.com/install", "https://www.dropbox.com/install"),
    ("read github.com/features", "github.com/features"),
    ("our bank at https://www.chase.com/personal", "https://www.chase.com/personal"),
])
def test_J2_I1_public_url_opaque(text, url):
    assert (url, False) in personal(text)
