"""Audit I8 / I12, gate J2 — places and companies the models found but a
gate dropped on the dev2 split, plus two structural ORG false positives.

* Pass D dropped every lower-case place name, even when the writer
  capitalises nothing ("my doctor ... in dayton").
* The relation gate missed "my boss at X", "I'm a ... at the X plant",
  "who is X? she ..." and devices named after their owner ("Lucas-iPhone").
* A model place span with a preposition in it ("uit Zwolle").
* Pass A read "the difference between an LLC" and "a clothing company" as
  company names; a bare legal form ("S-Corp") was kept as a named firm.
Model-free.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.detection.pipeline import (
    _detect_structural_orgs, _is_proper_capitalized, _snap_to_words, _writes_lowercase,
)
from surrogateshield.core.entities import DetectedEntity


def ent(text, span, kind, source="ner"):
    i = text.index(span)
    return DetectedEntity(span, i, i + len(span), kind, 0.85, source)


def kept(text, span, kind):
    k, _ = rg.gate(text, [ent(text, span, kind)])
    return [e.text for e in k] == [span]


# ── lower-case writers ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, place", [
    ("im 41 and my doctor at riverside family clinic in dayton put me on metformin", "dayton"),
    ("how much shoud i tip movers in denver, there moving me", "denver"),
])
def test_I8_lowercase_place_kept_for_lowercase_writer(text, place):
    assert _writes_lowercase(text)
    assert _is_proper_capitalized(place, text)


def test_I8_lowercase_common_noun_still_skipped_for_normal_writer():
    text = "Our office said the phoenix bird logo is fine."
    assert not _writes_lowercase(text)
    assert not _is_proper_capitalized("phoenix", text)


# ── relation gate ties ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, span, kind", [
    ("Can my boss at Piedmont Roofing legally hold back two weeks of pay?",
     "Piedmont Roofing", "GPE"),
    ("I'm a Unite shop steward at the Dagenham plant.", "Dagenham", "GPE"),
    ("who is Thandeka? she keeps liking my posts", "Thandeka", "ORG"),
    ("DHCPACK on 192.168.1.23 (Lucas-iPhone) via br0", "Lucas-iPhone", "ORG"),
])
def test_I12_tied_places_and_orgs_kept(text, span, kind):
    assert kept(text, span, kind)


@pytest.mark.parametrize("text, span, kind", [
    ("Is the boss fight in Elden Ring at Limgrave hard?", "Limgrave", "GPE"),
    ("What's the difference between an LLC and an S-Corp for a freelancer?", "S-Corp", "ORG"),
    ("who is Tesla? a company or a person", "Tesla", "ORG"),
])
def test_I12_untied_places_and_orgs_dropped(text, span, kind):
    assert not kept(text, span, kind)


def test_I1_preposition_cut_from_place():
    text = "Hallo, ik ben Sanne Wijnberg uit Zwolle (8011 PK)."
    out = _snap_to_words([ent(text, "uit Zwolle", "GPE")], text)[0]
    assert out.text == "Zwolle" and text[out.start:out.end] == "Zwolle"


# ── Pass A ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "What's the difference between an LLC and an S-Corp?",
    "I bought it from a clothing company last week.",
    "The insurance company denied my claim.",
])
def test_I8_structural_org_needs_a_name(text):
    assert _detect_structural_orgs(text, []) == []


@pytest.mark.parametrize("text, name", [
    ("I used to work at the Phoenix Group in Leeds.", "Phoenix"),
    ("i used to work for the target corporation", "target"),
])
def test_I8_structural_org_still_found(text, name):
    assert name in [e.text for e in _detect_structural_orgs(text, [])]
