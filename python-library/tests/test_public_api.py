"""Public API of the pip package: config / scan / mask / unmask / flush.

Fast tests cover config, versioning, and session plumbing. Tests that run the
full detection cascade (spaCy + HuggingFace) are marked heavy:
    python -m pytest python-library/tests/ -m heavy
"""

import re
from pathlib import Path

import pytest

import surrogateshield as ss
from surrogateshield._state import cfg, session


@pytest.fixture(autouse=True)
def _clean_session():
    ss.config(detailed_view=False)
    ss.flush()
    yield
    ss.flush()
    ss.config(detailed_view=False)


# ── Version ───────────────────────────────────────────────────────────────────

def test_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert ss.__version__ == declared == "2.0.0"


def test_public_all():
    assert set(ss.__all__) == {"config", "scan", "pii_finder", "mask", "unmask", "flush"}
    assert ss.pii_finder is ss.scan


# ── Session plumbing (no models needed) ───────────────────────────────────────

def test_flush_rotates_session():
    old_id = session.id
    session.get_shadow_map().update({"a": "b"})
    ss.flush()
    assert session.id != old_id
    assert session.get_shadow_map().lookup_original("b") is None


def test_unmask_plain_string_without_mappings():
    assert ss.unmask("hello world") == "hello world"


def test_unmask_accepts_sdk_like_objects():
    class FakeBlock:
        text = "the response text"

    class FakeAnthropicResponse:
        content = [FakeBlock()]

    assert ss.unmask(FakeAnthropicResponse()) == "the response text"


# ── Full pipeline (heavy: loads spaCy + HF models) ────────────────────────────

heavy = pytest.mark.heavy


@heavy
def test_mask_unmask_roundtrip_shift_mode():
    original = "Ship it to 789 Crescent Row Apt 4B, Tempe, AZ 85281-1234 and bill John Smith."
    masked = ss.mask(original)
    assert "789 Crescent Row" not in masked
    assert "John Smith" not in masked
    restored = ss.unmask(masked)
    assert restored == original


@heavy
def test_mask_shift_keeps_street_visible():
    masked = ss.mask("I live at 789 Crescent Row, Tempe, AZ 85281.")
    assert "Crescent Row" in masked
    assert "789" not in masked


@heavy
def test_mask_replace_mode_hides_components():
    ss.config(detailed_view=False, address_mode="replace")
    masked = ss.mask("I live at 789 Crescent Row, Tempe, AZ 85281.")
    assert "789" not in masked and "Tempe" not in masked and "85281" not in masked


@heavy
def test_mask_consistency_across_calls():
    m1 = ss.mask("addr: 789 Crescent Row, Tempe, AZ 85281")
    m2 = ss.mask("again: 789 Crescent Row, Tempe, AZ 85281")
    s1 = m1.split("addr: ")[1]
    s2 = m2.split("again: ")[1]
    assert s1 == s2


@heavy
def test_pii_off_respected():
    ss.config(detailed_view=False, pii_off=["address"])
    masked = ss.mask("my address is 789 Crescent Row, Tempe, AZ 85281")
    assert "789 Crescent Row" in masked  # detected but NOT replaced


@heavy
def test_scan_reports_types():
    found = ss.scan("email jane@example.com, addr 789 Crescent Row, Tempe, AZ")
    assert found.get("jane@example.com") == "email"
    assert found.get("789 Crescent Row, Tempe, AZ") == "address"
