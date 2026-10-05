"""Public API of the pip package: config / scan / mask / unmask / flush.

Fast tests cover config, versioning, and session plumbing. Tests that run the
full detection cascade (spaCy + HuggingFace) are marked heavy:
    python -m pytest python-library/tests/ -m heavy
"""

import re
from pathlib import Path

import pytest

import surrogateshield as ss


@pytest.fixture(autouse=True)
def _clean_session():
    ss.flush()
    yield


# ── Version ───────────────────────────────────────────────────────────────────

def test_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert ss.__version__ == declared


def test_public_all():
    assert set(ss.__all__) == {
        "config", "scan", "pii_finder", "mask", "mask_result", "unmask", "forget", "flush",
        "Session", "Config", "Detection", "MaskResult", "current_session", "use_session",
        "DetectorUnavailable", "StorageError", "__version__",
    }
    assert ss.pii_finder is ss.scan
    for name in ss.__all__:
        assert hasattr(ss, name), name


# ── Session plumbing (no models needed) ───────────────────────────────────────

def test_flush_rotates_session():
    old = ss.current_session()
    old._shadow.update({"a": "b"})
    ss.flush()
    new = ss.current_session()
    assert new is not old and new.id != old.id
    assert new.mappings == {}
    assert old._shadow.get_all() == {}            # the closed session is wiped too


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
    ss.config(detailed_view=False, address_mode="shift")
    masked = ss.mask("I live at 789 Crescent Row, Tempe, AZ 85281.")
    assert "Crescent Row" in masked
    assert "789" not in masked


@heavy
def test_I22_default_auto_replaces_outside_service_queries():
    masked = ss.mask("I live at 789 Crescent Row, Tempe, AZ 85281.")
    assert "Crescent Row" not in masked and "Tempe" not in masked


@heavy
def test_I22_default_auto_shifts_service_queries():
    masked = ss.mask("Find a coffee shop near 789 Crescent Row, Tempe, AZ 85281.")
    assert "Crescent Row, Tempe, AZ 85281" in masked
    assert "789 Crescent" not in masked


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
    found = ss.scan("email jane@example.com, addr 789 Crescent Row, Tempe, AZ", as_dict=True)
    assert found.get("jane@example.com") == "email"
    assert found.get("789 Crescent Row, Tempe, AZ") == "address"
