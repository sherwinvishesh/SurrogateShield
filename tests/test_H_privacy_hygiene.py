"""Workstream H/J hygiene: no value reaches a log, the library installs a
NullHandler, the tagger fails closed on a runtime error, the DOB rule's year
is a seam, and masking is idempotent on a message that quotes surrogates
back. Model-free except the one heavy test at the end."""

import logging
import re

import pytest

import surrogateshield as ss
from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.detection import pii_tagger as T
from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.errors import DetectorUnavailable
from surrogateshield.core.generation import mimic
from surrogateshield.core.generation.mimic import MimicGen

ADDRESS = "789 Crescent Row, Tempe, AZ 85281"


def _types(ents):
    return {e.type: e.text for e in ents}


# ── H: logs carry counts, never a value ──────────────────────────────────────

def test_H_shift_exhaustion_logs_no_address(caplog, monkeypatch):
    monkeypatch.setattr(mimic, "shift_house_number", lambda *a, **k: None)
    ent = next(e for e in pattern_scan.scan(f"Ship it to {ADDRESS}.") if e.type == "address")
    with caplog.at_level(logging.DEBUG, logger="surrogateshield"):
        out = MimicGen(seed=3)._gen_address(ent, mode="shift")
    assert out != ent.text and "789" not in out
    assert "shift collision exhausted" in caplog.text          # the fallback is still logged
    for piece in ("789", "Crescent", "Tempe", "85281"):
        assert piece not in caplog.text


NAMES = ["Alice Brown", "Bob Okafor"]
_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|" + "|".join(map(re.escape, NAMES)))


def _stub_cascade(text, skip_values=None, **_kw):
    skip = skip_values or set()
    found = [DetectedEntity(m.group(), m.start(), m.end(),
                            "email" if "@" in m.group() else "person", 0.99, "stub")
             for m in _PII.finditer(text) if m.group() not in skip]
    return found, []


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(pipeline, "run_cascade", _stub_cascade)


def test_H_no_value_reaches_a_log_at_info_or_above(stub, caplog):
    text = f"Alice Brown <alice.brown@corp.example> lives at {ADDRESS}; her SSN is 123-45-6789."
    with caplog.at_level(logging.INFO):
        s = ss.Session()
        masked = s.mask(text)
        s.unmask(masked)
    assert masked != text
    for value in ("Alice Brown", "alice.brown", "123-45-6789", "789", "Crescent"):
        assert value not in caplog.text


# ── H: a library logger with a NullHandler ───────────────────────────────────

def test_H_library_logger_has_a_null_handler_and_nothing_else():
    lg = logging.getLogger("surrogateshield")
    assert lg.handlers and all(isinstance(h, logging.NullHandler) for h in lg.handlers)
    assert logging.getLogger("surrogateshield.core.generation.mimic").hasHandlers()
    assert lg.propagate                                       # the application's handlers still see it


# ── H: the tagger fails closed on a runtime error ────────────────────────────

class _Broken:
    labels = T.LABELS

    def __init__(self, exc):
        self.exc = exc

    def token_probs(self, texts, window, stride, batch):
        raise self.exc


def _tagger(model):
    t = T.PIITagger.__new__(T.PIITagger)
    t.model, t.window, t.stride, t.batch, t.merge_address = model, 256, 64, 8, True
    return t


def test_H_tagger_runtime_error_is_detector_unavailable():
    t = _tagger(_Broken(RuntimeError("cuda gone")))
    with pytest.raises(DetectorUnavailable, match="cuda gone") as info:
        t.detect("Ana Ruiz, 34.", None)
    assert "Ana" not in str(info.value) and "not masked" in str(info.value)
    with pytest.raises(DetectorUnavailable, match="IndexError"):
        _tagger(_Broken(IndexError("index out of range"))).detect_many(["Ana Ruiz", "Bo Li"])


def test_H_tagger_passes_its_own_unavailable_through_and_skips_blank_text():
    own = DetectorUnavailable("weights gone")
    with pytest.raises(DetectorUnavailable) as info:
        _tagger(_Broken(own)).detect("Ana Ruiz", None)
    assert info.value is own
    assert _tagger(_Broken(RuntimeError("never called"))).detect("  \n", None) == []


# ── J: the DOB rule's year is a seam ─────────────────────────────────────────

def test_J_dob_without_a_cue_follows_the_injected_year(monkeypatch):
    text = "We met on 04/12/2023 at the office."
    monkeypatch.setattr(pattern_scan, "_this_year", lambda: 2026)
    assert "dob" not in _types(pattern_scan.scan(text))          # three years old: a date
    monkeypatch.setattr(pattern_scan, "_this_year", lambda: 2040)
    assert _types(pattern_scan.scan(text)).get("dob") == "04/12/2023"


def test_J_two_digit_year_pivot_follows_the_injected_year(monkeypatch):
    text = "renewal 03/15/30 please"
    monkeypatch.setattr(pattern_scan, "_this_year", lambda: 2026)
    assert _types(pattern_scan.scan(text)).get("dob") == "03/15/30"   # 1930
    monkeypatch.setattr(pattern_scan, "_this_year", lambda: 2033)
    assert "dob" not in _types(pattern_scan.scan(text))              # 2030, three years back


def test_J_year_is_read_at_call_time():
    assert not hasattr(pattern_scan, "_THIS_YEAR")
    assert pattern_scan._this_year() >= 2026


# ── H: masking is idempotent on quoted-back surrogates ───────────────────────

def test_H_masking_the_masked_text_again_changes_nothing(stub):
    original = "Alice Brown wrote to bob.okafor@corp.example and cc'd Bob Okafor."
    s = ss.Session()
    first = s.mask(original)
    assert "Alice Brown" not in first and "bob.okafor@" not in first
    again = s.mask_result(first)
    assert again.text == first and again.replacements == {}     # a surrogate of a surrogate is never made
    assert s.mask(first) == first
    assert s.unmask(first) == original


@pytest.mark.heavy
def test_H_masking_the_masked_text_again_changes_nothing_with_the_models():
    original = ("Hi, I'm Priya Raghunathan from Leeds. Mail me at priya.r@mailhost.example "
                "or call +44 7700 900123; I live at 14 Elm Grove, Leeds LS6 2AB.")
    s = ss.Session()
    first = s.mask(original)
    assert "Raghunathan" not in first and "priya.r@" not in first
    again = s.mask_result(first)
    assert again.replacements == {} and again.text == first
    assert s.unmask(first) == original
