"""Presidio baseline (presidio/engine.py, presidio/detect.py) — audit A5, I29.

Model-free: the analyzer is replaced by a fake.
"""

import inspect
from types import SimpleNamespace

import pytest

import presidio.detect as pdetect
import presidio.engine as pengine
from presidio.detect import PresidioEntity, resolve_overlaps


def test_A5_presidio_longest_span_wins():
    phone = PresidioEntity("PHONE_NUMBER", "+44 7911 123456", 10, 25, 0.75)
    date = PresidioEntity("DATE_TIME", "7911 123456", 14, 25, 0.85)   # higher score, shorter
    other = PresidioEntity("PERSON", "Ann", 0, 3, 0.85)
    kept = resolve_overlaps([date, phone, other])
    assert kept == [other, phone]


def test_A5_equal_length_keeps_higher_score():
    a = PresidioEntity("LOCATION", "Paris", 0, 5, 0.6)
    b = PresidioEntity("PERSON", "Paris", 0, 5, 0.85)
    assert resolve_overlaps([a, b]) == [b]


class _FakeAnalyzer:
    def __init__(self, results=(), exc=None):
        self.results, self.exc, self.calls = list(results), exc, []

    def analyze(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.results


def test_I29_presidio_default_config(monkeypatch):
    # Shipped defaults: no configuration file is passed to the provider.
    src = inspect.getsource(pengine.get_analyzer)
    assert "AnalyzerEngineProvider().create_engine()" in src
    cfg = pengine.baseline_config()
    assert cfg["score_threshold"] == pengine.SCORE_THRESHOLD
    assert "shipped defaults" in cfg["config"]

    fake = _FakeAnalyzer([SimpleNamespace(entity_type="PERSON", start=0, end=3, score=0.85)])
    monkeypatch.setattr(pengine, "get_analyzer", lambda: fake)
    out = pdetect.detect("Ann went home")
    assert [(e.entity_type, e.text) for e in out] == [("PERSON", "Ann")]
    # every entity enabled, explicit threshold
    assert fake.calls[0]["entities"] is None
    assert fake.calls[0]["score_threshold"] == pengine.SCORE_THRESHOLD


def test_I29_presidio_exceptions_propagate(monkeypatch):
    monkeypatch.setattr(pengine, "get_analyzer", lambda: _FakeAnalyzer(exc=ValueError("boom")))
    with pytest.raises(ValueError):
        pdetect.detect("Ann went home")       # never "found nothing"


def test_I29_presidio_unavailable_is_none(monkeypatch):
    monkeypatch.setattr(pengine, "get_analyzer", lambda: None)
    assert pdetect.detect("Ann went home") is None
