"""Audit I17 — detection fails closed: a missing or failing model raises
DetectorUnavailable instead of returning "no entities". Model-free."""

import logging

import pytest

from surrogateshield.core.detection import context_guard as cg
from surrogateshield.core.detection import entity_trace as et
from surrogateshield.core.detection import pipeline as pl
from surrogateshield.core.errors import DetectorUnavailable

TEXT = "Alice Johnson lives in Springfield and works at Initech."


@pytest.fixture
def no_spacy_cache(monkeypatch):
    monkeypatch.setattr(et, "_nlp", {})


@pytest.fixture
def no_ner_cache(monkeypatch):
    monkeypatch.setattr(cg, "_ner_pipelines", {})


def test_I17_missing_spacy_model_raises(no_spacy_cache):
    with pytest.raises(DetectorUnavailable, match="python -m spacy download"):
        et.trace(TEXT, spacy_model="xx_no_such_model_sm")


def test_I17_missing_model_is_cached_and_raises_every_time(no_spacy_cache):
    for _ in range(2):
        with pytest.raises(DetectorUnavailable):
            et._get_nlp("xx_no_such_model_sm")
    assert isinstance(et._nlp["xx_no_such_model_sm"], DetectorUnavailable)


def test_I17_spacy_error_on_input_raises(no_spacy_cache):
    def broken(_text):
        raise ValueError("boom")
    et._nlp["fake"] = broken
    with pytest.raises(DetectorUnavailable, match="boom"):
        et.trace(TEXT, spacy_model="fake")


def test_I17_context_guard_missing_model_raises(no_ner_cache, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with pytest.raises(DetectorUnavailable, match="ContextGuard"):
        cg.guard(TEXT, [], model_name="no-such-org/no-such-model-xyz", enabled=True)


def test_I17_context_guard_inference_error_raises(no_ner_cache):
    def broken(_text):
        raise RuntimeError("cuda gone")
    cg._ner_pipelines[("fake", -1)] = broken
    with pytest.raises(DetectorUnavailable, match="cuda gone"):
        cg.guard(TEXT, [], model_name="fake", enabled=True)


def test_I17_context_guard_disabled_needs_no_model(no_ner_cache):
    # Switching the stage off explicitly is the documented way to run without it.
    confirmed, uncertain = cg.guard(TEXT, [], model_name="no-such-model", enabled=False)
    assert confirmed == [] and uncertain == []


def test_I17_cascade_propagates(monkeypatch):
    def unavailable(*a, **k):
        raise DetectorUnavailable("spaCy model missing")
    monkeypatch.setattr(pl.entity_trace, "trace", unavailable)
    with pytest.raises(DetectorUnavailable):
        pl.run_cascade(TEXT, context_guard_enabled=False)


def test_I17_library_mask_raises_and_returns_nothing(monkeypatch):
    import surrogateshield as ss

    def unavailable(*a, **k):
        raise DetectorUnavailable("spaCy model missing")
    monkeypatch.setattr(ss._pipeline, "run_cascade", unavailable)
    with pytest.raises(ss.DetectorUnavailable):
        ss.mask(TEXT)
    with pytest.raises(ss.DetectorUnavailable):
        ss.scan(TEXT)


def test_I17_app_pipeline_sends_nothing(monkeypatch):
    import pipeline as app_pipeline
    from detection import logic as sentinel

    def unavailable(*a, **k):
        raise DetectorUnavailable("spaCy model missing")
    assert app_pipeline.sentinel_layer is sentinel
    monkeypatch.setattr(sentinel, "run_cascade", unavailable)
    with pytest.raises(DetectorUnavailable):
        app_pipeline.anonymise_text(TEXT)


def test_I17_runner_aborts_instead_of_recording_rows():
    import inspect
    import json_tester
    src = inspect.getsource(json_tester)
    assert "except (SendMismatch, DetectorUnavailable):" in src


def test_I17_root_logger_never_below_warning():
    import inspect
    import logging
    import main
    src = inspect.getsource(main)
    assert "getLogger().setLevel(logging.ERROR" not in src
    assert "getLogger().setLevel(logging.INFO" not in src
    root = logging.getLogger()
    before = root.level
    try:
        main._set_detailed_logging(True)
        assert root.level == before                    # root untouched
        assert logging.getLogger("surrogateshield.x").getEffectiveLevel() == logging.INFO
        assert logging.getLogger("pipeline").getEffectiveLevel() == logging.INFO
        main._set_detailed_logging(False)
        assert root.level == before
        assert logging.getLogger("surrogateshield.x").getEffectiveLevel() == before
    finally:
        main._set_detailed_logging(False)
