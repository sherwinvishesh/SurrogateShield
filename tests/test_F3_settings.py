"""Audit F3 — configuration knobs are honest: each detection threshold
changes an outcome, dead knobs are gone, and the Nominatim existence check
(which sent the street to a third party and never changed the output) is
removed. Model-free: spaCy is replaced by a stub that emits fixed entities.
"""

import types
import warnings

import pytest

import surrogateshield as ss
from surrogateshield.core.detection import entity_trace, pipeline

TEXT = "Dana Price met Acme in Tempe near Lake Pleasant at Gate Arena."
LABELS = {"Dana Price": "PERSON", "Acme": "ORG", "Tempe": "GPE",
          "Lake Pleasant": "LOC", "Gate Arena": "FAC"}


class _Doc:
    def __init__(self, text):
        self.ents = [types.SimpleNamespace(text=v, label_=lab, start_char=text.index(v),
                                           end_char=text.index(v) + len(v))
                     for v, lab in LABELS.items() if v in text]


@pytest.fixture
def stub_models(monkeypatch):
    from surrogateshield.core.detection import context_guard
    monkeypatch.setattr(entity_trace, "_get_nlp", lambda *_a, **_k: _Doc)
    monkeypatch.setattr(context_guard, "_get_ner", lambda *_a, **_k: (lambda _t: []))


def _types(guard=True, **kw):
    conf, nc = pipeline.run_cascade(TEXT, use_context_guard=guard, use_post_passes=False, **kw)
    return ({e.type for e in conf if e.source == "ner"},
            {e.type for e in nc if e.source == "ner"})


# Type scores: PERSON 0.88, GPE/ORG 0.85, LOC 0.74, FAC 0.70.

def test_F3_context_guard_threshold_changes_outcome(stub_models):
    conf, nc = _types()
    assert conf == {"PERSON", "GPE", "ORG", "LOC", "FAC"}
    conf, nc = _types(context_guard_threshold=0.75)
    assert {"LOC", "FAC"} <= nc and not {"LOC", "FAC"} & conf


def test_F3_high_threshold_changes_outcome(stub_models):
    conf, nc = _types(context_guard_threshold=0.86)
    assert {"GPE", "ORG"} <= conf                       # 0.85 ≥ high 0.85 → confirmed
    conf, nc = _types(context_guard_threshold=0.86, entity_trace_high_threshold=0.86)
    assert {"GPE", "ORG"} <= nc and not {"GPE", "ORG"} & conf


def test_F3_low_threshold_changes_outcome(stub_models):
    conf, nc = _types(entity_trace_low_threshold=0.75)
    assert not {"LOC", "FAC"} & (conf | nc)             # below low → dropped


def test_F3_fallback_threshold_applies_without_context_guard(stub_models):
    conf, _ = _types(guard=False)
    assert {"LOC", "FAC"} <= conf                       # ≥ fallback 0.65 → promoted
    conf, _ = _types(guard=False, entity_trace_fallback_threshold=0.72)
    assert "LOC" in conf and "FAC" not in conf


def test_F3_dead_knobs_removed():
    import config
    for name in ("SERVICE_QUERY_VERIFY_ADDRESSES", "CONTEXT_GUARD_FALLBACK_TO_OLLAMA",
                 "LOG_LEVEL", "SHOW_DETECTION_TABLE"):
        assert not hasattr(config, name), name
    from surrogateshield.core.detection import address_parser
    assert not hasattr(address_parser, "verify_address_exists")


def test_F3_verify_addresses_is_deprecated_no_op():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ss.config(verify_addresses=True)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert not hasattr(ss.Config(), "verify_addresses")
