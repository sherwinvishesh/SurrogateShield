"""One implementation (audit F4, J14): the app's detection/, generation/ and
reconstruction/ paths are the package's modules, not copies of them."""

import importlib
from pathlib import Path

import pytest

import config
import util
from surrogateshield.core import entities
from surrogateshield.core.detection import pipeline
from surrogateshield.core.reconstruction import resolve

ROOT = Path(__file__).resolve().parent.parent

ALIASES = {
    "detection.address_parser": "surrogateshield.core.detection.address_parser",
    "detection.geo_data": "surrogateshield.core.detection.geo_data",
    "detection.pattern_scan": "surrogateshield.core.detection.pattern_scan",
    "detection.quasi_identifier": "surrogateshield.core.detection.quasi_identifier",
    "detection.service_query": "surrogateshield.core.detection.service_query",
    "detection.entity_trace": "surrogateshield.core.detection.entity_trace",
    "detection.context_guard": "surrogateshield.core.detection.context_guard",
    "generation.logic": "surrogateshield.core.generation.mimic",
}


@pytest.mark.parametrize("app_path,package_path", sorted(ALIASES.items()))
def test_F4_app_module_is_package_module(app_path, package_path):
    assert importlib.import_module(app_path) is importlib.import_module(package_path)


def test_F4_entities_and_helpers_shared():
    for name in ("DetectedEntity", "mask_spans", "remove_span_overlap",
                 "plan_substitutions", "splice", "apply_entity_surrogates"):
        assert getattr(util, name) is getattr(entities, name), name


def test_F4_cascade_wrapper_binds_config(monkeypatch):
    from detection import logic

    assert logic.deduplicate is pipeline.deduplicate
    assert logic._filter_topical_geo_entities is pipeline._filter_topical_geo_entities
    seen = {}
    monkeypatch.setattr(pipeline, "run_cascade", lambda *a, **kw: seen.update(kw) or ([], []))
    monkeypatch.setattr(config, "ENTITY_TRACE_FALLBACK_THRESHOLD", 0.42)
    logic.run_cascade("x", use_post_passes=False, context_guard_enabled=False)
    assert seen["entity_trace_fallback_threshold"] == 0.42
    assert seen["context_guard_enabled"] is False          # explicit kwarg wins
    assert seen["use_post_passes"] is False
    assert seen["spacy_model"] == config.SPACY_MODEL


def test_F4_resolver_is_package_resolver(monkeypatch):
    from reconstruction.logic import ResolvePass

    assert issubclass(ResolvePass, resolve.ResolvePass)
    seen = {}
    monkeypatch.setattr(resolve.ResolvePass, "resolve",
                        lambda self, t, m, f=85, current=None, sent=None: seen.setdefault("f", f) and t)
    monkeypatch.setattr(config, "FUZZY_MATCH_THRESHOLD", 93)
    ResolvePass().resolve("t", {"a": "b"})
    assert seen["f"] == 93


def test_F4_no_implementation_left_in_app_tree():
    """Root modules are aliases/wrappers only: short, no regexes, no classes
    other than the config-binding ResolvePass."""
    for pkg in ("detection", "generation", "reconstruction"):
        for path in (ROOT / pkg).glob("*.py"):
            src = path.read_text(encoding="utf-8")
            assert len(src.splitlines()) <= 60, path
            assert "re.compile" not in src, path
            assert src.count("class ") <= 1, path
