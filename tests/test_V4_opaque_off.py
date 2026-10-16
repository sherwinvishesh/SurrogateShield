"""V4 Phase 1 step 6: an off PatternScan contributes nothing, its URL rule
included. Before, ``opaque_spans`` ran with the stage off and every
non-pattern candidate inside a URL was dropped, so ``h9/no_patterns`` leaked
every URL by construction. The default (PatternScan on) is unchanged.
Model-free; every text is invented."""

from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pipeline

NO_MODELS = dict(use_entity_trace=False, use_context_guard=False, use_tagger=False)
TEXT = "My page is https://example.org/u/jroe and my mail jane.roe@example.org"


def _opaque(cfg):
    trace = []
    pipeline.run_cascade(TEXT, config=cfg, trace=trace, **NO_MODELS)
    return trace[0]["opaque"]


def test_pattern_scan_on_keeps_urls_opaque():
    s = TEXT.index("https://")
    assert _opaque(C.preset("balanced")) == [[s, s + len("https://example.org/u/jroe")]]


def test_pattern_scan_off_has_no_opaque_spans():
    assert _opaque(C.preset("balanced").with_stage("pattern_scan", enabled=False)) == []
    assert _opaque(C.preset("balanced").with_stage("canonicaliser", enabled=False)) != []
