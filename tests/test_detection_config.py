"""DetectionConfig (V3 §3.6): presets, JSON, hash, environment, validation,
routing, actions, gate bypass, plugins, and that the default reproduces the
benchmark configuration. Model-free: the model stages are off or stubbed."""

import json
from pathlib import Path

import pytest

import config as root_config
from surrogateshield import Config, Session
from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pipeline, plugins
from surrogateshield.core.detection.plugins import Candidate
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.errors import DetectorUnavailable
from surrogateshield.core.generation.mimic import MimicGen

ROOT = Path(__file__).resolve().parent.parent

# The hashes of the shipped configs. A change to a default changes one of
# these on purpose: update it, and say so in the commit (the benchmark's is
# recorded in every ss .meta.json and in realdata_test2.json).
BALANCED_HASH = "68b6f9e8aba3a23f"
BENCHMARK_HASH = "8e463c3c6b7562fd"

NO_MODELS = dict(use_entity_trace=False, use_context_guard=False)
TEXT = ("Mail jane.roe@example.org or call +44 20 7946 0958; "
        "account ID 4471-2290-8813, see https://example.org/u/jroe")


# ── presets, JSON, hash ──────────────────────────────────────────────────────

def test_shipped_hashes_are_pinned():
    assert C.preset("balanced").config_hash()[:16] == BALANCED_HASH
    assert C.benchmark().config_hash()[:16] == BENCHMARK_HASH


def test_benchmark_hash_is_the_one_scored_on_test2():
    doc = ROOT / "bench" / "results" / "realdata_test2.json"
    if not doc.exists():
        pytest.skip("test-2 not scored yet")
    assert json.loads(doc.read_text())["config_hashes"]["ss"] == C.benchmark().config_hash()


def test_default_is_config_py_and_benchmark_only_pins_the_address_mode():
    from_root = C.from_settings(
        None, spacy_model=root_config.SPACY_MODEL, context_guard_enabled=root_config.CONTEXT_GUARD_ENABLED,
        entity_trace_high_threshold=root_config.ENTITY_TRACE_HIGH_THRESHOLD,
        entity_trace_low_threshold=root_config.ENTITY_TRACE_LOW_THRESHOLD,
        context_guard_threshold=root_config.CONTEXT_GUARD_CONFIDENCE_THRESHOLD,
        entity_trace_fallback_threshold=root_config.ENTITY_TRACE_FALLBACK_THRESHOLD,
        context_guard_model=root_config.CONTEXT_GUARD_MODEL,
        context_guard_device=root_config.CONTEXT_GUARD_DEVICE,
        address_mode=root_config.ADDRESS_MODE, address_shift_range=root_config.ADDRESS_SHIFT_RANGE,
        service=root_config.SERVICE_QUERY_DETECTION_ENABLED)
    assert from_root == C.preset("balanced") == C.DetectionConfig()
    assert C.benchmark() == C.preset("balanced").with_actions(ADDRESS="replace")
    assert C.benchmark().address_mode == "replace"
    # the library's default settings are the same config
    assert Session().detection_config == C.preset("balanced")


@pytest.mark.parametrize("name", C.PRESETS + C.ABLATIONS)
def test_presets_round_trip_through_json(name):
    cfg = C.preset(name)
    back = C.DetectionConfig.from_json(cfg.to_json())
    assert back == cfg and back.config_hash() == cfg.config_hash()
    assert json.loads(cfg.to_json())["preset"] == name


def test_presets_differ_where_documented():
    fast, strict = C.preset("fast"), C.preset("strict")
    assert not fast.enabled("entity_trace") and fast.enabled("context_guard")
    assert strict.stage("entity_trace").thresholds["high"] < C.preset("balanced").stage("entity_trace").thresholds["high"]
    assert all(strict.bypasses_gate(t) for t in C.PUBLIC_TYPES)
    assert not C.preset("balanced").bypasses_gate("PERSON")
    assert len({C.preset(n).config_hash() for n in C.PRESETS + C.ABLATIONS}) == len(C.PRESETS + C.ABLATIONS)
    with pytest.raises(ValueError, match="unknown preset"):
        C.preset("turbo")


def test_hash_ignores_key_order_but_not_values():
    d = C.preset("balanced").to_dict()
    shuffled = dict(reversed(list(d.items())))
    assert C.DetectionConfig.from_dict(shuffled).config_hash() == C.preset("balanced").config_hash()
    assert C.preset("balanced").with_stage("context_guard", thresholds={"accept": 0.71}).config_hash() \
        != C.preset("balanced").config_hash()


# ── validation ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("change, match", [
    (lambda c: c.with_actions(EMAIL="scramble"), "action"),
    (lambda c: c.with_actions(EMAIL="shift"), "shift"),
    (lambda c: c.with_stage("context_guard", thresholds={"accept": 1.5}), "threshold"),
    (lambda c: c.with_stage("entity_trace", thresholds={"PASSPORT": 0.5}), "threshold"),
    (lambda c: c.replace(gate_bypass=("SSN",)), "gate_bypass"),
    (lambda c: c.replace(type_conflicts={"PERSON|ORG": "score"}), "type_conflicts"),
    (lambda c: c.replace(address_shift_range=0), "address_shift_range"),
    (lambda c: c.with_sources(EMAIL=["nowhere"]), "type_sources"),
    (lambda c: c.replace(detectors=c.detectors + c.detectors[:1]), "duplicate|twice|unique"),
    (lambda c: c.with_stage("context_guard", options={"labels": ["FIRSTNAME"]}), "labels"),
    (lambda c: c.with_stage("context_guard", options={"labels": {}}), "labels"),
    (lambda c: c.with_stage("context_guard", options={"labels": {"FIRSTNAME": "NAME"}}), "unknown type"),
    (lambda c: c.with_stage("pii_tagger", options={"gate_above": 1.5}), "gate_above"),
    (lambda c: c.with_stage("pii_tagger", options={"gate_above": True}), "gate_above"),
])
def test_invalid_configs_are_refused(change, match):
    with pytest.raises((ValueError, TypeError), match=match):
        change(C.preset("balanced"))


def test_unknown_fields_are_refused():
    d = C.preset("balanced").to_dict()
    d["detectors"][0]["colour"] = "red"
    with pytest.raises((ValueError, TypeError)):
        C.DetectionConfig.from_dict(d)
    with pytest.raises(TypeError, match="unknown detection setting"):
        C.from_settings(None, spacy_mdoel="x")


# ── environment ──────────────────────────────────────────────────────────────

def test_environment_selects_a_preset_and_merges_a_file(tmp_path):
    assert not C.env_selected({})
    assert C.from_env(environ={C.ENV_PRESET: "strict"}) == C.preset("strict")
    f = tmp_path / "det.json"
    f.write_text(json.dumps({"preset": "fast", "type_actions": {"EMAIL": "redact"},
                             "detectors": [{"name": "context_guard", "thresholds": {"accept": 0.8}}]}))
    cfg = C.from_env(environ={C.ENV_FILE: str(f)})
    assert cfg.preset == "fast" and not cfg.enabled("entity_trace")
    assert cfg.redacts("email") and cfg.stage("context_guard").thresholds["accept"] == 0.8
    assert cfg.stage("context_guard").model == C.CONTEXT_GUARD_MODEL       # the rest kept


def test_cascade_follows_the_environment(monkeypatch):
    monkeypatch.setenv(C.ENV_PRESET, "no_models")
    import detection.logic as shim
    conf, _ = shim.run_cascade(TEXT)
    assert {e.text for e in conf} == {e.text for e in pipeline.run_cascade(TEXT, **NO_MODELS)[0]}


# ── run_cascade with a config ────────────────────────────────────────────────

def _spans(ents):
    return sorted((e.start, e.end, e.type, e.source) for e in ents)


@pytest.mark.parametrize("text", [TEXT, "Ship to 22 Baker Street, London NW1 6XE for Ms. Ada Byrne",
                                  "my ssn is 078-05-1120 and dob 03/04/1987"])
def test_flat_switches_and_the_config_give_the_same_result(text):
    flat, _ = pipeline.run_cascade(text, **NO_MODELS)
    cfg, _ = pipeline.run_cascade(text, config=C.preset("no_models"))
    assert _spans(flat) == _spans(cfg)


def test_type_routing():
    base = C.preset("no_models")
    found = {e.type for e in pipeline.run_cascade(TEXT, config=base)[0]}
    assert "email" in found
    no_email = base.with_sources(EMAIL=["canonicaliser"])          # only on a view
    assert "email" not in {e.type for e in pipeline.run_cascade(TEXT, config=no_email)[0]}


def test_keep_drops_the_type_from_the_result():
    cfg = C.preset("no_models").with_actions(EMAIL="keep")
    conf, _ = pipeline.run_cascade(TEXT, config=cfg)
    assert conf and "email" not in {e.type for e in conf}


def test_gate_bypass_reaches_the_relation_gate(monkeypatch):
    seen = []
    real = pipeline.relation_gate.gate

    def spy(text, gated, others, **kw):
        seen.append({e.type for e in gated})
        return real(text, gated, others, **kw)
    monkeypatch.setattr(pipeline.relation_gate, "gate", spy)
    text = "Dear Ms. Ada Byrne, thanks"
    pipeline.run_cascade(text, config=C.preset("no_models"))
    n = len(seen)
    pipeline.run_cascade(text, config=C.preset("no_models").replace(gate_bypass=("PERSON",)))
    assert "PERSON" in seen[0] and not any("PERSON" in s for s in seen[n:])


def test_gate_above_vouches_for_a_stages_sure_places(monkeypatch):
    from surrogateshield.core.detection import plugins

    class Places:
        def detect(self, text, view):
            return [plugins.Candidate(text.index("Lyon"), text.index("Lyon") + 4, "LOCATION", 0.95),
                    plugins.Candidate(text.index("Porto"), text.index("Porto") + 5, "LOCATION", 0.6)]
    seen = []
    real = pipeline.relation_gate.gate

    def spy(text, gated, others, **kw):
        seen.extend(e.text for e in gated)
        return real(text, gated, others, **kw)
    monkeypatch.setattr(pipeline.relation_gate, "gate", spy)
    plugins.register_detector("places", lambda stage: Places())
    try:
        text = "Lyon and Porto are lovely in spring"
        base = C.preset("no_models").with_stage("places", enabled=True)
        conf, _ = pipeline.run_cascade(text, config=base)
        assert {"Lyon", "Porto"} <= set(seen) and not {"Lyon", "Porto"} & {e.text for e in conf}
        seen.clear()
        conf, _ = pipeline.run_cascade(text, config=base.with_stage("places", options={"gate_above": 0.9}))
        assert {"Lyon", "Porto"} <= set(seen)                   # the gate still sees both
        assert "Lyon" in {e.text for e in conf} and "Porto" not in {e.text for e in conf}
    finally:
        plugins.unregister_detector("places")


def test_stage_of():
    e = lambda typ, src, view=None: DetectedEntity("x", 0, 1, typ, 1.0, src, view=view)
    assert pipeline.stage_of(e("email", "pattern")) == "pattern_scan"
    assert pipeline.stage_of(e("email", "pattern", "spelled")) == "canonicaliser"
    assert pipeline.stage_of(e("PERSON", "pattern")) == "structural"
    assert pipeline.stage_of(e("ORG", "structural")) == "structural"
    assert pipeline.stage_of(e("GPE", "ner")) == "entity_trace"
    assert pipeline.stage_of(e("PERSON", "slm")) == "context_guard"
    assert pipeline.stage_of(e("ID", "employee_ids")) == "employee_ids"


def test_budget_overrun_is_recorded_not_dropped(monkeypatch):
    clock = iter(range(0, 10_000, 5))
    cfg = C.preset("no_models").with_plugin("slow", types=["ID"], max_latency_ms=1)

    class Slow:
        def detect(self, text, view):
            return [Candidate(text.index("4471"), text.index("4471") + 14, "ID", 0.9)]
    plugins.register_detector("slow", lambda st: Slow(), replace=True)
    try:
        timings = {}
        monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(clock) / 1000 * 1000)
        conf, _ = pipeline.run_cascade(TEXT, config=cfg, timings=timings)
        assert "slow_over_budget_ms" in timings
        assert any(e.source == "slow" for e in conf)
    finally:
        plugins.unregister_detector("slow")


# ── plugins ──────────────────────────────────────────────────────────────────

def test_example_plugin_through_the_cascade():
    import sys
    sys.path.insert(0, str(ROOT / "examples"))
    from employee_id_detector import factory
    plugins.register_detector("employee_ids", factory, replace=True)
    try:
        cfg = C.preset("no_models").with_plugin("employee_ids", types=["ID"], options={"prefix": "EMP"})
        text = "Ask EMP-204816 or ＥＭＰ-２０４８１７ (https://hr.example.org/EMP-204818)"
        conf, _ = pipeline.run_cascade(text, config=cfg)
        mine = sorted(e.text for e in conf if e.source == "employee_ids")
        assert mine == ["EMP-204816", "ＥＭＰ-２０４８１７"]      # not the one inside the URL
        assert all(e.type == "id_number" for e in conf if e.source == "employee_ids")
        # routed like a model: a type the stage may not report, or below the
        # stage's threshold for it (the plugin scores 0.99), is dropped
        for off in (cfg.with_sources(ID=["pattern_scan"]), cfg.with_stage("employee_ids", thresholds={"ID": 0.995})):
            assert not [e for e in pipeline.run_cascade(text, config=off)[0] if e.source == "employee_ids"]
    finally:
        plugins.unregister_detector("employee_ids")


def test_unregistered_plugin_fails_closed():
    cfg = C.preset("no_models").with_plugin("nobody_home")
    with pytest.raises(DetectorUnavailable, match="nobody_home"):
        pipeline.run_cascade(TEXT, config=cfg)


def test_registry_rules():
    with pytest.raises(ValueError, match="built-in"):
        plugins.register_detector("entity_trace", lambda st: None)
    f = lambda st: None
    plugins.register_detector("x_plugin", f)
    try:
        plugins.register_detector("x_plugin", f)                     # same factory: fine
        with pytest.raises(ValueError, match="already registered"):
            plugins.register_detector("x_plugin", lambda st: None)
    finally:
        plugins.unregister_detector("x_plugin")


def test_bad_candidates_are_dropped():
    ents = plugins.candidates_to_entities("p", "hello world", [
        Candidate(0, 5, "PERSON"), Candidate(3, 99, "ID"), Candidate(5, 6, "ID"), Candidate(6, 11, "phone_us")])
    assert [(e.text, e.type, e.source) for e in ents] == [("hello", "PERSON", "p"), ("world", "phone_us", "p")]


# ── actions: redact, and the send paths ─────────────────────────────────────

def test_redact_gives_numbered_placeholders():
    text = "write to a@example.org, b@example.org; [EMAIL_1] is a literal"
    ents = [DetectedEntity("a@example.org", 9, 22, "email", 1.0, "pattern"),
            DetectedEntity("b@example.org", 24, 37, "email", 1.0, "pattern")]
    m = MimicGen(seed=1).generate_all(ents, text=text, redact=lambda t: t == "email")
    assert m == {"a@example.org": "[EMAIL_2]", "b@example.org": "[EMAIL_3]"}


def test_prepare_send_with_a_config():
    from json_tester import prepare_send
    cfg = C.preset("no_models").with_actions(EMAIL="redact", PHONE="keep")
    p = prepare_send(TEXT, MimicGen(seed=3), config=cfg)
    assert "[EMAIL_1]" in p.sanitized and "jane.roe@example.org" not in p.sanitized
    assert "+44 20 7946 0958" in p.sanitized


def test_session_detection_setting():
    s = Session(config=Config(detection="no_models"))
    assert s.detection_config == C.preset("no_models")
    # a flat setting changed from its default applies on top of the preset
    s = Session(config=Config(detection="strict", address_mode="shift"))
    assert s.detection_config.address_mode == "shift"
    assert s.detection_config.stage("entity_trace").thresholds["high"] == 0.70
    s = Session(config=Config(detection=C.preset("no_models").with_actions(EMAIL="keep", PHONE="redact")))
    r = s.mask_result(TEXT)
    emails = [d for d in r.detections if d.type == "email"]
    assert emails and not any(d.masked for d in emails)               # reported, sent as typed
    assert "jane.roe@example.org" in r.text and "[PHONE_1]" in r.text
    assert s.unmask(r.text) == TEXT


def test_library_config_accepts_presets_and_dicts():
    import surrogateshield as ss
    saved = ss._state.cfg.detection
    try:
        assert ss.config(detection="fast").detection == C.preset("fast")
        assert ss.config(detection=C.preset("strict").to_dict()).detection == C.preset("strict")
        with pytest.raises(ValueError):
            ss.config(detection=3)
    finally:
        ss.config(detection=saved)


def test_partial_dicts_merge_onto_their_preset(tmp_path, monkeypatch):
    part = {"type_actions": {"EMAIL": "keep"}}
    merged = C.from_partial(part)
    assert merged.action_for("ADDRESS") == "auto" and merged.keeps("EMAIL")    # ADDRESS kept
    assert C.from_partial({"preset": "strict", "gate": False}) == C.preset("strict").replace(gate=False)
    for name in C.PRESETS + C.ABLATIONS:                                       # whole: unchanged
        assert C.from_partial(C.preset(name).to_dict()) == C.preset(name)
    with pytest.raises(ValueError):
        C.from_partial(["fast"])
    import surrogateshield as ss
    saved = ss._state.cfg.detection
    try:
        assert ss.config(detection=part).detection == merged
    finally:
        ss.config(detection=saved)
    assert Session(config=Config(detection=part)).detection_config == merged
    f = tmp_path / "det.json"
    f.write_text(json.dumps(part))
    monkeypatch.delenv(C.ENV_PRESET, raising=False)
    monkeypatch.setenv(C.ENV_FILE, str(f))
    assert C.from_env() == merged
