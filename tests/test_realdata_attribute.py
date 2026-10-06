"""Model-free tests for bench/realdata/attribute.py (PROMPT_FOR_OPUS_V3 §3.1):
each cause on hand-built prepared outputs and traces, the parts left by a
partial edit, agreement with the scorer's leak rule, and the trace contract
of ``run_cascade`` (candidates after every pass, gate rules). All texts and
values are invented."""

from types import SimpleNamespace

import pytest

from bench import realworld as rw
from bench.realdata import attribute as A


def _ent(text, value, typ, source="ner"):
    s = text.index(value)
    return SimpleNamespace(text=value, start=s, end=s + len(value), type=typ, source=source)


def _prep(text, ents, edits, mapping=None):
    return SimpleNamespace(confirmed=ents, edits=edits, surrogate_map={e.text: "X" for e in ents} if mapping is None else mapping)


def _edit(text, value, rep="Zzz"):
    s = text.index(value)
    return (s, s + len(value), value, rep)


def _stage(name, *ents, bucket="confirmed"):
    return {"stage": name, "entities": [[e.start, e.end, e.type, e.source, bucket] for e in ents]}


TEXT = "Hi, I'm Ada Byrne, 12 Harbour Rd, Cork T12 X4Y2. Mail ada.byrne@example.org."
ADA = {"value": "Ada Byrne", "type": "PERSON", "fmt": "plain"}
ADDR = {"value": "12 Harbour Rd, Cork T12 X4Y2", "type": "ADDRESS", "fmt": "plain"}


def test_protected_and_policy():
    e = _ent(TEXT, "Ada Byrne", "PERSON")
    p = _prep(TEXT, [e], [_edit(TEXT, "Ada Byrne")])
    assert A.attribute_value(TEXT, ADA, False, p, []) == ("protected", "")
    p = _prep(TEXT, [], [])
    assert A.attribute_value(TEXT, ADDR, True, p, []) == ("policy", "")
    assert A.attribute_value(TEXT, ADDR, False, None, []) == ("refused", "")


def test_surrogate_that_contains_the_value():
    e = _ent(TEXT, "Ada Byrne", "PERSON")
    p = _prep(TEXT, [e], [_edit(TEXT, "Ada Byrne", "Dr Ada Byrne")])
    assert A.attribute_value(TEXT, ADA, False, p, []) == ("surrogate", "")


def test_partial_address_names_the_parts_left():
    street = _ent(TEXT, "12 Harbour Rd", "ADDRESS", "pattern")
    p = _prep(TEXT, [street], [_edit(TEXT, "12 Harbour Rd")])
    assert A.attribute_value(TEXT, ADDR, False, p, []) == ("partial", "locality+postcode")
    num = _ent(TEXT, "12", "ADDRESS", "pattern")
    p = _prep(TEXT, [num], [_edit(TEXT, "12", "13")])
    assert A.attribute_value(TEXT, ADDR, False, p, []) == ("partial", "locality+postcode+street")


def test_partial_person_and_email():
    first = _ent(TEXT, "Ada", "PERSON")
    p = _prep(TEXT, [first], [_edit(TEXT, "Ada")])
    assert A.attribute_value(TEXT, ADA, False, p, []) == ("partial", "last")
    t = "Write to Byrne, Ada today."
    sf = {"value": "Byrne, Ada", "type": "PERSON", "fmt": "surname-first"}
    p = _prep(t, [_ent(t, "Byrne", "PERSON")], [_edit(t, "Byrne")])
    assert A.attribute_value(t, sf, False, p, []) == ("partial", "first")
    mail = {"value": "ada.byrne@example.org", "type": "EMAIL", "fmt": "plain"}
    p = _prep(TEXT, [_ent(TEXT, "example.org", "URL", "pattern")], [_edit(TEXT, "example.org")])
    assert A.attribute_value(TEXT, mail, False, p, []) == ("partial", "local")


def test_parts_left_generic_positions():
    v = "AB-1234-XY"
    left = [c.isalnum() and i >= 7 for i, c in enumerate(v)]
    assert A.parts_left("ID", "", v, left) == "suffix"
    left = [c.isalnum() and i < 2 for i, c in enumerate(v)]
    assert A.parts_left("ID", "", v, left) == "prefix"
    left = [c.isalnum() and 3 <= i < 7 for i, c in enumerate(v)]
    assert A.parts_left("ID", "", v, left) == "inner"
    assert A.parts_left("ID", "", v, [c.isalnum() for c in v]) == "all"


def test_unplanned_when_a_final_entity_has_no_edit():
    e = _ent(TEXT, "Ada Byrne", "PERSON")
    p = _prep(TEXT, [e], [], mapping={})
    assert A.attribute_value(TEXT, ADA, False, p, []) == ("unplanned", "no_surrogate")


def test_gated_names_the_pass_the_rule_and_the_source():
    e = _ent(TEXT, "Ada Byrne", "PERSON", "slm")
    trace = [_stage("pattern_scan"), _stage("context_guard", e), _stage("relation_gate"),
             {"stage": "relation_gate_rules", "dropped": [[e.start, e.end, "public_person"]]}]
    p = _prep(TEXT, [], [])
    assert A.attribute_value(TEXT, ADA, False, p, trace) == ("gated", "relation_gate:public_person <- slm")
    trace = [_stage("pattern_scan"), _stage("entity_trace", e, bucket="borderline"), _stage("context_guard")]
    assert A.attribute_value(TEXT, ADA, False, p, trace) == ("gated", "context_guard <- slm")


def test_gated_by_prepare_send_steps():
    e = _ent(TEXT, "Ada Byrne", "PERSON")
    trace = [_stage("pattern_scan"), _stage("pii_off", e, bucket="needs_confirmation")]
    assert A.attribute_value(TEXT, ADA, False, _prep(TEXT, [], []), trace) == ("gated", "needs_confirmation <- ner")


def test_missed_and_in_url():
    p = _prep(TEXT, [], [])
    assert A.attribute_value(TEXT, ADA, False, p, [_stage("pattern_scan")]) == ("missed", "")
    t = "see https://example.org/u/AdaByrne99 now"
    h = {"value": "AdaByrne99", "type": "HANDLE", "fmt": "plain"}
    s = t.index("https")
    trace = [{"stage": "opaque", "opaque": [[s, s + len("https://example.org/u/AdaByrne99")]]}, _stage("pattern_scan")]
    assert A.attribute_value(t, h, False, _prep(t, [], []), trace) == ("missed", "in_url")


@pytest.mark.parametrize("edits", [[], [(8, 11, "Ada", "Zed")], [(8, 17, "Ada Byrne", "Ada Byrne Jr")],
                                   [(8, 17, "Ada Byrne", "Zed Q")]])
def test_first_failure_agrees_with_the_scorer(edits):
    gold = {"id": "m", "text": TEXT, "service_query": False, "protect": [ADA], "sensitive": [], "optional": [], "keep": []}
    leaked = bool(rw.score_message(gold, SimpleNamespace(edits=edits))["leaked"])
    assert (A.first_failure(TEXT, ADA["value"], edits)[0] is not None) == leaked


def test_spurious_sources_count_only_edits_outside_gold():
    t = "Ask Ada Byrne about Python and Leeds."
    gold = {"protect": [{"value": "Ada Byrne", "type": "PERSON"}], "sensitive": [], "optional": []}
    ents = [_ent(t, "Ada Byrne", "PERSON"), _ent(t, "Python", "ORG", "slm"), _ent(t, "Leeds", "LOCATION")]
    p = _prep(t, ents, [_edit(t, "Ada Byrne"), _edit(t, "Python"), _edit(t, "Leeds")])
    assert A.spurious_sources(t, gold, p) == {("slm", "ORG"): 1, ("ner", "LOCATION"): 1}


def test_summary_has_counts_and_no_text():
    rows = {("oasst1", "PERSON", "intro", "plain", "injected_single", "missed", ""): 2,
            ("oasst1", "PERSON", "intro", "plain", "injected_single", "protected", ""): 6,
            ("oasst1", "ADDRESS", "form", "plain", "shift", "policy", ""): 1}
    doc = A.summarise("dev", "dev", "test1", ["oasst1"], rows, {("oasst1", "ner", "PERSON"): 3}, {"equal": 9}, {"messages": 4})
    assert doc["injected"]["by_type"]["PERSON"]["leak_rate"] == 0.25
    assert doc["injected"]["by_type"]["ADDRESS"]["leak_rate"] is None
    assert doc["natural"]["spurious_by_source_type"] == {"ner:PERSON": 3}
    assert doc["injected"]["by_format"]["PERSON:plain"]["values"] == 8
    md = A.markdown({**doc, "git": None})
    assert "| PERSON | 8 | 0.25 |" in md
