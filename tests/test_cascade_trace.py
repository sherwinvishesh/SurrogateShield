"""The cascade's trace (PROMPT_FOR_OPUS_V3 §3.1), read by
bench/realdata/attribute.py: ``run_cascade(trace=[...])`` records the live
candidates after every pass and the relation gate's rule for each drop, and
changes nothing in the result. Model-free; texts and values are invented."""

import pytest

from surrogateshield.core.detection import pipeline
from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity

STAGES = ["pattern_scan", "canonicaliser", "entity_trace", "service_query_geo", "context_guard", "reanchor",
          "structural_org", "structural_person", "implausible_org", "merge_persons", "org_assembly", "card_brand_org",
          "email_username", "person_components", "inside_url", "topical_geo", "sentence_frame",
          "relation_gate", "structural", "gender_follows_name", "partial_overlaps", "pii_off"]


def _one(text, value, typ):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, "ner")


@pytest.mark.parametrize("text, value, typ, rule", [
    ("Can you remind me about the SLA for Q4?", "SLA", "ORG", "junk"),
    ("What did Taylor Swift say in her 2024 statement?", "Taylor Swift", "PERSON", "public_person"),
    ("I tripped over the matt again", "matt", "PERSON", "word_name"),
    ("Is Tesla stock a buy?", "Tesla", "ORG", "public_org"),
    ("Which terminal is best for this?", "terminal", "GPE", "common_noun"),
    ("How far is Tempe from Phoenix?", "Tempe", "GPE", "untied"),
])
def test_gate_names_the_rule_of_each_drop(text, value, typ, rule):
    reasons = {}
    kept, dropped = rg.gate(text, [_one(text, value, typ)], (), reasons=reasons)
    assert not kept and [reasons[id(e)] for e in dropped] == [rule]
    plain = rg.gate(text, [_one(text, value, typ)], ())
    assert [len(x) for x in plain] == [len(kept), len(dropped)]


def test_gate_records_nothing_for_a_kept_entity():
    reasons = {}
    kept, dropped = rg.gate("Hannah approved it", [_one("Hannah approved it", "Hannah", "PERSON")], (), reasons=reasons)
    assert len(kept) == 1 and not dropped and reasons == {}


TEXTS = [
    "Hi, I'm Ada Byrne. Email ada.byrne@example.org or call +1 415 555 0134. See https://example.org/u/ada99",
    "Is Tesla stock a buy? My card 4111 1111 1111 1111 expires 04/29.",
    "nothing personal here, just a question about loops",
]


@pytest.mark.parametrize("text", TEXTS)
def test_trace_lists_every_pass_and_changes_nothing(text):
    kw = dict(use_entity_trace=False, use_context_guard=False)
    plain, plain_nc = pipeline.run_cascade(text, **kw)
    trace = []
    traced, traced_nc = pipeline.run_cascade(text, trace=trace, **kw)
    key = lambda ents: [(e.start, e.end, e.type, e.source, e.text) for e in ents]
    assert key(traced) == key(plain) and key(traced_nc) == key(plain_nc)
    assert traced._skip_reasons == plain._skip_reasons
    assert [t["stage"] for t in trace] == STAGES
    final = trace[-1]["entities"]
    assert sorted(r[:4] for r in final if r[4] == "confirmed") == sorted(
        [e.start, e.end, e.type, e.source] for e in traced)
    assert all(len(r) == 5 and r[4] in ("confirmed", "borderline", "needs_confirmation")
               for t in trace for r in t["entities"])
    assert all(len(d) == 3 for d in trace[STAGES.index("relation_gate")]["dropped"])
    if "https://" in text:
        s = text.index("https://")
        assert [s, len(text)] in trace[0]["opaque"]


def test_trace_off_by_default_keeps_the_signature():
    trace = None
    conf, _ = pipeline.run_cascade(TEXTS[0], use_entity_trace=False, use_context_guard=False, trace=trace)
    assert any(e.type.lower() == "email" for e in conf)


def test_a_vouched_place_is_tied_but_a_public_org_is_still_dropped():
    text = "Lyon is lovely and Google is big"
    lyon, goog = _one(text, "Lyon", "GPE"), _one(text, "Google", "ORG")
    reasons = {}
    kept, _ = rg.gate(text, [lyon, goog], reasons=reasons)
    assert kept == [] and [reasons[id(lyon)], reasons[id(goog)]] == ["untied", "public_org"]
    kept, dropped = rg.gate(text, [lyon, goog], vouched=[lyon, goog])
    assert kept == [lyon] and dropped == [goog]
