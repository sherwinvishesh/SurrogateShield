"""The resolver (V3 §3.5): one candidate per value, by source priority, then
the type-conflict table, then score. Model-free."""

import random

from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pipeline, resolver
from surrogateshield.core.entities import DetectedEntity

BAL = C.preset("balanced")
# the rule deduplicate used before the resolver
SCORE_ONLY = BAL.replace(source_priority=(), type_conflicts={})


def ent(text, typ, src, score, start=0, view=None):
    return DetectedEntity(text, start, start + len(text), typ, score, src, view=view)


def old_rule(group):
    best = group[0]
    for e in group[1:]:
        if e.score > best.score:
            best = e
    return best


def test_empty_priority_and_score_table_is_the_old_rule():
    rng = random.Random(7)
    kinds = [("PERSON", "ner"), ("PERSON", "slm"), ("ORG", "ner"), ("GPE", "ner"), ("LOC", "slm"),
             ("email", "pattern"), ("PERSON", "pattern"), ("hostname", "structural"), ("ID", "plug")]
    for _ in range(500):
        group = [ent("Ana Bell", *rng.choice(kinds), rng.choice([0.6, 0.7, 0.85, 0.9, 1.0]), start=i)
                 for i in range(rng.randint(1, 5))]
        assert resolver.pick(group, SCORE_ONLY) is old_rule(group)


def test_source_priority_beats_score():
    model = ent("4471-2290-8813", "PERSON", "slm", 0.99)
    pattern = ent("4471-2290-8813", "id_number", "pattern", 0.5, start=20)
    assert resolver.pick([model, pattern], BAL) is pattern
    assert resolver.pick([model, pattern], SCORE_ONLY) is model
    structural = ent("mail.acme.io", "hostname", "structural", 0.6)
    spacy = ent("mail.acme.io", "PERSON", "ner", 0.9, start=30)
    assert resolver.pick([spacy, structural], BAL) is structural


def test_priority_is_configurable():
    spacy = ent("Reno", "GPE", "ner", 0.7)
    distil = ent("Reno", "LOC", "slm", 0.9, start=10)
    assert resolver.pick([spacy, distil], BAL) is distil
    first = BAL.replace(source_priority=("entity_trace",) + tuple(
        s for s in BAL.source_priority if s != "entity_trace"))
    assert resolver.pick([spacy, distil], first) is spacy


def test_type_conflict_table_decides_within_a_rank():
    gpe = ent("Phoenix", "GPE", "ner", 0.9)
    org = ent("Phoenix", "ORG", "ner", 0.7, start=40)
    assert resolver.pick([gpe, org], BAL) is gpe                       # "score"
    org_wins = BAL.replace(type_conflicts={"LOCATION|ORG": "ORG"})
    assert resolver.pick([gpe, org], org_wins) is org
    # the table never overrides the rank
    pattern = ent("Phoenix", "handle", "pattern", 0.5, start=60)
    assert resolver.pick([gpe, org, pattern], org_wins) is pattern


def test_three_types_and_a_cycle():
    a, b, c = ent("X Y", "PERSON", "ner", 0.6), ent("X Y", "ORG", "ner", 0.9), ent("X Y", "GPE", "ner", 0.8)
    table = {"ORG|PERSON": "PERSON", "LOCATION|ORG": "ORG", "LOCATION|PERSON": "score"}
    assert resolver.pick([a, b, c], BAL.replace(type_conflicts=table)) is a   # ORG and LOCATION out
    cycle = {"ORG|PERSON": "PERSON", "LOCATION|ORG": "ORG", "LOCATION|PERSON": "LOCATION"}
    assert resolver.pick([a, b, c], BAL.replace(type_conflicts=cycle)) is b   # all out: score


def test_unlisted_stages_share_the_last_rank():
    cfg = BAL.with_plugin("ids_a", types=["ID"]).with_plugin("ids_b", types=["ID"])
    assert cfg.rank("ids_a") == cfg.rank("ids_b") == len(cfg.source_priority)
    a, b = ent("EMP-123456", "ID", "ids_a", 0.8), ent("EMP-123456", "ID", "ids_b", 0.95, start=20)
    assert resolver.pick([a, b], cfg) is b
    pat = ent("EMP-123456", "id_number", "pattern", 0.1, start=40)
    assert resolver.pick([a, b, pat], cfg) is pat


def test_deduplicate_keeps_order_tags_and_values():
    ents = pipeline._TaggedList([
        ent("Ana Bell", "PERSON", "ner", 0.9, start=30),
        ent("ana@x.org", "email", "pattern", 1.0, start=5),
        ent("Ana Bell ", "PERSON", "slm", 0.95, start=60),
    ])
    ents._qi_matches, ents._skipped_entities, ents._skip_reasons = ["q"], ["s"], {"k": "v"}
    out = pipeline.deduplicate(ents, BAL)
    assert [e.text for e in out] == ["ana@x.org", "Ana Bell "]               # by start
    assert out[1].source == "slm"                                          # context_guard outranks spaCy
    assert (out._qi_matches, out._skipped_entities, out._skip_reasons) == (["q"], ["s"], {"k": "v"})
    assert pipeline.deduplicate(ents) == pipeline.deduplicate(ents, BAL)   # default: balanced


def test_cascade_output_goes_through_the_resolver(monkeypatch):
    seen = []
    real = resolver.pick
    monkeypatch.setattr(resolver, "pick", lambda g, c: seen.append(c) or real(g, c))
    cfg = C.preset("no_models").replace(source_priority=("structural", "pattern_scan"))
    found, _ = pipeline.run_cascade("Mail jane.roe@example.org", config=cfg)
    pipeline.deduplicate(found, cfg)
    assert seen and all(c is cfg for c in seen)
