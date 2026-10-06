"""Model-free tests for bench/realdata/label.py (Phase 2 silver labels): the
prompt quotes GUIDE.md verbatim, validation is the J2 lint, requests keep
conversations whole, committed rows carry offsets only. Toy text throughout."""

import json

import pytest

from bench import realworld as rw
from bench.realdata import label as L

TEXT = "hi I'm Ana Ruiz, mail ana.ruiz@example.org about order 4471 for my clinic in Tempe."
GOOD = {"task": "business", "service_query": False,
        "protect": [{"value": "Ana Ruiz", "type": "PERSON"}, {"value": "ana.ruiz@example.org", "type": "EMAIL"},
                    {"value": "Tempe", "type": "LOCATION"}],
        "sensitive": [], "optional": [], "keep": ["4471"]}


def test_system_prompt_quotes_the_guide_verbatim():
    guide = L.GUIDE.read_text(encoding="utf-8")
    sec = L.guide_sections()
    assert sec.startswith("## Record format (rule)\n\nEvery `value` must be")
    body = sec.split("\n\n", 1)[1]                       # after our sub-heading
    rule, rest = body.split("## The four lists", 1)
    lists, sq = rest.split("## service_query", 1)
    for part in (rule, "## The four lists" + lists.rstrip(), "## service_query" + sq.rstrip()):
        assert part in guide
    assert sec in L.system_prompt() and "## Mix" not in L.system_prompt()


def test_tool_types_are_the_scorer_types():
    props = L.TOOL["input_schema"]["properties"]["messages"]["items"]["properties"]
    assert set(props["protect"]["items"]["properties"]["type"]["enum"]) == rw.PROTECT_TYPES
    assert set(props["sensitive"]["items"]["properties"]["type"]["enum"]) == rw.SENSITIVE_TYPES
    assert props["task"]["enum"] == list(L.TASKS)


def test_validate_accepts_a_good_annotation():
    assert L.validate(GOOD, TEXT) == []


@pytest.mark.parametrize("change", [
    {"protect": [{"value": "Ana ruiz", "type": "PERSON"}]},           # case differs
    {"protect": [{"value": "Temp", "type": "LOCATION"}]},             # glued to a letter
    {"protect": [{"value": "Ana Ruiz", "type": "NAME"}]},             # not a protect type
    {"sensitive": [{"value": "clinic", "type": "PERSON"}]},           # not a sensitive type
    {"keep": ["Ana"]},                                                # keep overlaps protect
    {"task": "chat"},
    {"keep": [{"value": "4471"}]},
])
def test_validate_rejects_rule_breaks(change):
    assert L.validate({**GOOD, **change}, TEXT)


def test_validate_requires_every_candidate_decided():
    assert L.validate(GOOD, TEXT, ["Ana Ruiz", "4471"]) == []
    assert L.validate(GOOD, TEXT, ["Ruiz, mail"]) == []                 # broken boundary, overlaps a value
    probs = L.validate(GOOD, TEXT, ["Ana Ruiz", "clinic"])
    assert probs == ["candidate[1] in no list"]


def test_validate_reports_missing_fields_without_text():
    probs = L.validate({"task": "qa"}, TEXT)
    assert "protect missing" in probs and not any("Ana" in p for p in probs)


def _msg(items):
    return {"content": [{"type": "tool_use", "name": L.TOOL_NAME, "input": {"messages": items}}]}


def test_parse_maps_local_ids_and_flags_missing_and_duplicate():
    local = {"m1": "d/a#t0", "m2": "d/b#t0", "m3": "d/c#t0"}
    texts = {"d/a#t0": TEXT, "d/b#t0": "nothing personal here", "d/c#t0": TEXT}
    empty = {"task": "qa", "service_query": False, "protect": [], "sensitive": [], "optional": [], "keep": []}
    ok, bad = L.parse(_msg([{"id": "m1", **GOOD}, {"id": "m2", **empty}, {"id": "m2", **empty}]), local, texts)
    assert set(ok) == {"d/a#t0"} and ok["d/a#t0"]["protect"] == GOOD["protect"]
    assert bad == {"d/b#t0": ["annotated more than once"], "d/c#t0": ["no annotation for this id"]}
    ok, bad = L.parse(_msg([{"id": "m1", **GOOD}]), {"m1": "d/a#t0"}, texts, {"d/a#t0": ["clinic"]})
    assert not ok and bad == {"d/a#t0": ["candidate[0] in no list"]}
    ok, bad = L.parse({"content": [{"type": "text", "text": "sorry"}]}, local, texts)
    assert not ok and len(bad) == 3


def _m(conv, turn, n=10, kind=None):
    return {"id": f"d/{conv}#t{turn}", "text": "x" * n, "conv": conv, "turn": turn,
            "kind": kind or ("multi" if conv.startswith("c") else "single")}


def test_units_and_groups_keep_conversations_whole():
    msgs = [_m("s1", 0), _m("c1", 0), _m("c1", 1), _m("c1", 2), _m("s2", 0), _m("c2", 0), _m("c2", 1)]
    us = L.units(msgs)
    assert [len(u) for u in us] == [1, 3, 1, 2]
    gs = L.groups(us, per_call=4, budget=10 ** 6)
    assert [[m["id"] for m in g] for g in gs] == [["d/s1#t0", "d/c1#t0", "d/c1#t1", "d/c1#t2"],
                                                   ["d/s2#t0", "d/c2#t0", "d/c2#t1"]]
    gs = L.groups(us, per_call=10, budget=25)
    assert all(sum(len(m["text"]) for m in g) <= 30 for g in gs) and sum(len(g) for g in gs) == 7


def test_user_content_uses_local_ids_conversations_and_feedback():
    g = [_m("s1", 0), _m("c1", 0), _m("c1", 1)]
    text, local = L.user_content(g, {"d/s1#t0": ["Ana"]}, feedback={"d/c1#t1": ["protect[0] bad type"]})
    assert local == {"m1": "d/s1#t0", "m2": "d/c1#t0", "m3": "d/c1#t1"}
    assert '<message id="m2" conversation="c1" turn="1">' in text and 'turn="2"' in text
    assert '<candidates for="m1">["Ana"]</candidates>' in text and "- m3: protect[0] bad type" in text
    assert "d/c1" not in text


def test_params_force_the_tool_and_cache_the_system_prompt():
    p, _ = L.params([_m("s1", 0)], {}, "claude-sonnet-4-6")
    assert p["tool_choice"] == {"type": "tool", "name": L.TOOL_NAME} and p["temperature"] == 0
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_public_row_holds_offsets_only_and_round_trips():
    row = L.public_row("d/a#t0", TEXT, GOOD, "ok", 1)
    assert "Ana" not in json.dumps(row) and "Tempe" not in json.dumps(row)
    assert row["protect"][0] == {"type": "PERSON", "occ": [[7, 15]]}
    assert L.from_public(row, TEXT) == GOOD
    failed = L.public_row("d/b#t0", TEXT, None, "failed", 2, ["protect[0] bad type"])
    assert failed["label_status"] == "failed" and "protect" not in failed


def test_candidates_pool_every_arm_without_attribution(tmp_path):
    msgs = [{"id": "d/a#t0", "text": TEXT}, {"id": "d/b#t0", "text": "plain"}]
    t = TEXT.index("Tempe")
    for arm, edits in [("x", [[7, 15, "PERSON", 8, 0]]), ("y", [[7, 15, "NAME", 8, 0], [t, t + 5, "GPE", 5, 0]])]:
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "n.jsonl").write_text(json.dumps({"id": "d/a#t0", "edits": edits, "ms": 1}) + "\n"
                                                + json.dumps({"id": "d/b#t0", "edits": [], "ms": 1}) + "\n")
    c = L.candidates(msgs, "n", spans=tmp_path, arms=("x", "y"))
    assert c == {"d/a#t0": ["Ana Ruiz", "Tempe"], "d/b#t0": []}
    with pytest.raises(SystemExit):
        L.candidates(msgs[:1], "n", spans=tmp_path, arms=("x",))


def test_merged_spans_snap_to_words_and_join_overlaps():
    t = "hi I'm Ana Ruiz, from Tempe."
    a = t.index("Ana")
    assert L.merged_spans(t, [(a, a + 3), (a + 1, a + 8)]) == [(a, a + 8)]           # "Ana" + "na Ruiz"
    assert L.merged_spans(t, [(a, a + 3), (a + 4, a + 8)]) == [(a, a + 3), (a + 4, a + 8)]   # adjacent stay apart
    assert L.merged_spans(t, [(a - 1, a + 2)]) == [(a, a + 3)]                         # leading space dropped
    assert L.merged_spans(t, []) == []


def _lab(protect=(), sensitive=(), task="qa", sq=False):
    return {"task": task, "service_query": sq, "protect": [{"value": v, "type": t} for v, t in protect],
            "sensitive": [{"value": v, "type": t} for v, t in sensitive], "optional": [], "keep": []}


def test_prevalence_counts_by_type_task_and_kind():
    labs = {"d": {"a": _lab([("Ana", "PERSON"), ("Bo", "PERSON")], task="advice"), "b": _lab(),
                  "c": _lab(sensitive=[("asthma", "HEALTH")]), "e": None}}
    kinds = {"a": "single", "b": "single", "c": "multi", "e": "single"}
    p = L.prevalence(labs, kinds)["d"]
    assert (p["messages"], p["labelled"], p["failed"], p["with_protect"], p["with_pii"]) == (4, 3, 1, 1, 2)
    assert p["messages_with_type"] == {"HEALTH": 1, "PERSON": 1} and p["values_by_type"]["PERSON"] == 2
    assert p["by_task"]["advice"] == {"messages": 1, "with_pii": 1}
    assert p["by_kind"]["multi"] == {"messages": 1, "with_pii": 1}


def test_pii_free_adjudicated_and_literal_rules(tmp_path):
    msgs = [_m("s1", 0), _m("s2", 0), _m("c1", 0), _m("c1", 1)]
    labs = {"d/s1#t0": _lab(), "d/s2#t0": _lab([("Ana", "PERSON")]), "d/c1#t0": _lab(), "d/c1#t1": _lab()}
    for arm in L.LITERAL_ARMS:
        (tmp_path / arm).mkdir()
        rows = [{"id": m["id"], "edits": [[0, 1, "T", 1, 0]] if (arm == "ss" and m["id"] == "d/c1#t1") else [],
                 "ms": 1} for m in msgs]
        (tmp_path / arm / "n.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    free = L.pii_free(msgs, labs, tmp_path, "n")
    assert free["single"] == {"adjudicated": ["s1"], "literal": ["s1", "s2"]}
    assert free["multi"] == {"adjudicated": ["c1"], "literal": []}


def test_human_check_draw_is_balanced_and_seeded():
    labels = {ds: {f"{ds}/{i}#t0": (_lab([("Ana", "PERSON")]) if i % 3 == 0 else _lab()) for i in range(90)}
              for ds in ("oasst1", "sharegpt", "wildchat")}
    a = L.draw_human_check(labels, 100)
    assert a == L.draw_human_check(labels, 100) and len(a) == len(set(a)) == 100
    pos = sum(L.has_pii(labels[m.split("/")[0]][m]) for m in a)
    assert pos == 51 and {m.split("/")[0] for m in a} == set(labels)


def test_agreement_per_type_and_kappa():
    s1, h1 = _lab([("Ana", "PERSON"), ("Tempe", "LOCATION")]), _lab([("Ana", "PERSON")])
    s2, h2 = _lab(), _lab([("4471", "ID")])
    s3, h3 = _lab(), _lab()
    rows = [{**h, "sonnet": s, "checked": True} for s, h in [(s1, h1), (s2, h2), (s3, h3)]]
    rows.append({**h3, "sonnet": _lab([("x", "ID")]), "checked": False})      # unchecked rows ignored
    r = L.agreement(rows)
    assert r["rows"] == 3
    assert r["types"]["PERSON"]["precision"] == 1.0 and r["types"]["LOCATION"]["precision"] == 0.0
    assert r["types"]["ID"]["recall"] == 0.0
    ml = r["message_level"]
    assert (ml["both"], ml["neither"], ml["human_only"], ml["sonnet_only"]) == (1, 1, 1, 0)
    assert ml["kappa"] == pytest.approx((2 / 3 - 4 / 9) / (1 - 4 / 9))


def test_test2_labels_and_prevalence_go_to_their_own_files():
    from bench.realdata.common import COLLECTIONS, TEST1
    t2 = COLLECTIONS["test2"]
    assert L.labels_dir(TEST1) == L.LABELS and L.prevalence_file(TEST1) == L.PREVALENCE
    assert L.labels_dir(t2) == t2.build / "labels" != L.LABELS
    assert L.prevalence_file(t2).name == "realdata_prevalence_test2.json"
