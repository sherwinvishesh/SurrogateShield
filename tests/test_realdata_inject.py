"""Model-free tests for bench/realdata/inject.py (Phase 3 injected slice):
edits apply exactly or not at all, the word metric, each verifier rule with
an accepting and a rejecting case, the detector rule, answer parsing, the
request's shape, a deterministic plan, the two-round flow with a stubbed
batch runner, and records that lint clean. No model, no network; every text
here is made up."""

import json
import os
import re
from pathlib import Path

import pytest

from bench import realworld as rw
from bench.realdata import common
from bench.realdata import identities as I
from bench.realdata import inject as J

LAB = {"task": "qa", "service_query": False, "protect": [], "sensitive": [], "optional": [], "keep": []}
ANA = [("Ana Ruiz", "PERSON"), ("ana.ruiz@example.org", "EMAIL")]
TRIP = "Can you help me plan a trip to the coast this summer? Any ideas are welcome."


def row(values=ANA, layout="inline", key="oasst1/s1", kind="single", split="dev", shift=False):
    return {"key": key, "dataset": key.split("/")[0], "source_id": key.split("/")[1], "split": split, "kind": kind,
            "task": "qa", "shift": shift, "layout": layout,
            "identity": {"locale": "en_US", "country": "US", "shift": shift,
                         "values": [{"value": v, "type": t, "fmt": "plain"} for v, t in values]}}


def edit(find, replace, turn=1):
    return {"turn": turn, "find": find, "replace": replace}


INTRO = edit("Can you help", "Hi, I'm Ana Ruiz (ana.ruiz@example.org). Can you help")


# ── apply_edits, inserted, word_change ───────────────────────────────────────

def test_apply_edits_applies_in_text_order_and_reports_regions():
    text = "one two three four"
    new, regions, probs = J.apply_edits(text, [edit("four", "four!"), edit("one", "zero one")])
    assert probs == [] and new == "zero one two three four!"
    assert regions == [(0, 3, 0, 8), (14, 18, 19, 24)]
    assert [new[ns:ne] for _s, _e, ns, ne in regions] == ["zero one", "four!"]


@pytest.mark.parametrize("edits, needle", [
    ([edit("", "x")], "find is empty"),
    ([edit("five", "x")], "not an exact substring"),
    ([edit("o", "x")], "more than once"),
    ([edit("one two", "x"), edit("two three", "y")], "overlap"),
])
def test_apply_edits_rejects_bad_finds(edits, needle):
    new, regions, probs = J.apply_edits("one two three four", edits)
    assert new is None and regions == [] and any(needle in p for p in probs), probs


def test_inserted_marks_only_the_new_characters():
    text = "Plan a trip. Thanks"
    new, regions, _ = J.apply_edits(text, [edit("Thanks", "Thanks, Ana Ruiz")])
    assert [new[a:b] for a, b in J.inserted(text, new, regions)] == [", Ana Ruiz"]


def test_word_change_counts_a_value_as_one_word_and_frames_as_added():
    text = "Plan a trip. Thanks"
    new, regions, _ = J.apply_edits(text, [edit("Thanks", "Thanks, my email is ana.ruiz@example.org")])
    vs = J._spans(new, ["ana.ruiz@example.org"])
    assert J.word_change(text, new, regions, vs) == (1.0, 3)
    new, regions, _ = J.apply_edits(text, [edit("Plan a trip.", "Organise one holiday.")])
    kept, added = J.word_change(text, new, regions)
    assert kept == pytest.approx(1 / 4) and added == 3


def test_word_change_widens_a_cut_inside_a_word():
    text = "please summarise this"
    new, regions, _ = J.apply_edits(text, [edit("summar", "summar")])
    assert J.word_change(text, new, regions) == (1.0, 0)
    new, regions, _ = J.apply_edits(text, [edit("mmarise", "mmarize")])
    assert J.word_change(text, new, regions) == (pytest.approx(2 / 3), 1)


# ── check: one accepting and one rejecting case per rule ─────────────────────

def test_check_accepts_a_minimal_placement():
    turns, info, probs = J.check(row(), [TRIP], [LAB], [INTRO])
    assert probs == [] and turns[0].startswith("Hi, I'm Ana Ruiz (ana.ruiz@example.org). Can you help me")
    assert info["kept"] == [1.0] and info["added"] == [3]


@pytest.mark.parametrize("rep, needle", [
    ("Hi, I'm ana ruiz (ana.ruiz@example.org). Can you help", "not in turn 1 exactly as given"),
    ("Hi, I'm Ana Ruiz2 (ana.ruiz@example.org). Can you help", "not in turn 1 exactly as given"),
    ("Hi, I'm Ana Ruiz (ana.ruiz@example.org, cc bob@example.com). Can you help", "e-mail, link or long number"),
    ("Hi, I'm Ana Ruiz (ana.ruiz@example.org), member 4471902. Can you help", "e-mail, link or long number"),
    ("Hi, I'm Ana Ruiz (ana.ruiz@example.org), key ghp_q81. Can you help", "real provider's token"),
    ("Hi, I'm Ana Ruiz (ana.ruiz@example.org). " + "Well " * 25 + "Can you help", "words added"),
])
def test_check_rejects_each_broken_turn_one_rule(rep, needle):
    _t, _i, probs = J.check(row(), [TRIP], [LAB], [edit("Can you help", rep)])
    assert any(needle in p for p in probs), probs


def test_check_rejects_a_rewrite_of_the_users_words():
    rewrite = "Hi, I'm Ana Ruiz (ana.ruiz@example.org). Suggest seaside holiday plans for July, please; ideas needed."
    _t, info, probs = J.check(row(), [TRIP], [LAB], [edit(TRIP, rewrite)])
    assert info["kept"][0] < J.KEEP_MIN and any("words are kept" in p for p in probs), probs


def test_check_rejects_an_edit_for_a_turn_that_does_not_exist():
    turns, _i, probs = J.check(row(), [TRIP], [LAB], [INTRO, edit("Any", "Any", turn=2)])
    assert turns is None and "names turn 2" in probs[0]


def test_check_json_layout_needs_the_name_inside_an_object():
    inside = edit("welcome.", 'welcome.\n{"name": "Ana Ruiz", "email": "ana.ruiz@example.org"}')
    outside = edit("welcome.", 'welcome. Ana Ruiz\n{"email": "ana.ruiz@example.org"}')
    assert J.check(row(layout="json"), [TRIP], [LAB], [inside])[2] == []
    assert any("layout json" in p for p in J.check(row(layout="json"), [TRIP], [LAB], [outside])[2])
    assert J.check(row(layout="inline"), [TRIP], [LAB], [outside])[2] == []


def test_in_json():
    assert J.in_json('x {"a": "Ana Ruiz"} y', "Ana Ruiz")
    assert not J.in_json('x {"a": 1} Ana Ruiz {"b": 2}', "Ana Ruiz")
    assert not J.in_json("Ana Ruiz", "Ana Ruiz")


LETTER = ["Can you draft a cover letter for a barista job?", "Make it shorter please.", "Now add a closing line."]
FIRST = edit("Can you draft", "I'm Ana Ruiz, ana.ruiz@example.org. Can you draft")


def multi(t2, t3):
    return J.check(row(kind="multi"), LETTER, [LAB] * 3, [FIRST, *t2, *t3])


def test_check_multi_turn_accepts_turns_that_refer_back():
    turns, _i, probs = multi([edit("please.", "please, and sign it with my name.", 2)],
                             [edit("closing line.", "closing line with my email from before.", 3)])
    assert probs == [] and turns[1] == "Make it shorter please, and sign it with my name."


@pytest.mark.parametrize("t2, t3, needle", [
    ([], [edit("closing line.", "closing line with my email from before.", 3)], "turn 2 has no edit"),
    ([edit("please.", "please, signed Ruiz.", 2)], [edit("line.", "line, as before.", 3)], "repeats part of a detail"),
    ([edit("please.", "please, signed ana ruiz.", 2)], [edit("line.", "line, as before.", 3)], "turn 2 repeats"),
    ([edit("please.", "please.", 2)], [edit("line.", "line to ana.ruiz@example.org.", 3)], "turn 3 repeats"),
])
def test_check_multi_turn_rejects_a_later_turn_that_repeats_or_is_untouched(t2, t3, needle):
    _t, _i, probs = multi(t2, t3)
    assert any(needle in p for p in probs), probs


def test_name_parts_cover_name_words_and_digit_groups():
    vals = [{"type": "PERSON", "value": "Li Wei-Chen"}, {"type": "PHONE", "value": "(415) 867-2290"},
            {"type": "AGE", "value": "34"}]
    assert J.name_parts(vals) == ["2290", "415", "867", "Chen", "Li", "Wei"]


# ── detector rule, problem kinds ─────────────────────────────────────────────

def test_detector_problems_need_two_arms_on_added_non_value_text():
    new = "Hi, I'm Ana Ruiz. Plan a trip to Leeds."
    ins = [(0, 18)]
    vs = J._spans(new, ["Ana Ruiz"])
    two = lambda a, b: J.detector_problems(new, ins, vs, {"ss": [list(a)], "gliner_pii": [list(b)]})
    assert two((8, 16, "PERSON", "x"), (8, 16, "person", "x")) == []           # the placed value
    assert two((33, 38, "GPE", "x"), (33, 38, "location", "x")) == []          # the user's own text
    assert J.detector_problems(new, ins, vs, {"gliner_pii": [[4, 5, "person", "x"]]}) == []   # one arm: "I"
    probs = two((0, 2, "PERSON", "x"), (0, 3, "person", "x"))                   # "Hi", added, two arms
    assert len(probs) == 1 and '"Hi,"' in probs[0] and "2 detectors (gliner_pii: person, ss: PERSON)" in probs[0]
    assert J.detector_problems(new, ins, vs, {"ss": [[0, 2, "PERSON", "x"]]}, quorum=1)


def test_problem_kind_drops_quoted_text_and_numbers():
    assert J.problem_kind('turn 2 repeats "Ana Ruiz"') == "turn N repeats <text>"
    assert J.problem_kind("turn 1: only 40% of the user's words are kept (at least 65%); x") == \
        "turn N: only N of the user's words are kept (at least N); x"


# ── answers and requests ─────────────────────────────────────────────────────

def message(items):
    return {"content": [{"type": "tool_use", "name": J.TOOL_NAME, "input": {"items": items}}], "stop_reason": "tool_use"}


def test_parse_keys_by_row_and_reports_missing_duplicate_and_malformed_items():
    local = {"p1": "a/1", "p2": "a/2", "p3": "a/3", "p4": "a/4"}
    ok, bad = J.parse(message([{"id": "p1", "edits": [edit("x", "y")]},
                               {"id": "p3", "edits": []}, {"id": "p3", "edits": []},
                               {"id": "p4", "edits": [{"turn": "1", "find": "x", "replace": "y"}]}]), local)
    assert ok == {"a/1": [edit("x", "y")]}
    assert bad == {"a/2": ["no answer for this item"], "a/3": ["answered more than once"],
                   "a/4": ["edits must be objects with turn, find and replace"]}
    ok, bad = J.parse({"content": [{"type": "text", "text": "hi"}]}, {"p1": "a/1"})
    assert ok == {} and bad == {"a/1": ["no place_values call in the answer"]}


def test_params_force_the_tool_cache_the_rules_and_list_values_with_meanings():
    r = row(values=[("Ana Ruiz", "PERSON"), ("4111 2222 3333 4444", "ID")], layout="json")
    r["identity"]["values"][1]["fmt"] = "card-spaced"
    p, local = J.params([r], {r["key"]: [TRIP]}, "m", feedback={r["key"]: ["turn 1: lint: x"]})
    assert local == {"p1": r["key"]} and p["temperature"] == 0 and p["model"] == "m"
    assert p["tool_choice"] == {"type": "tool", "name": J.TOOL_NAME} and p["tools"] == [J.TOOL]
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"} and p["system"][0]["text"] == J.INSTRUCTIONS
    content = p["messages"][0]["content"]
    assert '<item id="p1" layout="json" turns="1">' in content and TRIP in content
    assert "- p1: turn 1: lint: x" in content
    vals = json.loads(re.search(r"<values>(.*)</values>", content).group(1))
    assert vals == [{"value": "Ana Ruiz", "means": "the user's name"},
                    {"value": "4111 2222 3333 4444", "means": "the user's payment card number"}]


def test_every_type_and_shift_format_has_a_meaning_and_the_rules_name_every_layout():
    assert set(J.MEANS) == set(I.TYPES)
    for layout in (*J.LAYOUTS, "json"):
        assert f"- {layout}:" in J.INSTRUCTIONS
    assert not I.FORBIDDEN.search(J.INSTRUCTIONS) and len(J.prompt_version()) == 16


def test_groups_respect_items_and_characters_per_request():
    rows = [row(key=f"d/{i}") for i in range(12)]
    sizes = {2: 5000, 4: 1500}                       # items 0-3 hold 5030 characters; item 4 would cross 6000
    texts = {r["key"]: ["x" * sizes.get(i, 10)] for i, r in enumerate(rows)}
    gs = J.groups(rows, texts, per_call=5, budget=6000)
    assert [len(g) for g in gs] == [4, 5, 3] and [r for g in gs for r in g] == rows
    big = {r["key"]: ["x" * 7000] for r in rows[:2]}
    assert [len(g) for g in J.groups(rows[:2], big, budget=6000)] == [1, 1]


# ── plan ─────────────────────────────────────────────────────────────────────

WORDS = "please explain how tides work for a curious beginner who likes the sea and long walks".split()


@pytest.fixture(scope="module")
def synthetic():
    units, single, multi_ = {}, [], []
    for i in range(380):
        kind = "single" if i < 300 else "multi"
        split = "dev" if (i < 300 and i < 70) or 300 <= i < 315 else "test"
        n = 3 if kind == "multi" else 1
        turns = [" ".join(WORDS[(i + k) % 5:][: 6 + (i * 7 + k) % 12]) + "?" for k in range(n)]
        sid = f"s{i:03d}"
        units[sid] = {"source_id": sid, "split": split, "kind": kind, "words": [len(t.split()) for t in turns],
                      "turns": turns, "labels": [dict(LAB, task=("qa", "coding", "advice")[i % 3]) for _ in turns]}
        (single if kind == "single" else multi_).append(sid)
    return units, {"single": {"adjudicated": single}, "multi": {"adjudicated": multi_}}


def test_plan_is_deterministic_and_matches_the_counts(synthetic):
    units, free = synthetic
    a, b = J.plan("oasst1", units, free), J.plan("oasst1", units, free)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    single = [r for r in a if r["kind"] == "single"]
    multi_ = [r for r in a if r["kind"] == "multi"]
    assert len(single) == 250 and len(multi_) == 50
    assert sum(r["split"] == "dev" for r in single) == 50 and sum(r["split"] == "dev" for r in multi_) == 10
    assert sum(r["shift"] for r in single) == 40 and not any(r["shift"] for r in multi_)
    js = [r for r in a if r["layout"] == "json"]
    assert len(js) == 13 and all(r["shift"] and "PERSON" in r["types"] for r in js)
    for r in a:
        text = "\n".join(units[r["source_id"]]["turns"])
        assert r["identity"]["shift"] == r["shift"]
        assert not any(rw.occurrences(text, v["value"]) for v in r["identity"]["values"])
    assert json.dumps(J.plan("sharegpt", units, free)) != json.dumps(a)


def test_plan_refuses_when_a_split_has_too_few_bases(synthetic):
    units, free = synthetic
    with pytest.raises(SystemExit, match="dev sources"):
        J.plan("oasst1", units, {**free, "single": {"adjudicated": free["single"]["adjudicated"][60:]}})


@pytest.fixture(scope="module")
def synthetic2():
    units, single, multi_ = {}, [], []
    for i in range(480):
        kind = "single" if i < 400 else "multi"
        n = 3 if kind == "multi" else 1
        turns = [" ".join((WORDS * 6)[(i + k) % 5:][: 20 + (i * 7 + k) % 50]) + "?" for k in range(n)]
        sid = f"t{i:03d}"
        units[sid] = {"source_id": sid, "split": "test2", "kind": kind, "words": [len(t.split()) for t in turns],
                      "turns": turns, "labels": [dict(LAB, task=("qa", "coding", "advice")[i % 3]) for _ in turns]}
        (single if kind == "single" else multi_).append(sid)
    return units, {"single": {"adjudicated": single}, "multi": {"adjudicated": multi_}}


def test_test2_plan_has_its_own_sizes_seeds_and_only_evaluation_pool_identities(synthetic2):
    t2 = common.COLLECTIONS["test2"]
    units, free = synthetic2
    a = J.plan("oasst1", units, free, coll=t2)
    assert json.dumps(a, sort_keys=True) == json.dumps(J.plan("oasst1", units, free, coll=t2), sort_keys=True)
    single = [r for r in a if r["kind"] == "single"]
    assert len(single) == 300 and len(a) == 360 and {r["split"] for r in a} == {"test2"}
    assert sum(r["shift"] for r in single) == 48
    named = [r for r in single if r["shift"] and "PERSON" in r["types"]]
    assert [r for r in a if r["layout"] == "json"] == [r for r in named if r["layout"] == "json"]
    assert sum(r["layout"] == "json" for r in a) == min(len(named), round(48 * J.JSON_SHARE)) > 0
    for t in I.TYPES:
        assert sum(t in r["types"] for r in single) >= J.TARGET_TEST2, t
    assert all(r["identity"]["pool"] == "eval" for r in a)
    assert all(I.token_half(w) == "eval" for r in a for w in r["identity"]["pool_tokens"])
    assert t2.seed("inject", "oasst1") != common.derive_seed("inject", "oasst1")


def test_collections_pick_their_sizes_pool_and_summary():
    t1, t2 = common.TEST1, common.COLLECTIONS["test2"]
    assert J.sizes(t1) == (J.N_SINGLE, J.N_SHIFT, J.N_MULTI, I.TARGET)
    assert J.sizes(t2) == ({"test2": 300}, {"test2": 48}, {"test2": 60}, 60)
    assert J.pool_of(t1) is None and J.pool_of(t2) == "eval"
    assert J.summary_file(t1) == J.SUMMARY and J.summary_file(t2).name == "realdata_injection_test2.json"


def test_avoid_values_skip_ones_without_two_letters_or_digits():
    lab = dict(LAB, keep=["/", "C", "Python", "C#"], optional=[{"value": "UTC", "type": "LOCATION"}])
    assert J.avoid_values([lab]) == ["Python", "UTC"]


# ── records, summary, two rounds ─────────────────────────────────────────────

def test_turn_record_drops_a_keep_value_a_placed_value_covers():
    vals = [{"value": "12 Leeds Road", "type": "ADDRESS", "fmt": "plain"}]
    lab = dict(LAB, keep=["Leeds", "tides"])
    rec = J.turn_record("I live at 12 Leeds Road; explain tides.", lab, vals)
    assert rec["protect"] == vals and rec["keep"] == ["tides"] and rec["sensitive"] == []
    assert J.turn_record("no values here", lab, vals)["protect"] == []


def stub_batch(answer):
    """A run_batch stand-in: answers each item of each request with
    ``answer(round_name, item_id, turn_texts, values)`` → edits."""
    item = re.compile(r'<item id="(p\d+)"[^>]*>\n(.*?)<values>(.*?)</values>', re.S)
    turn = re.compile(r'<turn n="\d+">\n(.*?)\n</turn>', re.S)

    def run_batch(cl, ledger, name, phase, kind, requests, log=print, **kw):
        out = {}
        for q in requests:
            content = q["params"]["messages"][0]["content"]
            items = [{"id": lid, "edits": answer(name, turn.findall(body), [v["value"] for v in json.loads(vals)])}
                     for lid, body, vals in item.findall(content)]
            out[q["custom_id"]] = {"status": "succeeded", "message": message(items)}
        return out
    return run_batch


def append_values(turns, values, skip_last=False):
    t = turns[0]
    tail = values[:-1] if skip_last else values
    return [edit(t, t + " Thanks, " + ", ".join(tail))] + [edit(x, x + " Use the details I gave.", k)
                                                          for k, x in enumerate(turns[1:], 2)]


@pytest.fixture
def three(monkeypatch, tmp_path):
    texts = {"oasst1/a": ["Explain tides simply."], "oasst1/b": ["Explain waves simply, flaky."],
             "oasst1/c": ["Explain wind simply, broken."],
             "oasst1/m": ["Draft a short note.", "Make it warmer."]}
    rows = [row(key="oasst1/a"), row(key="oasst1/b", split="test"), row(key="oasst1/c"),
            row(key="oasst1/m", kind="multi")]
    labels = {k: [LAB] * len(t) for k, t in texts.items()}
    monkeypatch.setattr(J, "load_all", lambda datasets, coll=None: (rows, texts, labels))
    monkeypatch.setattr(J, "INJECT", tmp_path)

    def answer(name, turns, values):
        broken = "broken" in turns[0] or ("flaky" in turns[0] and name.endswith("-r1"))
        return append_values(turns, values, skip_last=broken)
    monkeypatch.setattr("bench.realdata.provider.run_batch", stub_batch(answer))
    return rows, labels, tmp_path


def test_run_accepts_re_asks_once_then_drops(three):
    rows, labels, tmp = three
    seen = []
    final = J.run(["oasst1"], model="m", log=lambda s: None, detector=lambda texts, name: seen.append(name) or {})
    assert {k: (v["status"], v["round"]) for k, v in final.items()} == {
        "oasst1/a": ("accepted", 1), "oasst1/b": ("accepted", 2), "oasst1/c": ("dropped", 2),
        "oasst1/m": ("accepted", 1)}
    assert final["oasst1/b"]["first_problems"] and final["oasst1/c"]["problems"]
    assert len(seen) == 2 and seen[0].endswith("-r1") and seen[1].endswith("-r2")
    res = tmp / f"results-{J.prompt_version()}.jsonl"
    assert oct(os.stat(res).st_mode & 0o777) == "0o600"

    recs = J.records(rows, labels, final)
    assert sorted(recs) == [("oasst1", "dev"), ("oasst1", "test")]
    dev = recs[("oasst1", "dev")]
    assert [r["id"] for r in dev] == ["rd-oasst1-dev-0001", "rd-oasst1-dev-0002"]
    assert [r["source_id"] for r in dev] == ["a", "m"] and dev[0]["turns"] is None
    assert recs[("oasst1", "test")][0]["id"] == "rd-oasst1-test-0001"
    for r in [x for v in recs.values() for x in v]:
        assert rw.lint([r]) == [] and r["slice"] == "injected" and r["sensitive"] == []
        assert [p["value"] for p in r["protect"]] == ["Ana Ruiz", "ana.ruiz@example.org"]
    m = dev[1]
    assert len(m["turns"]) == 2 and m["turns"][1]["protect"] == [] and m["text"] == m["turns"][0]["text"]

    s = J.summary(rows, final)["oasst1"]
    assert s["single"] == {"planned": 3, "accepted_round1": 1, "accepted_round2": 1, "dropped": 1,
                           "acceptance": 0.6667}
    assert s["multi"]["accepted_round1"] == 1 and s["types_single"]["rows"]["PERSON"] == 2
    blob = json.dumps(s)
    assert "Ana" not in blob and "tides" not in blob and "<text>" in blob
