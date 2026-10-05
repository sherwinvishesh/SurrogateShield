"""Model-free tests for bench/realdata/score.py (Phase 4b, E5): frozen-hash
refusal, natural records rebuilt from offset labels, message units and slices,
span checks, per-message scoring (leak, policy, spurious, refusal), Wilson
intervals, the paired cluster bootstrap on toy sets, the pre-registered H1 / H2
checks, and an end-to-end run on a synthetic benchmark with stub arms: no text
in the output, byte-identical reruns. All texts and values are invented."""

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from bench import realworld as rw
from bench.realdata import score as S
from bench.realdata.common import file_sha256, sha256

DS = "oasst1"


def _gold(mid, text, protect=(), optional=(), keep=(), sq=False, sensitive=()):
    return {"id": mid, "text": text, "service_query": sq, "protect": [dict(p) for p in protect],
            "sensitive": [dict(p) for p in sensitive], "optional": [dict(o) for o in optional], "keep": list(keep)}


def _unit(gold, conv=None, task="advice", slices=("injected", "injected_single")):
    return {"mid": gold["id"], "conv": conv or gold["id"], "dataset": DS, "task": task, "slices": slices,
            "turn": 0, "gold": gold}


# ── per message ──────────────────────────────────────────────────────────────

def test_score_unit_leak_policy_spurious_and_keep():
    text = "Email Ada Byrne at ada@example.org, I live in Leeds; Python question."
    g = _gold("m", text, protect=[{"value": "Ada Byrne", "type": "PERSON"}, {"value": "ada@example.org", "type": "EMAIL"},
                                  {"value": "Leeds", "type": "LOCATION"}], keep=["Python"], sq=True)
    a, b = text.index("Ada"), text.index("Python")
    row = {"edits": [[a, a + 3, "PERSON", "Zed"], [b, b + 6, "LANG", "[X]"]]}
    s = S.score_unit(_unit(g), row)
    assert s["values"] == Counter(PERSON=1, EMAIL=1, LOCATION=1)
    assert s["leaked"] == Counter(PERSON=1, EMAIL=1)            # "Byrne" is outside the edit
    assert s["policy"] == Counter(LOCATION=1)                    # service query: location is policy, not leak
    assert s["edits"] == 2 and s["spurious"] == 1 and s["keep_hits"] == 1
    assert s["spurious_types"] == Counter(LANG=1) and s["refused"] == 0


def test_score_unit_copied_surrogate_leaks_and_refusal_sends_nothing():
    text = "Call Ada Byrne today."
    g = _gold("m", text, protect=[{"value": "Ada Byrne", "type": "PERSON"}])
    s, e = 5, 14
    assert S.score_unit(_unit(g), {"edits": [[s, e, "PERSON", "Dr Ada Byrne"]]})["leaked"] == Counter(PERSON=1)
    assert S.score_unit(_unit(g), {"edits": [[s, e, "PERSON", "Lou Park"]]})["leaked"] == Counter()
    r = S.score_unit(_unit(g), {"edits": [], "refused": "x"})
    assert r["refused"] == 1 and r["leaked"] == Counter() and r["edits"] == 0 and r["values"] == Counter(PERSON=1)


def test_optional_values_are_neither_leak_nor_spurious():
    text = "In my story the hero, Zorblax, meets a dragon."
    g = _gold("m", text, optional=[{"value": "Zorblax", "type": "PERSON"}])
    i = text.index("Zorblax")
    s = S.score_unit(_unit(g, slices=("natural",)), {"edits": [[i, i + 7, "PERSON", "[PERSON]"]]})
    assert s["spurious"] == 0 and s["leaked"] == Counter() and not s["values"]


# ── intervals and bootstrap ──────────────────────────────────────────────────

def test_wilson_matches_known_values():
    assert S.wilson(0, 0) is None
    assert S.wilson(5, 10) == [0.2366, 0.7634]
    assert S.wilson(0, 10) == [0.0, 0.2775]
    assert S.wilson(10, 10) == [0.7225, 1.0]


def test_bootstrap_identical_systems_give_zero():
    c = [f"c{i}" for i in range(30)]
    a = [(i % 3 == 0, 1) for i in range(30)]
    r = S.bootstrap(c, a, a, seed=1)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0] and not r["excludes_0"] and r["resamples"] == S.RESAMPLES


def test_bootstrap_separated_systems_exclude_zero_and_are_seeded():
    c = [f"c{i}" for i in range(40)]
    a = [(0, 1)] * 40
    b = [(int(i % 2 == 0), 1) for i in range(40)]
    r = S.bootstrap(c, a, b, seed=7)
    assert r["diff"] == -0.5 and r["excludes_0"] and r["ci95"][1] < 0
    assert r == S.bootstrap(c, a, b, seed=7)


def test_bootstrap_resamples_conversations_not_turns():
    # one conversation holds every leak: resampling the 23 turns would put the
    # lower bound near 0.7; resampling the 4 conversations drops it to 0
    c = ["k0"] * 20 + ["k1", "k2", "k3"]
    a = [(1, 1)] * 20 + [(0, 1)] * 3
    b = [(0, 1)] * 23
    r = S.bootstrap(c, a, b, seed=3)
    assert r["diff"] == round(20 / 23, 4) and r["ci95"][0] == 0.0 and r["ci95"][1] > 0.9
    turns = S.bootstrap([f"t{i}" for i in range(23)], a, b, seed=3)
    assert turns["ci95"][0] > 0.6


def test_bootstrap_without_denominator_reports_nothing():
    assert S.bootstrap(["a", "b"], [(0, 0)] * 2, [(0, 0)] * 2, seed=1) == {"diff": None, "ci95": None, "resamples": 0}


def test_aggregate_rates_types_tasks_and_universes():
    text = "Ada Byrne, 4 Elm Road, user ada_b."
    g = _gold("m1", text, protect=[{"value": "Ada Byrne", "type": "PERSON"}, {"value": "4 Elm Road", "type": "ADDRESS"},
                                   {"value": "ada_b", "type": "HANDLE"}])
    neg = _gold("m2", "Fix this loop please.")
    units = [_unit(g, task="coding"), _unit(neg, task="qa")]
    s1 = S.score_unit(units[0], {"edits": [[0, 9, "PERSON", "Lou Park"]]})
    s2 = S.score_unit(units[1], {"edits": []})
    agg = S.aggregate(units, [s1, s2])
    assert agg["leak"]["k"] == 2 and agg["leak"]["n"] == 3 and agg["message_leak"] == S.rate(1, 1)
    assert agg["negatives_untouched"] == S.rate(1, 1) and agg["spurious"]["k"] == 0
    assert agg["by_type"]["PERSON"]["leaked"] == 0 and agg["by_type"]["ADDRESS"]["leaked"] == 1
    assert agg["macro_leak_rate"] == round(2 / 3, 4)
    assert agg["by_task"]["coding"] == {"values": 3, "policy": 0, "leaked": 2, "leak_rate": round(2 / 3, 4)}
    assert agg["universes"]["shared_all_arms"] == S.rate(0, 1)
    assert agg["universes"]["ss_only_vs_presidio"] == S.rate(2, 2)


def test_claimed_types_are_guide_types():
    assert set(S.CLAIMED) == set(S.ARMS)
    for types in S.CLAIMED.values():
        assert set(types) <= rw.PROTECT_TYPES
    assert set(S.UNIVERSES["shared_all_arms"]) == {"PERSON", "EMAIL", "PHONE", "ID", "NETWORK"}
    assert not set(S.UNIVERSES["presidio_default"]) & set(S.UNIVERSES["ss_only_vs_presidio"])


def _res(ss_leak, base_leak):
    return {a: {"leak": S.rate(k, 100)} for a, k in (("ss", ss_leak), ("presidio_default", base_leak))}


def test_hypotheses_need_a_lower_point_and_an_interval_below_zero():
    good = {"leak_rate": {"diff": -0.1, "ci95": [-0.2, -0.01]}, "spurious_rate": {"diff": -0.1, "ci95": [-0.2, -0.05]}}
    wide = {"leak_rate": {"diff": -0.1, "ci95": [-0.2, 0.01]}, "spurious_rate": {"diff": 0.1, "ci95": [0.05, 0.2]}}
    arms = ("ss", "presidio_default")
    h = S.hypotheses({DS: {"injected": _res(5, 15)}}, {DS: {"injected": {"presidio_default": good}}}, [DS], arms)
    assert h[DS]["H1"] and h[DS]["H2"] and not h[DS]["ss_leak_worse_than_presidio_default"]
    h = S.hypotheses({DS: {"injected": _res(5, 15)}}, {DS: {"injected": {"presidio_default": wide}}}, [DS], arms)
    assert not h[DS]["H1"] and not h[DS]["H2"]
    h = S.hypotheses({DS: {"injected": _res(20, 15)}}, {DS: {"injected": {"presidio_default": good}}}, [DS], arms)
    assert not h[DS]["H1"] and h[DS]["ss_leak_worse_than_presidio_default"]


# ── a synthetic benchmark ────────────────────────────────────────────────────

INJ = [
    {"id": "rd-oasst1-dev-0001", "text": "Hi, I'm Ada Byrne. How do I bake bread?",
     "protect": [{"value": "Ada Byrne", "type": "PERSON", "fmt": "plain"}], "shift": False, "turns": None},
    {"id": "rd-oasst1-dev-0002", "text": "write to ada dot byrne at mailbox dot test about the rota",
     "protect": [{"value": "ada dot byrne at mailbox dot test", "type": "EMAIL", "fmt": "spelled"}], "shift": True,
     "turns": None},
]
MULTI = ["My number is 0161 496 0000, can you plan my week?", "Thanks, now add a gym slot."]
NAT = {"s1": ["Explain recursion with a Python example."],
       "s2": ["Tell my friend Ola Nwosu happy birthday in French.", "Now make it formal."],
       "s3": ["Summarise this article."]}


def _injected():
    out = []
    for r in INJ:
        out.append({**r, "category": "qa", "lang": "en", "service_query": False, "sensitive": [], "optional": [],
                    "keep": [], "dataset": DS, "source_id": r["id"][-4:], "task": "qa", "slice": "injected",
                    "layout": "inline", "round": 1, "identity": {"values": r["protect"]}})
    t0 = {"text": MULTI[0], "service_query": False, "sensitive": [], "optional": [], "keep": [],
          "protect": [{"value": "0161 496 0000", "type": "PHONE", "fmt": "plain"}]}
    t1 = {"text": MULTI[1], "service_query": False, "sensitive": [], "optional": [], "keep": [], "protect": []}
    out.append({"id": "rd-oasst1-dev-0003", "category": "advice", "lang": "en", **t0, "dataset": DS,
                "source_id": "m9", "task": "advice", "slice": "injected", "shift": False, "layout": "inline",
                "round": 1, "identity": {"values": t0["protect"]}, "turns": [t0, t1]})
    return out


def _label_row(mid, text, status="ok", protect=(), keep=()):
    row = {"id": mid, "sha256": sha256(text), "label_status": status, "round": 1}
    if status != "ok":
        return {**row, "problems": []}
    occ = lambda v: [list(o) for o in rw.occurrences(text, v)]
    return {**row, "task": "writing", "service_query": False, "sensitive": [], "optional": [],
            "protect": [{"type": t, "occ": occ(v)} for v, t in protect], "keep": [{"occ": occ(v)} for v in keep]}


def make_bench(tmp_path):
    """A synthetic benchmark: rd (injected dev split, public pool, labels),
    build (private pool), a spans directory and the frozen hashes."""
    rd, build, spans = tmp_path / "rd", tmp_path / "build", tmp_path / "spans"
    (rd / DS).mkdir(parents=True)
    (build / DS).mkdir(parents=True)
    with open(rd / DS / "dev.jsonl", "w") as f:
        for r in _injected():
            f.write(json.dumps(r) + "\n")
    pool, priv, labels = [], [], []
    for sid, turns in NAT.items():
        split = "test" if sid == "s3" else "dev"
        kind = "multi" if len(turns) > 1 else "single"
        pool.append({"dataset": DS, "kind": kind, "meta": {}, "refs": [], "sha256": [sha256(t) for t in turns],
                     "source_id": sid, "split": split, "words": [len(t.split()) for t in turns]})
        priv.append({"dataset": DS, "kind": kind, "meta": {}, "source_id": sid, "split": split, "turns": turns})
    labels.append(_label_row(f"{DS}/s1#t0", NAT["s1"][0], keep=["Python"]))
    labels.append(_label_row(f"{DS}/s2#t0", NAT["s2"][0], protect=[("Ola Nwosu", "PERSON")]))
    labels.append(_label_row(f"{DS}/s2#t1", NAT["s2"][1], status="failed"))
    labels.append(_label_row(f"{DS}/s3#t0", NAT["s3"][0]))
    for path, rows in ((rd / DS / "pool.jsonl", pool), (build / DS / "pool.jsonl", priv), (rd / DS / "labels.jsonl", labels)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    frozen = {f"{DS}/{n}": file_sha256(rd / DS / n) for n in ("dev.jsonl", "pool.jsonl", "labels.jsonl")}
    return rd, build, spans, frozen


@pytest.fixture()
def bench(tmp_path):
    return make_bench(tmp_path)


def _runner(spans):
    """Stub arms: ``ss`` edits every capitalised word pair and digit group; ``presidio_default``
    edits only the word "Python" (a spurious edit) and nothing else."""
    import re

    def run(arm, src, name):
        rows = []
        for line in open(src):
            m = json.loads(line)
            if arm == "ss":
                pat = r"[A-Z][a-z]+ [A-Z][a-z]+|\d[\d ]*\d|ada dot \S+ at \S+ dot test"
            else:
                pat = r"Python"
            rows.append({"id": m["id"], "ms": 1.0,
                         "edits": [[x.start(), x.end(), "T", "Q" * 3] for x in re.finditer(pat, m["text"])]})
        out = spans / arm / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        Path(str(out) + ".meta.json").write_text(json.dumps({"input_sha256": file_sha256(src)}))
    return run


def _score(bench, tmp_path, reuse=False, **kw):
    rd, build, spans, frozen = bench
    return S.score_split("dev", [DS], ("ss", "presidio_default"), reuse, out=tmp_path / "res" / "realdata_dev.json",
                         rd=rd, build=build, frozen=frozen, runner=_runner(spans), spans=spans, log=lambda *_: None, **kw)


def test_natural_records_rebuild_values_and_skip_failed_turns(bench):
    rd, build, _spans, _frozen = bench
    recs = S.natural_records(DS, "dev", rd, build)
    assert [r["id"] for r in recs] == ["rd-oasst1-dev-1001", "rd-oasst1-dev-1002"]
    assert recs[0]["keep"] == ["Python"] and recs[0]["turns"] is None
    assert recs[1]["protect"] == [{"value": "Ola Nwosu", "type": "PERSON"}]
    assert [t["label_status"] for t in recs[1]["turns"]] == ["ok", "failed"]
    units = S.messages(recs)
    assert [u["mid"] for u in units] == ["rd-oasst1-dev-1001", "rd-oasst1-dev-1002#t0"]
    assert S.lint_records(recs) == []


def test_natural_records_refuse_a_changed_text(bench):
    rd, build, _spans, _frozen = bench
    rows = [json.loads(l) for l in open(build / DS / "pool.jsonl")]
    rows[0]["turns"][0] += " "
    (build / DS / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="frozen hash"):
        S.natural_records(DS, "dev", rd, build)


def test_messages_ids_and_slices():
    units = S.messages(_injected())
    assert [(u["mid"], u["slices"]) for u in units] == [
        ("rd-oasst1-dev-0001", ("injected", "injected_single")), ("rd-oasst1-dev-0002", ("injected", "shift")),
        ("rd-oasst1-dev-0003#t0", ("injected", "multi")), ("rd-oasst1-dev-0003#t1", ("injected", "multi"))]
    assert units[3]["gold"]["protect"] == [] and units[3]["conv"] == "rd-oasst1-dev-0003"


def test_check_frozen_refuses_a_changed_or_unfrozen_file(bench):
    rd, _build, _spans, frozen = bench
    assert S.check_frozen([DS], "dev", frozen, rd) == frozen
    with pytest.raises(SystemExit, match="not frozen"):
        S.check_frozen([DS], "dev", {k: v for k, v in frozen.items() if "labels" not in k}, rd)
    with open(rd / DS / "dev.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="refusing to score"):
        S.check_frozen([DS], "dev", frozen, rd)


def test_read_spans_refuses_another_input_missing_ids_and_bad_offsets(tmp_path):
    units = [_unit(_gold("a", "Ada Byrne")), _unit(_gold("b", "hello"))]
    full = tmp_path / "x.jsonl"
    meta = Path(str(full) + ".meta.json")
    full.write_text(json.dumps({"id": "a", "edits": [[0, 3, "T", "Q"]]}) + "\n" + json.dumps({"id": "b", "edits": []}) + "\n")
    meta.write_text(json.dumps({"input_sha256": "abc"}))
    assert set(S.read_spans(full, units, "abc")) == {"a", "b"}
    with pytest.raises(SystemExit, match="another input"):
        S.read_spans(full, units, "abd")
    with pytest.raises(SystemExit, match="ids differ"):
        S.read_spans(full, units[:1], "abc")
    full.write_text(json.dumps({"id": "a", "edits": [[0, 30, "T", "Q"]]}) + "\n" + json.dumps({"id": "b", "edits": []}) + "\n")
    with pytest.raises(SystemExit, match="outside"):
        S.read_spans(full, units, "abc")


def test_end_to_end_counts_privacy_and_byte_identical_rerun(bench, tmp_path):
    doc = _score(bench, tmp_path)
    out = tmp_path / "res" / "realdata_dev.json"
    first = out.read_bytes()
    r = doc["results"][DS]
    # injected: 3 values (name, spelled e-mail, phone); SS covers all, presidio none
    assert r["injected"]["ss"]["leak"] == S.rate(0, 3) and r["injected"]["presidio_default"]["leak"] == S.rate(3, 3)
    assert r["shift"]["ss"]["messages"] == 1 and r["multi"]["ss"]["messages"] == 2
    # natural (dev only: s1, s2 turn 0): presidio's edit of the keep word is spurious; SS covers Ola Nwosu
    assert r["natural"]["presidio_default"]["spurious"] == S.rate(1, 1) and r["natural"]["presidio_default"]["keep_hits"] == 1
    assert r["natural"]["ss"]["leak"] == S.rate(0, 1)
    assert doc["corpus"][DS]["natural_turns_excluded"] == 1 and doc["corpus"][DS]["natural_records"] == 2
    d = doc["differences"][DS]["injected"]["presidio_default"]["leak_rate"]
    assert d["diff"] == -1.0
    assert doc["command"].split()[-1].endswith("realdata_dev.json")
    assert len(doc["git"]["commit"]) == 40 and isinstance(doc["git"]["modified"], list)
    assert f"at commit `{doc['git']['commit'][:12]}`" in out.with_suffix(".md").read_text()
    assert doc["hypotheses"][DS]["H1_per_baseline"] == {"presidio_default": True}
    # counts only: no text and no value in the outputs
    blob = first.decode() + out.with_suffix(".md").read_text()
    for t in [x for r_ in INJ for x in (r_["text"], r_["protect"][0]["value"])] + MULTI + ["Ola Nwosu", "Explain recursion"]:
        assert t not in blob
    nat = bench[1] / DS / "natural-dev.jsonl"
    assert (os.stat(nat).st_mode & 0o777) == 0o600
    # rerun, then rescoring the saved spans: the same bytes
    _score(bench, tmp_path)
    assert out.read_bytes() == first
    _score(bench, tmp_path, reuse=True)
    assert out.read_bytes() == first


def test_reuse_refuses_spans_of_another_input(bench, tmp_path):
    _score(bench, tmp_path)
    rd, build, spans, frozen = bench
    meta = spans / "ss" / "dev-oasst1.jsonl.meta.json"
    meta.write_text(json.dumps({"input_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="another input"):
        _score(bench, tmp_path, reuse=True)


def test_commit_note_flags_a_dirty_tree():
    from bench.realdata.common import commit_note
    assert commit_note(None) == ""
    assert commit_note({"commit": "a" * 40, "modified": []}) == " at commit `aaaaaaaaaaaa`"
    assert commit_note({"commit": "a" * 40, "modified": ["x.py", "y.py"]}).endswith("with 2 modified tracked file(s)")
