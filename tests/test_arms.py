"""Model-free tests for the span producers in bench/arms/ (Phase 4, E4)."""

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from bench.arms import base, inputs, run
from bench.arms.gliner_pii import windows
from bench.arms.llm_guard import fix_faker_arity
from bench.arms.presidio_faker import generators, shape_like

ROOT = Path(__file__).resolve().parents[1]


def test_msg_seed_is_stable_and_keyed():
    a = base.msg_seed("ss", "oasst1/x#t0")
    assert a == base.msg_seed("ss", "oasst1/x#t0")
    assert a != base.msg_seed("ss", "oasst1/x#t1")
    assert a != base.msg_seed("llm_guard", "oasst1/x#t0")
    assert 0 <= a < 2 ** 32


def test_apply_edits_replaces_in_offset_order():
    text = "Call Ann at 555-0101 today."
    edits = [[12, 20, "PHONE", "555-0199"], [5, 8, "PERSON", "Bea"]]
    assert base.apply_edits(text, edits) == "Call Bea at 555-0199 today."
    assert base.apply_edits(text, []) == text


@pytest.mark.parametrize("edits", [
    [[0, 5, "T", "x"], [3, 8, "T", "y"]],        # overlap
    [[5, 8, "T", "x"], [0, 2, "T", "y"]],        # unsorted
    [[0, 99, "T", "x"]],                         # out of bounds
    [[0, 2, "T", None]],                         # replacement not a string
])
def test_check_edits_rejects_bad_edits(edits):
    with pytest.raises(ValueError):
        base.check_edits("0123456789", edits)


def test_check_edits_accepts_adjacent_and_empty():
    base.check_edits("0123456789", [[0, 3, "T", "a"], [3, 5, "T", "b"]])
    base.check_edits("", [])


def test_resolve_overlaps_longest_then_score():
    spans = [(0, 4, "A", 0.9), (0, 10, "B", 0.5), (12, 15, "C", 0.4), (12, 15, "D", 0.8), (20, 22, "E", 0.1)]
    assert base.resolve_overlaps(spans) == [(0, 10, "B", 0.5), (12, 15, "D", 0.8), (20, 22, "E", 0.1)]


def test_hf_revision_reads_local_cache(tmp_path, monkeypatch):
    ref = tmp_path / "models--org--model" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc123\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert base.hf_revision("org/model") == "abc123"
    assert base.hf_revision("org/other") == "not cached"


def _fake_arm(calls):
    def load():
        def fn(text, seed):
            calls.append((text, seed))
            return [[m.start(), m.end(), "NAME", "Zed"] for m in re.finditer(r"Ann|Bob", text)][::-1]
        return fn
    return load


def test_produce_writes_sorted_spans_and_meta(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(m) for m in [
        {"id": "a", "text": "Bob met Ann."}, {"id": "b", "text": "nothing here"}, {"id": "c", "text": "Ann"}]) + "\n")
    out = tmp_path / "out" / "spans.jsonl"
    calls = []
    meta = base.produce("fake", _fake_arm(calls), {"k": 1}, src, out, limit=2)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["edits"] == [[0, 3, "NAME", "Zed"], [8, 11, "NAME", "Zed"]]
    assert rows[1]["edits"] == [] and all(isinstance(r["ms"], float) for r in rows)
    assert calls[0][1] == 0 and len(calls) == 3                           # warm-up first, then 2 messages
    assert calls[1][1] == base.msg_seed("fake", "a")
    side = json.loads(Path(str(out) + ".meta.json").read_text())
    assert side == meta and meta["messages"] == 2 and meta["edits"] == 2 and meta["config"] == {"k": 1}
    assert "--limit 2" in meta["command"]
    assert meta["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()   # the real-data scorer checks it


def test_produce_refuses_overlapping_output(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": "a", "text": "abcdef"}) + "\n")
    load = lambda: (lambda text, seed: [[0, 4, "T", "x"], [2, 6, "T", "y"]])
    with pytest.raises(ValueError):
        base.produce("fake", load, {}, src, tmp_path / "o.jsonl")
    assert not (tmp_path / "o.jsonl").exists()


def test_produce_records_refusals_and_stops_on_other_errors(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps({"id": i, "text": t}) for i, t in [("a", "Ann"), ("b", "refuse me")]) + "\n")

    def load():
        def fn(text, seed):
            if text.startswith("refuse"):
                raise base.Refused("could not generate a surrogate")
            return [[0, 3, "NAME", "Zed"]] if text == "Ann" else []
        return fn
    meta = base.produce("fake", load, {}, src, tmp_path / "o.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert "refused" not in rows[0] and rows[1]["refused"].startswith("could not") and rows[1]["edits"] == []
    assert meta["refused"] == 1 and meta["edits"] == 1

    def broken():
        def fn(text, seed):
            if text.startswith("refuse"):
                raise RuntimeError("bug")
            return []
        return fn
    with pytest.raises(RuntimeError):
        base.produce("fake", broken, {}, src, tmp_path / "p.jsonl")
    assert not (tmp_path / "p.jsonl").exists()


def test_read_messages_requires_id_and_text(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": 3, "text": "x"}) + "\n")
    with pytest.raises(ValueError):
        base.read_messages(p)


def test_inputs_flatten_and_ids_round_trip():
    rows = [{"dataset": "sharegpt", "source_id": "Ab3x-9.q", "turns": ["one", "two", "three"]},
            {"dataset": "oasst1", "source_id": "1f2e", "turns": ["solo"]}]
    msgs = inputs.flatten(rows)
    assert [m["id"] for m in msgs] == ["sharegpt/Ab3x-9.q#t0", "sharegpt/Ab3x-9.q#t1", "sharegpt/Ab3x-9.q#t2", "oasst1/1f2e#t0"]
    assert [m["text"] for m in msgs] == ["one", "two", "three", "solo"]
    assert inputs.parse_id("sharegpt/Ab3x-9.q#t2") == ("sharegpt", "Ab3x-9.q", 2)


def test_public_rows_carry_no_text():
    texts = {"a": "Mail jo@x.io or Kim Lee, PIN 12."}
    rows = [{"id": "a", "ms": 1.5, "edits": [[5, 12, "EMAIL", "zz@y.io"], [16, 23, "PERSON", "Kim Lee"],
                                              [29, 31, "ID", "12"]]}]
    pub = run.public_rows(rows, texts)
    assert pub == [{"id": "a", "ms": 1.5, "edits": [[5, 12, "EMAIL", 7, 0], [16, 23, "PERSON", 7, 1], [29, 31, "ID", 2, 1]]}]
    assert "jo@x" not in json.dumps(pub) and "Kim" not in json.dumps(pub)
    refused = run.public_rows([{"id": "a", "ms": 2.0, "edits": [], "refused": "could not ... Kim Lee"}], texts)
    assert refused == [{"id": "a", "ms": 2.0, "edits": [], "refused": 1}]


@pytest.mark.parametrize("orig,rep,flag", [
    ("Kim Lee", "Kim Lee", 1), ("Kim Lee", "Dr Kim Lee Jr", 1), ("12", "12", 1),
    ("12", "123", 0), ("Kim Lee", "Ann Roe", 0), ("Kim", "Kimberly", 0),
])
def test_copied_follows_the_scorer_rule(orig, rep, flag):
    assert run.copied(orig, rep) == flag


def test_every_arm_has_a_module_and_an_interpreter():
    assert set(run.ARMS) == {"ss", "presidio_default", "presidio_faker", "presidio_transformers", "llm_guard", "gliner_pii",
                             "gliner_pii_tuned"}
    for arm, venv in run.ARMS.items():
        assert (ROOT / "bench" / "arms" / f"{arm}.py").exists()
        cmd = run.command(arm, Path("/in.jsonl"), Path("/out.jsonl"))
        assert cmd[0] == str(ROOT / venv / "bin" / "python") and cmd[2] == f"bench.arms.{arm}"


def test_committed_span_files_hold_offsets_only():
    files = sorted((ROOT / "bench" / "results" / "spans").glob("*/*.jsonl"))
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            assert set(r) - {"refused"} == {"id", "edits", "ms"} and r.get("refused", 1) == 1
            for e in r["edits"]:
                assert len(e) == 5 and all(isinstance(x, int) for x in (e[0], e[1], e[3], e[4]))
                # a type, or a canonical view's label ("age:worded"); never text
                assert re.fullmatch(r"[A-Za-z_' ]{1,40}(?::[a-z_]{1,20})?", e[2]), (f.name, e[2])


def test_gliner_windows_cover_every_word_with_overlap():
    short = "a few words only"
    assert windows(short) == [(0, len(short))]
    words = [f"w{i}" for i in range(450)]
    text = "  ".join(words)
    ws = windows(text, size=200, overlap=30)
    assert ws[0][0] == 0 and ws[-1][1] == len(text)
    covered = set()
    for s, e in ws:
        covered |= set(re.findall(r"w\d+", text[s:e]))
        assert len(text[s:e].split()) <= 200
    assert covered == set(words)
    assert all(ws[i + 1][0] < ws[i][1] for i in range(len(ws) - 1))   # consecutive windows overlap


def test_shape_like_keeps_shape_and_is_seeded():
    v = "AB-12cd 9"
    a, b = shape_like(v, random.Random(1)), shape_like(v, random.Random(1))
    assert a == b and len(a) == len(v)
    for x, y in zip(v, a):
        assert (x.isdigit(), x.isupper(), x.islower()) == (y.isdigit(), y.isupper(), y.islower())
        if not x.isalnum():
            assert x == y


def test_presidio_faker_types_cover_presidio_defaults():
    assert {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "CREDIT_CARD", "US_SSN", "IP_ADDRESS"} <= set(generators(None))


def test_fix_faker_arity_wraps_only_broken_entries():
    class Fake:
        def method(self, kind=None):
            return "m"
    fmap = {"OK": Fake().method, "BROKEN": lambda _: "b", "NESTED": lambda _: lambda _: "n", "PLAIN": lambda: "p"}
    assert fix_faker_arity(fmap) == ["BROKEN", "NESTED"]
    assert {k: f() for k, f in fmap.items()} == {"OK": "m", "BROKEN": "b", "NESTED": "n", "PLAIN": "p"}
    assert fix_faker_arity(fmap) == []


def test_record_arms_writes_configs_and_refuses_disagreeing_runs(tmp_path):
    from bench.arms import run
    from bench.realdata import manifest
    pub, mpath = tmp_path / "spans", tmp_path / "manifest.json"
    for name in ("natural-a", "natural-b"):
        (pub / "ss").mkdir(parents=True, exist_ok=True)
        (pub / "ss" / f"{name}.jsonl.meta.json").write_text(json.dumps(
            {"arm": "ss", "config": {"versions": {"surrogateshield": "2.0"}}, "seed": 1}))
    arms = run.record_arms(pub, mpath)
    assert arms == {"ss": {"config": {"versions": {"surrogateshield": "2.0"}}, "interpreter": ".venv", "seed": 1}}
    assert manifest.load(mpath)["arms"] == arms
    assert "### ss" in manifest.render(manifest.load(mpath))
    (pub / "ss" / "natural-b.jsonl.meta.json").write_text(json.dumps(
        {"arm": "ss", "config": {"versions": {"surrogateshield": "2.1"}}, "seed": 1}))
    with pytest.raises(SystemExit):
        run.record_arms(pub, mpath)


def test_natural_inputs_per_collection():
    assert run.natural() == run.NATURAL
    t2 = run.natural("test2")
    assert sorted(t2) == ["test2-natural-oasst1", "test2-natural-sharegpt", "test2-natural-wildchat"]
    assert t2["test2-natural-oasst1"].parts[-4:] == ("build", "test2", "oasst1", "messages.jsonl")


def test_record_arms_lists_historical_span_files_apart(tmp_path):
    from bench.arms import run
    from bench.realdata import manifest
    pub, mpath = tmp_path / "spans", tmp_path / "manifest.json"
    (pub / "lg").mkdir(parents=True)
    for name, dev in (("natural-a", None), ("natural-b", None), ("dev-a", "cpu"), ("test2-natural-a", "cpu")):
        (pub / "lg" / f"{name}.jsonl.meta.json").write_text(json.dumps(
            {"arm": "lg", "config": {"device": dev}, "seed": 1}))
    hist = {("lg", "natural-a"): "older run", ("lg", "natural-b"): "older run"}
    arms = run.record_arms(pub, mpath, hist)
    assert arms["lg"]["config"] == {"device": "cpu"}
    assert arms["lg"]["historical"] == [{"config": {"device": None}, "interpreter": None, "seed": 1,
                                         "why": "older run", "files": ["natural-a", "natural-b"]}]
    assert "Historical span files (`natural-a`, `natural-b`): older run." in manifest.render(manifest.load(mpath))
    with pytest.raises(SystemExit):                     # without the exception it still refuses
        run.record_arms(pub, mpath, {})


def test_an_arm_may_keep_several_historical_configurations_one_per_reason(tmp_path):
    from bench.arms import run
    from bench.realdata import manifest
    pub, mpath = tmp_path / "spans", tmp_path / "manifest.json"
    (pub / "ss").mkdir(parents=True)
    for name, v in (("dev-a", 1), ("natural-a", 1), ("devlarge-a", 2)):
        (pub / "ss" / f"{name}.jsonl.meta.json").write_text(json.dumps({"arm": "ss", "config": {"v": v}, "seed": 1}))
    hist = {("ss", "dev-a"): "v1 era", ("ss", "natural-a"): "v1 era", ("ss", "devlarge-a"): "v2 era"}
    arms = run.record_arms(pub, mpath, hist)
    assert arms["ss"]["config"] is None
    assert [(h["why"], h["config"], h["files"]) for h in arms["ss"]["historical"]] == [
        ("v1 era", {"v": 1}, ["dev-a", "natural-a"]), ("v2 era", {"v": 2}, ["devlarge-a"])]
    md = manifest.render(manifest.load(mpath))
    assert "No current span file" in md and "`devlarge-a`): v2 era." in md
    (pub / "ss" / "test2-a.jsonl.meta.json").write_text(json.dumps({"arm": "ss", "config": {"v": 3}, "seed": 1}))
    assert run.record_arms(pub, mpath, hist)["ss"]["config"] == {"v": 3}
    with pytest.raises(SystemExit, match="same historical reason"):
        run.record_arms(pub, mpath, {**hist, ("ss", "devlarge-a"): "v1 era"})


def test_ss_merges_a_config_file_on_the_benchmark_config(tmp_path, monkeypatch):
    # V3 §3.7: a configuration is measured through the product's config file
    # alone; without the file the arm's config is exactly the benchmark's
    from bench.arms import ss
    from surrogateshield.core.detection.config import ENV_FILE, ENV_PRESET, benchmark
    monkeypatch.delenv(ENV_FILE, raising=False)
    monkeypatch.delenv(ENV_PRESET, raising=False)
    assert ss.detection_config() == benchmark()
    assert "detection_config_file" not in ss.config()
    f = tmp_path / "cg.json"
    f.write_text(json.dumps({"detectors": [{"name": "context_guard", "model": "m/x",
                                            "options": {"labels": {"FIRSTNAME": "PERSON"}}}]}))
    monkeypatch.setenv(ENV_FILE, str(f))
    cfg = ss.detection_config()
    assert cfg.stage("context_guard").model == "m/x" and cfg.address_mode == benchmark().address_mode
    rec = ss.config()
    assert rec["detection_config_file"]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert rec["detection_config_hash"] != benchmark().config_hash()
    monkeypatch.setenv(ENV_PRESET, "strict")
    with pytest.raises(SystemExit, match="replace the benchmark"):
        ss.detection_config()
