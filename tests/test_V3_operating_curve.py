"""V3 §3.9 / E10 — the operating curve (``bench/realdata/operating_curve.py``):
its committed SS points, the dominance check, GLiNER-PII's points from one
scored run, and an end-to-end run with stub arms. Model-free; every text and
value is invented."""

import json

import pytest

from bench.realdata import components as K
from bench.realdata import operating_curve as O
from bench.realdata import score
from surrogateshield.core.detection import config as C
from tests.test_V3_components import _edits, _loaded, _stub


def _cfg(name):
    return C.from_partial(json.loads((K.CONFIGS / f"{name}.json").read_text()), C.benchmark())


def test_the_ss_points_move_only_the_model_thresholds_and_in_one_direction():
    assert sorted(f"curve/{p.stem}" for p in (K.CONFIGS / "curve").glob("*.json")) == sorted(
        n for n in O.OPS if n != O.DEFAULT)
    base = _cfg(O.DEFAULT).to_dict()
    tag, trace, above = [], [], []
    for n in O.OPS:
        c = _cfg(n)
        c.validate()
        d = c.to_dict()
        stages = {s["name"]: s for s in d.pop("detectors")}
        b_stages = {s["name"]: s for s in base["detectors"]}
        assert d == {k: v for k, v in base.items() if k != "detectors"}       # no other knob moves
        for name, s in stages.items():
            moved = {k for k in s if s[k] != b_stages[name][k]}
            assert moved <= {"pii_tagger": {"thresholds", "options"}, "entity_trace": {"thresholds"}}.get(name, set())
        assert set(stages["pii_tagger"]["options"]) == set(b_stages["pii_tagger"]["options"])
        tag.append(stages["pii_tagger"]["thresholds"])
        trace.append(stages["entity_trace"]["thresholds"])
        above.append(stages["pii_tagger"]["options"]["gate_above"])
    for seq in (tag, trace):
        for t in seq[0]:
            assert all(a[t] < b[t] for a, b in zip(seq, seq[1:])), t       # op1 protects most, op5 least
    assert above[-1] is None and all(a < b for a, b in zip(above[:-1], above[1:-1]))
    assert _cfg(O.DEFAULT).config_hash() == C.benchmark().config_hash()


def _p(leak, spurious, per_message):
    return {"leak": {"rate": leak}, "spurious": {"rate": spurious}, "spurious_per_message": {"value": per_message}}


def test_dominance_asks_for_an_ss_point_below_and_left_of_every_gliner_point():
    pts = {"ss@a": _p(0.01, 0.80, 0.9), "ss@b": _p(0.05, 0.50, 0.3),
           "g@1": _p(0.03, 0.85, 1.0), "g@2": _p(0.08, 0.60, 0.2)}
    d = O.dominance(pts, ["ss@a", "ss@b"], ["g@1", "g@2"], "ss@a")
    assert d["spurious"] == {"ss_dominates": True, "default_dominates": False,
                             "below_left_of": {"g@1": ["ss@a"], "g@2": ["ss@b"]}}
    assert d["spurious_per_message"]["ss_dominates"] is False                 # nothing under g@2's 0.2
    assert d["spurious_per_message"]["below_left_of"]["g@2"] == []
    tie = {"ss@a": _p(0.03, 0.5, 0.1), "g@1": _p(0.03, 0.9, 0.9)}
    assert O.dominance(tie, ["ss@a"], ["g@1"], "ss@a")["spurious"]["ss_dominates"] is False   # equal leak: not left
    none = {"ss@a": _p(None, 0.5, 0.1), "g@1": _p(0.03, 0.9, 0.9)}
    assert O.dominance(none, ["ss@a"], ["g@1"], "ss@a")["spurious"]["ss_dominates"] is False


def _gliner_files(v, private, sweep, run="dev", ds="oasst1", recorded_at=0.5):
    """A scored sweep at 0.3 (every protect value found at 0.6, a spurious
    span at 0.45 on natural messages) and the recorded gliner_pii arm."""
    rows = []
    for u in v["units"]:
        text, spans = u["gold"]["text"], []
        for p in u["gold"]["protect"]:
            a = text.index(p["value"])
            spans.append([a, a + len(p["value"]), p["type"].lower(), 0.6])
        if not u["gold"]["protect"]:
            a = text.index("Python")
            spans.append([a, a + 6, "organization", 0.45])
        rows.append({"id": u["mid"], "spans": spans})
    path = sweep / O.LABELS / f"{run}-{ds}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.with_name(path.name + ".meta.json").write_text(json.dumps({"input_sha256": v["input_sha"], "threshold": 0.3}))
    rec = private / "gliner_pii" / f"{run}-{ds}.jsonl"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text("".join(json.dumps({"id": r["id"], "edits": O.gliner_sweep.edits_at(r["spans"], recorded_at)})
                           + "\n" for r in rows))
    rec.with_name(rec.name + ".meta.json").write_text(json.dumps({"input_sha256": v["input_sha"]}))


def test_gliner_points_come_from_one_scored_run_checked_against_the_recorded_arm(tmp_path):
    v = _loaded(tmp_path)["oasst1"]
    _gliner_files(v, tmp_path / "private", tmp_path / "sweep")
    calls = []
    got = O.gliner_rows("dev", "oasst1", v, tmp_path / "sweep", tmp_path / "private", lambda *a: calls.append(a))
    assert calls == [] and list(got) == [O.gliner_label(t) for t in O.GLINER_AT]
    nat = [u["mid"] for u in v["units"] if "natural" in u["slices"]]
    assert all(len(got["gliner_pii@0.4"][m]["edits"]) == 1 for m in nat)    # the 0.45 span is kept at 0.4 ...
    assert all(not got["gliner_pii@0.5"][m]["edits"] for m in nat)          # ... and dropped at 0.5
    assert all(not r["edits"] for r in got["gliner_pii@0.6"].values())      # 0.6 is not above 0.6
    bad = tmp_path / "bad"
    _gliner_files(v, bad / "private", bad / "sweep", recorded_at=0.4)
    with pytest.raises(SystemExit, match="differs from the recorded gliner_pii arm on 3 messages"):
        O.gliner_rows("dev", "oasst1", v, bad / "sweep", bad / "private")


def _single(v, private, arm, run="dev", ds="oasst1"):
    rec = private / arm / f"{run}-{ds}.jsonl"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text("".join(json.dumps({"id": u["mid"], "edits": _edits(u, leak_person=True)}) + "\n"
                           for u in v["units"]))
    rec.with_name(rec.name + ".meta.json").write_text(json.dumps({"input_sha256": v["input_sha"]}))


def test_an_end_to_end_run_writes_every_point_and_its_command(tmp_path):
    loaded = _loaded(tmp_path, ("oasst1", "sharegpt"))
    private, sweep = tmp_path / "private", tmp_path / "sweep"
    for ds, v in loaded.items():
        _gliner_files(v, private, sweep, ds=ds)
        _single(v, private, "presidio_default", ds=ds)
    out = tmp_path / "operating_curve_dev.json"
    doc = O.operating_curve("dev", out, ("curve/op1", "balanced"), single=("presidio_default",),
                            spans=tmp_path / "spans", private=private, sweep=sweep, ss_runner=_stub(),
                            loaded=loaded, log=lambda *_: None)
    assert doc["command"].endswith(f"--split dev --ops curve/op1 balanced --out {out}")
    assert doc["ss_points"]["ss@op1"]["file"] == "bench/realdata/configs/curve/op1.json"
    assert doc["gliner_arms"] == {"gliner_pii": "gliner_pii@0.5", "gliner_pii_tuned": "gliner_pii@0.3"}
    pts = doc["points"]["all"]
    assert set(pts) == {"ss@op1", "ss@balanced", "presidio_default", *(O.gliner_label(t) for t in O.GLINER_AT)}
    assert pts["ss@balanced"]["leak"]["k"] == 16                       # the stub edits nothing
    assert pts["gliner_pii@0.3"]["leak"]["k"] == 0 and pts["gliner_pii@0.3"]["spurious"]["k"] == 6
    assert pts["gliner_pii@0.3"]["spurious_per_message"]["value"] == 1.0
    assert pts["presidio_default"]["leak"]["k"] == 8
    assert doc["dominance"]["all"]["spurious"]["ss_dominates"] is False
    assert doc["differences"]["all"]["gliner_pii@0.3"]["leak_rate"]["diff"] == 1.0
    assert doc["differences"]["all"]["gliner_pii@0.3"]["spurious_per_message"]["diff"] == -1.0
    assert json.loads(out.read_text()) == doc
    md = out.with_suffix(".md").read_text()
    assert "| `gliner_pii@0.9` |" in md and "Ada" not in md and "example.org" not in md
    with pytest.raises(SystemExit, match="default point"):
        O.operating_curve("dev", out, ("curve/op1",), loaded=loaded)
