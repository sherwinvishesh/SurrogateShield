"""V3 §3.6 / E9 — the presets and component ablations runner
(``bench/realdata/components.py``): its committed config files, the paired
differences from ``balanced``, the reuse of a run, and an end-to-end run with a
stub SS. Model-free; every text and value is invented."""

import json

import pytest

from bench.realdata import components as K
from bench.realdata import score
from bench.realdata.common import ROOT
from surrogateshield.core.detection import config as C


def _unit(mid, text, protect=(), slices=("injected", "injected_single"), ds="oasst1"):
    gold = {"id": mid, "text": text, "service_query": False, "protect": [dict(p) for p in protect],
            "sensitive": [], "optional": [], "keep": []}
    return {"mid": mid, "conv": mid, "dataset": ds, "task": "advice", "slices": list(slices), "turn": 0, "gold": gold}


def _units(ds):
    text = "Write to Ada Byrne at ada@example.org about the report."
    out = [_unit(f"{ds}-i{i}", text, [{"value": "Ada Byrne", "type": "PERSON"},
                                      {"value": "ada@example.org", "type": "EMAIL"}], ds=ds) for i in range(4)]
    out += [_unit(f"{ds}-n{i}", "How do I sort a list in Python?", slices=("natural",), ds=ds) for i in range(3)]
    return out


def _edits(unit, leak_person=False, spurious=False):
    text, edits = unit["gold"]["text"], []
    for p in unit["gold"]["protect"]:
        if leak_person and p["type"] == "PERSON":
            continue
        a = text.index(p["value"])
        edits.append([a, a + len(p["value"]), p["type"], "x"])
    if spurious and not unit["gold"]["protect"]:
        a = text.index("Python")
        edits.append([a, a + 6, "ORG", "x"])
    return sorted(edits)


def test_each_committed_config_is_its_preset_with_every_address_redrawn():
    assert sorted(p.stem for p in K.CONFIGS.glob("*.json")) == sorted(K.NAMES)
    for n in K.NAMES:
        assert json.loads((K.CONFIGS / f"{n}.json").read_text()) == {"preset": n, "type_actions": {"ADDRESS": "replace"}}
        info = K.config_info(n)
        assert info["preset"] == n and info["file"] == f"bench/realdata/configs/{n}.json"
    assert K.config_info("balanced")["detection_config_hash"] == C.benchmark().config_hash()
    # classic is the SS the tagger is measured against (bench/tagger/configs/ss-v2.json) but for its name
    v2 = C.from_partial(json.loads((ROOT / "bench" / "tagger" / "configs" / "ss-v2.json").read_text()), C.benchmark())
    classic = C.from_partial(json.loads((K.CONFIGS / "classic.json").read_text()), C.benchmark())
    assert {**v2.to_dict(), "preset": "classic"} == classic.to_dict()


@pytest.mark.parametrize("name", ["h9/no_patterns", "h9/no_patterns_unrouted"])
def test_the_h9_configs_are_balanced_with_every_patternscan_pattern_off(name):
    assert K.RUN == (*K.NAMES, "h9/no_patterns", "h9/no_patterns_unrouted")
    assert sorted(str(p.relative_to(K.CONFIGS)) for p in K.CONFIGS.glob("h9/*.json")) == \
        ["h9/no_patterns.json", "h9/no_patterns_unrouted.json"]
    info = K.config_info(name)
    assert info["preset"] == "balanced" and info["file"] == f"bench/realdata/configs/{name}.json"
    off = C.from_partial(json.loads((K.CONFIGS / f"{name}.json").read_text()), C.benchmark()).to_dict()
    on = C.benchmark().to_dict()
    stages = lambda d: {s["name"]: s for s in d["detectors"]}
    assert {n for n, s in stages(off).items() if not s["enabled"]} == \
        {n for n, s in stages(on).items() if not s["enabled"]} | {"pattern_scan", "canonicaliser"}
    for n, s in stages(on).items():
        assert stages(off)[n] == (s if n not in ("pattern_scan", "canonicaliser") else {**s, "enabled": False})
    routed = {t: [*v, "pii_tagger"] for t, v in on["type_sources"].items()
              if t in ("AGE", "DATE_OF_BIRTH", "EMAIL", "NETWORK", "URL")}
    assert len(routed) == 5 and all("pii_tagger" not in on["type_sources"][t] for t in routed)
    assert off["type_sources"] == {**on["type_sources"], **(routed if name == "h9/no_patterns" else {})}
    assert {**off, "detectors": None, "type_sources": None} == {**on, "detectors": None, "type_sources": None}


def test_differences_are_each_arm_minus_balanced_per_group_and_slice():
    units = {ds: _units(ds) for ds in ("oasst1", "wildchat")}
    rows = {"balanced": {}, "no_tagger": {}, "gliner_pii": {}}
    for ds, us in units.items():
        rows["balanced"][ds] = [_edits(u) for u in us]
        rows["no_tagger"][ds] = [_edits(u, leak_person=True) for u in us]
        rows["gliner_pii"][ds] = [_edits(u, spurious=True) for u in us]
    scores = {a: {ds: [score.score_unit(u, {"edits": e}) for u, e in zip(units[ds], r[ds])] for ds in r}
              for a, r in rows.items()}
    t = K.tables(units, scores, "dev")
    assert list(t["results"]) == ["oasst1", "wildchat", "all"]
    assert t["results"]["all"]["injected"]["no_tagger"]["leak"] == score.rate(8, 16)
    assert t["results"]["all"]["injected"]["no_tagger"]["leaked_by_type"] == {"EMAIL": 0, "PERSON": 8}
    assert t["results"]["all"]["natural"]["gliner_pii"]["spurious"]["k"] == 6
    d = t["differences"]["all"]
    assert "balanced" not in d["injected"] and "shift" not in d
    assert d["injected"]["no_tagger"]["leak_rate"]["diff"] == 0.5
    assert d["injected"]["gliner_pii"]["leak_rate"]["diff"] == 0.0
    assert d["natural"]["gliner_pii"]["spurious_rate"]["diff"] is None      # balanced made no natural edit
    with pytest.raises(SystemExit, match="'balanced', which was not run"):
        K.tables(units, {"no_tagger": scores["no_tagger"]}, "dev")


def _stub(hash_of=None, calls=None, ms=2.0):
    """An SS run: no edit, the config hash of *hash_of* (else of the file it is given)."""
    def run(cfg, src, out):
        calls is not None and calls.append(cfg.name)
        h = hash_of or C.from_partial(json.loads(cfg.read_text()), C.benchmark()).config_hash()
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in src.read_text().splitlines()]
        out.write_text("".join(json.dumps({"id": r["id"], "edits": [], "ms": ms}) + "\n" for r in rows))
        meta = {"config": {"detection_config_hash": h}, "input_sha256": score.file_sha256(src)}
        out.with_name(out.name + ".meta.json").write_text(json.dumps(meta))
    return run


def _loaded(tmp_path, datasets=("oasst1",)):
    out = {}
    for ds in datasets:
        us = _units(ds)
        src = tmp_path / "in" / f"{ds}.jsonl"
        out[ds] = {"units": us, "src": src, "input_sha": score.write_input(us, src), "corpus": {"messages": len(us)}}
    return out


def test_a_run_is_reused_only_on_the_same_code_from_the_same_file(tmp_path):
    v = _loaded(tmp_path)["oasst1"]
    info, calls = K.config_info("no_gate"), []
    go = lambda stamp, inf=info, reuse=True: K.run_config("no_gate", inf, "dev", "oasst1", v, stamp, reuse,
                                                          tmp_path / "spans", _stub(calls=calls), log=lambda *_: None)
    assert len(go("c0")) == len(v["units"]) and calls == ["no_gate.json"]
    go("c0")
    assert len(calls) == 1                                  # same code, same file: kept
    go("c1")
    go("c1", {**info, "sha256": "other"})
    go("c1", {**info, "sha256": "other"}, reuse=False)
    assert len(calls) == 4
    with pytest.raises(SystemExit, match="made with config 0000"):
        K.run_config("no_gate", info, "dev", "oasst1", v, "c2", True, tmp_path / "spans", _stub(hash_of="0" * 64),
                     log=lambda *_: None)


def test_an_end_to_end_run_writes_the_table_and_its_command(tmp_path):
    out = tmp_path / "presets_dev.json"
    doc = K.components("dev", out, ("balanced", "no_gate"), reference=(), spans=tmp_path / "spans",
                       runner=_stub(), loaded=_loaded(tmp_path, ("oasst1", "sharegpt")), log=lambda *_: None)
    assert doc["command"].endswith(f"--split dev --configs balanced no_gate --out {out}")
    assert doc["configs"]["no_gate"]["detection_config_hash"] == K.config_info("no_gate")["detection_config_hash"]
    assert doc["configs"]["balanced"]["in_run_p50_ms"] == 2.0 and doc["base"] == "balanced"
    assert doc["results"]["all"]["injected"]["balanced"]["leak"]["k"] == 16
    assert json.loads(out.read_text()) == doc
    md = out.with_suffix(".md").read_text()
    assert "| `no_gate` |" in md and "Ada" not in md and "example.org" not in md
