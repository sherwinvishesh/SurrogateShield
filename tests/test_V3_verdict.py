"""bench.realdata.verdict: the GO rule of HYPOTHESES_TEST2.md over stub result
files (no model, no data; every number below is made up)."""

import json

import pytest

from bench.realdata import verdict as V
from bench.realdata.common import file_sha256

DS = ("oasst1", "sharegpt", "wildchat")
GROUPS = (*DS, "all")
ARMS = ("ss", *V.SIX)


def r(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4)}


def d(diff, lo, hi):
    return {"diff": diff, "ci95": [lo, hi], "excludes_0": lo > 0 or hi < 0, "resamples": 2000}


def e5_doc():
    results, diffs = {}, {}
    for g in GROUPS:
        results[g], diffs[g] = {}, {}
        for sl in ("injected", "natural", "shift"):
            results[g][sl] = {a: {"leak": r(2 if a == "ss" else 20, 100), "protect_values": 100,
                                  "negatives_untouched": r(90 if a == "ss" else 80, 100),
                                  "by_type": {"EMAIL": {"values": 10, "policy": 0, "leaked": 0}}} for a in ARMS}
            diffs[g][sl] = {a: {"leak_rate": d(-0.18, -0.25, -0.1), "spurious_rate": d(-0.1, -0.2, -0.05)}
                            for a in V.SIX}
    by_type = {"all": {a: {"EMAIL": d(-0.1, -0.2, -0.01), "PERSON": d(0.01, 0.002, 0.02)} for a in V.SIX}}
    return {"split": "test2", "results": results, "differences": diffs, "differences_by_type": {"groups": by_type}}


def docs():
    e5 = e5_doc()
    comp = {g: {"injected": {"balanced": {"leak": r(2, 100), "protect_values": 100},
                             "no_tagger": {"leak": r(10, 100), "protect_values": 100},
                             V.NO_PATTERNS: {"leak": r(15, 100), "protect_values": 100},
                             V.UNROUTED: {"leak": r(30, 100), "protect_values": 100}}} for g in GROUPS}
    row = lambda arm, p50, rss: {"arm": arm, "p50_ms": p50, "p95_ms": 2 * p50, "peak_rss_mb": rss,
                                 "model_load_s": 1.0, "cold_start_s": 2.0}
    h8 = {"no_exact_single": True, "no_exact_conversation": True, "recovery_not_above_gliner_pii": True,
          "holds": True, "recovery": {"ss": r(1, 100), "gliner_pii": r(5, 100), "ss-gliner_pii": d(-0.04, -0.08, 0)}}
    ext = {g: {"leak_not_above": True, "span_f1_not_below": False, "holds": True} for g in ("gliner_pii", V.TUNED)}
    return {"e5": e5, "components": {"split": "test2", "results": comp},
            "perf": {"detection_config": None, "arms": [row("ss", 40.0, 1500.0), row("gliner_pii", 80.0, 2500.0)]},
            "utility": {"split": "test2", "e5b": {g: {"H7''": {"holds": True}} for g in GROUPS}},
            "multiturn": {"split": "test2", "results": {g: {"H7''": {"holds": True}} for g in GROUPS}},
            "attacker": {"split": "test2", "results": {g: {"H8''": dict(h8)} for g in GROUPS}},
            "external": {"hypotheses": {"H10''": ext}}}


def test_every_hypothesis_holding_is_go():
    v = V.verdict(docs())
    assert v["verdict"] == "GO" and v["not_run"] == []
    assert {k: x["holds"] for k, x in v["hypotheses"].items()} == {f"H{i}''": True for i in range(1, 11)}
    assert v["hypotheses"]["H1''"]["datasets_held"] == 3


def test_h1_failing_on_one_dataset_but_not_worse_than_either_gliner_arm_is_go_with_caveat():
    x = docs()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    v = V.verdict(x)
    assert v["verdict"] == "GO-with-caveat" and "sharegpt" in v["caveat"]
    h1 = v["hypotheses"]["H1''"]["per_dataset"]["sharegpt"]
    assert not h1["holds"] and not h1["per_baseline"]["gliner_pii"]["holds"] and h1["per_baseline"]["llm_guard"]["holds"]


def test_go_with_caveat_needs_the_third_dataset_not_worse_than_the_tuned_arm_too():
    x = docs()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    x["e5"]["differences"]["sharegpt"]["injected"][V.TUNED]["leak_rate"] = d(0.05, 0.01, 0.09)
    v = V.verdict(x)
    assert v["verdict"] == "NO-GO" and v["failed"] == ["H1''"]
    assert v["hypotheses"]["H1''"]["per_dataset"]["sharegpt"]["not_worse"] == {"gliner_pii": True, V.TUNED: False}


def test_go_with_caveat_needs_every_other_go_condition():
    x = docs()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    x["e5"]["differences"]["oasst1"]["injected"]["presidio_faker"]["spurious_rate"] = d(-0.01, -0.03, 0.01)
    v = V.verdict(x)
    assert v["verdict"] == "NO-GO" and v["failed"] == ["H1''", "H2''"]


def test_h1_on_one_dataset_only_is_no_go():
    x = docs()
    for ds in ("sharegpt", "wildchat"):
        x["e5"]["differences"][ds]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    assert V.verdict(x)["verdict"] == "NO-GO"


def test_h1_and_h2_ignore_a_baseline_the_hypothesis_does_not_name():
    x = docs()
    for ds in DS:   # H2'' names gliner_pii, presidio_default and presidio_faker only
        x["e5"]["differences"][ds]["injected"]["llm_guard"]["spurious_rate"] = d(0.1, 0.05, 0.2)
        x["e5"]["differences"][ds]["injected"][V.TUNED]["spurious_rate"] = d(0.1, 0.05, 0.2)
    v = V.verdict(x)
    assert v["verdict"] == "GO"
    assert v["hypotheses"]["H2''"]["per_dataset"]["oasst1"]["beside"][V.TUNED]["lower"] is False


def test_h3_a_tie_with_any_baseline_fails():
    x = docs()
    x["e5"]["results"]["oasst1"]["natural"]["llm_guard"]["negatives_untouched"] = r(90, 100)
    v = V.verdict(x)
    assert v["verdict"] == "NO-GO" and v["failed"] == ["H3''"]
    assert "oasst1: fails (llm_guard)" in V._detail("H3''", v["hypotheses"]["H3''"])


@pytest.mark.parametrize("diff, lo, fails", [(0.03, 0.001, True), (0.03, -0.01, False), (0.019, 0.005, False),
                                             (0.021, 0.0, False)])
def test_h5_a_type_fails_above_the_margin_with_its_interval_above_zero(diff, lo, fails):
    x = docs()
    x["e5"]["differences_by_type"]["groups"]["all"]["gliner_pii"]["PERSON"] = d(diff, lo, diff + 0.02)
    h = V.verdict(x)["hypotheses"]["H5''"]
    assert h["failing_types"] == (["PERSON"] if fails else []) and h["holds"] is (not fails)


def test_h5_the_tuned_arm_is_reported_beside():
    x = docs()
    x["e5"]["differences_by_type"]["groups"]["all"][V.TUNED]["PERSON"] = d(0.05, 0.01, 0.1)
    v = V.verdict(x)
    assert v["verdict"] == "GO" and v["hypotheses"]["H5''"]["beside"][V.TUNED]["failing_types"] == ["PERSON"]


def test_h5_without_per_type_differences_is_not_run():
    x = docs()
    del x["e5"]["differences_by_type"]
    v = V.verdict(x)
    assert v["hypotheses"]["H5''"]["holds"] is None and v["verdict"] == "incomplete" and v["missing"] == ["H5''"]


@pytest.mark.parametrize("change", ["p50", "rss", "no_record", "partial_config"])
def test_h6_needs_the_default_pipeline_at_most_gliner_on_both(change):
    x = docs()
    ss = x["perf"]["arms"][0]
    if change == "p50":
        ss["p50_ms"] = 80.1
    elif change == "rss":
        ss["peak_rss_mb"] = 2500.5
    elif change == "no_record":
        del x["perf"]["detection_config"]
    else:
        x["perf"]["detection_config"] = {"path": "x.json", "sha256": "0" * 64}
    v = V.verdict(x)
    assert v["hypotheses"]["H6''"]["holds"] is False and v["verdict"] == "NO-GO" and v["failed"] == ["H6''"]


def test_h6_equal_is_not_above():
    x = docs()
    x["perf"]["arms"][0].update(p50_ms=80.0, peak_rss_mb=2500.0)
    assert V.verdict(x)["hypotheses"]["H6''"]["holds"] is True


def test_h7_needs_both_live_runs_and_does_not_decide():
    x = docs()
    x["multiturn"]["results"]["all"]["H7''"]["holds"] = False
    v = V.verdict(x)
    assert v["hypotheses"]["H7''"]["holds"] is False and v["verdict"] == "GO"


def test_h8_is_read_pooled():
    x = docs()
    x["attacker"]["results"]["all"]["H8''"].update(no_exact_single=False, holds=False)
    x["attacker"]["results"]["oasst1"]["H8''"]["holds"] = True
    v = V.verdict(x)
    assert v["verdict"] == "NO-GO" and v["failed"] == ["H8''"]
    assert v["hypotheses"]["H8''"]["per_dataset"]["oasst1"] is True


def test_h9_both_ablations_below_presidio_default_pooled_and_unrouted_beside():
    x = docs()
    h = V.verdict(x)["hypotheses"]["H9''"]
    assert h["a"] and h["b"] and h["holds"]
    assert h["per_group"]["all"]["beside_unrouted"] == {"leak": r(30, 100), "below": False}
    x["components"]["results"]["all"]["injected"][V.NO_PATTERNS]["leak"] = r(20, 100)    # a tie with Presidio
    v = V.verdict(x)
    assert v["hypotheses"]["H9''"]["a"] is False and v["hypotheses"]["H9''"]["holds"] is False
    assert v["verdict"] == "GO"                                                         # H9'' is not in the rule


def test_h10_decides_on_gliner_pii_at_its_default():
    x = docs()
    x["external"]["hypotheses"]["H10''"][V.TUNED]["holds"] = False
    assert V.verdict(x)["hypotheses"]["H10''"]["holds"] is True


def test_a_missing_input_leaves_its_hypotheses_not_run():
    x = docs()
    x["attacker"] = None
    v = V.verdict(x)
    assert v["verdict"] == "incomplete" and v["missing"] == ["H8''"] and v["not_run"] == ["H8''"]
    x = docs()
    x["utility"] = x["external"] = None
    v = V.verdict(x)
    assert v["verdict"] == "GO" and v["not_run"] == ["H7''", "H10''"]
    x = docs()
    x["e5"] = None
    v = V.verdict(x)
    assert v["verdict"] == "incomplete" and v["not_run"] == ["H1''", "H2''", "H3''", "H4''", "H5''", "H9''"]


def _write(results, x, sealed):
    for key, name in V.INPUTS.items():
        if x.get(key) is not None:
            doc = dict(x[key], **({"freeze_sha256": sealed} if key in V.SEALED else {}))
            (results / name).write_text(json.dumps(doc))


def test_load_checks_the_freeze_and_the_split(tmp_path):
    freeze = tmp_path / "FREEZE.json"
    freeze.write_text('{"frozen": true}\n')
    x = docs()
    _write(tmp_path, x, file_sha256(freeze))
    got = V.load(tmp_path, freeze)
    assert set(got) == set(V.INPUTS) and all(got.values())
    assert got["e5"]["_input"]["file"] == "bench/results/realdata_test2.json"
    (tmp_path / "attacker_realdata_test2.json").unlink()
    assert V.load(tmp_path, freeze)["attacker"] is None
    (tmp_path / "utility_realdata_test2.json").write_text(json.dumps({**x["utility"], "split": "test"}))
    with pytest.raises(SystemExit, match="not test2"):
        V.load(tmp_path, freeze)
    _write(tmp_path, x, "0" * 64)
    with pytest.raises(SystemExit, match="not a test-2 result"):
        V.load(tmp_path, freeze)
    _write(tmp_path, x, file_sha256(freeze))
    freeze.unlink()
    with pytest.raises(SystemExit, match="not a test-2 result"):
        V.load(tmp_path, freeze)


def test_run_writes_the_verdict_and_its_table(tmp_path):
    freeze = tmp_path / "FREEZE.json"
    freeze.write_text('{"frozen": true}\n')
    x = docs()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    _write(tmp_path, x, file_sha256(freeze))
    out = tmp_path / "out" / "verdict_test2.json"
    doc = V.run(out, tmp_path, freeze)
    back = json.loads(out.read_text())
    assert back["verdict"] == doc["verdict"] == "GO-with-caveat" and back["freeze_sha256"] == file_sha256(freeze)
    md = out.with_suffix(".md").read_text()
    assert md.startswith("# Test-2 verdict: GO-with-caveat\n")
    assert "Caveat: H1'' fails on sharegpt" in md
    assert sum(line.startswith("| H") for line in md.splitlines()) == 10
    assert "| H6'' | yes | holds | p50 40.0 vs 80.0 ms; peak RSS 1500.0 vs 2500.0 MB |" in md


# ── test-3 (--split test3): the same rule, test-3's files, H1''' ... H10''' ──

P3 = "'" * 3


def docs3():
    """docs() as test-3's files carry them: split test3, the live and external
    blocks named H7''', H8''', H10''', the external benchmark at test-3."""
    x = docs()
    for k in ("e5", "components", "utility", "multiturn", "attacker"):
        x[k]["split"] = "test3"
    x["utility"]["e5b"] = {g: {f"H7{P3}": v["H7''"]} for g, v in x["utility"]["e5b"].items()}
    x["multiturn"]["results"] = {g: {f"H7{P3}": v["H7''"]} for g, v in x["multiturn"]["results"].items()}
    x["attacker"]["results"] = {g: {f"H8{P3}": v["H8''"]} for g, v in x["attacker"]["results"].items()}
    x["external"] = {"collection": "test3", "hypotheses": {f"H10{P3}": x["external"]["hypotheses"]["H10''"]}}
    return x


def test_test3_reads_its_own_blocks_and_names_every_hypothesis_with_three_primes():
    v = V.verdict(docs3(), split="test3")
    assert v["verdict"] == "GO" and v["not_run"] == []
    assert {k: x["holds"] for k, x in v["hypotheses"].items()} == {f"H{i}{P3}": True for i in range(1, 11)}
    assert V.SPLITS["test3"]["inputs"]["perf"] == "perf_arms_v4.json"
    assert set(V.SPLITS["test3"]["inputs"].values()) - {"external_nemotron_pii.json", "perf_arms_v4.json"} == {
        f"{n}_test3.json" for n in ("realdata", "realdata_components")} | {
        f"{n}_realdata_test3.json" for n in ("utility", "multiturn", "attacker")}


def test_test3_applies_test2s_rule_unchanged():
    x = docs3()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    v = V.verdict(x, split="test3")
    assert v["verdict"] == "GO-with-caveat" and f"H1{P3}" in v["caveat"] and "H1'' " not in v["caveat"]
    x["attacker"]["results"]["all"][f"H8{P3}"].update(no_exact_single=False, holds=False)
    v = V.verdict(x, split="test3")
    assert v["verdict"] == "NO-GO" and v["failed"] == [f"H1{P3}", f"H8{P3}"]     # the caveat needs every other part
    x["attacker"] = None
    v = V.verdict(x, split="test3")
    assert v["missing"] == [f"H8{P3}"] and v["not_run"] == [f"H8{P3}"]
    x = docs3()
    x["e5"]["differences_by_type"]["groups"]["all"]["gliner_pii"]["PERSON"] = d(0.03, 0.001, 0.05)
    v = V.verdict(x, split="test3")
    assert v["failed"] == [f"H5{P3}"] and v["hypotheses"][f"H5{P3}"]["failing_types"] == ["PERSON"]


def _write3(results, x, sealed):
    for key, name in V.SPLITS["test3"]["inputs"].items():
        if x.get(key) is not None:
            doc = dict(x[key], **({"freeze_sha256": sealed} if key in V.SPLITS["test3"]["sealed"] else {}))
            (results / name).write_text(json.dumps(doc))


def test_test3_load_needs_test3s_freeze_on_e5_components_and_the_external_benchmark(tmp_path):
    freeze = tmp_path / "test3" / "FREEZE.json"
    freeze.parent.mkdir()
    freeze.write_text('{"frozen": "test-3"}\n')
    x = docs3()
    _write3(tmp_path, x, file_sha256(freeze))
    got = V.load(tmp_path, freeze, "test3")
    assert set(got) == set(V.SPLITS["test3"]["inputs"]) and all(got.values())
    assert got["e5"]["_input"]["file"] == "bench/results/realdata_test3.json"
    (tmp_path / "external_nemotron_pii.json").write_text(json.dumps({**x["external"], "freeze_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="external_nemotron_pii.json: not a test-3 result"):
        V.load(tmp_path, freeze, "test3")                         # scored at another freeze (test-2's)
    (tmp_path / "external_nemotron_pii.json").write_text(
        json.dumps({**x["external"], "collection": "test2", "freeze_sha256": file_sha256(freeze)}))
    with pytest.raises(SystemExit, match="not a test-3 result"):
        V.load(tmp_path, freeze, "test3")
    _write3(tmp_path, x, file_sha256(freeze))
    (tmp_path / "utility_realdata_test3.json").write_text(json.dumps({**x["utility"], "split": "test2"}))
    with pytest.raises(SystemExit, match="not test3"):
        V.load(tmp_path, freeze, "test3")
    _write3(tmp_path, x, file_sha256(freeze))
    (tmp_path / "realdata_test3.json").write_text(json.dumps({**x["e5"], "split": "test2",
                                                              "freeze_sha256": file_sha256(freeze)}))
    with pytest.raises(SystemExit, match="realdata_test3.json: not a test-3 result"):
        V.load(tmp_path, freeze, "test3")


def test_test3_run_writes_its_own_verdict_and_table(tmp_path):
    freeze = tmp_path / "test3" / "FREEZE.json"
    freeze.parent.mkdir()
    freeze.write_text('{"frozen": "test-3"}\n')
    x = docs3()
    x["e5"]["differences"]["sharegpt"]["injected"]["gliner_pii"]["leak_rate"] = d(-0.03, -0.06, 0.01)
    _write3(tmp_path, x, file_sha256(freeze))
    out = tmp_path / "out" / "verdict_test3.json"
    doc = V.run(out, tmp_path, freeze, "test3")
    back = json.loads(out.read_text())
    assert back["verdict"] == doc["verdict"] == "GO-with-caveat" and back["split"] == "test3"
    assert back["freeze_sha256"] == file_sha256(freeze) and back["prereg"]["file"].endswith("HYPOTHESES_TEST3.md")
    assert " -m bench.realdata.verdict --split test3 --out " in back["command"]
    md = out.with_suffix(".md").read_text()
    assert md.startswith("# Test-3 verdict: GO-with-caveat\n")
    assert f"Caveat: H1{P3} fails on sharegpt" in md
    assert sum(line.startswith("| H") for line in md.splitlines()) == 10
    assert f"| H6{P3} | yes | holds | p50 40.0 vs 80.0 ms; peak RSS 1500.0 vs 2500.0 MB |" in md
    assert f"| H7{P3} | no | holds |" in md and f"| H8{P3} | yes | holds |" in md
