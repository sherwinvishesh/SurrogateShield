"""V3 §3.3 — choosing the tagger's encoder and its learning curve
(``bench/tagger/select.py``), on hand-made evaluation results. Model-free."""

import hashlib
import json

import pytest

from bench.tagger import select as S


def _arm(k, spurious, n=100):
    return {"leak": {"k": k, "n": n, "rate": k / n, "wilson95": [0.0, 1.0]}, "leaked_by_type": {"ORG": k},
            "values_by_type": {"ORG": n}, "natural_spurious": spurious, "natural_edits": spurious + 5}


def _run(sha, k, spurious, code):
    """An ``evaluate --ss`` result: ORG's bound is GLiNER's 0.08; pooled 0.06; ss spurious 50."""
    return {"model": {"weights_sha256": sha}, "command": "... -m bench.tagger.evaluate", "git": {"commit": "x"},
            "with_ss": {"ss_tagger": _arm(k, spurious), "ss": _arm(2, 50), "gliner_pii": _arm(9, 400),
                        "gliner_pii_tuned": _arm(8, 500), "config": {"code": code}},
            "sweep": {t: _arm(k + 1, 2 * spurious) for t in ("0.3", "0.5", "0.9")}}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    models, ev = tmp_path / "models", tmp_path / "eval"
    (ev / "perf").mkdir(parents=True)
    monkeypatch.setattr(S, "MODELS", models)
    monkeypatch.setattr(S, "BUILD", ev)
    return models, ev


def _model(dirs, name, size, k, spurious, windows=40000, p50=40.0, rss=1200.0, code="c0", pin=True):
    models, ev = dirs
    (models / name).mkdir(parents=True)
    weights = name.encode() + bytes(size)                  # distinct hash, size ~ size
    (models / name / "model.safetensors").write_bytes(weights)
    sha = hashlib.sha256(weights).hexdigest()
    meta = {"encoder": name, "hub_id": f"org/{name}", "revision": "r1", "licence": "MIT", "data_sha256": "d",
            "windows": windows, "args": {"epochs": 3, "lr": 1e-4, "batch": 16, "seed": 1},
            "best_val_f1": 0.99, "minutes": 1.0, "device": "mps"}
    (models / name / "train_meta.json").write_text(json.dumps(meta))
    for split in ("devlarge-calib", "dev"):
        (ev / f"{name}-{split}-{S.VARIANT}.json").write_text(json.dumps(_run(sha, k, spurious, code)))
    cfg = ev / name / "ss-devlarge.config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"detectors": [{"name": "pii_tagger", "revision": f"sha256:{sha if pin else 0}"}]}))
    perf = {"detection_config": {"path": str(cfg), "sha256": "-"}, "machine": {"cpus": 8},
            "arms": [{"p50_ms": p50, "p95_ms": 2 * p50, "mean_ms": p50, "peak_rss_mb": rss, "cold_start_s": 5.0}]}
    (ev / "perf" / f"{name}-{S.VARIANT}.json").write_text(json.dumps(perf))


def _select(tmp_path, *args):
    out = tmp_path / "out.json"
    S.main([*args, "--out", str(out)])
    return json.loads(out.read_text())


def test_the_smallest_eligible_model_is_kept_when_no_larger_one_dominates(dirs, tmp_path):
    _model(dirs, "xs", 10, k=4, spurious=40)
    _model(dirs, "s", 20, k=3, spurious=45)               # one fewer leak, more spurious: not enough
    _model(dirs, "b", 30, k=1, spurious=30, rss=1700.0)   # dominates, but over the RSS budget
    _model(dirs, "m", 25, k=7, spurious=20)               # pooled 0.07 fails §3.3 on calib
    doc = _select(tmp_path, "--models", "xs", "s", "b", "m")
    assert doc["chosen"] == "xs" and doc["why"] == "smallest eligible model"
    assert [r["eligible"] for r in doc["models"]] == [True, True, False, False]
    assert doc["reference"]["devlarge-calib"]["ss"]["natural_spurious"] == 50
    assert doc["rule"].startswith("Selection") and doc["budget"] == S.BUDGET
    assert doc["command"].endswith("--models xs s b m --out " + str(tmp_path / "out.json"))


def test_a_larger_model_replaces_it_only_by_dominating_on_calib(dirs, tmp_path):
    _model(dirs, "xs", 10, k=4, spurious=40)
    _model(dirs, "s", 20, k=3, spurious=40)
    doc = _select(tmp_path, "--models", "s", "xs")
    assert doc["chosen"] == "s" and doc["why"].startswith("Pareto-dominates xs")


def test_no_eligible_model_chooses_none(dirs, tmp_path):
    _model(dirs, "xs", 10, k=4, spurious=40, p50=61.0)
    assert _select(tmp_path, "--models", "xs")["chosen"] is None


def test_runs_on_other_code_or_other_weights_are_refused(dirs, tmp_path):
    models, ev = dirs
    _model(dirs, "xs", 10, k=4, spurious=40)
    _model(dirs, "s", 20, k=3, spurious=40, code="c1")
    with pytest.raises(SystemExit, match="different library code"):
        _select(tmp_path, "--models", "xs", "s")
    (models / "xs" / "model.safetensors").write_bytes(b"retrained")
    with pytest.raises(SystemExit, match="other weights"):
        _select(tmp_path, "--models", "xs")


def test_perf_must_be_measured_with_the_models_own_weights(dirs, tmp_path):
    _model(dirs, "xs", 10, k=4, spurious=40, pin=False)
    with pytest.raises(SystemExit, match="does not pin"):
        _select(tmp_path, "--models", "xs")


def test_the_learning_curve_needs_no_perf_and_reports_a_plateau(dirs, tmp_path):
    for i, (n, k) in enumerate(((80000, 3), (10000, 6), (20000, 4), (40000, 3))):
        _model(dirs, f"c{n // 1000}k", 10 + i, k=k, spurious=40, windows=n)
    doc = _select(tmp_path, "--curve", "--models", "c80k", "c10k", "c20k", "c40k")
    assert doc["curve"]["windows"] == [10000, 20000, 40000, 80000]
    assert doc["curve"]["calib_leak"] == [6, 4, 3, 3] and doc["curve"]["plateau"]
    assert "chosen" not in doc and "perf" not in doc["models"][0]
    assert doc["models"][0]["dev"]["raw_tagger"]["0.5"]["leak"]["k"] == 4
