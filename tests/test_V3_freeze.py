"""V3 Phase 3 — the freeze (``bench/realdata/freeze.py``) and the test-2
guard that reads it (``score.check_freeze``). Model-free: the tagger's pin is
checked on a few invented bytes."""

import hashlib
import json

import pytest

from bench.realdata import freeze as F
from bench.realdata import score
from bench.realdata.common import file_sha256
from surrogateshield.core.detection import config as C
from surrogateshield.core.detection.pii_tagger import DetectorUnavailable


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A hypotheses file, two config files and no models to load."""
    hyp, cfgs = tmp_path / "HYPOTHESES_TEST2.md", tmp_path / "configs"
    hyp.write_text("H1'' ...\n")
    (cfgs / "curve").mkdir(parents=True)
    (cfgs / "balanced.json").write_text('{"preset": "balanced"}\n')
    (cfgs / "curve" / "op1.json").write_text('{"preset": "balanced"}\n')
    monkeypatch.setattr(F, "models", lambda cfg: [{"stage": "pii_tagger", "model": "m", "revision": "sha256:0"}])
    monkeypatch.setattr(F, "CONFIGS", cfgs)
    return hyp, cfgs


def _freeze(tmp_path, tree, **kw):
    hyp, cfgs = tree
    return F.freeze(tmp_path / "FREEZE.json", cfgs, hyp, git=kw.get("git", {"commit": "c" * 40, "modified": []}),
                    today="2026-11-04")


def test_the_freeze_records_the_detector_and_is_written_once(tmp_path, tree):
    doc = _freeze(tmp_path, tree)
    assert doc["commit"] == "c" * 40 and doc["date"] == "2026-11-04"
    assert doc["detection_config_hash"] == C.benchmark().config_hash()
    assert doc["hypotheses_sha256"] == file_sha256(tree[0])
    assert doc["config_files"] == {str(tree[1] / "balanced.json"): file_sha256(tree[1] / "balanced.json"),
                                   str(tree[1] / "curve" / "op1.json"): file_sha256(tree[1] / "curve" / "op1.json")}
    assert doc["gliner_pii_tuned"] == {"label_set": "published", "threshold": 0.3} and doc["models"][0]["model"] == "m"
    assert json.loads((tmp_path / "FREEZE.json").read_text()) == doc
    with pytest.raises(SystemExit, match="frozen once"):
        _freeze(tmp_path, tree)


def test_a_dirty_tree_is_not_frozen(tmp_path, tree):
    with pytest.raises(SystemExit, match="not clean"):
        _freeze(tmp_path, tree, git={"commit": "c" * 40, "modified": ["python-library/x.py"]})
    with pytest.raises(SystemExit, match="not clean .bench/arms/new.py"):
        _freeze(tmp_path, tree, git={"commit": "c" * 40, "modified": [], "untracked": ["bench/arms/new.py"]})
    assert not (tmp_path / "FREEZE.json").exists()


def test_test2_is_refused_once_the_tree_moves_from_the_freeze(tmp_path, tree):
    hyp, cfgs = tree
    _freeze(tmp_path, tree)
    freeze = tmp_path / "FREEZE.json"
    assert score.check_freeze(freeze, hyp) == file_sha256(freeze)
    (cfgs / "curve" / "op1.json").write_text('{"preset": "strict"}\n')
    with pytest.raises(SystemExit, match="differs from .* in config_files"):
        score.check_freeze(freeze, hyp)
    (cfgs / "curve" / "op1.json").write_text('{"preset": "balanced"}\n')
    doc = json.loads(freeze.read_text())
    freeze.write_text(json.dumps({**doc, "code": "0" * 64, "detection_config_hash": "0" * 64}))
    with pytest.raises(SystemExit, match="in code, detection_config_hash"):
        score.check_freeze(freeze, hyp)
    freeze.write_text(json.dumps({"hypotheses_sha256": file_sha256(hyp)}))   # an older freeze: hypotheses only
    assert score.check_freeze(freeze, hyp)


def test_the_tagger_must_be_installed_with_its_pinned_weights(tmp_path, monkeypatch):
    folder = tmp_path / "models" / "tagger-x"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"invented weights")
    sha = hashlib.sha256(b"invented weights").hexdigest()
    monkeypatch.setenv("SURROGATESHIELD_MODELS", str(tmp_path / "models"))
    cfg = C.benchmark().with_stage("pii_tagger", model="tagger-x", revision=f"sha256:{sha}")
    rows = {r["stage"]: r for r in F.models(cfg)}
    assert rows["pii_tagger"]["weights_sha256_checked"] == sha
    assert rows["entity_trace"]["licence"] == "MIT" and "context_guard" not in rows    # disabled: not loaded
    with pytest.raises(DetectorUnavailable, match="not the pinned ones"):
        F.models(cfg.with_stage("pii_tagger", revision="sha256:" + "0" * 64))
    with pytest.raises(SystemExit, match="not installed"):
        F.models(cfg.with_stage("pii_tagger", model="tagger-y"))
