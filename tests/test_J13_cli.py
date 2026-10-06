"""Gate J13 — the ``surrogateshield`` command: scan, mask, unmask, bench,
doctor. Model-free: spaCy and distilbert are stubbed to find nothing, so only
PatternScan detects (an email, an SSN). Stored maps go to a temporary
SURROGATESHIELD_HOME.
"""

import io
import json
import stat
import types
from pathlib import Path

import pytest

from surrogateshield import cli
from surrogateshield.core.detection import context_guard, entity_trace
from surrogateshield.core.errors import DetectorUnavailable

pytestmark = pytest.mark.usefixtures("stub_tagger")    # balanced runs the tagger

TEXT = "Mail me at dana.w@example.com, SSN 219-09-9999."


@pytest.fixture(autouse=True)
def no_models(monkeypatch, tmp_path):
    monkeypatch.setattr(entity_trace, "_get_nlp", lambda *_a, **_k: (lambda t: types.SimpleNamespace(ents=[])))
    monkeypatch.setattr(context_guard, "_get_ner", lambda *_a, **_k: (lambda _t: []))
    monkeypatch.setenv("SURROGATESHIELD_HOME", str(tmp_path / "home"))


def run(*argv, stdin=None, monkeypatch=None):
    out = io.StringIO()
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return cli.main(list(argv), out=out), out.getvalue()


def test_J13_scan_json_lists_offsets():
    rc, out = run("scan", "--json", TEXT)
    found = {(d["type"], d["text"]) for d in json.loads(out)}
    assert rc == 0 and {("email", "dana.w@example.com"), ("ssn", "219-09-9999")} <= found
    d = next(d for d in json.loads(out) if d["type"] == "ssn")
    assert TEXT[d["start"]:d["end"]] == "219-09-9999"


def test_J13_mask_then_unmask_in_a_new_process(capsys, tmp_path):
    rc, sent = run("mask", "--seed", "3", TEXT)
    sid = capsys.readouterr().err.split("session: ")[1].split()[0]
    assert rc == 0 and "dana.w@example.com" not in sent and "219-09-9999" not in sent
    stored = tmp_path / "home" / "sessions" / f"{sid}.shadowmap"
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert stat.S_IMODE(stored.parent.stat().st_mode) == 0o700
    assert b"dana.w@example.com" not in stored.read_bytes()           # encrypted
    rc, back = run("unmask", "-s", sid, "--close", sent.strip() + " Done.")
    assert rc == 0 and back.strip() == TEXT + " Done."
    assert not stored.exists()                                       # --close erased it


def test_J13_mask_reads_stdin_and_no_store_writes_nothing(monkeypatch, tmp_path):
    rc, sent = run("mask", "--no-store", "-", stdin=TEXT, monkeypatch=monkeypatch)
    assert rc == 0 and "219-09-9999" not in sent
    assert not (tmp_path / "home" / "sessions").exists()


def test_J13_unmask_unknown_session_fails(capsys):
    rc, out = run("unmask", "-s", "nosuch", "hello")
    assert rc == cli.EXIT_FAIL and out == "" and "no stored session" in capsys.readouterr().err


def test_J13_bad_session_id_is_refused(capsys):
    rc, _ = run("unmask", "-s", "../etc", "hello")
    assert rc == cli.EXIT_FAIL


def test_J13_detector_unavailable_prints_nothing(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise DetectorUnavailable("spaCy model missing")
    monkeypatch.setattr(entity_trace, "_get_nlp", boom)
    rc, out = run("mask", "--no-store", TEXT)
    assert rc == cli.EXIT_DETECTOR and out == ""                     # fail closed
    assert "nothing was masked" in capsys.readouterr().err


def test_J13_doctor_exit_code_follows_required_checks(monkeypatch):
    monkeypatch.setattr(cli, "_check_models", lambda c: [("spaCy model", True, True, "")])
    rc, out = run("doctor", "--smoke")
    assert rc == 0 and "[  ok] smoke: mask hides an email and an SSN" in out
    monkeypatch.setattr(cli, "_check_models", lambda c: [("spaCy model", False, True, "missing")])
    rc, out = run("doctor")
    assert rc == cli.EXIT_FAIL and "[FAIL] spaCy model — missing" in out


def test_J13_doctor_flags_loose_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "device.key").write_bytes(b"x" * 32)
    (home / "device.key").chmod(0o644)
    monkeypatch.setattr(cli, "_check_models", lambda c: [])
    rc, out = run("doctor")
    assert rc == cli.EXIT_FAIL and "[FAIL] device key — chmod 600" in out


def test_J13_bench_reports_percentiles():
    rc, out = run("bench", "--no-context-guard", "--rounds", "1", "--json")
    (row,) = json.loads(out)
    assert rc == 0 and row["messages"] == len(cli.BENCH_TEXTS) and row["p50_ms"] <= row["p95_ms"]


def test_J13_packaging_has_entry_point_and_py_typed():
    root = Path(cli.__file__).resolve().parent
    assert (root / "py.typed").exists() and (root / "__main__.py").exists()
    toml = (root.parent / "pyproject.toml").read_text()
    assert 'surrogateshield = "surrogateshield.cli:main"' in toml and '"py.typed"' in toml


def test_J13_doctor_checks_the_taggers_folder_and_pin(tmp_path, monkeypatch):
    import hashlib
    from surrogateshield.core.detection import config as dconfig, pii_tagger
    monkeypatch.setenv(pii_tagger.MODELS_ENV, str(tmp_path))
    st = dconfig.Stage("pii_tagger", model="pii-tagger-x", revision="sha256:" + hashlib.sha256(b"w").hexdigest())
    _name, ok, required, note = cli._check_tagger(st, {"pii-tagger-x": "MIT"})
    assert not ok and required and "bench.tagger.install" in note and "--preset classic" in note
    (tmp_path / "pii-tagger-x").mkdir()
    (tmp_path / "pii-tagger-x" / "model.safetensors").write_bytes(b"w")
    _name, ok, _req, note = cli._check_tagger(st, {"pii-tagger-x": "MIT"})
    assert ok and note.endswith("licence MIT")
    (tmp_path / "pii-tagger-x" / "model.safetensors").write_bytes(b"x")
    assert not cli._check_tagger(st, {})[1]


def test_J13_doctor_prints_the_detection_config(monkeypatch, tmp_path):
    from surrogateshield.core.detection import config as dconfig
    monkeypatch.delenv(dconfig.ENV_PRESET, raising=False)
    monkeypatch.delenv(dconfig.ENV_FILE, raising=False)
    monkeypatch.setattr(cli, "_check_models", lambda det: [])
    rc, out = run("doctor")
    balanced = dconfig.preset("balanced").config_hash()[:16]
    assert rc == 0 and f"preset balanced, hash {balanced} (default)" in out
    assert "stage context_guard  off, dslim/distilbert-NER@dfa2838a1273" in out
    assert f"stage pii_tagger     on, {dconfig.PII_TAGGER_MODEL}@{dconfig.PII_TAGGER_REVISION[:12]}" in out
    rc, out = run("doctor", "--preset", "strict", "--show-config")
    strict = dconfig.preset("strict")
    assert rc == 0 and f"hash {strict.config_hash()[:16]} (--preset / --detection-config)" in out
    shown = out[out.index("{"):out.rindex("}") + 1]
    assert dconfig.DetectionConfig.from_json(shown) == strict
    partial = tmp_path / "det.json"
    partial.write_text(json.dumps({"gate_bypass": ["EMAIL"], "type_actions": {"PHONE": "redact"}}))
    rc, out = run("doctor", "--detection-config", str(partial))
    assert rc == 0 and "relation gate  bypassed by EMAIL" in out and "PHONE redact" in out
    partial.write_text(json.dumps({"type_actions": {"PHONE": "erase"}}))
    rc, out = run("doctor", "--detection-config", str(partial))
    assert rc == cli.EXIT_FAIL and out.startswith("[FAIL] detection config")
    rc, out = run("doctor", "--detection-config", str(tmp_path / "missing.json"))
    assert rc == cli.EXIT_FAIL and "cannot read --detection-config" in out


def test_J13_detection_config_flag_reaches_scan_and_mask(tmp_path):
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"type_actions": {"EMAIL": "keep"}}))
    rc, out = run("scan", "--json", "--detection-config", str(keep), TEXT)
    masked = {d["text"]: d["masked"] for d in json.loads(out)}
    assert rc == 0 and masked == {"dana.w@example.com": False, "219-09-9999": True}
    rc, out = run("mask", "--no-store", "--detection-config", str(keep), TEXT)
    assert rc == 0 and "dana.w@example.com" in out and "219-09-9999" not in out
    rc, out = run("mask", "--no-store", "--seed", "1", "--preset", "fast", TEXT)
    assert rc == 0 and "dana.w@example.com" not in out
    keep.write_text("[1, 2]")
    assert run("scan", "--detection-config", str(keep), TEXT)[0] == cli.EXIT_FAIL


def test_J13_bench_preset_runs_that_config_only(monkeypatch):
    seen = []
    def fake(texts, *, context_guard, rounds=3, seed=0, detection=None):
        seen.append((context_guard, detection and detection.preset))
        return {"context_guard": context_guard, "messages": 1, "p50_ms": 1.0, "p95_ms": 1.0,
                "max_ms": 1.0, **({"preset": detection.preset} if detection else {})}
    monkeypatch.setattr(cli, "latency", fake)
    rc, out = run("bench", "--preset", "fast")
    from surrogateshield.core.detection import config as dconfig
    fast_cg = dconfig.preset("fast").enabled("context_guard")
    assert rc == 0 and seen == [(fast_cg, "fast")] and out.startswith("fast, with")
    seen.clear()
    run("bench")
    assert seen == [(False, None), (True, None)]
