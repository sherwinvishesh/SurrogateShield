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
