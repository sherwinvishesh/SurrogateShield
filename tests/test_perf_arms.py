"""Model-free tests for bench/perf_arms.py (E7, B4): the seeded, balanced
message draw, the warm statistics, the RSS unit, and a whole run with a stub
arm process standing in for the real ones. Texts are invented."""

import json
import os
import sys

import pytest

from bench import perf_arms as P

STUB = r'''
import argparse, json, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="src"); ap.add_argument("--out"); ap.add_argument("--limit", type=int)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.src)][:a.limit]
with open(a.out, "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"id": r["id"], "edits": [], "ms": float(i % 10 + 1)}) + "\n")
open(a.out + ".meta.json", "w").write(json.dumps({"refused": 0, "edits": 0, "load_seconds": 0.5}))
if "fail" in a.out:
    sys.exit(3)
'''


def _units(n_per=100):
    return {ds: [{"mid": f"rd-{ds}-dev-{i:04d}", "gold": {"text": f"message {i} of {ds}"}} for i in range(1, n_per + 1)]
            for ds in ("oasst1", "sharegpt", "wildchat")}


def test_sample_is_balanced_seeded_and_independent_of_order():
    u = _units()
    s = P.sample(u)
    assert len(s) == 200 and len({m["id"] for m in s}) == 200
    per = {ds: sum(m["id"].startswith(f"rd-{ds}-") for m in s) for ds in u}
    assert per == {"oasst1": 67, "sharegpt": 67, "wildchat": 66}
    assert s == P.sample({ds: list(reversed(v)) for ds, v in u.items()})
    assert s != P.sample(u, seed=1)


def test_warm_stats_and_rss_units(monkeypatch):
    w = P.warm_stats([float(x) for x in range(1, 101)])
    assert w == {"p50_ms": 50.5, "p95_ms": 96.0, "mean_ms": 50.5, "max_ms": 100.0}
    monkeypatch.setattr(sys, "platform", "darwin")
    assert P.rss_mb(512 * 2 ** 20) == 512.0
    monkeypatch.setattr(sys, "platform", "linux")
    assert P.rss_mb(512 * 2 ** 10) == 512.0


def _cmd(tmp_path):
    script = tmp_path / "stub_arm.py"
    script.write_text(STUB)
    return lambda arm, src, out: [sys.executable, str(script), "--in", str(src), "--out", str(out)]


def test_run_all_with_a_stub_arm(tmp_path):
    out = tmp_path / "res" / "perf_arms.json"
    doc = P.run_all(["ss", "llm_guard"], out, _units(), perf=tmp_path / "perf", cmd=_cmd(tmp_path), cold_runs=2,
                    log=lambda *_: None)
    assert [r["arm"] for r in doc["arms"]] == ["ss", "llm_guard"]
    r = doc["arms"][0]
    assert r["messages"] == 200 and r["p50_ms"] == 5.5 and r["model_load_s"] == 0.5
    assert len(r["cold_start_runs_s"]) == 2 and r["peak_rss_mb"] > 0 and r["venv"] == ".venv"
    assert doc["per_dataset"] == {"oasst1": 67, "sharegpt": 67, "wildchat": 66}
    assert doc["command"].split()[-1].endswith("perf_arms.json")
    blob = out.read_text() + out.with_suffix(".md").read_text()
    assert "message 1 of" not in blob
    assert (os.stat(tmp_path / "perf" / "messages.jsonl").st_mode & 0o777) == 0o600
    assert (os.stat(tmp_path / "perf" / "spans" / "ss.jsonl").st_mode & 0o777) == 0o600


def test_a_failing_arm_stops_the_run(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": "a", "text": "hello"}) + "\n")
    with pytest.raises(SystemExit, match="exit 3"):
        P.measure("fail", src, tmp_path / "spans", _cmd(tmp_path))
