"""E7 (B4) — latency of every arm on the same 200 messages, one machine, offline.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out bench/results/perf_arms.json

Messages: a seeded draw of 200 messages of the real-data dev split (the E5
units, ``score.load_split``), as even as possible across the three
datasets, written to git-ignored ``build/perf/messages.jsonl`` (0600). Each
arm runs as its own subprocess in its own interpreter
(``bench.arms.run.command``), one after another:

* warm: the arm's own per-message time (``ms``, after its warm-up message)
  on the 200 messages: p50, p95 (as ``bench/perf.py``: the sorted value at
  index ⌊0.95 n⌋), mean, max;
* model load: the seconds the arm reports for building its pipeline;
* cold start: wall time of a fresh process that loads the arm and handles
  its warm-up message and one message, median of 3;
* peak RSS of the 200-message process (``os.wait4``).

Spans of these runs stay private (``build/perf/spans/``); the output holds
timings and counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from bench.arms.base import arm_env
from bench.arms.run import ARMS, command
from bench.realdata.common import BUILD, DATASETS, ROOT, derive_seed, file_sha256, read_jsonl, write_jsonl

N = 200
COLD_RUNS = 3
PERF = BUILD / "perf"


def sample(units_by_ds: Dict[str, Sequence[dict]], n: int = N, seed: int = None) -> List[dict]:
    """*n* messages, ⌈n/k⌉ or ⌊n/k⌋ per dataset, seeded."""
    rng = random.Random(derive_seed("perf-arms") if seed is None else seed)
    dss = sorted(units_by_ds)
    out = []
    for i, ds in enumerate(dss):
        k = n // len(dss) + (1 if i < n % len(dss) else 0)
        pool = sorted(units_by_ds[ds], key=lambda u: u["mid"])
        out += [{"id": u["mid"], "text": u["gold"]["text"]} for u in rng.sample(pool, k)]
    return out


def rss_mb(maxrss: int) -> float:
    """``ru_maxrss`` is bytes on macOS and kilobytes on Linux."""
    return round(maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10), 1)


def spawn(cmd: List[str]) -> dict:
    env = arm_env()
    t = time.perf_counter()
    with open(os.devnull, "w") as null, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=null,
                                                         stderr=subprocess.PIPE, text=True) as p:
        err = p.stderr.read()
        _pid, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t
    if p.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:3])} failed (exit {p.returncode}):\n" + "\n".join(err.strip().splitlines()[-15:]))
    return {"wall_s": wall, "peak_rss_mb": rss_mb(usage.ru_maxrss)}


def warm_stats(ms: Sequence[float]) -> dict:
    s = sorted(ms)
    return {"p50_ms": round(statistics.median(s), 1), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
            "mean_ms": round(statistics.mean(s), 1), "max_ms": round(s[-1], 1)}


def measure(arm: str, src: Path, spans: Path, cmd: Callable = command, cold_runs: int = COLD_RUNS) -> dict:
    out = spans / f"{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o700)
    run = spawn(cmd(arm, src, out))
    os.chmod(out, 0o600)
    rows = read_jsonl(out)
    meta = json.loads(Path(str(out) + ".meta.json").read_text())
    cold = []
    one = spans / f"{arm}-cold.jsonl"
    for _ in range(cold_runs):
        cold.append(spawn(cmd(arm, src, one) + ["--limit", "1"])["wall_s"])
    os.chmod(one, 0o600)
    return {"arm": arm, "venv": ARMS.get(arm, "?"), "messages": len(rows), "refused": meta["refused"],
            "edits": meta["edits"], "model_load_s": meta["load_seconds"],
            "cold_start_s": round(statistics.median(cold), 1), "cold_start_runs_s": [round(c, 1) for c in cold],
            "peak_rss_mb": run["peak_rss_mb"], "process_wall_s": round(run["wall_s"], 1),
            **warm_stats([r["ms"] for r in rows])}


def machine() -> dict:
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpus": os.cpu_count(), "memory_gb": round(mem / 2 ** 30, 1), "python": platform.python_version()}


def run_all(arms: Sequence[str], out: Path, units_by_ds: Optional[Dict[str, Sequence[dict]]] = None,
            perf: Path = PERF, cmd: Callable = command, cold_runs: int = COLD_RUNS, log=print) -> dict:
    if units_by_ds is None:
        from bench.realdata.score import load_split
        _hashes, loaded = load_split("dev")
        units_by_ds = {ds: loaded[ds]["units"] for ds in DATASETS}
    msgs = sample(units_by_ds)
    src = perf / "messages.jsonl"
    write_jsonl(src, msgs, private=True)
    rows = []
    for arm in arms:
        r = measure(arm, src, perf / "spans", cmd, cold_runs)
        log(f"{arm:22} p50 {r['p50_ms']:>7.1f} ms  p95 {r['p95_ms']:>7.1f}  load {r['model_load_s']:>5.1f} s  "
            f"cold {r['cold_start_s']:>5.1f} s  RSS {r['peak_rss_mb']:>7.1f} MB")
        rows.append(r)
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.perf_arms --out {_rel(out)}",
           "messages": len(msgs), "per_dataset": {ds: sum(m["id"].startswith(f"rd-{ds}-") for m in msgs) for ds in sorted(units_by_ds)},
           "input_sha256": file_sha256(src), "machine": machine(), "cold_runs": cold_runs,
           "p95": "sorted value at index floor(0.95 n), as bench/perf.py", "arms": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def markdown(doc: dict) -> str:
    m = doc["machine"]
    lines = ["# Latency of every arm (E7)", "", f"Command: `{doc['command']}`", "",
             f"{doc['messages']} real-data dev messages ({', '.join(f'{k} {v}' for k, v in doc['per_dataset'].items())}), "
             f"offline, one arm at a time on {m['platform']}, {m['cpus']} CPUs, {m['memory_gb']} GB.", "",
             "| arm | venv | warm p50 ms | warm p95 ms | mean ms | max ms | model load s | cold start s | peak RSS MB | refused |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["arms"]:
        lines.append(f"| `{r['arm']}` | {r['venv']} | {r['p50_ms']} | {r['p95_ms']} | {r['mean_ms']} | {r['max_ms']} "
                     f"| {r['model_load_s']} | {r['cold_start_s']} | {r['peak_rss_mb']} | {r['refused']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "perf_arms.json")
    args = ap.parse_args(argv)
    run_all(args.arms, args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
