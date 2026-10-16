"""V3 §3.3: choose the tagger's encoder on development runs, and its learning curve.

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.select \\
        --models dv3xs-40k dv3s-40k mb-40k --out bench/results/tagger_selection.json
    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.select --curve \\
        --models dv3xs-c10k dv3xs-c20k dv3xs-c40k dv3xs-c80k --out bench/results/tagger_learning_curve.json

Per model ``m`` (a folder under ``bench/tagger/build/models/``) it reads the
git-ignored runs under ``bench/tagger/build/eval/``:
``<m>-devlarge-calib-<variant>.json`` and ``<m>-dev-<variant>.json``
(``bench.tagger.evaluate --ss``: SS with the tagger beside the ``ss`` and
GLiNER arms on the same messages, plus the raw tagger's threshold sweep) and,
for a selection, ``perf/<m>-<variant>.json`` (``bench.perf_arms --arms ss``
with that run's config, on a quiet machine). Every run must be on the same
library code and on the model folder's current weights. The variant is the
configuration ``balanced`` runs, ``sel3ga9-spacyloc``.

Selection (fixed in the tracker before the runs it decides were seen): a model
is eligible if SS with it passes §3.3 on dev-large's calib half (every type,
pooled, natural spurious edits at most the current SS's) and the perf budget
(p50 at most 60 ms, peak RSS at most 1,600 MB). Of the eligible models the one
with the smallest weights is kept, unless a larger one is at least as good on
calib leak and on natural spurious edits and better on one of them.
Dev-large's val half is not read.

``--curve`` reports the same numbers per training size, and whether calib
leak still fell at the largest size. Counts only: no text and no value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

from bench.realdata import score
from bench.realdata.common import ROOT, git_state
from bench.tagger.evaluate import BUILD, acceptance, model_sha

MODELS = ROOT / "bench" / "tagger" / "build" / "models"
VARIANT = "sel3ga9-spacyloc"
BUDGET = {"p50_ms": 60.0, "peak_rss_mb": 1600.0}
REFERENCE = ("ss", "gliner_pii", "gliner_pii_tuned")
RAW_AT = ("0.5", "0.9")
RULE = " ".join(__doc__.split("\n\n")[3].split())          # the "Selection ..." paragraph above


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {score.rel(path)}: run bench.tagger.evaluate (or bench.perf_arms) for it first")
    return json.loads(path.read_text())


def arm(r: dict) -> dict:
    return {"leak": r["leak"], "leaked_by_type": r["leaked_by_type"],
            "natural_spurious": r["natural_spurious"], "natural_edits": r["natural_edits"]}


def split_row(doc: dict) -> dict:
    """One split's SS + tagger counts, its §3.3 check and the raw tagger's."""
    w = doc["with_ss"]
    acc = acceptance(w)
    return {"ss_tagger": arm(w["ss_tagger"]),
            "acceptance": {"ok": acc["ok"], "checks": acc["checks"],
                           "failing_types": sorted(t for t, v in acc["by_type"].items() if not v["ok"])},
            "raw_tagger": {t: arm(doc["sweep"][t]) for t in RAW_AT}}


def perf_row(m: str, variant: str, sha: str, eval_dir: Path) -> dict:
    doc = _load(eval_dir / "perf" / f"{m}-{variant}.json")
    cfg = doc.get("detection_config")
    if not cfg:
        raise SystemExit(f"perf/{m}-{variant}.json was not run with the variant's config")
    tagger = [d for d in json.loads((ROOT / cfg["path"]).read_text()).get("detectors", [])
              if d.get("name") == "pii_tagger"]
    if not tagger or tagger[0].get("revision") != f"sha256:{sha}":
        raise SystemExit(f"perf/{m}-{variant}.json: its config does not pin {m}'s weights")
    r = {k: doc["arms"][0][k] for k in ("p50_ms", "p95_ms", "mean_ms", "peak_rss_mb", "cold_start_s")}
    r.update(file=score.rel(eval_dir / "perf" / f"{m}-{variant}.json"), config=cfg, machine=doc["machine"],
             ok=r["p50_ms"] <= BUDGET["p50_ms"] and r["peak_rss_mb"] <= BUDGET["peak_rss_mb"])
    return r


def model_row(m: str, variant: str, eval_dir: Path, models_dir: Path, perf: bool) -> Tuple[dict, Dict[str, dict]]:
    meta = _load(models_dir / m / "train_meta.json")
    sha = model_sha(models_dir / m)
    docs = {s: _load(eval_dir / f"{m}-{s}-{variant}.json") for s in ("devlarge-calib", "dev")}
    for s, d in docs.items():
        if d["model"]["weights_sha256"] != sha:
            raise SystemExit(f"{m}-{s}-{variant}.json was run on other weights than {m}'s")
    a = meta["args"]
    row = {"name": m, "encoder": meta["encoder"], "hub_id": meta["hub_id"], "revision": meta["revision"],
           "licence": meta["licence"], "weights_sha256": sha,
           "weights_bytes": (models_dir / m / "model.safetensors").stat().st_size,
           "training": {"data_sha256": meta["data_sha256"], "windows": meta["windows"], "epochs": a["epochs"],
                        "lr": a["lr"], "batch": a["batch"], "seed": a["seed"], "best_val_f1": meta["best_val_f1"],
                        "minutes": meta["minutes"], "device": meta["device"]},
           "runs": {s: {"file": score.rel(eval_dir / f"{m}-{s}-{variant}.json"), "command": d["command"],
                        "git": d["git"]} for s, d in docs.items()},
           "calib": split_row(docs["devlarge-calib"]), "dev": split_row(docs["dev"])}
    if perf:
        row["perf"] = perf_row(m, variant, sha, eval_dir)
        row["eligible"] = row["calib"]["acceptance"]["ok"] and row["perf"]["ok"]
    return row, docs


def _key(row: dict):
    r = row["calib"]["ss_tagger"]
    return r["leak"]["k"], r["natural_spurious"]


def choose(rows: Sequence[dict]):
    """The smallest eligible model, replaced by a larger one only if it
    Pareto-dominates on calib (leak, natural spurious)."""
    eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: r["weights_bytes"])
    if not eligible:
        return None, "no model passes §3.3 on calib and the perf budget"
    best, why = eligible[0], "smallest eligible model"
    for r in eligible[1:]:
        (lk, ls), (bk, bs) = _key(r), _key(best)
        if lk <= bk and ls <= bs and (lk, ls) != (bk, bs):
            best, why = r, f"Pareto-dominates {best['name']} on calib (leak {lk} vs {bk}, spurious {ls} vs {bs})"
    return best["name"], why


def references(docs_by_model: dict) -> dict:
    """The ss and GLiNER arms of each split: the same for every model."""
    out = {}
    for s in ("devlarge-calib", "dev"):
        refs = {m: {a: arm(d[s]["with_ss"][a]) for a in REFERENCE} for m, d in docs_by_model.items()}
        first = next(iter(refs.values()))
        if any(v != first for v in refs.values()):
            raise SystemExit(f"the reference arms differ between models on {s}")
        out[s] = first
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="folders under bench/tagger/build/models/")
    ap.add_argument("--curve", action="store_true", help="a learning curve: no perf, no choice")
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if len(set(a.models)) != len(a.models):
        raise SystemExit("a model is given twice")
    rows, docs_by_model = [], {}
    for m in a.models:
        row, docs = model_row(m, a.variant, BUILD, MODELS, perf=not a.curve)
        rows.append(row)
        docs_by_model[m] = docs
    codes = {d["with_ss"]["config"]["code"] for docs in docs_by_model.values() for d in docs.values()}
    if len(codes) != 1:
        raise SystemExit(f"the runs are on different library code: {sorted(c[:12] for c in codes)}")
    doc = {"command": "... -m bench.tagger.select " + ("--curve " if a.curve else "")
                      + ("" if a.variant == VARIANT else f"--variant {a.variant} ")
                      + "--models " + " ".join(a.models) + f" --out {score.rel(a.out.resolve())}",
           "git": git_state(), "variant": a.variant, "code": codes.pop(),
           "reference": references(docs_by_model), "models": rows}
    for r in rows:
        c, d = r["calib"], r["dev"]
        perf = f"  p50 {r['perf']['p50_ms']} ms RSS {r['perf']['peak_rss_mb']} MB" if "perf" in r else ""
        print(f"{r['name']:12s} {r['training']['windows']:>6d} windows {r['weights_bytes'] / 2 ** 20:>6.1f} MB  "
              f"calib leak {c['ss_tagger']['leak']['k']}/{c['ss_tagger']['leak']['n']} spurious "
              f"{c['ss_tagger']['natural_spurious']} §3.3 {c['acceptance']['ok']}  dev leak "
              f"{d['ss_tagger']['leak']['k']}/{d['ss_tagger']['leak']['n']} spurious {d['ss_tagger']['natural_spurious']}"
              f"  raw dev @0.5 {d['raw_tagger']['0.5']['leak']['k']}{perf}")
    if a.curve:
        by_size = sorted(rows, key=lambda r: r["training"]["windows"])
        leaks = [r["calib"]["ss_tagger"]["leak"]["k"] for r in by_size]
        doc["curve"] = {"windows": [r["training"]["windows"] for r in by_size], "calib_leak": leaks,
                        "dev_leak": [r["dev"]["ss_tagger"]["leak"]["k"] for r in by_size],
                        "calib_spurious": [r["calib"]["ss_tagger"]["natural_spurious"] for r in by_size],
                        "plateau": len(leaks) > 1 and leaks[-1] >= leaks[-2]}
        print(f"curve {doc['curve']}")
    else:
        doc["budget"] = BUDGET
        doc["rule"] = RULE
        doc["chosen"], doc["why"] = choose(rows)
        print(f"chosen {doc['chosen']}: {doc['why']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
