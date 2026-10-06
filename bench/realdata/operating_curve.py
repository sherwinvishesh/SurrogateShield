"""V3 §3.9 / E10: the operating curve, injected leak against over-redaction on natural text.

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.operating_curve \\
        --split dev --out bench/results/operating_curve_dev.json
    ... --split test2 --out bench/results/operating_curve_test2.json      # once, after FREEZE.json (Phase 3)

SS's points are committed config files (``OPS``): ``balanced``, the default,
and four under ``bench/realdata/configs/curve/`` that move the model-based
sources together: the tagger's per-type thresholds, spaCy's place
thresholds and the score from which the tagger vouches for a place or an
organisation at the relation gate (``gate_above``; op5 never vouches).
op1, op2 and op3 protect more, op5 edits less; patterns, the canonicaliser
and the structural passes do not move. ``balanced`` was the point now in
op3 until, with deberta-v3-small, the point then in op4 met the default
rule on dev (same leak, fewer harmless edits, §3.3 acceptance passed); the
five points are the same values as before. They run as in
``bench.realdata.components`` (same span files, same reuse rule).
GLiNER-PII's points are its thresholds (``gliner_pii@0.5`` is the
``gliner_pii`` arm, ``@0.3`` is ``gliner_pii_tuned``), from one run at the
lowest with every window span's score, as in ``bench.realdata.gliner_sweep``,
checked against the recorded ``gliner_pii`` arm at 0.5. The Presidio and LLM
Guard arms are single points from their recorded runs.

x is the injected leak rate. y is the natural slice's spurious-edit rate
(``score.aggregate``'s ``spurious``: edits touching no gold value, over
edits). Beside it is a named metric that does not shrink with how much an
arm edits: spurious edits per natural message. Each point gets a Wilson 95 %
interval for the rates and a conversation-cluster bootstrap for the
per-message ratio.

The dominance check, per dataset and pooled, was fixed before test-2: SS
dominates if every GLiNER-PII point has an SS point below and left of it
(lower leak and lower y, point estimates). SS's default dominates if
``balanced`` alone is below and left of every GLiNER-PII point. Each check
runs for both y metrics, beside the paired bootstrap differences of
``balanced`` against each GLiNER-PII point. Counts only: no text and no value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from bench.arms import gliner_pii, gliner_pii_tuned as tuned
from bench.arms.run import PRIVATE
from bench.realdata import components as K
from bench.realdata import gliner_sweep, score
from bench.realdata.common import COLLECTIONS, DATASETS, derive_seed, git_state
from bench.tagger.evaluate import code_stamp

OPS = ("curve/op1", "curve/op2", "curve/op3", "balanced", "curve/op5")
DEFAULT = "balanced"
LABELS = tuned.LABEL_SET
GLINER_AT = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SINGLE = ("presidio_default", "presidio_faker", "presidio_transformers", "llm_guard")
Y = ("spurious", "spurious_per_message")


def per_message(s: dict) -> tuple:
    """A message's (spurious edits, 1): the bootstrap's pair for spurious per message."""
    return s["spurious"], 1


def ss_label(name: str) -> str:
    return "ss@" + name.split("/")[-1]


def gliner_label(t: float) -> str:
    return f"gliner_pii@{t}"


def gliner_rows(run: str, ds: str, v: dict, sweep: Path = gliner_sweep.SWEEP, private: Path = PRIVATE,
                runner: Optional[Callable] = None) -> Dict[str, Dict[str, dict]]:
    """GLiNER-PII at every threshold of ``GLINER_AT`` on one dataset, from
    its scored window spans (run once if missing or made from another input)."""
    path = sweep / LABELS / f"{run}-{ds}.jsonl"
    meta = Path(str(path) + ".meta.json")
    if not (path.exists() and meta.exists() and json.loads(meta.read_text())["input_sha256"] == v["input_sha"]):
        (runner or gliner_sweep.run_spans)(LABELS, v["src"], path)
    lowest = json.loads(meta.read_text())["threshold"]
    if lowest > min(GLINER_AT):
        raise SystemExit(f"{score.rel(path)}: spans kept from {lowest}, above the curve's {min(GLINER_AT)}")
    spans = gliner_sweep.read_sweep(path, v["units"], v["input_sha"])
    ref = score.read_spans(private / "gliner_pii" / f"{run}-{ds}.jsonl", v["units"], v["input_sha"])
    differ = [m for m, sp in spans.items()
              if gliner_sweep.edits_at(sp, gliner_pii.THRESHOLD) != [list(e) for e in ref[m]["edits"]]]
    if differ:
        raise SystemExit(f"{ds}: GLiNER-PII @ {gliner_pii.THRESHOLD} from the scored spans differs from the "
                         f"recorded gliner_pii arm on {len(differ)} messages")
    return {gliner_label(t): {m: {"edits": gliner_sweep.edits_at(sp, t)} for m, sp in spans.items()}
            for t in GLINER_AT}


def ratio_ci(clusters: Sequence[str], pairs: Sequence[tuple], seed: int) -> dict:
    """A ratio of sums with a conversation-cluster bootstrap 95 % interval
    (``score.bootstrap`` against a zero ratio)."""
    b = score.bootstrap(clusters, pairs, [(0, 1)] * len(pairs), seed)
    return {"k": sum(p[0] for p in pairs), "n": sum(p[1] for p in pairs), "value": b["diff"], "ci95": b["ci95"]}


def point(units: Sequence[dict], scores: Sequence[dict], seed: int) -> dict:
    inj = [i for i, u in enumerate(units) if "injected" in u["slices"]]
    nat = [i for i, u in enumerate(units) if "natural" in u["slices"]]
    a = score.aggregate([units[i] for i in inj], [scores[i] for i in inj])
    n = score.aggregate([units[i] for i in nat], [scores[i] for i in nat])
    return {"leak": a["leak"], "spurious": n["spurious"], "negatives_untouched": n["negatives_untouched"],
            "spurious_per_message": ratio_ci([units[i]["conv"] for i in nat], [per_message(scores[i]) for i in nat],
                                             seed)}


def _y(p: dict, y: str) -> Optional[float]:
    return p[y]["rate"] if y == "spurious" else p[y]["value"]


def dominance(points: Dict[str, dict], ss: Sequence[str], gliner: Sequence[str], default: str) -> dict:
    """For each y: which SS points are below and left of each GLiNER point."""
    out = {}
    for y in Y:
        def below_left(s, g):
            ps, pg = points[s], points[g]
            if None in (ps["leak"]["rate"], pg["leak"]["rate"], _y(ps, y), _y(pg, y)):
                return False
            return ps["leak"]["rate"] < pg["leak"]["rate"] and _y(ps, y) < _y(pg, y)
        covered = {g: [s for s in ss if below_left(s, g)] for g in gliner}
        out[y] = {"ss_dominates": all(covered.values()), "default_dominates": all(default in c for c in covered.values()),
                  "below_left_of": covered}
    return out


def curve(units_by_ds: Dict[str, List[dict]], scores: Dict[str, Dict[str, List[dict]]], data: str,
          ss: Sequence[str], gliner: Sequence[str], default: str) -> dict:
    datasets = list(units_by_ds)
    groups = datasets + (["all"] if len(datasets) > 1 else [])
    points, dom, diffs = {}, {}, {}
    for g in groups:
        dss = datasets if g == "all" else [g]
        units = [u for ds in dss for u in units_by_ds[ds]]
        sc = {a: [s for ds in dss for s in scores[a][ds]] for a in scores}
        seed = derive_seed("operating-curve", data, g)
        points[g] = {a: point(units, sc[a], seed) for a in sc}
        dom[g] = dominance(points[g], ss, gliner, default)
        diffs[g] = {}
        for sl, metrics in (("injected", {"leak_rate": score.METRICS["leak_rate"]}),
                            ("natural", {"spurious_rate": score.METRICS["spurious_rate"],
                                         "spurious_per_message": per_message})):
            keep = [i for i, u in enumerate(units) if sl in u["slices"]]
            clusters = [units[i]["conv"] for i in keep]
            for gl in gliner:
                for m, f in metrics.items():
                    diffs[g].setdefault(gl, {})[m] = score.bootstrap(
                        clusters, [f(sc[default][i]) for i in keep], [f(sc[gl][i]) for i in keep], seed)
    return {"points": points, "dominance": dom, "differences": diffs}


def operating_curve(run: str, out: Path, ops: Sequence[str] = OPS, datasets: Sequence[str] = DATASETS,
                    single: Sequence[str] = SINGLE, reuse: bool = False, configs: Path = K.CONFIGS,
                    spans: Path = K.SPANS, private: Path = PRIVATE, sweep: Path = gliner_sweep.SWEEP,
                    ss_runner: Optional[Callable] = None, gliner_runner: Optional[Callable] = None,
                    loaded: Optional[dict] = None, log=print) -> dict:
    if DEFAULT not in ops:
        raise SystemExit(f"the curve needs SS's default point, {DEFAULT}")
    data, coll_name, role = score.RUNS[run]
    coll = COLLECTIONS[coll_name]
    sealed = score.check_freeze() if coll.prefix else None
    hashes = None
    if loaded is None:
        hashes, loaded = score.load_split(data, datasets, coll.rd, coll.build, None, coll.prefix)
    infos = {n: K.config_info(n, configs) for n in ops}
    stamp = code_stamp()
    rows: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for ds, v in loaded.items():
        for n in ops:
            rows.setdefault(ss_label(n), {})[ds] = K.run_config(n, infos[n], run, ds, v, stamp, reuse, spans,
                                                                ss_runner, log)
        for label, r in gliner_rows(run, ds, v, sweep, private, gliner_runner).items():
            rows.setdefault(label, {})[ds] = r
        for a in single:
            rows.setdefault(a, {})[ds] = score.read_spans(private / a / f"{run}-{ds}.jsonl", v["units"], v["input_sha"])
    units_by_ds = {ds: v["units"] for ds, v in loaded.items()}
    scores = {a: {ds: [score.score_unit(u, r[ds][u["mid"]]) for u in units_by_ds[ds]] for ds in r}
              for a, r in rows.items()}
    ss, gl = [ss_label(n) for n in ops], [gliner_label(t) for t in GLINER_AT]
    some = "" if list(ops) == list(OPS) else f"--ops {' '.join(ops)} "
    doc = {"command": f"PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python "
                      f"-m bench.realdata.operating_curve --split {run} {some}--out {score.rel(out)}",
           "git": git_state(), "code": stamp, "split": run, "role": role,
           **({"data_split": data, "collection": coll.name} if run not in ("dev", "test") else {}),
           **({"freeze_sha256": sealed} if sealed else {}),
           **({"frozen": hashes} if hashes else {}),
           "ss_points": {ss_label(n): infos[n] for n in ops}, "default": ss_label(DEFAULT),
           "gliner_points": {gliner_label(t): {"label_set": LABELS, "threshold": t} for t in GLINER_AT},
           "gliner_arms": {"gliner_pii": gliner_label(gliner_pii.THRESHOLD),
                           "gliner_pii_tuned": gliner_label(tuned.THRESHOLD)},
           "single_points": list(single),
           "x": "injected leak rate (Wilson 95 %)",
           "y": {"spurious": "natural-slice spurious edits / natural-slice edits (score.aggregate; Wilson 95 %)",
                 "spurious_per_message": "natural-slice spurious edits / natural-slice messages "
                                         "(conversation-cluster bootstrap 95 %)"},
           "rule": "per y: SS dominates if every GLiNER-PII point has an SS point with lower leak and lower y "
                   "(point estimates); the default dominates if balanced alone does",
           "bootstrap": {"resamples": score.RESAMPLES, "unit": "conversation (a single-turn prompt is its own)",
                         "seed": "derive_seed('operating-curve', data split, group)", "ci": "percentile 2.5 / 97.5",
                         "difference": "ss@balanced − gliner_pii@t"},
           **curve(units_by_ds, scores, data, ss, gl, ss_label(DEFAULT))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def _ci(r: dict) -> str:
    if r.get("value") is None:
        return "–"
    lo, hi = r["ci95"]
    return f"{r['value']:.2f} [{lo:.2f}, {hi:.2f}]"


def markdown(doc: dict) -> str:
    lines = [f"# Operating curve — {doc['split']}", "", f"`{doc['command']}`", "",
             f"Commit {doc['git'].get('commit', '?')[:12]}, library code {doc['code'][:12]}. x = {doc['x']}; "
             f"y = {doc['y']['spurious']}; y' = {doc['y']['spurious_per_message']}. Rule: {doc['rule']}.", ""]
    groups = list(doc["points"])
    for g in ([groups[-1]] + groups[:-1] if "all" in groups else groups):
        pts, dom = doc["points"][g], doc["dominance"][g]
        lines += [f"## {g}", "",
                  f"SS dominates: y {dom['spurious']['ss_dominates']}, y' {dom['spurious_per_message']['ss_dominates']}; "
                  f"the default alone: y {dom['spurious']['default_dominates']}, "
                  f"y' {dom['spurious_per_message']['default_dominates']}.", "",
                  "| point | injected leak | natural spurious / edits | spurious per natural message | negatives untouched |",
                  "|---|---|---|---|---|"]
        for a, p in pts.items():
            lines.append(f"| `{a}` | {p['leak']['k']}/{p['leak']['n']} {score._pct(p['leak'])} "
                         f"| {p['spurious']['k']}/{p['spurious']['n']} {score._pct(p['spurious'])} "
                         f"| {_ci(p['spurious_per_message'])} "
                         f"| {p['negatives_untouched']['k']}/{p['negatives_untouched']['n']} |")
        lines += ["", f"Paired differences, {doc['default']} − GLiNER-PII point (percentage points; per-message "
                      "ratio ×100):", "", "| GLiNER point | Δ leak | Δ spurious rate | Δ spurious per message |",
                  "|---|---|---|---|"]
        for gl, d in doc["differences"][g].items():
            lines.append(f"| `{gl}` | {score._diff(d['leak_rate'])} | {score._diff(d['spurious_rate'])} "
                         f"| {score._diff(d['spurious_per_message'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(score.RUNS))
    ap.add_argument("--ops", nargs="+", default=list(OPS), help=f"SS's points: files under {score.rel(K.CONFIGS)}/")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--reuse", action="store_true", help="keep SS span files made on this code from the same file")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = operating_curve(a.split, a.out.resolve(), a.ops, a.datasets, reuse=a.reuse)
    g = "all" if "all" in doc["points"] else next(iter(doc["points"]))
    for arm, p in doc["points"][g].items():
        print(f"{arm:24s} leak {p['leak']['k']:>4}/{p['leak']['n']}  spurious {p['spurious']['k']:>5}/"
              f"{p['spurious']['n']:<5} per msg {p['spurious_per_message']['value']}")
    print({y: {k: v for k, v in d.items() if k != "below_left_of"} for y, d in doc["dominance"][g].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
