"""V3 §3.6 / E9: the detector's presets and component ablations on one run, beside GLiNER-PII.

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.components \\
        --split dev --out bench/results/presets_dev.json
    ... --split test2 --out bench/results/realdata_components_test2.json     # after FREEZE.json (Phase 3)

Each configuration is a committed file under ``bench/realdata/configs/``: a
preset or an ablation (``DetectionConfig.preset``) with every address
redrawn, as ``benchmark()`` does, so ``balanced.json`` is the benchmark
config. ``h9/no_patterns.json`` (``EXTRA``) is ``balanced`` with the
PatternScan and canonicaliser stages off, so no pattern of PatternScan
matches anything, and AGE, DATE_OF_BIRTH, EMAIL, NETWORK and URL routed to
the tagger as the other structured types are ("the tagger carries structured
types as free text when regex is gone"): H9''(a) of ``HYPOTHESES_TEST2.md``
(H9''(b), the tagger removed, is ``no_tagger``). ``h9/no_patterns_unrouted``
keeps the default routing, under which nothing reports those five types, and
is reported beside it. SS runs once per file and dataset through ``bench.arms.ss`` with
``$SURROGATESHIELD_DETECTION_CONFIG`` set to the file, and the run's recorded
config hash must be the file's. Span files stay under the git-ignored
``bench/realdata/build/spans/components/``; ``--reuse`` keeps one only if it
was made on the same library code from the same file. ``gliner_pii`` and
``gliner_pii_tuned`` are read from their recorded runs (on ``dev`` the tuned
arm comes from the frozen sweep, as in the yardstick).

Per slice: every configuration's leak and spurious counts, and a paired
cluster bootstrap of its difference from ``balanced`` (arm − balanced: a
positive leak difference is leakage the switched-off part was preventing).
The in-run p50 is not a quiet measurement: the budget is
``bench.perf_arms``'s. Counts only: no text and no value.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from bench.arms.run import PRIVATE
from bench.realdata import score, yardstick
from bench.realdata.common import COLLECTIONS, DATASETS, ROOT, derive_seed, file_sha256, git_state
from bench.tagger.evaluate import code_stamp, tuned_rows
from surrogateshield.core.detection import config as C

CONFIGS = ROOT / "bench" / "realdata" / "configs"
NAMES = ("balanced", "no_tagger", "no_canonicaliser", "no_gate", "no_models", "no_structural",
         "fast", "strict", "classic")
EXTRA = ("h9/no_patterns", "h9/no_patterns_unrouted")   # not presets: a subfolder, named by path
RUN = (*NAMES, *EXTRA)
BASE = "balanced"
REFERENCE = ("gliner_pii", "gliner_pii_tuned")
SPANS = PRIVATE / "components"
KEEP = ("messages", "protect_values", "leak", "macro_leak_rate", "message_leak", "edits", "spurious",
        "messages_with_spurious", "negatives_untouched", "refused")
DIFFS = ("leak_rate", "spurious_rate")


def config_info(name: str, configs: Path = CONFIGS) -> dict:
    """The committed file of configuration *name* and the config it makes."""
    path = configs / f"{name}.json"
    cfg = C.from_partial(json.loads(path.read_text()), C.benchmark())
    return {"file": score.rel(path), "sha256": file_sha256(path), "preset": cfg.preset,
            "detection_config_hash": cfg.config_hash()}


def run_config(name: str, info: dict, run: str, ds: str, v: dict, stamp: str, reuse: bool,
               spans: Path = SPANS, runner: Optional[Callable] = None, log=print) -> Dict[str, dict]:
    """SS under configuration *name* on one dataset's arm input; its checked rows."""
    path = spans / name / f"{run}-{ds}.jsonl"
    mark = path.with_name(path.name + ".code")
    want = f"{stamp} {info['sha256']}"
    if not (reuse and path.exists() and mark.exists() and mark.read_text() == want):
        (runner or yardstick.run_ss)(ROOT / info["file"], v["src"], path)
        mark.write_text(want)
        log(f"{run} {ds:9} {name:17} ran on {len(v['units'])} messages")
    got = score.config_hash(path)
    if got != info["detection_config_hash"]:
        raise SystemExit(f"{score.rel(path)}: made with config {str(got)[:16]}, not {info['file']}'s "
                         f"{info['detection_config_hash'][:16]}")
    return score.read_spans(path, v["units"], v["input_sha"])


def reference_rows(arm: str, run: str, ds: str, v: dict) -> Dict[str, dict]:
    if arm == "gliner_pii_tuned":
        return tuned_rows(run, ds, v)
    return score.read_spans(PRIVATE / arm / f"{run}-{ds}.jsonl", v["units"], v["input_sha"])


def brief(units: Sequence[dict], scores: Sequence[dict]) -> dict:
    a = score.aggregate(units, scores)
    out = {k: a[k] for k in KEEP}
    out["leaked_by_type"] = {t: r["leaked"] for t, r in a["by_type"].items()}
    out["values_by_type"] = {t: r["values"] - r["policy"] for t, r in a["by_type"].items()}
    return out


def tables(units_by_ds: Dict[str, List[dict]], scores: Dict[str, Dict[str, List[dict]]], data: str,
           base: str = BASE) -> dict:
    """Per group (each dataset, then ``all``) and slice: every arm's counts and,
    for every arm but *base*, the paired bootstrap of arm − *base*."""
    if base not in scores:
        raise SystemExit(f"the differences are taken from {base!r}, which was not run")
    datasets = list(units_by_ds)
    groups = datasets + (["all"] if len(datasets) > 1 else [])
    results, diffs = {}, {}
    for g in groups:
        dss = datasets if g == "all" else [g]
        units = [u for ds in dss for u in units_by_ds[ds]]
        sc = {a: [s for ds in dss for s in scores[a][ds]] for a in scores}
        results[g], diffs[g] = {}, {}
        for sl in score.SLICES:
            keep = [i for i, u in enumerate(units) if sl in u["slices"]]
            if not keep:
                continue
            us = [units[i] for i in keep]
            per = {a: [sc[a][i] for i in keep] for a in sc}
            clusters = [u["conv"] for u in us]
            seed = derive_seed("components-bootstrap", data, g, sl)
            results[g][sl] = {a: brief(us, per[a]) for a in per}
            diffs[g][sl] = {a: {m: score.bootstrap(clusters, [score.METRICS[m](s) for s in per[a]],
                                                    [score.METRICS[m](s) for s in per[base]], seed)
                                for m in DIFFS}
                            for a in per if a != base}
    return {"results": results, "differences": diffs}


def p50(rows_by_ds: Dict[str, Dict[str, dict]]) -> Optional[float]:
    ms = [r["ms"] for rows in rows_by_ds.values() for r in rows.values() if "ms" in r]
    return round(statistics.median(ms), 1) if ms else None


def components(run: str, out: Path, names: Sequence[str] = RUN, datasets: Sequence[str] = DATASETS,
               reference: Sequence[str] = REFERENCE, reuse: bool = False, configs: Path = CONFIGS,
               spans: Path = SPANS, runner: Optional[Callable] = None, loaded: Optional[dict] = None,
               log=print) -> dict:
    data, coll_name, role = score.RUNS[run]
    coll = COLLECTIONS[coll_name]
    sealed = score.seal(coll)
    hashes = None
    if loaded is None:
        hashes, loaded = score.load_split(data, datasets, coll.rd, coll.build, None, coll.prefix)
    infos = {n: config_info(n, configs) for n in names}
    stamp = code_stamp()
    rows: Dict[str, Dict[str, Dict[str, dict]]] = {a: {} for a in [*names, *reference]}
    for ds, v in loaded.items():
        for n in names:
            rows[n][ds] = run_config(n, infos[n], run, ds, v, stamp, reuse, spans, runner, log)
        for a in reference:
            rows[a][ds] = reference_rows(a, run, ds, v)
    units_by_ds = {ds: v["units"] for ds, v in loaded.items()}
    scores = {a: {ds: [score.score_unit(u, r[ds][u["mid"]]) for u in units_by_ds[ds]] for ds in r}
              for a, r in rows.items()}
    for n in names:
        infos[n]["in_run_p50_ms"] = p50(rows[n])
    some = "" if list(names) == list(RUN) else f"--configs {' '.join(names)} "
    doc = {"command": f"PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python "
                      f"-m bench.realdata.components --split {run} {some}--out {score.rel(out)}",
           "git": git_state(), "code": stamp, "split": run, "role": role,
           **({"data_split": data, "collection": coll.name} if run not in ("dev", "test") else {}),
           **({"freeze_sha256": sealed} if sealed else {}),
           **({"frozen": hashes} if hashes else {}),
           "corpus": {ds: v.get("corpus") for ds, v in loaded.items()},
           "configs": infos, "reference": list(reference), "base": BASE, "slices": list(score.SLICES),
           "bootstrap": {"resamples": score.RESAMPLES, "unit": "conversation (a single-turn prompt is its own)",
                         "seed": "derive_seed('components-bootstrap', data split, group, slice)",
                         "ci": "percentile 2.5 / 97.5", "difference": f"arm − {BASE}"},
           **tables(units_by_ds, scores, data)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def _count(r: Optional[dict]) -> str:
    return "–" if not r else f"{r['k']}/{r['n']}"


def markdown(doc: dict) -> str:
    g = "all" if "all" in doc["results"] else next(iter(doc["results"]))
    res, dif = doc["results"][g], doc["differences"][g]
    arms = list(doc["configs"]) + doc["reference"]
    lines = [f"# Presets and component ablations — {doc['split']}", "",
             f"`{doc['command']}`", "",
             f"Commit {doc['git'].get('commit', '?')[:12]}, library code {doc['code'][:12]}. Group `{g}`. "
             f"Leak = protect values left in the text (injected slice); spurious = edits on natural text "
             f"that touch no gold value. Δ = arm − {doc['base']}, paired cluster bootstrap 95 % CI "
             f"(* excludes 0). The p50 is in-run (not quiet).", "",
             "| arm | config hash | injected leak | Δ leak | shift leak | multi leak | natural spurious "
             "| Δ spurious rate | negatives untouched | in-run p50 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        info = doc["configs"].get(a, {})
        inj, nat = res["injected"][a], res.get("natural", {}).get(a)
        d = dif.get("injected", {}).get(a, {})
        dn = dif.get("natural", {}).get(a, {})
        shift = res.get("shift", {}).get(a, {}).get("leak")
        multi = res.get("multi", {}).get(a, {}).get("leak")
        lines.append(f"| `{a}` | {info.get('detection_config_hash', '')[:16] or '–'} "
                     f"| {_count(inj['leak'])} {score._pct(inj['leak'])} | {score._diff(d.get('leak_rate'))} "
                     f"| {_count(shift)} | {_count(multi)} "
                     f"| {_count(nat['spurious']) if nat else '–'} | {score._diff(dn.get('spurious_rate'))} "
                     f"| {_count(nat['negatives_untouched']) if nat else '–'} "
                     f"| {info.get('in_run_p50_ms') if info.get('in_run_p50_ms') is not None else '–'} |")
    types = sorted({t for a in arms for t in res["injected"][a]["values_by_type"]})
    lines += ["", "Leaked values by type (injected):", "",
              "| arm | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for a in arms:
        r = res["injected"][a]
        lines.append(f"| `{a}` | " + " | ".join(f"{r['leaked_by_type'].get(t, 0)}/{r['values_by_type'].get(t, 0)}"
                                                for t in types) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(score.RUNS))
    ap.add_argument("--configs", nargs="+", default=list(RUN), help=f"files under {score.rel(CONFIGS)}/")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--reuse", action="store_true", help="keep span files made on this code from the same file")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if BASE not in a.configs:
        raise SystemExit(f"--configs must include {BASE} (the differences are taken from it)")
    doc = components(a.split, a.out.resolve(), a.configs, a.datasets, reuse=a.reuse)
    for arm, r in doc["results"].get("all", next(iter(doc["results"].values())))["injected"].items():
        print(f"{arm:18s} leak {r['leak']['k']}/{r['leak']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
