"""V4 §3.1 A: the tagger's AGE threshold, chosen on dev and dev-large's calib half.

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.tagger.age_sweep \\
        --model bench/tagger/build/models/dv3s-40k-v4 --out bench/results/age_sweep_dev.json

SurrogateShield runs over dev and the whole of dev-large (``evaluate.with_ss``,
the benchmark config of this checkout with the tagger at *--model* switched
on) once with AGE left to the patterns (variant ``age-off``) and once per
threshold with the tagger allowed for AGE beside them at that threshold
(``age<t>``). Only dev and the calib half are scored here: the val half is the
acceptance run's (``evaluate --as-configured``, §3.3).

The reference is ``balanced`` at ``a1abaa92142b0db9`` on the same messages:
the spans ``evaluate --as-configured --variant balanced`` made with
``dv3s-40k``, kept under the git-ignored ``bench/tagger/build/eval/`` and
checked here by the config hash in their metadata.

The rule (V4 §4 Phase 1 step 4, §3.3 G1): the highest threshold at which AGE
leaks at most max(0.02, GLiNER-PII's rate) of the AGE values on dev and on
calib (0 of 33 on dev) is chosen; ties go to fewer natural AGE edits. If none
meets G1, none is chosen. G2 (no type's leaked count rises against the
reference) and G3 (natural spurious edits not above the reference's,
negatives untouched not fallen, OASST1's included) are recorded on the choice
data beside it; they are gates on dev and dev-large's val half, in the
acceptance run.

Counts only: no text and no value is written. Natural edits of type AGE are
counted by label, ``age`` and ``age:worded`` apart (§5.6).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from bench.realdata import score
from bench.realdata.common import DATASETS, git_state
from bench.tagger import evaluate as E
from surrogateshield.core.detection import config as C

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
REFERENCE = ("dv3s-40k", "balanced", "a1abaa92142b0db9")
CHOICE = (("dev", None), ("devlarge", "calib"))
AGE_ROUTED = ["pattern_scan", "canonicaliser", "structural", "pii_tagger"]
AGE_OFF = ["pattern_scan", "canonicaliser", "structural"]


def variant(t: Optional[float]) -> str:
    return "age-off" if t is None else f"age{round(t * 100)}"


def extra(t: Optional[float]) -> dict:
    """The partial config merged on the tagger's for one variant."""
    if t is None:
        return {"type_sources": {"AGE": AGE_OFF}}
    return {"type_sources": {"AGE": AGE_ROUTED}, "detectors": [{"name": "pii_tagger", "thresholds": {"AGE": t}}]}


def spans(model: str, split: str, name: str, loaded: dict, need_hash: Optional[str] = None) -> dict:
    out = {}
    for ds, v in loaded.items():
        path = E.BUILD / model / f"ss-{split}-{name}-{ds}.jsonl"
        if need_hash:
            got = json.loads(Path(str(path) + ".meta.json").read_text())["config"]["detection_config_hash"]
            if not got.startswith(need_hash):
                raise SystemExit(f"{score.rel(path)} was made with config {got[:16]}, not {need_hash}")
        out[ds] = score.read_spans(path, v["units"], v["input_sha"])
    return out


def is_age(label: str) -> bool:
    return C.public_type(label.split(":")[0]) == "AGE"


def measure(units_by_ds: Dict[str, List[dict]], rows_by_ds: dict) -> dict:
    """Per dataset and pooled: leaks by type, natural spurious edits, negatives
    untouched, natural AGE edits by label."""
    out = {}
    for ds in [*units_by_ds, "all"]:
        dss = list(units_by_ds) if ds == "all" else [ds]
        s = E.score_rows({d: units_by_ds[d] for d in dss}, {d: rows_by_ds[d] for d in dss})["all"]
        inj, nat = s["injected"], s["natural"]
        age_edits = Counter(e[2] for d in dss for u in units_by_ds[d] if "natural" in u["slices"]
                            for e in rows_by_ds[d][u["mid"]]["edits"] if is_age(e[2]))
        out[ds] = {"leak": inj["leak"], "leaked_by_type": inj["leaked_by_type"],
                   "values_by_type": inj["values_by_type"], "natural_spurious": nat["spurious"]["k"],
                   "negatives_untouched": nat["negatives_untouched"],
                   "natural_age_edits": dict(sorted(age_edits.items()))}
    return out


def rate(m: dict, t: str) -> float:
    n = m["values_by_type"].get(t, 0)
    return m["leaked_by_type"].get(t, 0) / n if n else 0.0


def checks(got: dict, ref: dict, gliner: dict, per_type: float = 0.02) -> dict:
    """§3.3 G1–G3 for one variant on one part of the choice data."""
    a, r = got["all"], ref["all"]
    bound = max(per_type, rate(gliner["all"], "AGE"))
    risen = {t: [r["leaked_by_type"].get(t, 0), k] for t, k in a["leaked_by_type"].items()
             if k > r["leaked_by_type"].get(t, 0)}
    fallen = {ds: [ref[ds]["negatives_untouched"]["k"], got[ds]["negatives_untouched"]["k"]]
              for ds in got if got[ds]["negatives_untouched"]["k"] < ref[ds]["negatives_untouched"]["k"]}
    return {"G1": {"ok": rate(a, "AGE") <= bound, "age": [a["leaked_by_type"].get("AGE", 0),
                                                         a["values_by_type"].get("AGE", 0)], "bound": round(bound, 4)},
            "G2": {"ok": not risen, "risen": risen},
            "G3": {"ok": a["natural_spurious"] <= r["natural_spurious"] and not fallen,
                   "spurious": [r["natural_spurious"], a["natural_spurious"]], "negatives_fallen": fallen}}


def natural_age_edits(v: dict) -> int:
    return sum(sum(p["all"]["natural_age_edits"].values()) for p in v.get("parts", {}).values())


def choose(results: dict) -> dict:
    """The highest threshold meeting G1 on every part of the choice data,
    ties to fewer natural AGE edits; with the thresholds meeting all three
    checks there too (a diagnostic)."""
    passing = [t for t, v in results.items() if t != "off" and all(p["G1"]["ok"] for p in v["checks"].values())]
    clean = [t for t in passing if all(c["ok"] for p in results[t]["checks"].values() for c in p.values())]
    if not passing:
        return {"threshold": None, "passing": [], "all_checks": []}
    best = max(passing, key=lambda t: (float(t), -natural_age_edits(results[t])))
    return {"threshold": float(best), "passing": sorted(passing, key=float), "all_checks": sorted(clean, key=float)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reuse", action="store_true", help="keep SS spans made on this code from the same config")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    model = a.model.resolve()
    results: Dict[str, dict] = {}
    for split, half in CHOICE:
        tag = split + (f"-{half}" if half else "")
        loaded = E.load(split, DATASETS)
        units = E.units_for(loaded, half)
        ref = measure(units, spans(REFERENCE[0], split, REFERENCE[1], loaded, REFERENCE[2]))
        gliner = measure(units, {ds: score.read_spans(E.PRIVATE / "gliner_pii" / f"{split}-{ds}.jsonl",
                                                      v["units"], v["input_sha"]) for ds, v in loaded.items()})
        results.setdefault("reference", {})[tag] = ref
        results.setdefault("gliner_pii", {})[tag] = gliner
        for t in [None, *a.thresholds]:
            key = "off" if t is None else str(t)
            E.with_ss(model, split, tag, loaded, units, a.device, a.reuse, variant=variant(t), extra=extra(t))
            got = measure(units, spans(model.name, split, variant(t), loaded))
            res = results.setdefault(key, {"variant": variant(t), "extra": extra(t), "parts": {}, "checks": {}})
            res["parts"][tag] = got
            res["checks"][tag] = checks(got, ref, gliner)
            c = res["checks"][tag]
            print(f"{tag:14} {variant(t):8} AGE {c['G1']['age'][0]}/{c['G1']['age'][1]} G1 {c['G1']['ok']} "
                  f"G2 {c['G2']['ok']} {c['G2']['risen']} G3 {c['G3']['ok']} spurious {c['G3']['spurious']} "
                  f"natural age edits {got['all']['natural_age_edits']}")
    sweep = {k: v for k, v in results.items() if k not in ("reference", "gliner_pii")}
    doc = {"command": f"... -m bench.tagger.age_sweep --model {score.rel(model)} --out {score.rel(a.out.resolve())}",
           "git": git_state(), "code": E.code_stamp(),
           "model": {"path": score.rel(model), "weights_sha256": E.model_sha(model)},
           "reference": {"model": REFERENCE[0], "variant": REFERENCE[1], "config_hash": REFERENCE[2],
                         "parts": results["reference"]},
           "gliner_pii": results["gliner_pii"], "sweep": sweep, "choice": choose(sweep)}
    print(f"chosen: {doc['choice']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    a.out.with_suffix(".md").write_text(markdown(doc))
    return 0


def markdown(doc: dict) -> str:
    lines = ["# The tagger's AGE threshold (V4 §3.1 A)", "",
             f"`{doc['command']}`, model weights `{doc['model']['weights_sha256'][:12]}`, code "
             f"`{doc['code'][:12]}`. Reference: `balanced` at `{doc['reference']['config_hash']}`. "
             "Choice data only: dev and dev-large's calib half.", "",
             "| variant | part | AGE leaked | G1 | G2 (types risen) | natural spurious (ref → got) | G3 "
             "| natural AGE edits |", "|---|---|---|---|---|---|---|---|"]
    for key, v in doc["sweep"].items():
        for part, c in v["checks"].items():
            lines.append(f"| `{v['variant']}` | {part} | {c['G1']['age'][0]}/{c['G1']['age'][1]} "
                         f"(bound {c['G1']['bound']}) | {'yes' if c['G1']['ok'] else 'no'} | "
                         f"{'yes' if c['G2']['ok'] else 'no ' + json.dumps(c['G2']['risen'])} | "
                         f"{c['G3']['spurious'][0]} → {c['G3']['spurious'][1]} | {'yes' if c['G3']['ok'] else 'no'} | "
                         f"{v['parts'][part]['all']['natural_age_edits'] or 0} |")
    ch = doc["choice"]
    lines += ["", f"Chosen by G1: **{ch['threshold']}** (meeting G1: {', '.join(ch['passing'])}; meeting G1–G3 on "
              f"the choice data: {', '.join(ch['all_checks']) or 'none'})." if ch["threshold"]
              else "No threshold meets G1 on the choice data.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
