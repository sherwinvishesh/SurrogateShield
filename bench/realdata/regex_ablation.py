"""E6 — regex robustness (A1, H6): SurrogateShield with PatternScan
recogniser families dropped or made wrong, on one frozen split.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.regex_ablation --split dev --out bench/results/realdata_regex_ablation_dev.json
    ... --split test2 --out bench/results/realdata_regex_ablation_test2.json   # E6'': once, after FREEZE.json (V3 Phase 3)
    ... --split test3 --out bench/results/realdata_regex_ablation_test3.json   # E6''': once, after test3/FREEZE.json (V4)
    ... --reuse        # rescore saved spans whose input hash still matches; runs nothing

1. The units and arm inputs are E5's (``score.load_split``): the same frozen
   files, the same messages, the same input hashes. ``test2`` and ``test3``
   are refused unless their freeze passes (``score.seal``), as E5 is.
2. ``bench.arms.ss_ablate`` runs as a subprocess in ``.venv`` on each
   dataset's input and writes one private span file per condition
   (``build/spans/ss_ablate/<split>-<ds>/<condition>.jsonl``, 0600). This
   script never imports it; it reads the condition list the arm wrote.
3. Every condition is scored with ``score.score_unit`` / ``score.aggregate``
   (the E5 definitions). Condition ``none`` is compared edit for edit with
   arm ``ss``'s E5 spans of the same input.
4. Leak under a condition minus leak under ``none``: paired bootstrap over
   conversations (``score.bootstrap``, 2,000 resamples).
5. Attribution: each protect value that ``none`` protects with a PatternScan
   edit (source ``pattern``) is followed under the condition: leaked, or
   protected by an edit of ``pattern`` (another family), ``ner``
   (EntityTrace), ``slm`` (ContextGuard), ``structural`` or ``repeat``.
6. H6 as pre-registered: SS's leak rate under every condition is below
   Presidio-default's unablated leak rate (E5 spans of the same input), per
   dataset, on the injected slice. Counts only; no text leaves this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

from bench import realworld as rw
from bench.arms.base import arm_env
from bench.realdata import score as S
from bench.realdata.common import BUILD, COLLECTIONS, DATASETS, ROOT, commit_note, derive_seed, git_state, read_jsonl

PRIVATE = BUILD / "spans"
REFERENCE = "presidio_default"
DRAW_SEED = derive_seed("regex-ablation-draws")       # one set of random draws for every split and dataset
SLICES = S.SLICES
FIELDS = ("leak", "message_leak", "macro_leak_rate", "edits", "spurious", "refused", "messages", "protect_values")


def run_ablate(src: Path, out_dir: Path, seed: int) -> None:
    env = arm_env()
    cmd = [str(ROOT / ".venv" / "bin" / "python"), "-m", "bench.arms.ss_ablate", "--in", str(src),
           "--out-dir", str(out_dir), "--seed", str(seed)]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"ss_ablate failed on {S.rel(src)} (exit {proc.returncode}):\n{tail}")


def value_status(unit: dict, row: dict) -> List[dict]:
    """Per protect value of *unit*: ``leaked``, ``policy`` and the cascade
    stages of the edits that cover it (``"+"``-joined, sorted)."""
    gold = unit["gold"]
    text = gold["text"]
    if row.get("refused"):
        return [{"type": x["type"], "leaked": False, "policy": False, "by": "refused"} for x in gold["protect"]]
    edits = [(s, e, text[s:e], rep) for s, e, _t, rep in row["edits"]]
    r = rw.score_message(gold, SimpleNamespace(edits=edits))
    sources = row.get("sources") or ["unknown"] * len(edits)
    out = []
    for x in gold["protect"]:
        occ = rw.occurrences(text, x["value"])
        by = {src for (a, b, _o, _r), src in zip(edits, sources) for s, e in occ if a < e and s < b}
        out.append({"type": x["type"], "leaked": any(x is y for y in r["leaked"]),
                    "policy": any(x is y for y in r["policy"]), "by": "+".join(sorted(by)) or "-"})
    return out


def attribution(base: Sequence[List[dict]], cond: Sequence[List[dict]]) -> dict:
    """Values ``none`` protects with a PatternScan edit, followed under the
    condition; and every value lost (protected → leaked) or gained."""
    caught, now, lost, gained = 0, Counter(), Counter(), 0
    for vb, vc in zip(base, cond):
        for b, c in zip(vb, vc):
            if b["policy"] or c["policy"]:
                continue
            if not b["leaked"] and c["leaked"]:
                lost[b["type"]] += 1
            if b["leaked"] and not c["leaked"]:
                gained += 1
            if not b["leaked"] and "pattern" in b["by"].split("+"):
                caught += 1
                now["leaked" if c["leaked"] else ("pattern" if "pattern" in c["by"].split("+") else c["by"])] += 1
    return {"pattern_caught_in_none": caught, "under_condition": dict(sorted(now.items())),
            "recovered_by_other_stage": sum(v for k, v in now.items() if k not in ("leaked", "pattern")),
            "lost": dict(sorted(lost.items())), "lost_total": sum(lost.values()), "gained": gained}


def _trim(agg: dict) -> dict:
    out = {k: agg[k] for k in FIELDS}
    out["by_type"] = {t: {"values": v["values"], "leaked": v["leaked"]} for t, v in agg["by_type"].items()}
    return out


def _same_edits(a: Dict[str, dict], b: Dict[str, dict]) -> int:
    return sum(1 for mid in a if a[mid]["edits"] != b[mid]["edits"] or bool(a[mid].get("refused")) != bool(b[mid].get("refused")))


def ablate_split(split: str, datasets: Sequence[str] = DATASETS, reuse: bool = False, out: Optional[Path] = None,
                 rd: Optional[Path] = None, build: Optional[Path] = None, frozen: Optional[dict] = None,
                 runner: Optional[Callable] = None, spans: Optional[Path] = None, log=print,
                 freeze: Optional[Path] = None, prereg: Optional[Path] = None) -> dict:
    runner = runner or run_ablate
    spans = spans or PRIVATE
    data, coll_name, _role = S.RUNS[split]
    coll = COLLECTIONS[coll_name]
    rd, build = rd or coll.rd, build or coll.build
    sealed = S.seal(coll, freeze, prereg)
    hashes, loaded = S.load_split(data, datasets, rd, build, frozen, coll.prefix)
    names: Optional[List[str]] = None
    run_info = {}
    units: Dict[str, List[dict]] = {}
    scores: Dict[str, Dict[str, List[dict]]] = {}
    status: Dict[str, Dict[str, List[List[dict]]]] = {}
    ref: Dict[str, List[dict]] = {}
    none_vs_ss = {}
    for ds in datasets:
        u, src, sha = loaded[ds]["units"], loaded[ds]["src"], loaded[ds]["input_sha"]
        d = spans / "ss_ablate" / f"{split}-{ds}"
        if not reuse:
            runner(src, d, DRAW_SEED)
        info = json.loads((d / "run.json").read_text())
        ds_names = [c["name"] for c in info["conditions"]]
        if names is not None and ds_names != names:
            raise SystemExit(f"{ds}: conditions differ from the other datasets'")
        names, run_info[ds] = ds_names, info
        units[ds], scores[ds], status[ds] = u, {}, {}
        rows_by = {}
        for c in names:
            rows = S.read_spans(d / f"{c}.jsonl", u, sha)
            meta = json.loads(Path(str(d / f"{c}.jsonl") + ".meta.json").read_text())
            if meta.get("seed") != DRAW_SEED:
                raise SystemExit(f"{ds}/{c}: drawn with seed {meta.get('seed')}, not {DRAW_SEED}")
            rows_by[c] = rows
            scores[ds][c] = [S.score_unit(x, rows[x["mid"]]) for x in u]
            status[ds][c] = [value_status(x, rows[x["mid"]]) for x in u]
        pd_rows = S.read_spans(spans / REFERENCE / f"{split}-{ds}.jsonl", u, sha)
        ref[ds] = [S.score_unit(x, pd_rows[x["mid"]]) for x in u]
        ss_file = spans / "ss" / f"{split}-{ds}.jsonl"
        none_vs_ss[ds] = (_same_edits(rows_by["none"], S.read_spans(ss_file, u, sha)) if ss_file.exists() else None)
        log(f"{split} {ds:9} {len(names)} conditions scored on {len(u)} messages; "
            f"none vs ss: {none_vs_ss[ds]} messages differ")
    groups = list(datasets) + (["all"] if len(datasets) > 1 else [])
    results, reference = {}, {}
    for g in groups:
        dss = datasets if g == "all" else [g]
        us = [x for ds in dss for x in units[ds]]
        results[g], reference[g] = {}, {}
        for sl in SLICES:
            keep = [i for i, x in enumerate(us) if sl in x["slices"]]
            sub = [us[i] for i in keep]
            clusters = [x["conv"] for x in sub]
            ref_sc = [s for ds in dss for s in ref[ds]]
            reference[g][sl] = _trim(S.aggregate(sub, [ref_sc[i] for i in keep]))
            base_sc = [s for ds in dss for s in scores[ds]["none"]]
            base_st = [v for ds in dss for v in status[ds]["none"]]
            base_pairs = [S.METRICS["leak_rate"](base_sc[i]) for i in keep]
            results[g][sl] = {}
            for c in names:
                sc = [s for ds in dss for s in scores[ds][c]]
                st = [v for ds in dss for v in status[ds][c]]
                row = _trim(S.aggregate(sub, [sc[i] for i in keep]))
                row["attribution"] = attribution([base_st[i] for i in keep], [st[i] for i in keep])
                if c != "none":
                    row["vs_none"] = S.bootstrap(clusters, [S.METRICS["leak_rate"](sc[i]) for i in keep], base_pairs,
                                                 derive_seed("regex-ablation", split, g, sl, c))
                results[g][sl][c] = row
    h6 = {}
    for ds in datasets:
        pd_leak = reference[ds]["injected"]["leak"]["rate"]
        leaks = {c: results[ds]["injected"][c]["leak"]["rate"] for c in names}
        worst = max((c for c in names if leaks[c] is not None), key=lambda c: (leaks[c], c), default=None)
        above = sorted(c for c in names if leaks[c] is not None and pd_leak is not None and leaks[c] >= pd_leak)
        h6[ds] = {"presidio_default_leak": pd_leak, "worst_condition": worst,
                  "worst_leak": leaks[worst] if worst else None, "conditions_at_or_above": above,
                  "H6": pd_leak is not None and worst is not None and not above}
    out = out or ROOT / "bench" / "results" / f"realdata_regex_ablation_{split}.json"
    first = run_info[datasets[0]]
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.regex_ablation "
                      f"--split {split} --out {S.rel(out)}",
           "git": git_state(),
           "split": split, "role": "the paper's robustness numbers" if split in ("test", "test2", "test3") else "diagnosis only",
           **({"freeze_sha256": sealed} if sealed else {}),
           "frozen": hashes, "draw_seed": DRAW_SEED, "families": first["families"],
           "conditions": first["conditions"], "versions": first["versions"],
           "corpus": {ds: loaded[ds]["corpus"] for ds in datasets},
           "none_vs_ss_messages_differing": none_vs_ss, "reference_arm": REFERENCE,
           "bootstrap": {"resamples": S.RESAMPLES, "unit": "conversation", "difference": "condition − none",
                         "seed": "derive_seed('regex-ablation', split, group, slice, condition)"},
           "reference": reference, "results": results, "H6": h6}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def _pct(r) -> str:
    return S._pct(r)


def markdown(doc: dict) -> str:
    groups = [g for g in doc["results"]]
    g0 = "all" if "all" in groups else groups[0]
    res, ref = doc["results"][g0], doc["reference"][g0]
    names = [c["name"] for c in doc["conditions"]]
    lines = [f"# Regex robustness (E6), split `{doc['split']}` ({doc['role']})", "",
             f"Command: `{doc['command']}`" + commit_note(doc.get("git")), "",
             f"Families: {len(doc['families'])}; conditions: {len(names)}; draw seed {doc['draw_seed']}. "
             f"`none` vs arm `ss` (messages whose edits differ): {doc['none_vs_ss_messages_differing']}.", "",
             f"## Pooled ({g0}): leak rate per condition", "",
             f"Reference, {doc['reference_arm']} unablated: injected {_pct(ref['injected']['leak'])}, "
             f"shift {_pct(ref['shift']['leak'])}, natural {_pct(ref['natural']['leak'])}.", "",
             "| condition | injected leak | Δ vs none [95 % CI] | shift leak | multi leak | natural leak "
             "| pattern-caught values | recovered by another stage | lost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in names:
        r = res["injected"][c]
        a = r["attribution"]
        lines.append(f"| `{c}` | {_pct(r['leak'])} | {S._diff(r.get('vs_none'))} | {_pct(res['shift'][c]['leak'])} "
                     f"| {_pct(res['multi'][c]['leak'])} | {_pct(res['natural'][c]['leak'])} "
                     f"| {a['pattern_caught_in_none']} | {a['recovered_by_other_stage']} | {a['lost_total']} |")
    lines += ["", "## Where values caught by a regex went (injected, pooled)", "",
              "| condition | " + " | ".join(("leaked", "pattern", "ner", "slm", "other")) + " |", "|---|---|---|---|---|---|"]
    for c in names:
        u = res["injected"][c]["attribution"]["under_condition"]
        other = sum(v for k, v in u.items() if k not in ("leaked", "pattern", "ner", "slm"))
        lines.append(f"| `{c}` | " + " | ".join(str(u.get(k, 0)) for k in ("leaked", "pattern", "ner", "slm"))
                     + f" | {other} |")
    lines += ["", "## H6 per dataset (injected slice)", "",
              "| dataset | Presidio-default leak | worst condition | its leak | conditions at or above | H6 |",
              "|---|---|---|---|---|---|"]
    for ds, h in doc["H6"].items():
        lines.append(f"| {ds} | {h['presidio_default_leak']} | `{h['worst_condition']}` | {h['worst_leak']} "
                     f"| {', '.join(h['conditions_at_or_above']) or '—'} | {'yes' if h['H6'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=("dev", "test", "test2", "test3"), required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    doc = ablate_split(args.split, args.datasets, args.reuse, args.out)
    print(json.dumps(doc["H6"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
