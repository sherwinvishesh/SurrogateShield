"""V3 §3.8: the GLiNER-PII sweep on dev that picks the ``gliner_pii_tuned``
baseline, so that H1'' is tested against GLiNER at its best configuration on
our data and not only at the model card's.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.gliner_sweep --out bench/results/gliner_sweep_dev.json

Grid: thresholds ``gliner_pii_tuned.THRESHOLDS`` × label sets
``gliner_pii_tuned.LABEL_SETS`` (the model card's 22 labels; our 13 types in
words with natural aliases). GLiNER decodes a window by taking every span
whose probability exceeds the threshold and keeping, highest score first, the
spans that overlap no kept span (flat NER). A span above a higher threshold is
therefore kept or dropped exactly as at the lowest one, so each label set is
run once, at the lowest threshold with every window span's score kept
(``gliner_pii_tuned --sweep``, in ``.venv-baselines``), and each threshold is
that output filtered to ``score > threshold`` (compared in float32, as the
model compares) before the arm's own overlap rule and ``[LABEL]`` edits. The
filter is checked against the committed arm: the model card's labels at 0.5
must give ``gliner_pii``'s dev edits on every message.

Each configuration is scored on dev's injected and natural slices with
``score.score_unit`` / ``score.aggregate`` (the frozen definitions). Chosen:
the lowest pooled injected leak rate; ties go to fewer natural spurious edits,
then to the higher threshold. Counts only: no text and no value leaves this
script; the span files stay under the private build directory.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from bench.arms import gliner_pii, gliner_pii_tuned as tuned
from bench.arms.base import arm_env
from bench.arms.run import PRIVATE
from bench.realdata import score
from bench.realdata.common import DATASETS, ROOT, git_state, read_jsonl

SPLIT = "dev"
SWEEP = PRIVATE / "gliner_sweep"


def f32(x: float) -> float:
    """*x* rounded to float32, the precision GLiNER compares scores in."""
    return struct.unpack("f", struct.pack("f", x))[0]


def edits_at(spans: Sequence[Sequence], threshold: float) -> List[list]:
    """The arm's edits at *threshold* from the scored window spans found at
    a lower one."""
    t = f32(threshold)
    return gliner_pii.to_edits([tuple(x) for x in spans if x[3] > t])


def choose(configs: Sequence[dict]) -> dict:
    """Lowest pooled injected leak; then fewer natural spurious edits; then
    the higher threshold."""
    return min(configs, key=lambda c: (c["all"]["injected"]["leak"]["rate"],
                                       c["all"]["natural"]["spurious"]["k"], -c["threshold"]))


def spans_path(labels: str, ds: str) -> Path:
    return SWEEP / labels / f"{SPLIT}-{ds}.jsonl"


def run_spans(labels: str, src: Path, out: Path) -> None:
    cmd = [str(ROOT / ".venv-baselines" / "bin" / "python"), "-m", "bench.arms.gliner_pii_tuned",
           "--sweep", labels, "--in", str(src), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, env=arm_env(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"gliner sweep {labels} failed:\n{proc.stderr[-2000:]}")


def read_sweep(path: Path, units: Sequence[dict], input_sha: str) -> Dict[str, list]:
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    if meta["input_sha256"] != input_sha:
        raise SystemExit(f"{path.parent.name}/{path.name}: made from another input; rerun without --reuse")
    rows = {r["id"]: r["spans"] for r in read_jsonl(path)}
    if set(rows) != {u["mid"] for u in units}:
        raise SystemExit(f"{path.parent.name}/{path.name}: ids differ from the input")
    return rows


def summary(units: Sequence[dict], scores: Sequence[dict]) -> dict:
    out = {}
    for sl in ("injected", "natural"):
        keep = [i for i, u in enumerate(units) if sl in u["slices"]]
        a = score.aggregate([units[i] for i in keep], [scores[i] for i in keep])
        out[sl] = {k: a[k] for k in ("messages", "protect_values", "leak", "macro_leak_rate", "message_leak",
                                     "edits", "spurious", "messages_with_spurious", "negatives_untouched")}
        out[sl]["leaked_by_type"] = {t: v["leaked"] for t, v in a["by_type"].items()}
    return out


def sweep(out: Path, labels: Sequence[str] = tuple(tuned.LABEL_SETS), datasets: Sequence[str] = DATASETS,
          reuse: bool = False, log=print) -> dict:
    hashes, loaded = score.load_split(SPLIT, datasets)
    spans: Dict[str, Dict[str, Dict[str, list]]] = {}
    span_meta = {}
    for ls in labels:
        spans[ls], span_meta[ls] = {}, {}
        for ds in datasets:
            path = spans_path(ls, ds)
            if not reuse:
                run_spans(ls, loaded[ds]["src"], path)
            spans[ls][ds] = read_sweep(path, loaded[ds]["units"], loaded[ds]["input_sha"])
            m = json.loads(Path(str(path) + ".meta.json").read_text())
            span_meta[ls][ds] = {k: m[k] for k in ("model_revision", "threshold", "messages", "load_seconds",
                                                   "offline", "versions")}
            log(f"{ls:9} {ds:9} {len(spans[ls][ds])} messages")

    check = {}
    if "published" in labels:
        for ds in datasets:
            ref = score.read_spans(PRIVATE / "gliner_pii" / f"{SPLIT}-{ds}.jsonl", loaded[ds]["units"],
                                   loaded[ds]["input_sha"])
            got = {mid: edits_at(sp, gliner_pii.THRESHOLD) for mid, sp in spans["published"][ds].items()}
            differ = sorted(mid for mid in got if got[mid] != [list(e) for e in ref[mid]["edits"]])
            check[ds] = {"messages": len(got), "equal": len(got) - len(differ)}
            if differ:
                raise SystemExit(f"published@{gliner_pii.THRESHOLD} differs from gliner_pii on {ds}: "
                                 f"{len(differ)} messages; the threshold filter is not exact")

    configs = []
    for ls in labels:
        for t in tuned.THRESHOLDS:
            c = {"label_set": ls, "threshold": t}
            pooled_u, pooled_s = [], []
            for ds in datasets:
                units = loaded[ds]["units"]
                sc = [score.score_unit(u, {"edits": edits_at(spans[ls][ds][u["mid"]], t)}) for u in units]
                c[ds] = summary(units, sc)
                pooled_u += units
                pooled_s += sc
            c["all"] = summary(pooled_u, pooled_s)
            configs.append(c)
            inj = c["all"]["injected"]
            log(f"{ls:9} {t:.1f} injected leak {inj['leak']['k']}/{inj['leak']['n']} = {inj['leak']['rate']}, "
                f"natural spurious {c['all']['natural']['spurious']['k']}")
    best = choose(configs)
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.gliner_sweep "
                      f"--out {score.rel(out)}",
           "git": git_state(), "split": SPLIT, "role": "selects the gliner_pii_tuned baseline (V3 §3.8); dev only",
           "frozen": hashes, "datasets": list(datasets),
           "inputs": {ds: {"arm_input_sha256": loaded[ds]["input_sha"], "messages": len(loaded[ds]["units"])}
                      for ds in datasets},
           "model": gliner_pii.MODEL, "window_words": gliner_pii.WINDOW, "overlap_words": gliner_pii.OVERLAP,
           "thresholds": list(tuned.THRESHOLDS),
           "label_sets": {ls: tuned.LABEL_TYPES[ls] for ls in labels},
           "method": "one run per label set at the lowest threshold with window-span scores; each threshold "
                     "keeps score > threshold (float32), then gliner_pii's overlap rule and [LABEL] edits",
           "span_runs": span_meta,
           "filter_check": {"config": f"published@{gliner_pii.THRESHOLD}",
                            "reference": f"gliner_pii {SPLIT} span files", "datasets": check},
           "rule": "lowest pooled injected leak rate; ties: fewer natural spurious edits, then higher threshold",
           "configs": configs,
           "chosen": {"label_set": best["label_set"], "threshold": best["threshold"]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def _rate(r: dict) -> str:
    if r["rate"] is None:
        return "–"
    lo, hi = r["wilson95"]
    return f"{r['rate']:.4f} ({r['k']}/{r['n']}; {lo:.3f}–{hi:.3f})"


def markdown(doc: dict) -> str:
    types = sorted({t for c in doc["configs"] for t in c["all"]["injected"]["leaked_by_type"]})
    ch = doc["chosen"]
    lines = [f"# GLiNER-PII sweep on {doc['split']} (V3 §3.8)", "",
             f"`{doc['command']}`", "",
             f"git {doc['git']['commit'][:7]}{' (modified: ' + ', '.join(doc['git']['modified']) + ')' if doc['git']['modified'] else ''}; "
             f"model `{doc['model']}`; {doc['method']}.", "",
             "Filter check (" + doc["filter_check"]["config"] + " vs the committed arm): " +
             (", ".join(f"{ds} {v['equal']}/{v['messages']}" for ds, v in doc["filter_check"]["datasets"].items())
              or "not run") + " messages equal.", "",
             f"**Chosen: `{ch['label_set']}` at {ch['threshold']}** ({doc['rule']}).", "",
             "| label set | threshold | injected leak (95 %) | macro | message leak | natural spurious edits | negatives untouched |",
             "|---|---|---|---|---|---|---|"]
    for c in doc["configs"]:
        inj, nat = c["all"]["injected"], c["all"]["natural"]
        mark = "**" if (c["label_set"], c["threshold"]) == (ch["label_set"], ch["threshold"]) else ""
        lines.append(f"| {mark}{c['label_set']}{mark} | {c['threshold']} | {_rate(inj['leak'])} | "
                     f"{inj['macro_leak_rate']} | {_rate(inj['message_leak'])} | {nat['spurious']['k']} of "
                     f"{nat['edits']} | {_rate(nat['negatives_untouched'])} |")
    lines += ["", "Injected values leaked by type (pooled):", "",
              "| label set | threshold | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types)]
    for c in doc["configs"]:
        lb = c["all"]["injected"]["leaked_by_type"]
        lines.append(f"| {c['label_set']} | {c['threshold']} | " + " | ".join(str(lb.get(t, 0)) for t in types) + " |")
    lines += ["", "Per dataset (injected leak):", "",
              "| label set | threshold | " + " | ".join(doc["datasets"]) + " |", "|---|---|" + "---|" * len(doc["datasets"])]
    for c in doc["configs"]:
        lines.append(f"| {c['label_set']} | {c['threshold']} | " +
                     " | ".join(_rate(c[ds]["injected"]["leak"]) for ds in doc["datasets"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", nargs="+", choices=sorted(tuned.LABEL_SETS), default=list(tuned.LABEL_SETS))
    ap.add_argument("--reuse", action="store_true", help="read existing sweep span files instead of running GLiNER")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "gliner_sweep_dev.json")
    a = ap.parse_args(argv)
    doc = sweep(a.out.resolve(), a.labels, reuse=a.reuse)
    ch = doc["chosen"]
    print(f"chosen {ch['label_set']} @ {ch['threshold']} -> {score.rel(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
