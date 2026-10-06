"""V3 §3.7: an off-the-shelf yardstick on dev. SurrogateShield with its
ContextGuard model swapped, through a config file alone
(``bench/arms/yardstick_isotonic.json``, read through the product's
``SURROGATESHIELD_DETECTION_CONFIG``), for
``Isotonic/deberta-v3-base_finetuned_ai4privacy_v2`` with its labels mapped
onto our types, next to SS as configured and the two GLiNER-PII arms.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.yardstick --out bench/results/yardstick_dev.json

The model is CC-BY-NC-4.0 and its training set
(``ai4privacy/pii-masking-200k``) declares no licence: a local reference for
non-commercial evaluation only, never a default, never shipped, never
fine-tuned from, distilled or used to label data (``LICENCES``). Both SS runs
are made here at the current commit with the same seeds; ``gliner_pii`` is its
committed dev span files and ``gliner_pii_tuned`` the sweep's chosen
configuration on the sweep's span files. Counts only: no text and no value
leaves this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from bench.arms import gliner_pii_tuned as tuned
from bench.arms.base import arm_env
from bench.arms.run import PRIVATE
from bench.realdata import gliner_sweep, score
from bench.realdata.common import DATASETS, ROOT, file_sha256, git_state, read_jsonl

SPLIT = "dev"
CONFIG = ROOT / "bench" / "arms" / "yardstick_isotonic.json"
ENV_FILE, ENV_PRESET = "SURROGATESHIELD_DETECTION_CONFIG", "SURROGATESHIELD_PRESET"
OUT = PRIVATE / "yardstick"
MODEL = "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2"
LICENCES = {
    "model": {"repo": MODEL, "revision": "9814d1113e03f72e8d383423188a51b43d5534c9",
              "licence": "cc-by-nc-4.0", "source": "model card front matter (local cache)"},
    "training_data": {"repo": "ai4privacy/pii-masking-200k", "revision": "7c22e6816b8fc24b7781e9f4e22016c698842dd0",
                      "licence": "none declared",
                      "source": "Hub API tags and card front matter, checked 2026-10-05"},
    "use": "local non-commercial evaluation only (V3 §3.7); never a default, shipped, fine-tuned from, "
           "distilled or used to label data",
}
# run name -> config file merged on the benchmark config (None: as configured)
SS_RUNS: Dict[str, Optional[Path]] = {"ss": None, "ss_isotonic": CONFIG}


def run_ss(config: Optional[Path], src: Path, out: Path) -> None:
    env = arm_env()
    env.pop(ENV_PRESET, None)
    env.pop(ENV_FILE, None)
    if config is not None:
        env[ENV_FILE] = str(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / ".venv" / "bin" / "python"), "-m", "bench.arms.ss", "--in", str(src), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ss ({config or 'as configured'}) failed:\n{proc.stderr[-2000:]}")


def yardstick(out: Path, datasets: Sequence[str] = DATASETS, reuse: bool = False, log=print) -> dict:
    hashes, loaded = score.load_split(SPLIT, datasets)
    sweep_doc = json.loads((ROOT / "bench" / "results" / "gliner_sweep_dev.json").read_text())
    chosen = sweep_doc["chosen"]
    if (tuned.LABEL_SET, tuned.THRESHOLD) != (chosen["label_set"], chosen["threshold"]):
        raise SystemExit("gliner_pii_tuned is not frozen at the sweep's choice")
    arms = list(SS_RUNS) + ["gliner_pii", "gliner_pii_tuned"]
    edits: Dict[str, Dict[str, dict]] = {a: {} for a in arms}
    meta: Dict[str, dict] = {a: {} for a in arms}
    for ds in datasets:
        units, src, sha = loaded[ds]["units"], loaded[ds]["src"], loaded[ds]["input_sha"]
        for name, cfg in SS_RUNS.items():
            path = OUT / name / f"{SPLIT}-{ds}.jsonl"
            if not reuse:
                run_ss(cfg, src, path)
            edits[name][ds] = score.read_spans(path, units, sha)
            m = json.loads(Path(str(path) + ".meta.json").read_text())
            meta[name][ds] = {"detection_config_hash": m["config"]["detection_config_hash"],
                              "load_seconds": m["load_seconds"],
                              "p50_ms": round(statistics.median(r["ms"] for r in read_jsonl(path)), 1)}
            log(f"{name:12} {ds:9} {len(units)} messages")
        edits["gliner_pii"][ds] = score.read_spans(PRIVATE / "gliner_pii" / f"{SPLIT}-{ds}.jsonl", units, sha)
        sw = gliner_sweep.read_sweep(gliner_sweep.spans_path(chosen["label_set"], ds), units, sha)
        edits["gliner_pii_tuned"][ds] = {mid: {"edits": gliner_sweep.edits_at(sp, chosen["threshold"])}
                                         for mid, sp in sw.items()}
    for name in SS_RUNS:
        seen = {v["detection_config_hash"] for v in meta[name].values()}
        if len(seen) != 1:
            raise SystemExit(f"{name}: span files made with different detection configs")
    if meta["ss"][datasets[0]]["detection_config_hash"] == meta["ss_isotonic"][datasets[0]]["detection_config_hash"]:
        raise SystemExit("the yardstick config did not change the detection config")
    results = {}
    for a in arms:
        per, pu, ps = {}, [], []
        for ds in datasets:
            units = loaded[ds]["units"]
            sc = [score.score_unit(u, edits[a][ds][u["mid"]]) for u in units]
            per[ds] = gliner_sweep.summary(units, sc)
            pu += units
            ps += sc
        results[a] = {"all": gliner_sweep.summary(pu, ps), **per}
        inj = results[a]["all"]["injected"]
        log(f"{a:16} injected leak {inj['leak']['k']}/{inj['leak']['n']} = {inj['leak']['rate']}")
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.yardstick "
                      f"--out {score.rel(out)}",
           "git": git_state(), "split": SPLIT, "role": "off-the-shelf reference (V3 §3.7); dev only",
           "frozen": hashes, "datasets": list(datasets),
           "inputs": {ds: {"arm_input_sha256": loaded[ds]["input_sha"], "messages": len(loaded[ds]["units"])}
                      for ds in datasets},
           "licences": LICENCES,
           "yardstick_config": {"file": score.rel(CONFIG), "sha256": file_sha256(CONFIG),
                                "via": f"${ENV_FILE} merged on the benchmark config (bench/arms/ss.py)"},
           "ss_runs": meta, "gliner_pii_tuned": chosen, "arms": arms, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def markdown(doc: dict) -> str:
    r = doc["results"]
    types = sorted({t for a in doc["arms"] for t in r[a]["all"]["injected"]["leaked_by_type"]})
    rate = gliner_sweep._rate
    lines = [f"# Off-the-shelf yardstick on {doc['split']} (V3 §3.7)", "", f"`{doc['command']}`", "",
             f"git {doc['git']['commit'][:7]}"
             f"{' (modified: ' + ', '.join(doc['git']['modified']) + ')' if doc['git']['modified'] else ''}. "
             f"`ss_isotonic` is SS with `{doc['yardstick_config']['file']}` "
             f"(sha256 {doc['yardstick_config']['sha256'][:12]}) merged on the benchmark config; "
             f"`gliner_pii_tuned` is `{doc['gliner_pii_tuned']['label_set']}` at "
             f"{doc['gliner_pii_tuned']['threshold']}.", "",
             f"Licences: model {doc['licences']['model']['licence']}; training data "
             f"{doc['licences']['training_data']['licence']}. {doc['licences']['use']}.", "",
             "| arm | injected leak (95 %) | macro | message leak | natural spurious edits | negatives untouched | p50 ms |",
             "|---|---|---|---|---|---|---|"]
    for a in doc["arms"]:
        inj, nat = r[a]["all"]["injected"], r[a]["all"]["natural"]
        p50 = (" / ".join(str(v["p50_ms"]) for v in doc["ss_runs"][a].values()) if a in doc["ss_runs"] else "–")
        lines.append(f"| {a} | {rate(inj['leak'])} | {inj['macro_leak_rate']} | {rate(inj['message_leak'])} | "
                     f"{nat['spurious']['k']} of {nat['edits']} | {rate(nat['negatives_untouched'])} | {p50} |")
    lines += ["", "Injected values leaked by type (pooled):", "",
              "| arm | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for a in doc["arms"]:
        lb = r[a]["all"]["injected"]["leaked_by_type"]
        lines.append(f"| {a} | " + " | ".join(str(lb.get(t, 0)) for t in types) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reuse", action="store_true", help="read existing yardstick span files")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "yardstick_dev.json")
    a = ap.parse_args(argv)
    yardstick(a.out.resolve(), reuse=a.reuse)
    print(f"-> {score.rel(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
