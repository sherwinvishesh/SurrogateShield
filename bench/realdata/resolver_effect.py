"""What the resolver (V3 §3.5) changes: per value group with more than one
candidate at ``deduplicate``, the old winner (highest score, first on a tie)
against the resolver's, through the SS arm's send path and config.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.resolver_effect \
        --split dev devlarge --out bench/results/resolver_effect.json

Per slice (injected, natural): the groups, how many keep the same winner, and
each change by kind (public type, internal type, or only which source or
occurrence stands for the value) with its type/stage transition. For an
injected value (a gold value's text is the group's text), a public-type
change is counted as old right, new right or neither against the gold
type. Counts only; no text is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from bench.realdata import attribute as AT, score as S
from bench.realdata.common import ROOT, git_state


def old_rule(group):
    best = group[0]
    for e in group[1:]:
        if e.score > best.score:
            best = e
    return best


def run(splits):
    import detection.logic as L
    import json_tester
    from bench.arms.ss import detection_config
    from generation.logic import MimicGen
    from surrogateshield.core.detection import resolver as R
    from surrogateshield.core.detection.config import public_type

    cfg = detection_config()
    groups = []
    real = L.deduplicate

    def spy(ents, config=None):
        by = {}
        for e in ents:
            by.setdefault(e.text.strip(), []).append(e)
        groups.extend(g for g in by.values() if len(g) > 1)
        return real(ents, config)

    L.deduplicate = spy
    out = {}
    try:
        for split in splits:
            data_split, coll_name, _ = S.RUNS[split]
            coll = AT.COLLECTIONS[coll_name]
            _h, loaded = S.load_split(data_split, AT.DATASETS, coll.rd, coll.build, prefix=coll.prefix)
            c = {"injected": Counter(), "natural": Counter()}
            for ds in AT.DATASETS:
                for u in loaded[ds]["units"]:
                    key = "natural" if "natural" in u["slices"] else "injected"
                    gold = {it["value"].strip(): it["type"] for it in u["gold"].get("protect", [])}
                    groups.clear()
                    try:
                        json_tester.prepare_send(u["gold"]["text"], MimicGen(seed=AT.msg_seed("ss", u["mid"])),
                                                 config=cfg)
                    except RuntimeError:
                        c[key]["refused messages"] += 1
                    for g in groups:
                        c[key]["groups"] += 1
                        o, n = old_rule(g), R.pick(g, cfg)
                        if o is n:
                            c[key]["same winner"] += 1
                            continue
                        po, pn = public_type(o.type), public_type(n.type)
                        kind = ("public type" if po != pn else
                                "internal type" if o.type != n.type else "source or occurrence only")
                        c[key][f"changed, {kind}: {po}/{R.stage_of(o)} -> {pn}/{R.stage_of(n)}"] += 1
                        gt = gold.get(g[0].text.strip()) if key == "injected" else None
                        if gt and po != pn:
                            c[key]["gold value, public type " + ("old right" if po == gt else
                                                                 "new right" if pn == gt else "neither")] += 1
            out[split] = {k: dict(sorted(v.items())) for k, v in c.items()}
    finally:
        L.deduplicate = real
    return {"splits": out, "config_hash": cfg.config_hash(), "source_priority": list(cfg.source_priority),
            "type_conflicts": dict(cfg.type_conflicts), "git": git_state()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", nargs="+", choices=list(S.RUNS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":          # as bench.realdata.attribute
        env = {**os.environ, "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-m", "bench.realdata.resolver_effect", *(argv or sys.argv[1:])],
                               env=env, cwd=ROOT)
    doc = run(args.split)
    out = args.out.resolve()
    doc["command"] = ("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.resolver_effect "
                      f"--split {' '.join(args.split)} --out {S.rel(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for split, d in doc["splits"].items():
        print(split, {k: (v["groups"], v["same winner"]) for k, v in d.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
