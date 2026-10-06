"""V3 §3.3 acceptance on the pooled development set: dev + dev-large's val half.

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.accept \\
        bench/tagger/build/eval/dv3s-40k-dev-<variant>.json \\
        bench/tagger/build/eval/dv3s-40k-devlarge-val-<variant>.json \\
        --out bench/results/tagger_acceptance_dev.json

Each input is a ``bench.tagger.evaluate --ss`` result for one split. The
arms' counts (leaked and injected values by type, pooled leak, natural
spurious edits) are summed over the inputs, and ``evaluate.acceptance`` is
applied to the sums, so a type's bound is set by the better GLiNER on the same
pooled rows. The inputs must share the tagger's weights, the SS + tagger
config and the library code; the per-split checks are kept beside the pooled
one. Counts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from bench.realdata import score
from bench.tagger.evaluate import acceptance

ARMS = ("ss_tagger", "ss", "gliner_pii", "gliner_pii_tuned")


def pool(arm_sets: Sequence[Dict[str, dict]]) -> Dict[str, dict]:
    """Per arm, the summed counts of several splits' ``with_ss`` results."""
    out = {}
    for arm in ARMS:
        rs = [a[arm] for a in arm_sets]
        types = sorted({t for r in rs for t in r["values_by_type"]})
        k, n = sum(r["leak"]["k"] for r in rs), sum(r["leak"]["n"] for r in rs)
        out[arm] = {"leak": {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": score.wilson(k, n)},
                    "leaked_by_type": {t: sum(r["leaked_by_type"].get(t, 0) for r in rs) for t in types},
                    "values_by_type": {t: sum(r["values_by_type"].get(t, 0) for r in rs) for t in types},
                    "natural_spurious": sum(r["natural_spurious"] for r in rs),
                    "natural_edits": sum(r["natural_edits"] for r in rs)}
    return out


def same(docs: Sequence[dict], *path: str) -> str:
    vals = set()
    for d in docs:
        v = d
        for p in path:
            v = v[p]
        vals.add(v)
    if len(vals) != 1:
        raise SystemExit(f"inputs differ in {'.'.join(path)}: {sorted(vals)}")
    return vals.pop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", type=Path, nargs="+", help="evaluate --ss results, one per split")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    docs: List[dict] = [json.loads(p.read_text()) for p in a.inputs]
    if any("with_ss" not in d for d in docs):
        raise SystemExit("every input must come from evaluate --ss")
    if len({d["split"] for d in docs}) != len(docs):
        raise SystemExit("a split is given twice")
    same(docs, "model", "weights_sha256")
    same(docs, "with_ss", "config", "code")
    same(docs, "with_ss", "config", "variant")
    pooled = pool([d["with_ss"] for d in docs])
    doc = {"command": "... -m bench.tagger.accept " + " ".join(score.rel(p.resolve()) for p in a.inputs)
                      + f" --out {score.rel(a.out.resolve())}",
           "inputs": [{"file": score.rel(p.resolve()), "command": d["command"], "split": d["split"],
                       "git": d["git"], "acceptance": d["acceptance"]} for p, d in zip(a.inputs, docs)],
           "model": docs[0]["model"], "config": docs[0]["with_ss"]["config"],
           "pooled": pooled, "acceptance": acceptance(pooled)}
    for arm, r in pooled.items():
        print(f"{arm:18s} leak {r['leak']['k']}/{r['leak']['n']} = {r['leak']['rate']}  "
              f"natural spurious {r['natural_spurious']} of {r['natural_edits']}  {r['leaked_by_type']}")
    acc = doc["acceptance"]
    print(f"acceptance {acc['ok']} {acc['checks']} failing types "
          f"{ {t: v for t, v in acc['by_type'].items() if not v['ok']} }")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
