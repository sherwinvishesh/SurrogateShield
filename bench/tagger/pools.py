"""The identity pools of the tagger's training data and of the sealed tests, and their overlap (V3 §5.3.1).

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.pools \\
        --data bench/tagger/build/data/train-40k.jsonl bench/tagger/build/data/train-80k.jsonl \\
        [--collection test2 test3] --out bench/results/tagger_pools.json

A training row records the pool words its values and third-party names were
built from (``meta.pool_tokens``, ``identity(pool="train")``). Test-2's
identities are drawn by ``inject.plan`` from its seed and frozen files, which
is what the injection itself draws (no provider call), and record theirs
(``identity.pool_tokens``, ``pool="eval"``). The report gives each pool's size,
the words of each that hash to the other half (0 by construction), the overlap
of every training file with test-2 (must be 0) and a hash of each pool's sorted
words; it exits 1 when any of these is not 0. Counts and hashes only: no word
and no text is written. ``--collection`` names the sealed tests to check
(default test-2 alone; PROMPT_FOR_OPUS_V4 checks test-2 and test-3, each under
its own key, ``overlap_with_<name>`` per training file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

from bench.realdata import identities as ids
from bench.realdata.common import COLLECTIONS, DATASETS, file_sha256, git_state, read_jsonl
from bench.realdata.score import rel


def training_pool(path: Path) -> set:
    return {t for r in read_jsonl(path) for t in r["meta"]["pool_tokens"]}


def eval_pools(datasets: Iterable[str] = DATASETS, plan: Callable = None, collection: str = "test2") -> Dict[str, set]:
    """Per dataset, the pool words of a sealed collection's injected identities."""
    if plan is None:
        from bench.realdata.inject import plan
    coll = COLLECTIONS[collection]
    return {ds: {t for r in plan(ds, coll=coll) for t in r["identity"]["pool_tokens"]} for ds in datasets}


def test2_pools(datasets: Iterable[str] = DATASETS, plan: Callable = None) -> Dict[str, set]:
    return eval_pools(datasets, plan, "test2")


def describe(words: set, half: str) -> dict:
    return {"words": len(words), "other_half": sum(ids.token_half(w) != half for w in words),
            "sha256": hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest()}


def report(training: Dict[str, set], test2: Dict[str, set] = None, **sealed: Dict[str, set]) -> dict:
    """Each training pool against each sealed collection's (``test2``, and any
    other given by name)."""
    sealed = {**({"test2": test2} if test2 is not None else {}), **sealed}
    evals = {name: set().union(*pools.values()) if pools else set() for name, pools in sealed.items()}
    train = {name: {**describe(ws, "train"), **{f"overlap_with_{c}": len(ws & e) for c, e in evals.items()}}
             for name, ws in training.items()}
    out = {"key": ids.POOL_KEY, "rule": "sha256(key:word)[0] >> 7 -> eval | train", "training": train,
           **{c: {"datasets": {ds: describe(ws, "eval") for ds, ws in pools.items()},
                  "all": describe(evals[c], "eval")} for c, pools in sealed.items()}}
    out["ok"] = (all(v["other_half"] == 0 and all(v[f"overlap_with_{c}"] == 0 for c in sealed)
                     for v in train.values())
                 and all(v["other_half"] == 0 for c in sealed for v in [*out[c]["datasets"].values(), out[c]["all"]]))
    return out


def main(argv: Sequence[str] = None, plan: Callable = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="training files (jsonl)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--collection", nargs="+", default=["test2"],
                    choices=[n for n, c in COLLECTIONS.items() if c.prefix])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = {"command": "python -m bench.tagger.pools " + " ".join(argv if argv is not None else []),
           "git": git_state(),
           "data": {rel(p): file_sha256(p) for p in a.data},
           **report({rel(p): training_pool(p) for p in a.data},
                    **{c: eval_pools(a.datasets, plan, c) for c in a.collection})}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for c in a.collection:
        t = doc[c]["all"]
        label = c.replace("test", "test-")
        for name, v in doc["training"].items():
            print(f"{name}: {v['words']} training pool words, {v[f'overlap_with_{c}']} shared with {label}'s "
                  f"{t['words']}; {v['other_half']} in the eval half")
        print(f"{label}: {t['other_half']} words in the train half")
    print("ok" if doc["ok"] else "NOT DISJOINT")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
