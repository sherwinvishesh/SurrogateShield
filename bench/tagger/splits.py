"""devlarge (test-1's test split) in two halves for the tagger (V3 §3.3, D3-3).

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.splits --out bench/tagger/splits.json

``calib`` is used only to set thresholds; ``val`` decides acceptance. No
half is ever trained on (D3-3: training text is generated, never real). A
record goes to a half by a keyed hash of its *source* conversation, so an
injected record and the natural record built on the same real conversation
always land together. ``splits.json`` holds the rule, the counts and a hash
of each half's member list, never the ids themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

from bench.realdata import score
from bench.realdata.common import DATASETS, read_jsonl

KEY = "surrogateshield-tagger-split-v1"
HALVES = ("calib", "val")
DATA = "test"                                     # devlarge's data split


def source(rec: dict) -> str:
    return rec.get("source_id") or rec["id"]


def half(dataset: str, rec: dict) -> str:
    return HALVES[hashlib.sha256(f"{KEY}:{dataset}:{source(rec)}".encode()).digest()[0] >> 7]


def records(dataset: str) -> List[dict]:
    coll = score.COLLECTIONS["test1"]
    return read_jsonl(coll.rd / dataset / f"{DATA}.jsonl") + score.natural_records(dataset, DATA, coll.rd, coll.build)


def membership(datasets: Iterable[str] = DATASETS) -> Dict[str, str]:
    """``{record id: half}`` over devlarge's injected and natural records."""
    return {r["id"]: half(ds, r) for ds in datasets for r in records(ds)}


def unit_half(unit: dict, members: Dict[str, str]) -> str:
    return members[unit["conv"]]


def summary(datasets: Iterable[str] = DATASETS) -> dict:
    out = {"key": KEY, "data_split": DATA, "rule": "sha256(key:dataset:source_id)[0] >> 7 -> calib | val",
           "use": {"calib": "thresholds only", "val": "acceptance"}, "datasets": {}}
    for ds in datasets:
        recs = records(ds)
        per = {}
        for h in HALVES:
            mine = sorted(r["id"] for r in recs if half(ds, r) == h)
            units = score.messages([r for r in recs if half(ds, r) == h])
            per[h] = {"records": len(mine),
                      "injected_units": sum("injected" in u["slices"] for u in units),
                      "natural_units": sum("natural" in u["slices"] for u in units),
                      "members_sha256": hashlib.sha256("\n".join(mine).encode()).hexdigest()}
        out["datasets"][ds] = per
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    s = summary()
    a.out.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    print(json.dumps({ds: {h: v[h]["records"] for h in HALVES} for ds, v in s["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
