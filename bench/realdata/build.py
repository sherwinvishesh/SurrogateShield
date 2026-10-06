"""Rebuild the natural-slice text from the committed index and the pinned raw files.

    .venv/bin/python -m bench.realdata.build            # all datasets
    .venv/bin/python -m bench.realdata.build --check    # verify only, write nothing
    .venv/bin/python -m bench.realdata.build --collection test2

Reads ``bench/realdata/<dataset>/pool.jsonl`` (ids, refs, SHA-256 per turn),
looks every turn up in ``bench/realdata/raw/`` (download those files at the
revisions in MANIFEST.md first) and writes ``bench/realdata/build/<dataset>/
pool.jsonl`` (git-ignored, 0600). Any hash mismatch stops the build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.realdata.common import COLLECTIONS, DATASETS, RAW, read_jsonl, sha256, write_jsonl
from bench.realdata.sources import LOOKUPS, resolve


class HashMismatch(Exception):
    pass


def rebuild(index_rows: List[dict], dataset: str, lookup) -> List[dict]:
    out = []
    for r in index_rows:
        turns = resolve(dataset, r["refs"], lookup)
        got = [sha256(t) for t in turns]
        if got != r["sha256"]:
            raise HashMismatch(f"{dataset} {r['source_id']}: text differs from the frozen hash")
        out.append({"source_id": r["source_id"], "dataset": dataset, "kind": r["kind"],
                    "split": r["split"], "turns": turns, "meta": r["meta"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    ap.add_argument("--check", action="store_true", help="verify hashes without writing")
    ap.add_argument("--collection", choices=list(COLLECTIONS), default="test1")
    args = ap.parse_args(argv)
    args.datasets = args.datasets or list(DATASETS)
    coll = COLLECTIONS[args.collection]
    for ds in args.datasets:
        rows = rebuild(read_jsonl(coll.rd / ds / "pool.jsonl"), ds, LOOKUPS[ds](RAW))
        out = coll.build / ds / "pool.jsonl"
        if not args.check:
            write_jsonl(out, rows, private=True)
        print(f"{ds}: {len(rows)} prompts, every hash matches"
              + ("" if args.check else f" -> {out.relative_to(coll.build.parent if coll.prefix else coll.build)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
