"""Flatten the rebuilt natural pools into span-producer input.

    .venv/bin/python -m bench.arms.inputs        # -> bench/realdata/build/<dataset>/messages.jsonl
    .venv/bin/python -m bench.arms.inputs --collection test2   # -> build/test2/<dataset>/messages.jsonl

One line per user turn, ``{"id": "<dataset>/<source_id>#t<k>", "text"}``: a
multi-turn conversation contributes each of its turns, and every arm sees each
turn as its own message. The file is private (0600, git-ignored), like the pool
it comes from.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

from bench.realdata.common import COLLECTIONS, DATASETS, read_jsonl, write_jsonl


def message_id(dataset: str, source_id: str, turn: int) -> str:
    return f"{dataset}/{source_id}#t{turn}"


def parse_id(mid: str) -> tuple:
    dataset, rest = mid.split("/", 1)
    source_id, turn = rest.rsplit("#t", 1)
    return dataset, source_id, int(turn)


def flatten(rows: Iterable[dict]) -> List[dict]:
    return [{"id": message_id(r["dataset"], r["source_id"], k), "text": t}
            for r in rows for k, t in enumerate(r["turns"])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    ap.add_argument("--collection", choices=list(COLLECTIONS), default="test1")
    args = ap.parse_args(argv)
    build = COLLECTIONS[args.collection].build
    for ds in args.datasets or list(DATASETS):
        msgs = flatten(read_jsonl(build / ds / "pool.jsonl"))
        write_jsonl(build / ds / "messages.jsonl", msgs, private=True)
        print(f"{ds}: {len(msgs)} messages -> {build / ds / 'messages.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
