"""Phase 1 (E1): filter, sample and split the natural prompts of each dataset.

    .venv/bin/python -m bench.realdata.pull            # all three datasets
    .venv/bin/python -m bench.realdata.pull oasst1     # one

Reads only the pinned files under ``bench/realdata/raw/`` (no network).
Writes, per dataset:

  bench/realdata/<dataset>/pool.jsonl          committed index: ids, refs,
                                               SHA-256 per turn, split; no text
  bench/realdata/build/<dataset>/pool.jsonl    the text (git-ignored, 0600)

and records filter counts, seeds and file hashes in ``manifest.json``
(rendered to MANIFEST.md). Procedure, fixed before any system ran:

1. source-specific exclusions (``sources.py``), then the turn filters in
   ``common.py``: 15–400 words, not only code / only a URL, no jailbreak
   template; follow-up turns of a multi-turn conversation ≤ 400 words, English,
   no jailbreak; a conversation keeps its first ≤ 3 user turns;
2. exact duplicates (normalised first turn) collapse to the smallest source id;
3. candidates are sorted by source id and shuffled with ``random.Random``;
   multi-turn conversations are drawn first, then single-turn prompts from the
   remaining conversations; a draw is rejected as a near duplicate when its
   normalised first turn has rapidfuzz ratio ≥ 95 with one already drawn;
4. split 20 % dev / 80 % test by a seeded draw over the drawn source prompts.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.realdata import manifest
from bench.realdata.common import (BUILD, DATASETS, RAW, RD, derive_seed, first_turn_ok,
                                   later_turn_ok, normalise, sha256, words, write_jsonl)
from bench.realdata.sources import ITERATORS

N_SINGLE = 500
N_MULTI = 100
MAX_TURNS = 3
DEV_FRACTION = 0.2
NEAR_DUP = 95


def filter_candidates(cands, english_check: bool, counts: Counter) -> Tuple[List[dict], List[dict]]:
    """(single-turn candidates, multi-turn candidates); every drop is counted."""
    single, multi = [], []
    for c in cands:
        reason = first_turn_ok(c["turns"][0], english_check)
        if reason:
            counts[f"drop_first_turn_{reason}"] += 1
            continue
        counts["first_turn_kept"] += 1
        single.append(c)
        later = c["turns"][1:MAX_TURNS]
        if not later:
            continue
        counts["multi_candidates"] += 1
        bad = next((r for r in (later_turn_ok(t, english_check) for t in later) if r), None)
        if bad:
            counts[f"drop_multi_later_turn_{bad}"] += 1
            continue
        counts["multi_kept"] += 1
        multi.append(c)
    return single, multi


def dedup_exact(cands: List[dict], counts: Counter, key: str) -> List[dict]:
    best: Dict[str, dict] = {}
    for c in cands:
        n = normalise(c["turns"][0])
        if n not in best or c["source_id"] < best[n]["source_id"]:
            best[n] = c
    counts[f"{key}_exact_duplicates"] += len(cands) - len(best)
    return sorted(best.values(), key=lambda c: c["source_id"])


def draw(cands: List[dict], n: int, rng: random.Random, taken_ids: set,
         taken_norm: List[str], counts: Counter, key: str) -> List[dict]:
    """Walk a seeded shuffle; accept until *n*, skipping near duplicates."""
    from rapidfuzz import fuzz, process
    order = list(cands)
    rng.shuffle(order)
    out = []
    for c in order:
        if len(out) == n:
            break
        if c["source_id"] in taken_ids:
            continue
        norm = normalise(c["turns"][0])
        if taken_norm and process.extractOne(norm, taken_norm, scorer=fuzz.ratio, score_cutoff=NEAR_DUP):
            counts[f"{key}_near_duplicates_skipped"] += 1
            continue
        out.append(c)
        taken_ids.add(c["source_id"])
        taken_norm.append(norm)
    if len(out) < n:
        raise SystemExit(f"only {len(out)} {key} prompts left after filtering (need {n})")
    return out


def split(rows: List[dict], dataset: str, kind: str) -> None:
    ids = sorted(r["source_id"] for r in rows)
    dev = set(random.Random(derive_seed(dataset, kind, "split")).sample(ids, round(len(ids) * DEV_FRACTION)))
    for r in rows:
        r["split"] = "dev" if r["source_id"] in dev else "test"


def sample(dataset: str, raw: Path = RAW) -> Tuple[List[dict], Counter]:
    counts: Counter = Counter()
    cands = list(ITERATORS[dataset](raw, counts))
    single, multi = filter_candidates(cands, english_check=(dataset == "sharegpt"), counts=counts)
    multi = dedup_exact(multi, counts, "multi")
    single = dedup_exact(single, counts, "single")
    taken_ids: set = set()
    taken_norm: List[str] = []
    picked_multi = draw(multi, N_MULTI, random.Random(derive_seed(dataset, "multi")),
                        taken_ids, taken_norm, counts, "multi")
    picked_single = draw(single, N_SINGLE, random.Random(derive_seed(dataset, "single")),
                         taken_ids, taken_norm, counts, "single")
    rows = []
    for kind, picked in (("multi", picked_multi), ("single", picked_single)):
        kind_rows = []
        for c in picked:
            n = MAX_TURNS if kind == "multi" else 1
            turns, refs = c["turns"][:n], c["refs"][:n]
            kind_rows.append({"source_id": c["source_id"], "dataset": dataset, "kind": kind,
                              "refs": refs, "turns": turns, "meta": c["meta"]})
        split(kind_rows, dataset, kind)
        rows.extend(kind_rows)
    counts["drawn_single"] = len(picked_single)
    counts["drawn_multi"] = len(picked_multi)
    return rows, counts


def index_row(r: dict) -> dict:
    """The committed form of a drawn prompt: everything but the text."""
    return {"source_id": r["source_id"], "dataset": r["dataset"], "kind": r["kind"],
            "split": r["split"], "refs": r["refs"], "meta": r["meta"],
            "sha256": [sha256(t) for t in r["turns"]],
            "words": [words(t) for t in r["turns"]]}


def text_row(r: dict) -> dict:
    return {"source_id": r["source_id"], "dataset": r["dataset"], "kind": r["kind"],
            "split": r["split"], "turns": r["turns"], "meta": r["meta"]}


def order_key(r: dict):
    return (r["kind"] != "single", r["split"] != "dev", r["source_id"])


def write(dataset: str, rows: List[dict], rd: Path = RD, build: Path = BUILD) -> Tuple[Path, Path]:
    rows = sorted(rows, key=order_key)
    index = rd / dataset / "pool.jsonl"
    text = build / dataset / "pool.jsonl"
    write_jsonl(index, (index_row(r) for r in rows))
    write_jsonl(text, (text_row(r) for r in rows), private=True)
    return index, text


def english_check(raw: Path = RAW, n: int = 20000) -> dict:
    """Agreement of ``common.is_english`` (used for ShareGPT, which has no
    language field) with WildChat's language label, on first user turns of
    15–400 words in a seeded sample of *n* WildChat conversations."""
    from bench.realdata.common import is_english
    from bench.realdata.sources import _wildchat_rows
    rows = list(_wildchat_rows(raw))
    random.Random(derive_seed("english_check")).shuffle(rows)
    c = Counter()
    for r in rows[:n]:
        msgs = r["conversation"] or []
        if not msgs or msgs[0]["role"] != "user" or not 15 <= words(msgs[0]["content"] or "") <= 400:
            continue
        c[(r["language"] == "English", is_english(msgs[0]["content"]))] += 1
    tp, fn, fp = c[(True, True)], c[(True, False)], c[(False, True)]
    return {"n": sum(c.values()), "true_pos": tp, "false_neg": fn, "false_pos": fp,
            "true_neg": c[(False, False)], "precision": round(tp / (tp + fp), 4),
            "recall": round(tp / (tp + fn), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    ap.add_argument("--english-check", action="store_true",
                    help="only measure the ShareGPT English heuristic against WildChat's labels")
    args = ap.parse_args(argv)
    args.datasets = args.datasets or list(DATASETS)
    m = manifest.load()
    if args.english_check:
        m["datasets"].setdefault("sharegpt", {})["english_heuristic_vs_wildchat_label"] = r = english_check()
        manifest.save(m)
        print(r)
        return 0
    for ds in args.datasets:
        rows, counts = sample(ds)
        index, _ = write(ds, rows)
        by = Counter((r["kind"], r["split"]) for r in rows)
        m["datasets"].setdefault(ds, {})
        m["datasets"][ds]["raw_files"] = manifest.raw_hashes(ds)
        m["datasets"][ds]["pull_counts"] = dict(sorted(counts.items()))
        m["datasets"][ds]["drawn"] = {f"{k}_{s}": v for (k, s), v in sorted(by.items())}
        m["datasets"][ds]["seeds"] = {k: derive_seed(ds, *k.split("/")) for k in
                                      ("multi", "single", "multi/split", "single/split")}
        m["frozen"][f"{ds}/pool.jsonl"] = manifest.file_hash(index)
        print(f"{ds}: drawn {dict(by)}  -> {index.relative_to(manifest.ROOT)}")
    manifest.save(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
