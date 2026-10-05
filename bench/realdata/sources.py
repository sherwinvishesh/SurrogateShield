"""Read the pinned raw downloads into candidate conversations (user turns only).

A candidate is ``{"source_id", "dataset", "refs", "turns", "meta"}`` where
``turns`` are the user's messages in order and ``refs`` locate each one in the
raw files again (so ``build.py`` can rebuild text that is never committed):

  oasst1    refs = [message_id, ...] along the best-ranked path from the root
  sharegpt  refs = [[conversation id, message index], ...]
  wildchat  refs = [[conversation_hash, message index], ...]

Source-specific exclusions are counted in the *counts* Counter passed in.
Assistant turns are never read into memory past this module.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

from bench.realdata.common import RAW, SOURCES


def _files(dataset: str, raw: Path) -> List[Path]:
    src = SOURCES[dataset]
    return [raw / src["local"] / f for f in src["files"]]


# ── OASST1 ────────────────────────────────────────────────────────────────────

_OASST_COLS = ["message_id", "parent_id", "text", "role", "lang", "deleted", "rank", "created_date"]


def _oasst_rows(raw: Path) -> List[dict]:
    import pyarrow.parquet as pq
    rows: List[dict] = []
    for f in _files("oasst1", raw):
        rows.extend(pq.read_table(f, columns=_OASST_COLS).to_pylist())
    return rows


def _oasst_child_key(m: dict):
    """Best reply first: lowest rank (ranked replies before unranked), then oldest, then id."""
    return (m["rank"] is None, m["rank"] if m["rank"] is not None else 0, str(m["created_date"]), m["message_id"])


def oasst_path(root: dict, children: Dict[str, List[dict]]) -> List[dict]:
    """The root and its best-ranked descendant chain (deleted replies skipped)."""
    path, node = [root], root
    while True:
        kids = [k for k in children.get(node["message_id"], []) if not k["deleted"]]
        if not kids:
            return path
        node = min(kids, key=_oasst_child_key)
        path.append(node)


def iter_oasst1(raw: Path = RAW, counts: Counter | None = None) -> Iterator[dict]:
    counts = counts if counts is not None else Counter()
    rows = _oasst_rows(raw)
    children: Dict[str, List[dict]] = defaultdict(list)
    for m in rows:
        if m["parent_id"] is not None:
            children[m["parent_id"]].append(m)
    roots = sorted((m for m in rows if m["parent_id"] is None), key=lambda m: m["message_id"])
    counts["source_roots"] += len(roots)
    for root in roots:
        if root["role"] != "prompter":
            counts["drop_root_not_prompter"] += 1
            continue
        if root["deleted"]:
            counts["drop_deleted"] += 1
            continue
        if root["lang"] != "en":
            counts["drop_not_english"] += 1
            continue
        user = [m for m in oasst_path(root, children) if m["role"] == "prompter"]
        # follow-up turns in another language end the usable conversation
        keep = [user[0]]
        for m in user[1:]:
            if m["lang"] != "en":
                break
            keep.append(m)
        yield {"source_id": root["message_id"], "dataset": "oasst1",
               "refs": [m["message_id"] for m in keep],
               "turns": [m["text"] for m in keep], "meta": {}}


def oasst1_lookup(raw: Path = RAW) -> Dict[str, str]:
    return {m["message_id"]: m["text"] for m in _oasst_rows(raw)}


# ── ShareGPT ──────────────────────────────────────────────────────────────────

_SG_USER = ("human", "user")


def _sharegpt_rows(raw: Path) -> List[dict]:
    with open(_files("sharegpt", raw)[0], encoding="utf-8") as f:
        return json.load(f)


def iter_sharegpt(raw: Path = RAW, counts: Counter | None = None) -> Iterator[dict]:
    counts = counts if counts is not None else Counter()
    for conv in sorted(_sharegpt_rows(raw), key=lambda c: c["id"]):
        counts["source_conversations"] += 1
        msgs = conv.get("conversations") or []
        start = 0
        while start < len(msgs) and msgs[start].get("from") == "system":
            start += 1
        if start >= len(msgs) or msgs[start].get("from") not in _SG_USER:
            # an exported continuation: the first message is the model's
            counts["drop_not_starting_with_user"] += 1
            continue
        idx = [i for i in range(start, len(msgs)) if msgs[i].get("from") in _SG_USER]
        yield {"source_id": conv["id"], "dataset": "sharegpt",
               "refs": [[conv["id"], i] for i in idx],
               "turns": [msgs[i].get("value") or "" for i in idx], "meta": {}}


def sharegpt_lookup(raw: Path = RAW) -> Dict[str, list]:
    return {c["id"]: c.get("conversations") or [] for c in _sharegpt_rows(raw)}


# ── WildChat ──────────────────────────────────────────────────────────────────

_WC_COLS = ["conversation_hash", "conversation", "language", "toxic", "redacted"]


def _wildchat_rows(raw: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for f in _files("wildchat", raw):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=_WC_COLS):
            yield from batch.to_pylist()


def iter_wildchat(raw: Path = RAW, counts: Counter | None = None) -> Iterator[dict]:
    """English, non-toxic conversations; the first occurrence of each hash.

    ``redacted`` (the publishers scrubbed PII they found) is kept and recorded
    in ``meta`` so natural-slice prevalence can be reported with and without it.
    """
    counts = counts if counts is not None else Counter()
    seen = set()
    for r in _wildchat_rows(raw):
        counts["source_conversations"] += 1
        if r["conversation_hash"] in seen:
            counts["drop_duplicate_hash"] += 1
            continue
        seen.add(r["conversation_hash"])
        if r["language"] != "English":
            counts["drop_not_english"] += 1
            continue
        if r["toxic"]:
            counts["drop_toxic"] += 1
            continue
        msgs = r["conversation"] or []
        idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if not idx or idx[0] != 0:
            counts["drop_not_starting_with_user"] += 1
            continue
        yield {"source_id": r["conversation_hash"], "dataset": "wildchat",
               "refs": [[r["conversation_hash"], i] for i in idx],
               "turns": [msgs[i]["content"] or "" for i in idx],
               "meta": {"redacted": bool(r["redacted"])}}


def wildchat_lookup(raw: Path = RAW) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for r in _wildchat_rows(raw):
        out.setdefault(r["conversation_hash"], [m["content"] for m in (r["conversation"] or [])])
    return out


ITERATORS = {"oasst1": iter_oasst1, "sharegpt": iter_sharegpt, "wildchat": iter_wildchat}


def resolve(dataset: str, refs: list, lookup) -> List[str]:
    """The user turns a committed ``refs`` list points at."""
    if dataset == "oasst1":
        return [lookup[r] for r in refs]
    if dataset == "sharegpt":
        return [lookup[cid][i].get("value") or "" for cid, i in refs]
    return [lookup[h][i] or "" for h, i in refs]


LOOKUPS = {"oasst1": oasst1_lookup, "sharegpt": sharegpt_lookup, "wildchat": wildchat_lookup}
