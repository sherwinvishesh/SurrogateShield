"""Span-producer interface shared by every arm (Phase 4, E4).

An arm reads a JSONL of messages ``{"id", "text"}`` and writes one line per
message ``{"id", "edits": [[start, end, type, replacement], ...], "ms"}``,
offsets into the original text, sorted, non-overlapping. One scorer then
reads every arm's spans, whatever environment produced them. A sidecar
``<out>.meta.json`` records the arm's configuration and package versions.

This module is imported by both interpreters (``.venv`` and
``.venv-baselines``), so it uses the standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261005

Edit = List  # [start, end, type, replacement]


class Refused(Exception):
    """The system declined to produce a sendable text for this message (it
    would not reach the provider). Raised by an arm only for a documented
    refusal of the system under test; any other exception stops the run."""


def msg_seed(arm: str, msg_id: str) -> int:
    """Per-message seed: stable across runs, interpreters and message order."""
    h = hashlib.sha256(f"{SEED}|{arm}|{msg_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def apply_edits(text: str, edits: Iterable[Edit]) -> str:
    out, pos = [], 0
    for s, e, _t, rep in sorted(edits, key=lambda x: (x[0], x[1])):
        out.append(text[pos:s])
        out.append(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def check_edits(text: str, edits: List[Edit]) -> None:
    """Raise if edits are out of bounds, unsorted or overlapping."""
    prev = 0
    for s, e, t, rep in edits:
        if not (0 <= s <= e <= len(text)):
            raise ValueError(f"edit [{s}, {e}) outside text of length {len(text)}")
        if s < prev:
            raise ValueError(f"edit [{s}, {e}) overlaps or precedes the previous edit")
        if not isinstance(t, str) or not isinstance(rep, str):
            raise ValueError("edit type and replacement must be strings")
        prev = e


def resolve_overlaps(spans: List[tuple]) -> List[tuple]:
    """``(start, end, type, score)``: keep the longest span, then the highest
    score, per overlapping region (as ``presidio.detect.resolve_overlaps``)."""
    kept: List[tuple] = []
    for sp in sorted(spans, key=lambda x: (-(x[1] - x[0]), -x[3], x[0])):
        if all(sp[1] <= k[0] or sp[0] >= k[1] for k in kept):
            kept.append(sp)
    return sorted(kept, key=lambda x: x[0])


def versions(*packages: str) -> dict:
    out = {"python": platform.python_version()}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def hf_revision(repo: str) -> str:
    """The commit of *repo* that an offline ``from_pretrained`` resolves to
    (``refs/main`` in the local Hugging Face cache)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
    ref = Path(hub) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "not cached"


def read_messages(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        if not isinstance(r.get("id"), str) or not isinstance(r.get("text"), str):
            raise ValueError(f"{path}: every line needs a string id and text")
    return rows


def arm_env() -> dict:
    """Environment for an arm subprocess: offline, and a fixed string-hash
    seed. Presidio breaks ties between same-span, same-score entities in set
    order, so without it a rerun can label one span NRP or URL and draw a
    different Faker value for it."""
    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0"}


def rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def produce(arm: str, load: Callable[[], Callable[[str, int], List[Edit]]], config: dict,
            src: Path, out: Path, limit: int | None = None) -> dict:
    """Run *arm* over *src*, write *out* and its meta sidecar; return the meta."""
    msgs = read_messages(src)[:limit]
    t0 = time.perf_counter()
    fn = load()
    load_s = time.perf_counter() - t0
    if msgs:
        fn("Warm-up message for Jane Roe, jane@example.org.", 0)  # first-call costs out of the timings
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    n_edits = n_refused = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for m in msgs:
            t = time.perf_counter()
            row = {"id": m["id"]}
            try:
                edits = [list(e) for e in fn(m["text"], msg_seed(arm, m["id"]))]
            except Refused as exc:
                edits, row["refused"] = [], str(exc)
                n_refused += 1
            ms = (time.perf_counter() - t) * 1000
            edits.sort(key=lambda e: (e[0], e[1]))
            check_edits(m["text"], edits)
            n_edits += len(edits)
            row.update(edits=edits, ms=round(ms, 2))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    with open(src, "rb") as f:
        input_sha256 = hashlib.sha256(f.read()).hexdigest()   # the scorer refuses spans of another input
    meta = {"arm": arm, "config": config, "input": rel(src), "input_sha256": input_sha256,
            "messages": len(msgs), "edits": n_edits, "refused": n_refused,
            "load_seconds": round(load_s, 1), "offline": os.environ.get("HF_HUB_OFFLINE") == "1", "seed": SEED,
            "command": f"{Path(sys.prefix).name}/bin/python -m bench.arms.{arm} --in {rel(src)} --out {rel(out)}"
                       + (f" --limit {limit}" if limit is not None else "")}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def cli(arm: str, load, config: Callable[[], dict]) -> int:
    ap = argparse.ArgumentParser(description=f"span producer: {arm}")
    ap.add_argument("--in", dest="src", type=Path, required=True, help="JSONL of {id, text}")
    ap.add_argument("--out", type=Path, required=True, help="JSONL of {id, edits, ms}")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    meta = produce(arm, load, config(), args.src.resolve(), args.out.resolve(), args.limit)
    print(f"{arm}: {meta['messages']} messages, {meta['edits']} edits, {meta['refused']} refused, "
          f"model load {meta['load_seconds']} s -> {args.out}")
    return 0
