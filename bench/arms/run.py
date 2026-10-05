"""Run span producers as subprocesses, each in its own interpreter, and publish
text-free copies of their spans.

    .venv/bin/python -m bench.arms.run --natural                       # every arm, the three natural pools
    .venv/bin/python -m bench.arms.run --in F.jsonl --name NAME [--arms ss llm_guard]
    .venv/bin/python -m bench.arms.run --record                        # arm configs -> manifest only

Full spans (with replacement strings) go to the git-ignored
``bench/realdata/build/spans/<arm>/<name>.jsonl`` (0600): a replacement can repeat
characters of a real value (no-op or format-preserving edits), so they stay
private. The committed copy ``bench/results/spans/<arm>/<name>.jsonl`` keeps
offsets and counts only: ``[start, end, type, replacement_length, copied]``,
``copied`` = 1 when the replacement equals the original or contains it
verbatim (original ≥ 4 characters; the scorer's leak rule); a message the
system refused to send carries ``refused: 1``. Both run offline. Each arm's
configuration (versions, model revisions, thresholds) is then recorded under
``arms`` in ``bench/realdata/manifest.json``; two runs of one arm with
different configurations stop the record.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arms.base import ROOT, arm_env, read_messages, rel

ARMS: Dict[str, str] = {
    "ss": ".venv",
    "presidio_default": ".venv",
    "presidio_faker": ".venv",
    "presidio_transformers": ".venv-baselines",
    "llm_guard": ".venv-baselines",
    "gliner_pii": ".venv-baselines",
}
PRIVATE = ROOT / "bench" / "realdata" / "build" / "spans"
PUBLIC = ROOT / "bench" / "results" / "spans"
NATURAL = {f"natural-{ds}": ROOT / "bench" / "realdata" / "build" / ds / "messages.jsonl"
           for ds in ("oasst1", "sharegpt", "wildchat")}


def copied(original: str, replacement: str) -> int:
    return int(replacement == original or (len(original) >= 4 and original in replacement))


def public_rows(span_rows: List[dict], texts: Dict[str, str]) -> List[dict]:
    out = []
    for r in span_rows:
        t = texts[r["id"]]
        row = {"id": r["id"], "ms": r["ms"],
               "edits": [[s, e, ty, len(rep), copied(t[s:e], rep)] for s, e, ty, rep in r["edits"]]}
        if "refused" in r:
            row["refused"] = 1          # the system's message can name a value type only; keep the flag
        out.append(row)
    return out


def command(arm: str, src: Path, out: Path) -> List[str]:
    return [str(ROOT / ARMS[arm] / "bin" / "python"), "-m", f"bench.arms.{arm}", "--in", str(src), "--out", str(out)]


def run_arm(arm: str, src: Path, name: str) -> dict:
    full = PRIVATE / arm / f"{name}.jsonl"
    full.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(PRIVATE, 0o700)
    env = arm_env()
    proc = subprocess.run(command(arm, src, full), cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"{arm} failed on {rel(src)} (exit {proc.returncode}):\n{tail}")
    os.chmod(full, 0o600)
    texts = {m["id"]: m["text"] for m in read_messages(src)}
    rows = [json.loads(l) for l in open(full, encoding="utf-8")]
    pub = PUBLIC / arm / f"{name}.jsonl"
    pub.parent.mkdir(parents=True, exist_ok=True)
    with open(pub, "w", encoding="utf-8") as f:
        for r in public_rows(rows, texts):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copyfile(str(full) + ".meta.json", str(pub) + ".meta.json")
    meta = json.loads(Path(str(full) + ".meta.json").read_text())
    return {"arm": arm, "name": name, "messages": len(rows), "with_edits": sum(bool(r["edits"]) for r in rows),
            "edits": meta["edits"], "refused": meta["refused"], "load_seconds": meta["load_seconds"],
            "median_ms": sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else None}


def record_arms(public: Path = PUBLIC, manifest_path: Optional[Path] = None) -> Dict[str, dict]:
    """Every arm's configuration, from the committed meta sidecars, into the
    real-data manifest; a sidecar that disagrees with another of its arm raises."""
    from bench.realdata import manifest
    arms: Dict[str, dict] = {}
    for meta_path in sorted(public.glob("*/*.jsonl.meta.json")):
        meta = json.loads(meta_path.read_text())
        rec = {"config": meta["config"], "interpreter": ARMS.get(meta["arm"]), "seed": meta["seed"]}
        if meta["arm"] in arms and arms[meta["arm"]] != rec:
            raise SystemExit(f"{rel(meta_path)}: configuration differs from another run of {meta['arm']}; "
                             "rerun every input with one version")
        arms[meta["arm"]] = rec
    path = manifest_path or manifest.PATH
    m = manifest.load(path)
    m["arms"] = arms
    manifest.save(m, path, manifest.MD if manifest_path is None else None)
    return arms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--natural", action="store_true", help="the three rebuilt natural pools")
    ap.add_argument("--arms", nargs="*", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--record", action="store_true", help="only record arm configurations in the manifest")
    args = ap.parse_args(argv)
    if args.record:
        print("recorded:", ", ".join(record_arms()))
        return 0
    if args.natural:
        jobs = list(NATURAL.items())
    elif args.src and args.name:
        jobs = [(args.name, args.src.resolve())]
    else:
        ap.error("give --natural, or --in and --name")
    for name, src in jobs:
        for arm in args.arms:
            s = run_arm(arm, src, name)
            print(f"{name:18} {arm:22} messages {s['messages']:4}  with edits {s['with_edits']:4}  "
                  f"edits {s['edits']:5}  refused {s['refused']}  load {s['load_seconds']:5} s  median {s['median_ms']} ms", flush=True)
    print("recorded:", ", ".join(record_arms()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
