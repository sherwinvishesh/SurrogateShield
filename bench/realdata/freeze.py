"""V3 Phase 3: freeze the detector before test-2 is scored.

    PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.freeze            # writes FREEZE.json
    PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.freeze --check    # is the tree still the frozen one?

``bench/realdata/FREEZE.json`` records the commit (of a clean tree, with
nothing untracked under ``python-library/`` or ``bench/``), the library code
stamp, the default detector (``benchmark()``: its config hash and the config
itself), every model it loads with its revision and licence (the tagger's
installed weights checked against their pin), the hypotheses file's hash, the
committed config files the test-2 runs read (E9 and E10) and the frozen
baseline choice (``gliner_pii_tuned``). It refuses to overwrite an existing
freeze. ``score.check_freeze`` refuses test-2 unless the hypotheses,
and, when FREEZE.json records them, the code, the default config and the
config files are still the frozen ones; ``--check`` prints the same.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from bench.arms import gliner_pii_tuned as tuned
from bench.realdata.common import file_sha256, git_state
from bench.realdata.score import FREEZE, HYPOTHESES, RD, rel

CONFIGS = RD / "configs"


def untracked() -> List[str]:
    """Files git does not track or ignore under the code a run imports."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "python-library", "bench"],
                         cwd=RD.parents[1], capture_output=True, text=True, check=True).stdout
    return out.split()


def config_files(configs: Optional[Path] = None) -> Dict[str, str]:
    return {rel(p): file_sha256(p) for p in sorted((configs or CONFIGS).rglob("*.json"))}


def models(cfg) -> List[dict]:
    """The models the enabled stages load; a local tagger folder is checked
    against its pinned weights."""
    from surrogateshield.core.detection import config as C
    from surrogateshield.core.detection import pii_tagger as T
    out = []
    for s in cfg.detectors:
        if not (s.enabled and s.model):
            continue
        row = {"stage": s.name, "model": s.model, "revision": s.revision, "licence": C.MODEL_LICENCES.get(s.model)}
        local = T.local_dir(s.model) if s.name == "pii_tagger" else None
        if s.name == "pii_tagger":
            if not local:
                raise SystemExit(f"the tagger {s.model!r} is not installed under {T.models_dir()}")
            T.check_pin(local, s.revision)
            row["weights_sha256_checked"] = T.weights_sha256(local)
        out.append(row)
    return out


def current(configs: Optional[Path] = None, hypotheses: Path = HYPOTHESES) -> dict:
    """What a freeze records, from the tree as it is now (no commit, no models)."""
    from bench.tagger.evaluate import code_stamp
    from surrogateshield.core.detection import config as C
    cfg = C.benchmark()
    return {"code": code_stamp(), "detection_config_hash": cfg.config_hash(), "detection_config": cfg.to_dict(),
            "hypotheses_sha256": file_sha256(hypotheses), "config_files": config_files(configs),
            "gliner_pii_tuned": {"label_set": tuned.LABEL_SET, "threshold": tuned.THRESHOLD}}


def differences(doc: dict, now: Optional[dict] = None, hypotheses: Path = HYPOTHESES) -> List[str]:
    """The frozen fields that the tree no longer matches."""
    now = now or current(hypotheses=hypotheses)
    return [k for k in ("code", "detection_config_hash", "hypotheses_sha256", "config_files", "gliner_pii_tuned")
            if k in doc and doc[k] != now[k]]


def freeze(out: Path = FREEZE, configs: Optional[Path] = None, hypotheses: Path = HYPOTHESES,
           git: Optional[dict] = None, today: Optional[str] = None) -> dict:
    if out.exists():
        raise SystemExit(f"{rel(out)} exists: the detector is frozen once")
    git = git or {**git_state(), "untracked": untracked()}
    dirty = git.get("modified", []) + git.get("untracked", [])
    if dirty:
        raise SystemExit(f"the tree is not clean ({', '.join(dirty[:5])}): commit first")
    from surrogateshield.core.detection import config as C
    doc = {"commit": git["commit"], "date": today or datetime.date.today().isoformat(),
           **current(configs, hypotheses), "models": models(C.benchmark()),
           "note": "test-2 is scored at this commit, once: E5'', E6'', E7'', E9, attribution, the external "
                   "benchmark and E10 (PROMPT_FOR_OPUS_V3 Phase 3)"}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the tree with FREEZE.json; write nothing")
    a = ap.parse_args(argv)
    if a.check:
        if not FREEZE.exists():
            raise SystemExit(f"{rel(FREEZE)} does not exist")
        diff = differences(json.loads(FREEZE.read_text()))
        print("frozen tree: ok" if not diff else f"differs from FREEZE.json in: {', '.join(diff)}")
        return 1 if diff else 0
    doc = freeze()
    print(f"frozen at {doc['commit'][:12]}: config {doc['detection_config_hash'][:16]}, code {doc['code'][:12]}, "
          f"{len(doc['models'])} models, {len(doc['config_files'])} config files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
