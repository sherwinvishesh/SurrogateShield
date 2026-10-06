"""Arm ``gliner_pii_tuned``: GLiNER-PII (``urchade/gliner_multi_pii-v1``,
Apache-2.0) at the configuration with the lowest pooled dev leak of the V3
§3.8 sweep (``bench/results/gliner_sweep_dev.json``): thresholds 0.3–0.6 ×
two label sets, the model card's 22 labels (``gliner_pii.LABELS``) and our 13
protect types in words with natural aliases (``OURS``), so the model is asked
for exactly what it is scored on (``LABEL_TYPES`` maps each label to its type). Frozen before test-2. Windows, overlap rule
and ``[LABEL]`` replacement are ``gliner_pii``'s. Runs in ``.venv-baselines``.

    .venv-baselines/bin/python -m bench.arms.gliner_pii_tuned --in F.jsonl --out G.jsonl
    .venv-baselines/bin/python -m bench.arms.gliner_pii_tuned --sweep ours --in F.jsonl --out G.jsonl

``--sweep LABELS`` writes, per message, every window's decoded spans with
their scores at the sweep's lowest threshold instead of edits; the sweep
(``bench/realdata/gliner_sweep.py``) derives every higher threshold from them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from bench.arms import gliner_pii as base
from bench.arms.base import cli, hf_revision, read_messages, rel, versions

ARM = "gliner_pii_tuned"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
# label -> the protect type it asks for
OURS: Dict[str, str] = {
    "person": "PERSON", "name": "PERSON",
    "organization": "ORG", "company": "ORG",
    "address": "ADDRESS", "street address": "ADDRESS",
    "location": "LOCATION", "city": "LOCATION",
    "age": "AGE",
    "date of birth": "DATE_OF_BIRTH",
    "email": "EMAIL", "email address": "EMAIL",
    "phone number": "PHONE",
    "id number": "ID", "identification number": "ID",
    "handle": "HANDLE", "username": "HANDLE",
    "ip address": "NETWORK", "mac address": "NETWORK",
    "url": "URL", "website": "URL",
    "credential": "CREDENTIAL", "password": "CREDENTIAL", "api key": "CREDENTIAL",
}
_ID = ("passport number", "social security number", "credit card number", "bank account number", "iban",
       "driver's license number", "national id number", "tax identification number", "health insurance id number")
PUBLISHED: Dict[str, str] = {
    label: ("ID" if label in _ID else
            {"person": "PERSON", "organization": "ORG", "email": "EMAIL", "phone number": "PHONE",
             "address": "ADDRESS", "location": "LOCATION", "date of birth": "DATE_OF_BIRTH", "age": "AGE",
             "ip address": "NETWORK", "url": "URL", "username": "HANDLE", "password": "CREDENTIAL",
             "api key": "CREDENTIAL"}[label])
    for label in base.LABELS}
LABEL_TYPES: Dict[str, Dict[str, str]] = {"published": PUBLISHED, "ours": OURS}
LABEL_SETS: Dict[str, List[str]] = {name: list(m) for name, m in LABEL_TYPES.items()}

# The sweep's choice, frozen before test-2: bench/results/gliner_sweep_dev.json
# at 4d32bf0, pooled dev injected leak 49/627 = 0.0781 (the model card's
# setting, published @ 0.5: 59/627 = 0.0941; ours is worse at every threshold).
LABEL_SET: Optional[str] = "published"
THRESHOLD: Optional[float] = 0.3


def load():
    if LABEL_SET is None or THRESHOLD is None:
        raise SystemExit(f"{ARM}: not frozen yet; run bench.realdata.gliner_sweep first")
    return base.load(THRESHOLD, LABEL_SETS[LABEL_SET])


def config() -> dict:
    return {**base.config(), "labels": LABEL_SETS[LABEL_SET] if LABEL_SET else None,
            "label_set": LABEL_SET, "threshold": THRESHOLD,
            "chosen_by": "bench/results/gliner_sweep_dev.json (lowest pooled dev injected leak)"}


def sweep(labels: str, src: Path, out: Path) -> dict:
    """Scored window spans of every message of *src* at ``THRESHOLDS[0]``."""
    msgs = read_messages(src)
    t0 = time.perf_counter()
    m = base.model()
    load_s = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for msg in msgs:
            spans = sorted(base.window_spans(m, msg["text"], LABEL_SETS[labels], THRESHOLDS[0]))
            f.write(json.dumps({"id": msg["id"], "spans": [list(x) for x in spans]}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    os.chmod(out, 0o600)
    with open(src, "rb") as f:
        input_sha256 = hashlib.sha256(f.read()).hexdigest()
    meta = {"arm": ARM, "mode": "sweep", "label_set": labels, "labels": LABEL_SETS[labels],
            "threshold": THRESHOLDS[0], "model": base.MODEL, "model_revision": hf_revision(base.MODEL),
            "window_words": base.WINDOW, "overlap_words": base.OVERLAP, "input": rel(src),
            "input_sha256": input_sha256, "messages": len(msgs), "load_seconds": round(load_s, 1),
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "versions": versions("gliner", "transformers", "torch")}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        ap = argparse.ArgumentParser(description=f"{ARM}: scored spans for the §3.8 sweep")
        ap.add_argument("--sweep", choices=sorted(LABEL_SETS), required=True)
        ap.add_argument("--in", dest="src", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args()
        meta = sweep(a.sweep, a.src.resolve(), a.out.resolve())
        print(f"{ARM} sweep {a.sweep}: {meta['messages']} messages -> {a.out}")
        raise SystemExit(0)
    raise SystemExit(cli(ARM, load, config))
