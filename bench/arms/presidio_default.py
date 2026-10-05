"""Arm ``presidio_default``: Presidio as it ships (``AnalyzerEngineProvider()``
defaults, en_core_web_lg), threshold 0.4, overlaps resolved longest-then-score,
every entity replaced by ``[ENTITY_TYPE]`` — exactly ``presidio/`` in this repo.
Runs in ``.venv``.
"""

from __future__ import annotations

import sys

from bench.arms.base import ROOT, cli

ARM = "presidio_default"


def load():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from presidio.detect import detect

    def fn(text: str, _seed: int):
        ents = detect(text)
        if ents is None:
            raise RuntimeError("presidio-analyzer or en_core_web_lg is not installed")
        return [[e.start, e.end, e.entity_type, f"[{e.entity_type}]"] for e in ents]
    return fn


def config() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from presidio.engine import baseline_config
    return {**baseline_config(), "replacement": "[ENTITY_TYPE] placeholder (presidio/redact.py)"}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
