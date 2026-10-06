"""Put a trained tagger where the library finds it (V3 §3.3; weights stay local).

    PYTHONPATH=.:python-library .venv/bin/python -m bench.tagger.install bench/tagger/build/models/dv3s-40k

The folder is copied to ``$SURROGATESHIELD_MODELS/<name>`` (default
``~/.cache/surrogateshield/models``). Without ``--name`` the name is the
default config's ``PII_TAGGER_MODEL`` and the weights must be the ones it
pins (``PII_TAGGER_REVISION``); with ``--name`` any model goes in under that
name, and the ``sha256:`` revision a config pins it with is printed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

from surrogateshield.core.detection import config as C
from surrogateshield.core.detection import pii_tagger as T

SKIP = shutil.ignore_patterns("checkpoint-*", "*.log", "runs")


def install(src: Path, name: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """Copy *src* to *root* (``models_dir()``) / *name*; the installed folder."""
    sha = T.weights_sha256(str(src))
    if name is None:
        name = C.PII_TAGGER_MODEL
        T.check_pin(str(src), C.PII_TAGGER_REVISION)
    dest = Path(root or T.models_dir()) / name
    if dest.exists():
        if T.weights_sha256(str(dest)) != sha:
            raise SystemExit(f"{dest} holds other weights; remove it to install these")
        return dest
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    shutil.copytree(src, part, ignore=SKIP)
    part.rename(dest)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path, help="a trained model folder (bench/tagger/build/models/<run>)")
    ap.add_argument("--name", help=f"install under this name (default {C.PII_TAGGER_MODEL}, pin checked)")
    a = ap.parse_args(argv)
    dest = install(a.model, a.name)
    print(f"{dest}\nrevision sha256:{T.weights_sha256(str(dest))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
