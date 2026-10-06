"""Arm ``ss``: SurrogateShield's single send path, ``json_tester.prepare_send``,
with ``MimicGen`` seeded per message. Runs in ``.venv``.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.arms.ss --in M.jsonl --out S.jsonl
"""

from __future__ import annotations

import os
import sys

from bench.arms.base import ROOT, Refused, cli, versions

ARM = "ss"
NO_SURROGATE = "could not generate a surrogate"


def detection_config():
    """The benchmark's DetectionConfig: ``balanced`` with ADDRESS "replace"
    (V3 §3.5). The product's "auto" shifts the house number inside a service
    query and keeps street, town and postcode verbatim within one
    whole-address edit, which the scorer counts as covered; "replace"
    re-draws every part, so a covered address leaks none of them."""
    if str(ROOT / "python-library") not in sys.path:
        sys.path.insert(0, str(ROOT / "python-library"))
    from surrogateshield.core.detection.config import benchmark
    return benchmark()


ADDRESS_MODE = "replace"


def effective_config(cascade_options: dict | None = None):
    """The config a run uses: the benchmark's, with any flat cascade
    switches applied (the hash in ``.meta.json`` is this one's)."""
    from surrogateshield.core.detection.pipeline import detection_config as resolve
    opts = {k: v for k, v in (cascade_options or {}).items() if k not in ("trace", "pii_off")}
    return resolve(detection_config(), **opts)


def load(cascade_options: dict | None = None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from generation.logic import MimicGen
    from json_tester import prepare_send
    cfg = detection_config()

    def fn(text: str, seed: int):
        try:
            p = prepare_send(text, MimicGen(seed=seed), cascade_options, config=cfg)
        except RuntimeError as exc:
            # MimicGen.generate_all found no surrogate that differs from every
            # original; the app does not send such a message (main.py lets it raise).
            if str(exc).startswith(NO_SURROGATE):
                raise Refused(str(exc)) from exc
            raise
        # a hit found on a canonical view reports it ("email:spelled"), so the
        # scorer's by_edit_type shows what each view adds
        types = {(s["start"], s["end"]): s["type"] + (f":{s['view']}" if s.get("view") else "") for s in p.spans()}
        return [[s, e, types.get((s, e), "unknown"), sur] for s, e, _orig, sur in p.edits]
    return fn


def config() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import config as cfg
    det = effective_config()
    return {"send_path": "json_tester.prepare_send(text, MimicGen(seed=msg_seed), config=benchmark())",
            "detection_config": det.to_dict(),
            "detection_config_hash": det.config_hash(),
            "ADDRESS_MODE": det.address_mode,
            "ADDRESS_MODE_product": cfg.ADDRESS_MODE,
            "SERVICE_QUERY_DETECTION_ENABLED": cfg.SERVICE_QUERY_DETECTION_ENABLED,
            "refusal": "RuntimeError('could not generate a surrogate ...') from MimicGen.generate_all -> refused row",
            "versions": versions("surrogateshield", "spacy", "transformers", "torch", "faker")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
