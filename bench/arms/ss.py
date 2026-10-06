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


def load(cascade_options: dict | None = None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from generation.logic import MimicGen
    from json_tester import prepare_send

    def fn(text: str, seed: int):
        try:
            p = prepare_send(text, MimicGen(seed=seed), cascade_options)
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
    return {"send_path": "json_tester.prepare_send(text, MimicGen(seed=msg_seed))",
            "ADDRESS_MODE": cfg.ADDRESS_MODE,
            "SERVICE_QUERY_DETECTION_ENABLED": cfg.SERVICE_QUERY_DETECTION_ENABLED,
            "refusal": "RuntimeError('could not generate a surrogate ...') from MimicGen.generate_all -> refused row",
            "versions": versions("surrogateshield", "spacy", "transformers", "torch", "faker")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
