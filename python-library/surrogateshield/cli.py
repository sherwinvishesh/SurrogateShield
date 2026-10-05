"""
surrogateshield/cli.py — the ``surrogateshield`` command.

    surrogateshield scan   [TEXT | -]            list the PII found
    surrogateshield mask   [TEXT | -]            print the text to send
    surrogateshield unmask [TEXT | -] -s ID      restore an answer
    surrogateshield bench                        mask latency, p50 / p95
    surrogateshield doctor                       check models, storage, keys

TEXT defaults to stdin. ``mask`` keeps the surrogate map encrypted under
``~/.surrogateshield/sessions`` (``SURROGATESHIELD_HOME`` moves it) so a later
``unmask -s ID`` can restore the answer. It prints the session id on stderr
when it opens a new one. ``--no-store`` keeps the map in memory only.
A low-entropy surrogate (a bare age, a gender word) is restored only in the
process that sent it, so ``unmask`` in a new process leaves those as they are.

Exit codes: 0 ok · 1 check failed / unknown session · 2 usage ·
3 a detector is unavailable (nothing is masked or printed: fail closed).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_DETECTOR = 0, 1, 2, 3

# Sample traffic for ``bench`` (invented values, no real person).
BENCH_TEXTS = (
    "Hi, I'm Dana Whitfield and my email is dana.w@example.com — can you fix my CV?",
    "My SSN is 219-09-9999 and I live at 1126 E Apache Blvd, Tempe, AZ 85281.",
    "Call me on +44 7700 900123 after 6pm, or text 480-555-0172.",
    "What is the difference between a list and a tuple in Python?",
    "Draft a letter to Mercy General Hospital about my son Kofi Mensah's bill.",
    "is there a pharmacy near 2200 N Central Ave, Phoenix?",
    "Server 10.0.4.17 rejected user ravi_k92 with password hunter2!",
    "Résumé: Inés Fernández, née le 04/11/1987, Lyon. Tél 06 12 34 56 78.",
)


def _sessions_dir() -> Path:
    from .core.storage.shadow_map import home
    return home() / "sessions"


def _read_text(arg: Optional[str]) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read()
    return arg


def _config(args):
    from ._state import cfg
    c = dataclasses.replace(cfg)
    if getattr(args, "no_context_guard", False):
        c.context_guard_enabled = False
    if getattr(args, "pii_off", None):
        c.pii_off = list(args.pii_off)
    return c


def _detection_dict(d) -> dict:
    return {"text": d.text, "type": d.type, "start": d.start, "end": d.end,
            "score": round(d.score, 3), "source": d.source, "masked": d.masked}


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_scan(args, out) -> int:
    from .session import Session
    text = _read_text(args.text)
    with Session(config=_config(args), seed=args.seed) as s:
        found = s.scan(text)
    if args.json:
        out.write(json.dumps([_detection_dict(d) for d in found], ensure_ascii=False) + "\n")
    else:
        for d in found:
            flag = "" if d.masked else "  (pii_off: sent as is)"
            out.write(f"{d.start:>5}-{d.end:<5} {d.type:<18} {d.source:<8} {d.text!r}{flag}\n")
    return EXIT_OK


def cmd_mask(args, out) -> int:
    from .session import Session
    text = _read_text(args.text)
    store = None if args.no_store else str(args.store or _sessions_dir())
    new = args.session is None
    s = Session(args.session, config=_config(args), storage_dir=store, seed=args.seed)
    result = s.mask_result(text)
    if new and store is not None:
        sys.stderr.write(f"session: {s.id}\n")
    if args.json:
        out.write(json.dumps({
            "session": s.id, "text": result.text, "service_query": result.service_query,
            "replacements": result.replacements,
            "detections": [_detection_dict(d) for d in result.detections],
        }, ensure_ascii=False) + "\n")
    else:
        out.write(result.text if result.text.endswith("\n") else result.text + "\n")
    return EXIT_OK


def cmd_unmask(args, out) -> int:
    from .core.storage.shadow_map import validate_id
    from .session import Session
    text = _read_text(args.text)
    store = str(args.store or _sessions_dir())
    sid = validate_id(args.session)
    if not (Path(store).expanduser() / f"{sid}.shadowmap").exists():
        sys.stderr.write(f"surrogateshield: no stored session {sid!r} under {store}\n")
        return EXIT_FAIL
    s = Session(sid, config=_config(args), storage_dir=store)
    restored = s.unmask(text)
    out.write(restored if restored.endswith("\n") else restored + "\n")
    if args.close:
        s.close()                                   # erase the stored map
    return EXIT_OK


def latency(texts: Sequence[str], *, context_guard: bool, rounds: int = 3,
            seed: int = 0) -> dict:
    """Wall time of ``Session.mask`` per message, in ms, after one warm-up
    pass that loads the models. Used by ``surrogateshield bench`` and
    ``bench/perf.py``."""
    from ._state import cfg
    from .session import Session
    c = dataclasses.replace(cfg, context_guard_enabled=context_guard)
    with Session(config=c, seed=seed) as s:
        for t in texts:
            s.mask(t)                               # warm-up: model loads
        ms: List[float] = []
        for _ in range(rounds):
            for t in texts:
                t0 = time.perf_counter()
                s.mask(t)
                ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"context_guard": context_guard, "messages": len(ms),
            "p50_ms": round(statistics.median(ms), 1),
            "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1),
            "max_ms": round(ms[-1], 1)}


def cmd_bench(args, out) -> int:
    modes = [False] if args.no_context_guard else [False, True]
    rows = [latency(BENCH_TEXTS, context_guard=m, rounds=args.rounds) for m in modes]
    if args.json:
        out.write(json.dumps(rows) + "\n")
    else:
        for r in rows:
            name = "with ContextGuard" if r["context_guard"] else "without ContextGuard"
            out.write(f"{name:<22} p50 {r['p50_ms']:>7.1f} ms   p95 {r['p95_ms']:>7.1f} ms"
                      f"   ({r['messages']} masks)\n")
    return EXIT_OK


def _check_models(c) -> List[tuple]:
    rows = []
    try:
        import spacy
        ok = spacy.util.is_package(c.spacy_model)
        rows.append(("spaCy model " + c.spacy_model, ok, True,
                     "" if ok else f"python -m spacy download {c.spacy_model}"))
    except ImportError as exc:
        rows.append(("spaCy", False, True, str(exc)))
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(c.context_guard_model, "config.json")
        ok = isinstance(hit, str)
        rows.append(("ContextGuard model " + c.context_guard_model, ok, c.context_guard_enabled,
                     "" if ok else "not in the local Hugging Face cache; the first scan downloads it"))
    except ImportError as exc:
        rows.append(("transformers / huggingface_hub", False, c.context_guard_enabled, str(exc)))
    return rows


def _check_storage() -> List[tuple]:
    from .core.storage.shadow_map import home
    rows = []
    h = home()
    if h.exists():
        loose = h.stat().st_mode & 0o077
        rows.append((f"storage dir {h}", not loose, True,
                     "group/other can read it; chmod 700" if loose else "mode 0700"))
        key = h / "device.key"
        if key.exists():
            loose = key.stat().st_mode & 0o077
            rows.append(("device key", not loose, True,
                         "chmod 600 device.key" if loose else "mode 0600"))
        else:
            rows.append(("device key", True, False, "created on first persistent session"))
    else:
        rows.append((f"storage dir {h}", True, False, "created on first persistent session"))
    return rows


def cmd_doctor(args, out) -> int:
    c = _config(args)
    rows = [("python " + ".".join(map(str, sys.version_info[:3])), sys.version_info >= (3, 9), True, "")]
    for mod in ("faker", "cryptography", "rapidfuzz", "spacy", "transformers", "torch"):
        try:
            __import__(mod)
            rows.append((f"import {mod}", True, True, ""))
        except ImportError as exc:
            rows.append((f"import {mod}", False, mod not in ("transformers", "torch")
                         or c.context_guard_enabled, str(exc)))
    rows += _check_models(c)
    rows += _check_storage()
    if args.smoke and all(ok or not req for _, ok, req, _ in rows):
        from .core.errors import DetectorUnavailable
        from .session import Session
        probe = "My email is dana.w@example.com and my SSN is 219-09-9999."
        try:
            with Session(config=c, seed=0) as s:
                sent = s.mask(probe)
            ok = "dana.w@example.com" not in sent and "219-09-9999" not in sent
            rows.append(("smoke: mask hides an email and an SSN", ok, True, ""))
        except DetectorUnavailable as exc:
            rows.append(("smoke: mask", False, True, str(exc)))
    failed = False
    for name, ok, required, note in rows:
        status = "ok" if ok else ("FAIL" if required else "warn")
        failed |= (not ok and required)
        out.write(f"[{status:>4}] {name}" + (f" — {note}" if note else "") + "\n")
    return EXIT_FAIL if failed else EXIT_OK


# ── entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surrogateshield",
                                description="Mask PII before text reaches an LLM.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, text=True):
        if text:
            sp.add_argument("text", nargs="?", help="the text (default: stdin)")
        sp.add_argument("--no-context-guard", action="store_true",
                        help="skip the distilbert-NER stage (faster, lower recall)")
        sp.add_argument("--pii-off", nargs="*", metavar="TYPE",
                        help="types to send as is (e.g. phone email)")

    sp = sub.add_parser("scan", help="list the PII found in TEXT")
    common(sp)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("mask", help="print TEXT with PII replaced by surrogates")
    common(sp)
    sp.add_argument("-s", "--session", help="reuse a stored session (default: a new one)")
    sp.add_argument("--store", help="directory for encrypted maps (default ~/.surrogateshield/sessions)")
    sp.add_argument("--no-store", action="store_true", help="keep the map in memory only")
    sp.add_argument("--seed", type=int, help="seed the surrogates (reproducible output)")
    sp.add_argument("--json", action="store_true", help="also print detections and replacements")
    sp.set_defaults(func=cmd_mask)

    sp = sub.add_parser("unmask", help="restore the originals in an LLM answer")
    common(sp)
    sp.add_argument("-s", "--session", required=True)
    sp.add_argument("--store", help="directory for encrypted maps (default ~/.surrogateshield/sessions)")
    sp.add_argument("--close", action="store_true", help="erase the stored map afterwards")
    sp.set_defaults(func=cmd_unmask)

    sp = sub.add_parser("bench", help="mask latency on built-in sample messages")
    common(sp, text=False)
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("doctor", help="check models, dependencies and key storage")
    common(sp, text=False)
    sp.add_argument("--smoke", action="store_true", help="also mask one sample message")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[Sequence[str]] = None, out=None) -> int:
    from .core.errors import DetectorUnavailable
    from .core.storage.shadow_map import StorageError
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, out)
    except DetectorUnavailable as exc:
        sys.stderr.write(f"surrogateshield: detector unavailable, nothing was masked: {exc}\n")
        return EXIT_DETECTOR
    except (StorageError, ValueError) as exc:
        sys.stderr.write(f"surrogateshield: {exc}\n")
        return EXIT_FAIL


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
