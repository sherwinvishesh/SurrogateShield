"""
surrogateshield/cli.py — the ``surrogateshield`` command.

    surrogateshield scan   [TEXT | -]            list the PII found
    surrogateshield mask   [TEXT | -]            print the text to send
    surrogateshield unmask [TEXT | -] -s ID      restore an answer
    surrogateshield bench                        mask latency, p50 / p95
    surrogateshield doctor                       check models, storage, keys;
                                                 print the detection config

``--preset NAME`` (fast / balanced / strict) or ``--detection-config FILE``
(a DetectionConfig as JSON, partial or whole) choose the detection config;
``SURROGATESHIELD_PRESET`` / ``SURROGATESHIELD_DETECTION_CONFIG`` do the same
from the environment. See CONFIGURATION.md.

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
    if getattr(args, "preset", None) or getattr(args, "detection_config", None):
        from .core.detection import config as dconfig
        from ._state import effective_detection
        det = dconfig.preset(args.preset) if args.preset else effective_detection(c)
        if args.detection_config:
            try:
                overrides = json.loads(Path(args.detection_config).read_text(encoding="utf-8"))
            except OSError as exc:
                raise ValueError(f"cannot read --detection-config: {exc}") from exc
            if args.preset and isinstance(overrides, dict) and overrides.get("preset", args.preset) != args.preset:
                raise ValueError(f"--preset {args.preset} and the file's preset "
                                 f"{overrides['preset']!r} disagree")
            det = dconfig.from_partial(overrides, det)
        c.detection = det
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
            seed: int = 0, detection=None) -> dict:
    """Wall time of ``Session.mask`` per message, in ms, after one warm-up
    pass that loads the models. Used by ``surrogateshield bench`` and
    ``bench/perf.py``. *detection* is a DetectionConfig (default: the
    configured one)."""
    from ._state import cfg
    from .session import Session
    c = dataclasses.replace(cfg, context_guard_enabled=context_guard)
    if detection is not None:
        c.detection = detection
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
            "max_ms": round(ms[-1], 1),
            **({"preset": detection.preset} if detection is not None else {})}


def cmd_bench(args, out) -> int:
    det = _config(args).detection
    if det is not None:                 # --preset / --detection-config: that config only
        cg = det.enabled("context_guard") and not args.no_context_guard
        rows = [latency(BENCH_TEXTS, context_guard=cg, rounds=args.rounds, detection=det)]
    else:
        modes = [False] if args.no_context_guard else [False, True]
        rows = [latency(BENCH_TEXTS, context_guard=m, rounds=args.rounds) for m in modes]
    if args.json:
        out.write(json.dumps(rows) + "\n")
    else:
        for r in rows:
            name = "with ContextGuard" if r["context_guard"] else "without ContextGuard"
            if "preset" in r:
                name = f"{r['preset']}, {name[:-12].strip()} CG"
            out.write(f"{name:<22} p50 {r['p50_ms']:>7.1f} ms   p95 {r['p95_ms']:>7.1f} ms"
                      f"   ({r['messages']} masks)\n")
    return EXIT_OK


def _spacy_meta(name: str) -> dict:
    import spacy
    path = spacy.util.get_package_path(name)
    meta = path / "meta.json"
    if not meta.exists():
        meta = next(path.glob("*/meta.json"))
    return json.loads(meta.read_text(encoding="utf-8"))


def _check_models(det) -> List[tuple]:
    """One row per model an enabled stage of *det* loads: installed or
    cached at its revision, with the version and licence."""
    from .core.detection.config import MODEL_LICENCES
    rows = []
    et, cg = det.stage("entity_trace"), det.stage("context_guard")
    try:
        import spacy
        ok = spacy.util.is_package(et.model)
        note = f"python -m spacy download {et.model}"
        if ok:
            meta = _spacy_meta(et.model)
            version = meta.get("version")
            note = (f"version {version}, licence {meta.get('license') or MODEL_LICENCES.get(et.model, '?')}"
                    + (f"; the config pins {et.revision}" if et.revision and version != et.revision else ""))
        rows.append(("spaCy model " + et.model, ok, et.enabled, note))
    except ImportError as exc:
        rows.append(("spaCy", False, et.enabled, str(exc)))
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(cg.model, "config.json", revision=cg.revision)
        ok = isinstance(hit, str)
        rev = f"@{cg.revision[:12]}" if cg.revision else ""
        rows.append((f"ContextGuard model {cg.model}{rev}", ok, cg.enabled,
                     f"licence {MODEL_LICENCES.get(cg.model, 'not recorded')}" if ok
                     else "not in the local Hugging Face cache at this revision; the first scan downloads it"))
    except ImportError as exc:
        rows.append(("transformers / huggingface_hub", False, cg.enabled, str(exc)))
    from .core.detection import plugins
    for st in det.plugin_stages():
        if st.name == "pii_tagger":
            rows.append(_check_tagger(st, MODEL_LICENCES))
            continue
        try:
            plugins.get_detector(st)
            rows.append((f"detector {st.name}", True, True, f"model {st.model}" if st.model else ""))
        except Exception as exc:     # noqa: BLE001 - reported, the row fails
            rows.append((f"detector {st.name}", False, True, str(exc)))
    return rows


def _check_tagger(st, licences) -> tuple:
    """The tagger's weights: a local folder whose weights match the pin, or
    a hub id in the local cache; not loaded (``--cold-start`` times that)."""
    from .core.detection import pii_tagger
    from .core.errors import DetectorUnavailable
    name = f"PIITagger model {st.model}"
    if not st.model:
        return (name, False, True, "the pii_tagger stage names no model")
    licence = f"licence {licences.get(st.model, 'not recorded')}"
    local = pii_tagger.local_dir(st.model)
    if local:
        try:
            pii_tagger.check_pin(local, st.revision)
        except DetectorUnavailable as exc:
            return (name, False, True, str(exc))
        return (name, True, True, f"{local}, " + (f"weights {st.revision[:19]}…, " if st.revision else "") + licence)
    if "/" not in st.model:
        return (name, False, True, f"no folder of that name under {pii_tagger.models_dir()} "
                                   f"(${pii_tagger.MODELS_ENV}); python -m bench.tagger.install puts it there, "
                                   "or --preset classic runs without it")
    try:
        from huggingface_hub import try_to_load_from_cache
        ok = isinstance(try_to_load_from_cache(st.model, "config.json", revision=st.revision), str)
    except ImportError as exc:
        return (name, False, True, str(exc))
    return (name, ok, True, licence if ok else "not in the local Hugging Face cache at this revision")


def _cold_start(det) -> List[tuple]:
    """Load each enabled model stage once and time it (a fresh process
    pays this on its first message)."""
    from .core.detection import context_guard, entity_trace, plugins
    rows = []
    for name, load in (
        ("entity_trace", lambda st: entity_trace._get_nlp(st.model)),
        ("context_guard", lambda st: context_guard._get_ner(st.model, st.device, st.revision)),
        ("pii_tagger", plugins.get_detector),
    ):
        st = det.stage(name)
        if not st.enabled:
            continue
        t = time.perf_counter()
        try:
            load(st)
            rows.append((f"cold start {name} ({st.model})", True, True,
                         f"{(time.perf_counter() - t) * 1000:.0f} ms"))
        except Exception as exc:     # noqa: BLE001 - reported, the row fails
            rows.append((f"cold start {name} ({st.model})", False, True, str(exc)))
    return rows


def _describe(det, source: str) -> List[str]:
    lines = [f"detection config: preset {det.preset}, hash {det.config_hash()[:16]} ({source})"]
    for st in det.detectors:
        bits = ["on" if st.enabled else "off"]
        if st.model:
            bits.append(st.model + (f"@{st.revision[:12]}" if st.revision else ""))
        if st.thresholds:
            bits.append(" ".join(f"{k}={v:g}" for k, v in sorted(st.thresholds.items())))
        if st.max_latency_ms is not None:
            bits.append(f"budget {st.max_latency_ms:g} ms")
        lines.append(f"  stage {st.name:14} " + ", ".join(bits))
    actions = ", ".join(f"{k} {v}" for k, v in sorted(det.type_actions.items()))
    lines.append(f"  actions        {actions or 'replace'} (others replace)")
    gate = "off" if not det.gate else ("bypassed by " + ", ".join(det.gate_bypass) if det.gate_bypass else "on")
    lines.append(f"  relation gate  {gate}")
    return lines


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
    from ._state import effective_detection
    from .core.detection import config as dconfig
    try:
        c = _config(args)
        det = effective_detection(c)
    except (ValueError, TypeError) as exc:
        out.write(f"[FAIL] detection config — {exc}\n")
        return EXIT_FAIL
    source = ("--preset / --detection-config" if getattr(args, "preset", None) or getattr(args, "detection_config", None)
              else "the environment" if c.detection is None and dconfig.env_selected() else
              "settings" if c.detection is not None else "default")
    for line in _describe(det, source):
        out.write(line + "\n")
    if args.show_config:
        out.write(det.to_json() + "\n")
    cg_on, tagger_on = det.enabled("context_guard"), det.enabled("pii_tagger")
    rows = [("python " + ".".join(map(str, sys.version_info[:3])), sys.version_info >= (3, 9), True, "")]
    for mod in ("faker", "cryptography", "rapidfuzz", "spacy", "transformers", "torch"):
        try:
            __import__(mod)
            rows.append((f"import {mod}", True, True, ""))
        except ImportError as exc:
            rows.append((f"import {mod}", False, (mod not in ("transformers", "torch") or cg_on or tagger_on)
                         and (mod != "spacy" or det.enabled("entity_trace")), str(exc)))
    rows += _check_models(det)
    if args.cold_start and all(ok or not req for _, ok, req, _ in rows):
        rows += _cold_start(det)
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
        sp.add_argument("--preset", choices=("fast", "balanced", "strict"),
                        help="detection preset (default: balanced, or $SURROGATESHIELD_PRESET)")
        sp.add_argument("--detection-config", metavar="FILE",
                        help="a DetectionConfig as JSON (partial or whole), see CONFIGURATION.md")

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
    sp.add_argument("--cold-start", action="store_true", help="also load each model and time it")
    sp.add_argument("--show-config", action="store_true", help="print the whole detection config as JSON")
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
