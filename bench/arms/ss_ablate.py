"""E6 span producer: arm ``ss`` with PatternScan recognisers removed or made
wrong, one span file per condition. Runs in ``.venv``; the cascade loads once.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.arms.ss_ablate --in M.jsonl --out-dir DIR [--conditions none,drop-email]

A *family* is every regex of one PatternScan entity type (``phone_us``,
``phone_uk`` and ``phone_intl`` are one family, as are ``zip_us`` and
``postcode_uk``), plus the two parsers that ``scan()`` runs before the regex
list: ``url`` (``find_urls``) and ``address`` (``address_parser.find_addresses``,
patched for PatternScan only, not for the service-query check). Conditions:

* ``none`` — unablated; its spans equal arm ``ss``'s on the same input.
* ``drop-<family>`` — that family never matches (*incomplete*).
* ``drop25-<k>``, ``drop50-<k>`` — a seeded random 25 % / 50 % of families
  dropped, draws k = 0..4.
* ``wrong-<family>`` — that family's regexes perturbed (*incorrect*): every
  counted repeat one longer (``\\d{3}`` → ``\\d{4}``, ``{2,4}`` → ``{3,5}``) and
  every separator swapped for one the pattern does not accept, inside the
  value group only (``v``/``w``, or group 1; see ``perturb``). ``wronglen-all``, ``wrongsep-all``, ``wrong-all``: one or both
  perturbations on every family at once.

Validators, context regexes and every later stage are untouched. Each row
also carries the cascade stage of each edit (``sources``: ``pattern``,
``ner`` = EntityTrace, ``slm`` = ContextGuard, ``structural``, ``repeat``) so
the scorer can say which stage caught a value the regexes no longer do.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from re import _compiler as sre_compile, _constants as C, _parser as sre_parse
from typing import Dict, List

from bench.arms.base import ROOT, Refused, check_edits, msg_seed, read_messages, rel, versions

ARM = "ss_ablate"
SEED_ARM = "ss"                       # per-message MimicGen seeds as arm ss: condition none == ss
DRAWS = 5
SHARES = {"drop25": 0.25, "drop50": 0.50}
MERGE = {"phone_us": "phone", "phone_uk": "phone", "phone_intl": "phone", "zip_us": "postcode",
         "postcode_uk": "postcode"}
PARSERS = ("url", "address")
SEPARATORS = "-./_"                   # substituted in; a space is never introduced
_SEP_IN = set(SEPARATORS) | {" "}     # recognised as a separator in a pattern
_NEXT = {"-": ".", ".": "/", "/": "_", "_": "-"}
NEVER = re.compile(r"(?!)")


def _ps():
    if str(ROOT / "python-library") not in sys.path:
        sys.path.insert(0, str(ROOT / "python-library"))
    from surrogateshield.core.detection import pattern_scan
    return pattern_scan


def families(ps=None) -> Dict[str, List[int]]:
    """family -> indices into ``pattern_scan._PATTERNS`` (parsers: [])."""
    ps = ps or _ps()
    out: Dict[str, List[int]] = {}
    for i, (etype, _rx, _v) in enumerate(ps._PATTERNS):
        out.setdefault(MERGE.get(etype, etype), []).append(i)
    for p in PARSERS:
        out[p] = []
    return dict(sorted(out.items()))


def conditions(fams: List[str], seed: int) -> List[dict]:
    rng = random.Random(seed)
    out = [{"name": "none", "drop": [], "wrong": [], "how": None}]
    out += [{"name": f"drop-{f}", "drop": [f], "wrong": [], "how": None} for f in fams]
    for tag, share in SHARES.items():
        k = round(share * len(fams))
        out += [{"name": f"{tag}-{d}", "drop": sorted(rng.sample(fams, k)), "wrong": [], "how": None}
                for d in range(DRAWS)]
    out += [{"name": f"wrong-{f}", "drop": [], "wrong": [f], "how": "both"} for f in fams]
    out += [{"name": f"{n}-all", "drop": [], "wrong": list(fams), "how": how}
            for n, how in (("wronglen", "lengths"), ("wrongsep", "separators"), ("wrong", "both"))]
    return out


# ── perturbation on the parse tree ───────────────────────────────────────────

_REPEATS = {C.MAX_REPEAT, C.MIN_REPEAT, C.POSSESSIVE_REPEAT}
_ASSERTS = {C.ASSERT, C.ASSERT_NOT}


def _counted(lo: int, hi) -> bool:
    """``{n}``, ``{n,m}``, ``{n,}`` with n ≥ 2 or a finite m ≥ 2; not ``?``, ``*``, ``+``."""
    return not (lo <= 1 and hi in (1, C.MAXREPEAT))


def _class(items: list) -> list:
    if any(op is C.NEGATE for op, _ in items):
        return items
    seps = {chr(av) for op, av in items if op is C.LITERAL and chr(av) in _SEP_IN}
    only_seps = bool(seps) and all((op is C.LITERAL and chr(av) in _SEP_IN) or (op is C.CATEGORY and av is C.CATEGORY_SPACE)
                    for op, av in items)
    if not seps and not only_seps:
        return items
    if only_seps:                               # [\s.-] -> the other separators, no whitespace
        seps |= {" "} if any(op is C.CATEGORY for op, _ in items) else set()
        rest = []
    else:                                       # [A-Za-z0-9._-]: swap its separators only
        rest = [(op, av) for op, av in items if not (op is C.LITERAL and chr(av) in _SEP_IN)]
    new = [c for c in SEPARATORS if c not in seps] or ["~"]
    return rest + [(C.LITERAL, ord(c)) for c in new]


def _walk(sub, lengths: bool, separators: bool, targets=None, active: bool = True) -> None:
    """Perturb *sub* in place. With *targets* (group numbers of the value),
    only the inside of those groups changes; lookarounds never do."""
    data = []
    for op, av in sub.data:
        if op is C.LITERAL and active and separators and chr(av) in _SEP_IN - {" "}:
            av = ord(_NEXT[chr(av)])
        elif op is C.IN and active and separators:
            av = _class(av)
        elif op in _REPEATS:
            lo, hi, inner = av
            _walk(inner, lengths, separators, targets, active)
            if active and lengths and _counted(lo, hi):
                lo, hi = lo + 1, hi if hi == C.MAXREPEAT else hi + 1
            av = (lo, hi, inner)
        elif op is C.SUBPATTERN:
            _walk(av[-1], lengths, separators, targets, active or (targets is not None and av[0] in targets))
        elif op is C.ATOMIC_GROUP:
            _walk(av, lengths, separators, targets, active)
        elif op is C.BRANCH:
            for b in av[1]:
                _walk(b, lengths, separators, targets, active)
        elif op is C.GROUPREF_EXISTS:
            for b in av[1:]:
                if b is not None:
                    _walk(b, lengths, separators, targets, active)
        # lookarounds (ASSERT, ASSERT_NOT) are context, not the value's shape: left as is
        data.append((op, av))
    sub.data = data


def value_groups(rx: "re.Pattern", etype: str = "", ps=None):
    """Group numbers holding the value (``v``/``w``, or group 1 for the
    types ``scan()`` takes from group 1), or None: the whole match."""
    named = {rx.groupindex[g] for g in ("v", "w") if g in rx.groupindex}
    if named:
        return named
    ps = ps or _ps()
    return {1} if etype in ps._GROUP1_TYPES and rx.groups >= 1 else None


def perturb(rx: "re.Pattern", how: str = "both", targets=None) -> "re.Pattern":
    """*rx* matching the wrong shape: counted repeats one longer (``lengths``),
    separators swapped (``separators``), or both, inside the value groups
    *targets* (None: the whole pattern). A literal ``- . / _`` becomes the
    next of them; a class of separators (at least one of ``- . / _`` or a
    space, plus optional ``\\s``) becomes the ones it did not accept, never
    whitespace; a class that mixes letters and separators keeps its letters
    and swaps its separators the same way. A bare space and a lone ``\\s``
    are word gaps, not separators, and stay; so do keyword scaffolds outside
    the value group and every lookaround."""
    tree = sre_parse.parse(rx.pattern, rx.flags)
    _walk(tree, how in ("lengths", "both"), how in ("separators", "both"), targets, targets is None)
    return sre_compile.compile(tree, rx.flags)


# ── applying a condition ─────────────────────────────────────────────────────

class _Proxy:
    def __init__(self, mod, **over):
        self._mod, self._over = mod, over

    def __getattr__(self, name):
        return self._over[name] if name in self._over else getattr(self._mod, name)


@contextlib.contextmanager
def applied(cond: dict, ps=None):
    """Patch PatternScan for *cond*; restore everything on exit."""
    ps = ps or _ps()
    ap = ps.address_parser
    fams = families(ps)
    saved = {"_PATTERNS": ps._PATTERNS, "find_urls": ps.find_urls, "_URL_RE": ps._URL_RE,
             "address_parser": ap, "_table_age_cells": ps._table_age_cells}
    table = list(ps._PATTERNS)
    try:
        for f in cond["wrong"]:
            for i in fams[f]:
                t, rx, v = table[i]
                table[i] = (t, perturb(rx, cond["how"], value_groups(rx, t, ps)), v)
            if f == "url":
                ps._URL_RE = perturb(saved["_URL_RE"], cond["how"])
            if f == "address":
                street, po = perturb(ap._STREET_ADDRESS_RE, cond["how"]), perturb(ap._PO_BOX_RE, cond["how"])

                def find(text, _s=street, _p=po):
                    old = ap._STREET_ADDRESS_RE, ap._PO_BOX_RE
                    ap._STREET_ADDRESS_RE, ap._PO_BOX_RE = _s, _p
                    try:
                        return ap.find_addresses(text)
                    finally:
                        ap._STREET_ADDRESS_RE, ap._PO_BOX_RE = old
                ps.address_parser = _Proxy(ap, find_addresses=find)
        for f in cond["drop"]:
            for i in fams[f]:
                t, _rx, v = table[i]
                table[i] = (t, NEVER, v)
            if f == "url":
                ps.find_urls = lambda text: []
            if f == "address":
                ps.address_parser = _Proxy(ap, find_addresses=lambda text: [])
            if f == "age":
                ps._table_age_cells = lambda text: []
        ps._PATTERNS = table
        yield
    finally:
        for k, v in saved.items():
            setattr(ps, k, v)


# ── producing spans ──────────────────────────────────────────────────────────

def load():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from generation.logic import MimicGen
    from json_tester import prepare_send
    from bench.arms.ss import NO_SURROGATE

    def fn(text: str, seed: int):
        try:
            p = prepare_send(text, MimicGen(seed=seed))
        except RuntimeError as exc:
            if str(exc).startswith(NO_SURROGATE):
                raise Refused(str(exc)) from exc
            raise
        by = {(s["start"], s["end"]): s for s in p.spans()}
        return [[s, e, by.get((s, e), {}).get("type", "unknown"), sur, by.get((s, e), {}).get("source", "unknown")]
                for s, e, _orig, sur in p.edits]
    return fn


def run(src: Path, out_dir: Path, names: List[str] | None = None, fn=None, seed: int = 0) -> dict:
    msgs = read_messages(src)
    with open(src, "rb") as f:
        input_sha256 = hashlib.sha256(f.read()).hexdigest()
    ps = _ps()
    fams = families(ps)
    conds = conditions(list(fams), seed)
    if names:
        unknown = set(names) - {c["name"] for c in conds}
        if unknown:
            raise SystemExit(f"unknown conditions: {sorted(unknown)}")
        conds = [c for c in conds if c["name"] in names]
    t0 = time.perf_counter()
    fn = fn or load()
    load_s = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    summary = {}
    for cond in conds:
        out = out_dir / f"{cond['name']}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        n_edits = n_refused = 0
        t = time.perf_counter()
        with applied(cond, ps), open(tmp, "w", encoding="utf-8") as f:
            os.chmod(tmp, 0o600)
            for m in msgs:
                row = {"id": m["id"]}
                try:
                    edits = sorted((list(e) for e in fn(m["text"], msg_seed(SEED_ARM, m["id"]))),
                                   key=lambda e: (e[0], e[1]))
                except Refused as exc:
                    edits, row["refused"] = [], str(exc)
                    n_refused += 1
                check_edits(m["text"], [e[:4] for e in edits])
                n_edits += len(edits)
                row.update(edits=[e[:4] for e in edits], sources=[e[4] for e in edits])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        meta = {"arm": ARM, "condition": cond, "input": rel(src), "input_sha256": input_sha256,
                "messages": len(msgs), "edits": n_edits, "refused": n_refused,
                "seconds": round(time.perf_counter() - t, 1), "seed": seed}
        Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        summary[cond["name"]] = {"edits": n_edits, "refused": n_refused}
        print(f"{ARM} {cond['name']:24} {len(msgs)} messages, {n_edits} edits, {meta['seconds']} s", flush=True)
    return {"families": {k: len(v) for k, v in fams.items()}, "conditions": conds, "load_seconds": round(load_s, 1),
            "summary": summary, "versions": versions("surrogateshield", "spacy", "transformers", "torch")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E6 span producer: ss with PatternScan families dropped or perturbed")
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--conditions", help="comma-separated names (default: all)")
    ap.add_argument("--seed", type=int, required=True, help="seed of the random family draws")
    args = ap.parse_args(argv)
    info = run(args.src.resolve(), args.out_dir.resolve(),
               args.conditions.split(",") if args.conditions else None, seed=args.seed)
    (args.out_dir / "run.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
