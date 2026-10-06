"""E5 — every arm on one frozen split of the real-data benchmark, one scorer.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --split dev --out bench/results/realdata_dev.json
    ... --split test --out bench/results/realdata_test.json     # once, at the end of Phase 5
    ... --reuse        # rescore saved spans whose input hash still matches; runs no arm
    ... --split devlarge   # test-1's test split, development data for the v3 detector
    ... --split test2      # the sealed second test, once, only after FREEZE.json is committed

The prompt names this ``bench/realdata.py``; a module of that name would
shadow the package ``bench.realdata`` (D15), so it lives here.

1. The frozen files (``<ds>/<split>.jsonl``, ``labels.jsonl``, ``pool.jsonl``)
   must match the manifest's hashes, and every rebuilt natural text its
   committed SHA-256, or nothing is scored.
2. Records: the injected slice as committed; the natural slice rebuilt from
   the committed offset labels into git-ignored ``build/<ds>/natural-<split>.jsonl``
   (0600), ids from ``rd-<ds>-<split>-1001``. A multi-turn record is one
   message per turn (``<id>#t<k>``); turns whose silver label failed are left out.
3. One private arm input per dataset (``build/score/<split>/<ds>.jsonl``);
   every arm runs on it as a subprocess (``bench.arms.run.run_arm``). The
   scorer reads span files; it never imports an arm. A span file is used only
   if its meta records the SHA-256 of that exact input.
4. Each message is scored by ``bench.realworld.score_message`` (the J2
   definitions: leaked, policy, spurious, keep hits, sensitive) on the arm's
   edits. A refused message reaches no provider: nothing leaks, nothing is
   edited, and it is counted under ``refused``.
5. Per dataset (and pooled), per slice, per arm: rates with Wilson 95 %
   intervals, per type, per task, per type universe; SS − arm differences
   with a paired bootstrap (2,000 resamples of conversations, the same draws
   for every arm). Counts only: no text and no value leaves this script.

Named runs (``RUNS``) beyond ``dev`` / ``test`` (PROMPT_FOR_OPUS_V3 §5):
``devlarge`` scores test-1's ``test`` files (same records, ids and bootstrap
draws as ``test``) under its own span-file and result names, because test-1
is development data for the v3 detector; ``test2`` scores the second
collection (``bench/realdata/test2/``, frozen keys ``test2/...``) and is
refused unless ``bench/realdata/FREEZE.json`` exists and records the
pre-registered hypotheses' current hash.

``--external NAME`` (V3 §5.4, H10'') scores the external benchmark built by
``bench.realdata.external`` through the same arms, span checks, per-message
scoring, aggregation and bootstrap (clusters: the source's record groups),
per stratum and pooled, beside a value-level span F1 (precision = edits
touching a gold value / edits, recall = 1 − leak rate). Also refused before
the freeze. H10'' (fixed before any arm ran on the data): pooled, SS's leak
rate ≤ the GLiNER arm's or SS's span F1 ≥ the GLiNER arm's (point
estimates; the intervals are reported beside), for ``gliner_pii`` (the
hypothesis) and ``gliner_pii_tuned``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from bench import realworld as rw
from bench.realdata.common import (BUILD, COLLECTIONS, DATASETS, RD, ROOT, commit_note, derive_seed, file_sha256,
                                   git_state, read_jsonl, sha256, write_jsonl)

ARMS = ("ss", "presidio_default", "presidio_faker", "presidio_transformers", "llm_guard", "gliner_pii",
        "gliner_pii_tuned")
BASELINES = ARMS[1:]
SLICES = ("injected", "injected_single", "shift", "multi", "natural")
# value formats the tagger never saw in training (bench.tagger.data.HELD_OUT; V3 §5.3.2), reported apart
HELD_OUT_FORMATS = (("PHONE", "words"), ("DATE_OF_BIRTH", "month-ordinal"))
RESAMPLES = 2000
NATURAL_FROM = 1001
# run name -> (data split, collection, role)
RUNS = {"dev": ("dev", "test1", "diagnosis only"),
        "test": ("test", "test1", "the paper's detection numbers"),
        "devlarge": ("test", "test1", "development only (test-1's test split)"),
        "test2": ("test2", "test2", "the paper's detection numbers (sealed second test)")}
FREEZE = RD / "FREEZE.json"
HYPOTHESES = RD / "HYPOTHESES_TEST2.md"

# GUIDE protect types each arm is configured to detect (its recogniser list or
# label set as recorded in bench/realdata/manifest.json -> arms), not what it
# happened to find. Presidio: DATE_TIME stands for DATE_OF_BIRTH, IP_ADDRESS
# for NETWORK, and the national-id / card / IBAN / bank recognisers for ID;
# it has no street-address, organisation (shipped spaCy config), age, handle
# or secret recogniser. LLM Guard's default entity list has no location, link
# or date. GLiNER-PII's 22 labels cover every type, as do both label sets of
# gliner_pii_tuned (V3 §3.8: the dev sweep's threshold and label set).
_ALL = tuple(sorted(rw.PROTECT_TYPES))
_PRESIDIO = ("DATE_OF_BIRTH", "EMAIL", "ID", "LOCATION", "NETWORK", "PERSON", "PHONE", "URL")
CLAIMED: Dict[str, Tuple[str, ...]] = {
    "ss": _ALL,
    "presidio_default": _PRESIDIO,
    "presidio_faker": _PRESIDIO,
    "presidio_transformers": tuple(sorted(_PRESIDIO + ("AGE", "ORG"))),
    "llm_guard": ("EMAIL", "ID", "NETWORK", "PERSON", "PHONE"),
    "gliner_pii": _ALL,
    "gliner_pii_tuned": _ALL,
}
UNIVERSES = {
    "shared_all_arms": tuple(sorted(set.intersection(*(set(v) for v in CLAIMED.values())))),
    "presidio_default": _PRESIDIO,
    "ss_only_vs_presidio": tuple(sorted(set(_ALL) - set(_PRESIDIO))),
}


# ── 1. frozen inputs ─────────────────────────────────────────────────────────

def frozen_files(datasets: Sequence[str], split: str) -> List[str]:
    return [f"{ds}/{name}" for ds in datasets for name in (f"{split}.jsonl", "labels.jsonl", "pool.jsonl")]


def check_frozen(datasets: Sequence[str], split: str, frozen: Dict[str, str], rd: Path = RD,
                 prefix: str = "") -> Dict[str, str]:
    """The hash of every frozen input, or SystemExit naming the first that
    changed. Files are read under *rd*; their manifest keys carry *prefix*
    (``test2/oasst1/pool.jsonl``)."""
    out = {}
    for name in frozen_files(datasets, split):
        key = f"{prefix}/{name}" if prefix else name
        if key not in frozen:
            raise SystemExit(f"{key}: not frozen in the manifest; refusing to score")
        got = file_sha256(rd / name)
        if got != frozen[key]:
            raise SystemExit(f"{key}: hash {got[:12]} differs from the frozen {frozen[key][:12]}; refusing to score")
        out[key] = got
    return out


def check_freeze(freeze: Path = FREEZE, prereg: Path = HYPOTHESES) -> str:
    """The SHA-256 of FREEZE.json, or SystemExit: test2 is scored only after
    the detector is frozen, with the hypotheses it pre-registered and, when
    the freeze records them, its code, default config and config files."""
    if not freeze.exists():
        raise SystemExit(f"{rel(freeze)} does not exist: test2 is scored only after the freeze (Phase 3)")
    doc = json.loads(freeze.read_text())
    want = doc.get("hypotheses_sha256")
    if want is None or file_sha256(prereg) != want:
        raise SystemExit(f"{rel(prereg)} does not match the hash recorded in {rel(freeze)}; refusing to score")
    if "code" in doc:                       # a bench.realdata.freeze record: the detector must still be the frozen one
        from bench.realdata.freeze import differences
        diff = differences(doc, hypotheses=prereg)
        if diff:
            raise SystemExit(f"the tree differs from {rel(freeze)} in {', '.join(diff)}; test2 is scored only "
                             "with the frozen detector")
    return file_sha256(freeze)


# ── 2. records ───────────────────────────────────────────────────────────────

def natural_records(ds: str, split: str, rd: Path = RD, build: Path = BUILD) -> List[dict]:
    """GUIDE records of the natural slice of *split*, from the committed offset
    labels and the rebuilt text (each turn's hash checked)."""
    from bench.realdata.label import from_public, load_messages
    msgs, committed = load_messages(ds, rd, build)
    labs = {r["id"]: r for r in read_jsonl(rd / ds / "labels.jsonl")}
    turns: Dict[str, List[dict]] = defaultdict(list)
    for m in msgs:
        turns[m["conv"]].append(m)
    out = []
    for sid, ms in turns.items():
        if committed[sid]["split"] != split:
            continue
        recs = []
        for m in ms:
            lab = labs[m["id"]]
            if lab["sha256"] != sha256(m["text"]):
                raise SystemExit(f"{m['id']}: label hash does not match the text; refusing to score")
            if lab["label_status"] != "ok":
                recs.append({"text": m["text"], "service_query": False, "protect": [], "sensitive": [],
                             "optional": [], "keep": [], "label_status": lab["label_status"]})
                continue
            v = from_public(lab, m["text"])
            recs.append({"text": m["text"], "service_query": v["service_query"], "protect": v["protect"],
                         "sensitive": v["sensitive"], "optional": v["optional"], "keep": v["keep"],
                         "label_status": "ok", "task": v["task"]})
        task = next((r["task"] for r in recs if "task" in r), "other")
        rec = {"id": f"rd-{ds}-{split}-{NATURAL_FROM + len(out):04d}", "category": task, "lang": "en",
               **{k: v for k, v in recs[0].items() if k != "task"}, "dataset": ds, "source_id": sid,
               "task": task, "slice": "natural", "shift": False,
               "turns": [{k: v for k, v in r.items() if k != "task"} for r in recs] if len(recs) > 1 else None}
        out.append(rec)
    return out


def messages(records: Iterable[dict]) -> List[dict]:
    """One scoring unit per user turn: ``{mid, conv, dataset, task, slices, gold}``.
    ``gold`` is a J2 record (``id`` = the message id); failed natural turns are skipped."""
    out = []
    for rec in records:
        turns = rec["turns"] or [rec]
        multi = rec["turns"] is not None
        for k, t in enumerate(turns):
            if t.get("label_status", "ok") != "ok":
                continue
            mid = f"{rec['id']}#t{k}" if multi else rec["id"]
            if rec["slice"] == "natural":
                slices = ("natural",)
            elif multi:
                slices = ("injected", "multi")
            else:
                slices = ("injected", "shift" if rec["shift"] else "injected_single")
            gold = {"id": mid, "text": t["text"], "service_query": t["service_query"]}
            gold.update({name: t.get(name, []) for name in rw.LISTS})
            out.append({"mid": mid, "conv": rec["id"], "dataset": rec["dataset"], "task": rec["task"],
                        "slices": slices, "turn": k, "gold": gold})
    return out


def lint_records(records: Sequence[dict]) -> List[Tuple[str, str]]:
    errors = []
    for r in records:
        for k, t in enumerate(r["turns"] or [r]):
            if t.get("label_status", "ok") == "ok":
                errors += [(r["id"], f"turn {k + 1}: {p}") for _i, p in
                           rw.lint([{"id": r["id"], "category": r["category"], "lang": "en", **t}])]
    return errors


# ── 3. spans ─────────────────────────────────────────────────────────────────

def write_input(units: Sequence[dict], path: Path) -> str:
    write_jsonl(path, [{"id": u["mid"], "text": u["gold"]["text"]} for u in units], private=True)
    return file_sha256(path)


def read_spans(full: Path, units: Sequence[dict], input_sha: str) -> Dict[str, dict]:
    """The span rows of *full*, refusing a file made from another input or one
    whose ids or offsets do not fit the messages."""
    meta_path = Path(str(full) + ".meta.json")
    if not full.exists() or not meta_path.exists():
        raise SystemExit(f"{full.name}: no span file")
    meta = json.loads(meta_path.read_text())
    if meta.get("input_sha256") != input_sha:
        raise SystemExit(f"{full.parent.name}/{full.name}: made from another input "
                         f"({str(meta.get('input_sha256'))[:12]} ≠ {input_sha[:12]}); rerun without --reuse")
    rows = {r["id"]: r for r in read_jsonl(full)}
    texts = {u["mid"]: u["gold"]["text"] for u in units}
    if set(rows) != set(texts):
        raise SystemExit(f"{full.parent.name}/{full.name}: ids differ from the input")
    for mid, r in rows.items():
        prev = 0
        for s, e, _t, _rep in r["edits"]:
            if not (prev <= s <= e <= len(texts[mid])):
                raise SystemExit(f"{full.parent.name}/{full.name}: {mid} has an edit outside its text or out of order")
            prev = e
    return rows


def config_hash(full: Path) -> Optional[str]:
    """The DetectionConfig hash an arm wrote into *full*'s ``.meta.json``
    (V3 §3.6; None for an arm without one)."""
    meta = json.loads(Path(str(full) + ".meta.json").read_text())
    return (meta.get("config") or {}).get("detection_config_hash")


def produce(arm: str, src: Path, name: str, reuse: bool, input_sha: str, units: Sequence[dict],
            runner: Optional[Callable] = None, spans: Optional[Path] = None) -> Dict[str, dict]:
    """Run *arm* on *src* (unless *reuse*) and read back its checked spans."""
    from bench.arms.run import PRIVATE, run_arm
    if not reuse:
        (runner or run_arm)(arm, src, name)
    return read_spans((spans or PRIVATE) / arm / f"{name}.jsonl", units, input_sha)


# ── 4. per-message scores (counts and types only) ────────────────────────────

def score_unit(unit: dict, row: dict) -> dict:
    gold = unit["gold"]
    text = gold["text"]
    values = Counter(x["type"] for x in gold["protect"])
    formats = Counter((x["type"], x["fmt"]) for x in gold["protect"] if x.get("fmt"))
    if "refused" in row:                     # nothing was sent, so nothing leaked and nothing was edited
        return {"values": values, "leaked": Counter(), "policy": Counter(), "edits": 0, "spurious": 0,
                "keep_hits": 0, "spurious_types": Counter(), "sensitive": len(gold["sensitive"]),
                "sensitive_leaked": 0, "refused": 1, "formats": formats, "formats_leaked": Counter(),
                "formats_policy": Counter()}
    r = rw.score_message(gold, SimpleNamespace(edits=[(s, e, text[s:e], rep) for s, e, _t, rep in row["edits"]]))
    gold_spans = [sp for name in ("protect", "sensitive", "optional") for x in gold[name]
                  for sp in rw.occurrences(text, x["value"])]
    spurious_types = Counter(t for s, e, t, _rep in row["edits"] if not any(s < ge and gs < e for gs, ge in gold_spans))
    if sum(spurious_types.values()) != len(r["spurious"]):
        raise RuntimeError(f"{unit['mid']}: spurious edits by type disagree with score_message")
    return {"values": values, "leaked": Counter(x["type"] for x in r["leaked"]),
            "policy": Counter(x["type"] for x in r["policy"]), "edits": len(row["edits"]),
            "spurious": len(r["spurious"]), "keep_hits": len(r["keep_hits"]), "spurious_types": spurious_types,
            "sensitive": len(gold["sensitive"]), "sensitive_leaked": len(r["sensitive_leaked"]), "refused": 0,
            "formats": formats, "formats_leaked": Counter((x["type"], x["fmt"]) for x in r["leaked"] if x.get("fmt")),
            "formats_policy": Counter((x["type"], x["fmt"]) for x in r["policy"] if x.get("fmt"))}


# ── 5. aggregation ───────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959964) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def _universe(scores: Sequence[dict], types: Iterable[str]) -> dict:
    types = set(types)
    n = sum(v for s in scores for t, v in s["values"].items() if t in types)
    p = sum(v for s in scores for t, v in s["policy"].items() if t in types)
    k = sum(v for s in scores for t, v in s["leaked"].items() if t in types)
    return rate(k, n - p)


def aggregate(units: Sequence[dict], scores: Sequence[dict]) -> dict:
    values = sum(sum(s["values"].values()) for s in scores)
    policy = sum(sum(s["policy"].values()) for s in scores)
    leaked = sum(sum(s["leaked"].values()) for s in scores)
    with_values = [s for s in scores if sum(s["values"].values()) > sum(s["policy"].values())]
    negatives = [s for s in scores if not s["values"]]
    by_type = {}
    for t in sorted({t for s in scores for t in s["values"]}):
        n = sum(s["values"][t] for s in scores)
        p = sum(s["policy"][t] for s in scores)
        k = sum(s["leaked"][t] for s in scores)
        by_type[t] = {"values": n, "policy": p, "leaked": k, "leak_rate": round(k / (n - p), 4) if n > p else None}
    by_task: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for u, s in zip(units, scores):
        by_task[u["task"]][0] += sum(s["values"].values())
        by_task[u["task"]][1] += sum(s["policy"].values())
        by_task[u["task"]][2] += sum(s["leaked"].values())
    typed = [v["leak_rate"] for v in by_type.values() if v["leak_rate"] is not None]
    edits = sum(s["edits"] for s in scores)
    return {
        "messages": len(scores), "conversations": len({u["conv"] for u in units}),
        "protect_values": values, "policy": policy, "leak": rate(leaked, values - policy),
        "macro_leak_rate": round(sum(typed) / len(typed), 4) if typed else None,
        "message_leak": rate(sum(1 for s in with_values if s["leaked"]), len(with_values)),
        "edits": edits, "spurious": rate(sum(s["spurious"] for s in scores), edits),
        "keep_hits": sum(s["keep_hits"] for s in scores),
        "messages_with_spurious": sum(1 for s in scores if s["spurious"]),
        "negatives_untouched": rate(sum(1 for s in negatives if not s["edits"]), len(negatives)),
        "sensitive_values": sum(s["sensitive"] for s in scores),
        "sensitive_leaked": sum(s["sensitive_leaked"] for s in scores),
        "refused": sum(s["refused"] for s in scores),
        "by_type": by_type,
        "by_task": {t: {"values": v[0], "policy": v[1], "leaked": v[2],
                        "leak_rate": round(v[2] / (v[0] - v[1]), 4) if v[0] > v[1] else None}
                    for t, v in sorted(by_task.items())},
        "universes": {name: _universe(scores, types) for name, types in UNIVERSES.items()},
        "spurious_by_edit_type": dict(sorted(Counter(t for s in scores for t in s["spurious_types"].elements()).items())),
    }


def held_out(scores: Sequence[dict], formats: Sequence[Tuple[str, str]] = HELD_OUT_FORMATS) -> dict:
    """Leak on the values written in a held-out format, per format and together, beside the other
    formats of the same types (V3 §5.3.2)."""
    def count(keep):
        n = sum(v for s in scores for f, v in s.get("formats", {}).items() if keep(f))
        p = sum(v for s in scores for f, v in s.get("formats_policy", {}).items() if keep(f))
        k = sum(v for s in scores for f, v in s.get("formats_leaked", {}).items() if keep(f))
        return rate(k, n - p)
    held, types = set(formats), {t for t, _f in formats}
    return {"by_format": {f"{t}/{f}": count(lambda x, tf=(t, f): x == tf) for t, f in formats},
            "held_out": count(lambda x: x in held),
            "same_types_other_formats": count(lambda x: x[0] in types and x not in held)}


METRICS = {   # name -> per-message (numerator, denominator)
    "leak_rate": lambda s: (sum(s["leaked"].values()), sum(s["values"].values()) - sum(s["policy"].values())),
    "message_leak_rate": lambda s: ((int(bool(s["leaked"])), 1) if sum(s["values"].values()) > sum(s["policy"].values())
                                    else (0, 0)),
    "spurious_rate": lambda s: (s["spurious"], s["edits"]),
    "negatives_untouched_rate": lambda s: ((int(not s["edits"]), 1) if not s["values"] else (0, 0)),
}


def bootstrap(clusters: Sequence[str], a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]],
              seed: int, resamples: int = RESAMPLES) -> dict:
    """Paired cluster bootstrap of ratio(a) − ratio(b): per-message
    (numerator, denominator) pairs, resampled by cluster with the same draws
    for both systems. Resamples with a zero denominator are dropped and counted."""
    import numpy as np
    keys = sorted(set(clusters))
    pos = {k: i for i, k in enumerate(keys)}
    agg = np.zeros((len(keys), 4))
    for c, (an, ad), (bn, bd) in zip(clusters, a, b):
        agg[pos[c]] += (an, ad, bn, bd)
    tot = agg.sum(0)
    point = (tot[0] / tot[1] - tot[2] / tot[3]) if tot[1] and tot[3] else None
    if point is None:
        return {"diff": None, "ci95": None, "resamples": 0}
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
    s = agg[idx].sum(1)
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    d = s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]
    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (float("nan"), float("nan"))
    return {"diff": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_0": bool(lo > 0 or hi < 0), "resamples": int(ok.sum())}


def differences(units: Sequence[dict], scores: Dict[str, Sequence[dict]], seed: int, arms: Sequence[str]) -> dict:
    clusters = [u["conv"] for u in units]
    out = {}
    for arm in arms:
        if arm == "ss":
            continue
        out[arm] = {m: bootstrap(clusters, [f(s) for s in scores["ss"]], [f(s) for s in scores[arm]], seed)
                    for m, f in METRICS.items()}
    return out


def differences_by_type(units: Sequence[dict], scores: Dict[str, Sequence[dict]], data: str, group: str,
                        arms: Sequence[str], sl: str = "injected") -> dict:
    """Per protect type: the paired cluster bootstrap of SS's leak rate of that
    type minus each arm's (V3 H5''), from per-message (leaked, values − policy)
    of the type; one seed per type, the same draws for every arm."""
    clusters = [u["conv"] for u in units]
    types = sorted({t for s in scores["ss"] for t in s["values"]})
    out: Dict[str, Dict[str, dict]] = {}
    for arm in arms:
        if arm == "ss":
            continue
        out[arm] = {}
        for t in types:
            def pair(s, t=t):
                return s["leaked"].get(t, 0), s["values"].get(t, 0) - s["policy"].get(t, 0)
            out[arm][t] = bootstrap(clusters, [pair(s) for s in scores["ss"]], [pair(s) for s in scores[arm]],
                                    derive_seed("score-bootstrap-type", data, group, sl, t))
    return out


def hypotheses(results: dict, diffs: dict, datasets: Sequence[str], arms: Sequence[str]) -> dict:
    """H1 and H2 as pre-registered, per dataset, on the injected slice. The
    verdict is the paper's only on the test split."""
    out = {}
    for ds in datasets:
        res, dif = results[ds]["injected"], diffs[ds]["injected"]
        base = [a for a in arms if a != "ss"]
        h1 = {a: res["ss"]["leak"]["rate"] is not None and res[a]["leak"]["rate"] is not None
              and res["ss"]["leak"]["rate"] < res[a]["leak"]["rate"]
              and dif[a]["leak_rate"]["ci95"] is not None and dif[a]["leak_rate"]["ci95"][1] < 0 for a in base}
        h2 = {a: dif[a]["spurious_rate"]["ci95"] is not None and dif[a]["spurious_rate"]["ci95"][1] < 0
              for a in ("presidio_default", "presidio_faker") if a in base}
        worse = ("presidio_default" in res and res["ss"]["leak"]["rate"] is not None
                 and res["ss"]["leak"]["rate"] > res["presidio_default"]["leak"]["rate"])
        out[ds] = {"H1_per_baseline": h1, "H1": bool(h1) and all(h1.values()),
                   "H2_per_baseline": h2, "H2": bool(h2) and all(h2.values()),
                   "ss_leak_worse_than_presidio_default": worse}
    return out


# ── 6. run ───────────────────────────────────────────────────────────────────

def load_split(split: str, datasets: Sequence[str] = DATASETS, rd: Path = RD, build: Path = BUILD,
               frozen: Optional[dict] = None, prefix: str = "") -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Check the frozen hashes, rebuild and lint the records, write one arm
    input per dataset. Returns the hashes and, per dataset, ``units``,
    ``src`` (the arm input), ``input_sha`` and ``corpus`` (counts). *split*
    is a data split; *rd* / *build* / *prefix* are its collection's."""
    if frozen is None:
        from bench.realdata import manifest
        frozen = manifest.load()["frozen"]
    hashes = check_frozen(datasets, split, frozen, rd, prefix)
    out = {}
    for ds in datasets:
        injected = read_jsonl(rd / ds / f"{split}.jsonl")
        natural = natural_records(ds, split, rd, build)
        errors = lint_records(injected) + lint_records(natural)
        if errors:
            raise SystemExit(f"{ds}/{split}: {len(errors)} lint errors, first {errors[:3]}")
        write_jsonl(build / ds / f"natural-{split}.jsonl", natural, private=True)
        units = messages(injected) + messages(natural)
        src = build / "score" / split / f"{ds}.jsonl"
        input_sha = write_input(units, src)
        corpus = {"injected_records": len(injected), "natural_records": len(natural),
                  "natural_turns_excluded": sum(1 for r in natural for t in (r["turns"] or [r])
                                                if t.get("label_status", "ok") != "ok"),
                  "messages": dict(sorted(Counter(sl for u in units for sl in u["slices"]).items())),
                  "arm_input_sha256": input_sha}
        out[ds] = {"units": units, "src": src, "input_sha": input_sha, "corpus": corpus}
    return hashes, out


def score_split(split: str, datasets: Sequence[str] = DATASETS, arms: Sequence[str] = ARMS, reuse: bool = False,
                out: Optional[Path] = None, rd: Optional[Path] = None, build: Optional[Path] = None,
                frozen: Optional[dict] = None, runner: Optional[Callable] = None, spans: Optional[Path] = None,
                log=print, freeze: Path = FREEZE, prereg: Path = HYPOTHESES) -> dict:
    """Score the run *split* (a key of ``RUNS``). Span files are named
    ``<run>-<dataset>``; records, private files and bootstrap draws follow the
    data split, so ``devlarge`` reproduces ``test`` for an unchanged arm."""
    data, coll_name, role = RUNS[split]
    coll = COLLECTIONS[coll_name]
    rd, build = rd or coll.rd, build or coll.build
    sealed = check_freeze(freeze, prereg) if coll.prefix else None
    hashes, loaded = load_split(data, datasets, rd, build, frozen, coll.prefix)
    all_units: Dict[str, List[dict]] = {}
    all_scores: Dict[str, Dict[str, List[dict]]] = {}
    corpus = {}
    hashes_seen: Dict[str, set] = defaultdict(set)
    from bench.arms.run import PRIVATE
    for ds in datasets:
        units, src, input_sha = loaded[ds]["units"], loaded[ds]["src"], loaded[ds]["input_sha"]
        all_units[ds] = units
        all_scores[ds] = {}
        for arm in arms:
            rows = produce(arm, src, f"{split}-{ds}", reuse, input_sha, units, runner, spans)
            hashes_seen[arm].add(config_hash((spans or PRIVATE) / arm / f"{split}-{ds}.jsonl"))
            all_scores[ds][arm] = [score_unit(u, rows[u["mid"]]) for u in units]
            log(f"{split} {ds:9} {arm:22} scored {len(units)} messages")
        corpus[ds] = loaded[ds]["corpus"]
    config_hashes = {}
    for arm, seen in hashes_seen.items():
        if len(seen) > 1:
            raise SystemExit(f"{arm}: span files made with different detection configs {sorted(map(str, seen))}")
        if None not in seen:
            config_hashes[arm] = next(iter(seen))
    pooled = "all" if len(datasets) > 1 else None
    groups = list(datasets) + ([pooled] if pooled else [])
    results, diffs, by_type = {}, {}, {}
    for g in groups:
        dss = datasets if g == "all" else [g]
        units = [u for ds in dss for u in all_units[ds]]
        scores = {a: [s for ds in dss for s in all_scores[ds][a]] for a in arms}
        results[g], diffs[g] = {}, {}
        for sl in SLICES:
            keep = [i for i, u in enumerate(units) if sl in u["slices"]]
            us = [units[i] for i in keep]
            sc = {a: [scores[a][i] for i in keep] for a in arms}
            results[g][sl] = {a: aggregate(us, sc[a]) for a in arms}
            diffs[g][sl] = differences(us, sc, derive_seed("score-bootstrap", data, g, sl), arms) if "ss" in arms else {}
            if sl == "injected" and "ss" in arms:
                by_type[g] = differences_by_type(us, sc, data, g, arms, sl)
    out = out or ROOT / "bench" / "results" / f"realdata_{split}.json"
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score "
                      f"--split {split} --out {rel(out)}",
           "git": git_state(),
           "split": split, "role": role,
           **({"data_split": data, "collection": coll.name} if split not in ("dev", "test") else {}),
           **({"freeze_sha256": sealed} if sealed else {}),
           "frozen": hashes, "arms": list(arms), "slices": list(SLICES), "corpus": corpus,
           **({"config_hashes": config_hashes} if config_hashes else {}),
           "claimed_types": {a: list(CLAIMED[a]) for a in arms}, "universes": {k: list(v) for k, v in UNIVERSES.items()},
           "bootstrap": {"resamples": RESAMPLES, "unit": "conversation (a single-turn prompt is its own)",
                         "seed": "derive_seed('score-bootstrap', split, dataset, slice)", "ci": "percentile 2.5 / 97.5",
                         "difference": "ss − arm"},
           "results": results, "differences": diffs,
           **({"differences_by_type": {"slice": "injected", "metric": "leak_rate", "difference": "ss − arm",
                                       "seed": "derive_seed('score-bootstrap-type', split, group, slice, type)",
                                       "groups": by_type}} if by_type else {}),
           "held_out_formats": {"formats": [f"{t}/{f}" for t, f in HELD_OUT_FORMATS], "slice": "injected",
                                "results": {g: {a: held_out([s for ds in (datasets if g == "all" else [g])
                                                             for u, s in zip(all_units[ds], all_scores[ds][a])
                                                             if "injected" in u["slices"]]) for a in arms}
                                            for g in groups}},
           "hypotheses": hypotheses(results, diffs, datasets, arms) if "ss" in arms else {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def span_f1(agg: dict) -> Optional[float]:
    """Value-level F1: recall 1 − leak rate, precision 1 − spurious rate."""
    if agg["leak"]["rate"] is None or agg["spurious"]["rate"] is None:
        return None
    p, r = 1 - agg["spurious"]["rate"], 1 - agg["leak"]["rate"]
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def h10(res: dict) -> dict:
    out = {}
    for g in ("gliner_pii", "gliner_pii_tuned"):
        if g not in res or "ss" not in res:
            continue
        ss, gl = res["ss"], res[g]
        leak = ss["leak"]["rate"] is not None and gl["leak"]["rate"] is not None and ss["leak"]["rate"] <= gl["leak"]["rate"]
        f1 = ss["span_f1"] is not None and gl["span_f1"] is not None and ss["span_f1"] >= gl["span_f1"]
        out[g] = {"leak_not_above": leak, "span_f1_not_below": f1, "holds": leak or f1}
    return out


def score_external(name: str, arms: Sequence[str] = ARMS, reuse: bool = False, out: Optional[Path] = None,
                   loaded: Optional[dict] = None, runner: Optional[Callable] = None, spans: Optional[Path] = None,
                   log=print, freeze: Path = FREEZE, prereg: Path = HYPOTHESES) -> dict:
    """Score the external benchmark *name* (``bench.realdata.external``) once, after the freeze."""
    sealed = check_freeze(freeze, prereg)
    if loaded is None:
        from bench.realdata import external
        if name != external.NAME:
            raise SystemExit(f"no external benchmark {name!r} (built: {external.NAME})")
        loaded = external.load()
    units, src, input_sha = loaded["units"], loaded["src"], loaded["input_sha"]
    from bench.arms.run import PRIVATE
    scores, config_hashes = {}, {}
    for arm in arms:
        rows = produce(arm, src, f"external-{name}", reuse, input_sha, units, runner, spans)
        h = config_hash((spans or PRIVATE) / arm / f"external-{name}.jsonl")
        if h:
            config_hashes[arm] = h
        scores[arm] = [score_unit(u, rows[u["mid"]]) for u in units]
        log(f"external {name} {arm:22} scored {len(units)} messages")
    groups = sorted({u["dataset"] for u in units}) + ["all"]
    results, diffs = {}, {}
    for g in groups:
        keep = [i for i, u in enumerate(units) if g == "all" or u["dataset"] == g]
        us = [units[i] for i in keep]
        sc = {a: [scores[a][i] for i in keep] for a in arms}
        results[g] = {a: aggregate(us, sc[a]) for a in arms}
        for a in arms:
            results[g][a]["span_f1"] = span_f1(results[g][a])
        diffs[g] = differences(us, sc, derive_seed("score-external", name, g), arms) if "ss" in arms else {}
    out = out or ROOT / "bench" / "results" / f"external_{name}.json"
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score "
                      f"--external {name} --out {rel(out)}",
           "git": git_state(), "external": name, "role": "secondary check (H10''), not used for tuning",
           "freeze_sha256": sealed, "source": loaded.get("description", {}).get("source"),
           "label_map": loaded.get("description", {}).get("label_map"), "arms": list(arms),
           "corpus": loaded["corpus"], **({"config_hashes": config_hashes} if config_hashes else {}),
           "claimed_types": {a: list(CLAIMED[a]) for a in arms},
           "bootstrap": {"resamples": RESAMPLES, "unit": "the source's record group (uid)",
                         "seed": "derive_seed('score-external', name, group)", "ci": "percentile 2.5 / 97.5",
                         "difference": "ss − arm"},
           "span_f1": "value-level: precision 1 − spurious rate, recall 1 − leak rate",
           "results": results, "differences": diffs, "hypotheses": {"H10''": h10(results["all"])}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown_external(doc))
    return doc


def rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ── markdown ─────────────────────────────────────────────────────────────────

def _pct(r: Optional[dict]) -> str:
    if not r or r.get("rate") is None:
        return "–"
    ci = r["wilson95"]
    return f"{100 * r['rate']:.1f} % [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _diff(d: Optional[dict]) -> str:
    if not d or d.get("diff") is None:
        return "–"
    lo, hi = d["ci95"]
    star = " *" if d.get("excludes_0") else ""
    return f"{100 * d['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{star}"


def markdown(doc: dict) -> str:
    arms, res, dif = doc["arms"], doc["results"], doc["differences"]
    lines = [f"# Real-data detection, split `{doc['split']}` ({doc['role']})", "",
             f"Command: `{doc['command']}`" + commit_note(doc.get("git")), "",
             "Rates with Wilson 95 % intervals; Δ = SS − arm in percentage points with a paired cluster-bootstrap "
             f"95 % interval ({doc['bootstrap']['resamples']} resamples); * = interval excludes 0. "
             "Leak = protect values with a letter or digit reaching the provider (policy values apart); "
             "spurious = edits touching no gold value.", ""]
    for g in res:
        for sl in doc["slices"]:
            r = res[g][sl]
            if not r[arms[0]]["messages"]:
                continue
            lines += [f"## {g} — {sl} ({r[arms[0]]['messages']} messages, {r[arms[0]]['protect_values']} protect values)", "",
                      "| arm | leaked / values | leak rate | Δ leak | message leak | edits | spurious rate | Δ spurious | negatives untouched |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for a in arms:
                x, d = r[a], dif[g][sl].get(a, {})
                lines.append(f"| {a} | {x['leak']['k']} / {x['leak']['n']} | {_pct(x['leak'])} | {_diff(d.get('leak_rate'))} "
                             f"| {_pct(x['message_leak'])} | {x['edits']} | {_pct(x['spurious'])} | {_diff(d.get('spurious_rate'))} "
                             f"| {_pct(x['negatives_untouched'])} |")
            lines.append("")
    if "all" in res:
        r = res["all"]["injected"]
        types = sorted({t for a in arms for t in r[a]["by_type"]})
        lines += ["## all — injected, leak rate by type (leaked / values)", "",
                  "| type | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for t in types:
            cells = []
            for a in arms:
                v = r[a]["by_type"].get(t)
                cells.append(f"{v['leaked']} / {v['values'] - v['policy']}" if v else "–")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines += ["", "## all — injected, by type universe", "", "| universe | types | " + " | ".join(arms) + " |",
                  "|---|---|" + "---|" * len(arms)]
        for u, types_u in doc["universes"].items():
            lines.append(f"| {u} | {', '.join(types_u)} | " + " | ".join(_pct(r[a]["universes"][u]) for a in arms) + " |")
        lines.append("")
    if doc.get("differences_by_type"):
        bt = doc["differences_by_type"]["groups"]
        g = "all" if "all" in bt else next(iter(bt))
        others = [a for a in arms if a in bt[g]]
        types = sorted({t for a in others for t in bt[g][a]})
        lines += [f"## {g} — injected, Δ leak rate by type (SS − arm, percentage points, paired cluster bootstrap 95 %)", "",
                  "| type | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for t in types:
            lines.append(f"| {t} | " + " | ".join(_diff(bt[g][a].get(t)) for a in others) + " |")
        lines.append("")
    if doc.get("held_out_formats"):
        h = doc["held_out_formats"]
        g = "all" if "all" in h["results"] else next(iter(h["results"]))
        r = h["results"][g]
        cols = list(r[arms[0]]["by_format"]) + ["held_out", "same_types_other_formats"]
        lines += [f"## {g} — injected, formats the tagger never saw in training (leaked / values)", "",
                  "| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for a in arms:
            cells = [r[a]["by_format"][c] if c in r[a]["by_format"] else r[a][c] for c in cols]
            lines.append(f"| {a} | " + " | ".join(f"{x['k']} / {x['n']}" for x in cells) + " |")
        lines.append("")
    if doc["hypotheses"]:
        lines += ["## Pre-registered checks (injected slice)", "", "| dataset | H1 | H2 | SS leak worse than Presidio-default |",
                  "|---|---|---|---|"]
        for ds, h in doc["hypotheses"].items():
            lines.append(f"| {ds} | {'yes' if h['H1'] else 'no'} | {'yes' if h['H2'] else 'no'} | "
                         f"{'yes' if h['ss_leak_worse_than_presidio_default'] else 'no'} |")
        lines.append("")
    return "\n".join(lines)


def _f1(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:.3f}"


def markdown_external(doc: dict) -> str:
    arms, res, dif = doc["arms"], doc["results"], doc["differences"]
    lines = [f"# External benchmark `{doc['external']}` ({doc['role']})", "",
             f"Command: `{doc['command']}`" + commit_note(doc.get("git")), "",
             f"Source: {(doc.get('source') or {}).get('repo')} @ `{str((doc.get('source') or {}).get('revision'))[:12]}` "
             f"({(doc.get('source') or {}).get('licence')}); {doc['corpus']['records']} records, "
             f"strata {doc['corpus']['strata']}. Same scorer as the real-data benchmark: leak = a protect value with a "
             "letter or digit reaching the provider; spurious = an edit touching no gold value; span F1 = "
             "value-level (precision 1 − spurious rate, recall 1 − leak rate). Δ = SS − arm, paired cluster "
             f"bootstrap ({doc['bootstrap']['resamples']} resamples); * = interval excludes 0.", ""]
    for g in res:
        r = res[g]
        lines += [f"## {g} ({r[arms[0]]['messages']} records, {r[arms[0]]['protect_values']} protect values)", "",
                  "| arm | leaked / values | leak rate | Δ leak | edits | spurious rate | Δ spurious | span F1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for a in arms:
            x, d = r[a], dif[g].get(a, {})
            lines.append(f"| {a} | {x['leak']['k']} / {x['leak']['n']} | {_pct(x['leak'])} | {_diff(d.get('leak_rate'))} "
                         f"| {x['edits']} | {_pct(x['spurious'])} | {_diff(d.get('spurious_rate'))} "
                         f"| {_f1(x['span_f1'])} |")
        lines.append("")
    r = res["all"]
    types = sorted({t for a in arms for t in r[a]["by_type"]})
    lines += ["## all — leak by type (leaked / values)", "", "| type | " + " | ".join(arms) + " |",
              "|---|" + "---|" * len(arms)]
    for t in types:
        cells = []
        for a in arms:
            v = r[a]["by_type"].get(t)
            cells.append(f"{v['leaked']} / {v['values'] - v['policy']}" if v else "–")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## H10'' (pooled; point estimates)", "", "| GLiNER arm | SS leak ≤ | SS span F1 ≥ | holds |",
              "|---|---|---|---|"]
    for g, h in doc["hypotheses"]["H10''"].items():
        lines.append(f"| {g} | {'yes' if h['leak_not_above'] else 'no'} | {'yes' if h['span_f1_not_below'] else 'no'} "
                     f"| {'yes' if h['holds'] else 'no'} |")
    if doc.get("label_map"):
        lines += ["", "## Label map (Nemotron-PII label → J2 list, type)", "", "| label | list | type |", "|---|---|---|"]
        for label, (lst, typ) in sorted(doc["label_map"].items(), key=lambda kv: (kv[1][0], kv[1][1] or "", kv[0])):
            lines.append(f"| {label} | {lst} | {typ or '–'} |")
    c = doc["corpus"]
    lines += ["", f"Values dropped (not a whole-word substring of their text): {sum(c['dropped_not_whole_word'].values())} "
              f"{c['dropped_not_whole_word']}; values given labels of different lists or types (first kept): "
              f"{sum(c['merged_duplicate_values'].values())}.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--split", choices=list(RUNS))
    which.add_argument("--external", help="an external benchmark built by bench.realdata.external (V3 §5.4)")
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--arms", nargs="*", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--reuse", action="store_true", help="rescore saved spans; run no arm")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.external:
        doc = score_external(args.external, args.arms, args.reuse, args.out)
        for g, h in doc["hypotheses"]["H10''"].items():
            print(f"H10'' vs {g}: {'holds' if h['holds'] else 'does not hold'} {h}")
        return 0
    doc = score_split(args.split, args.datasets, args.arms, args.reuse, args.out)
    for ds, h in doc["hypotheses"].items():
        r = doc["results"][ds]["injected"]
        print(f"{ds:9} " + "  ".join(f"{a} leak {r[a]['leak']['rate']} spur {r[a]['spurious']['rate']}" for a in doc["arms"]))
        print(f"{'':9} H1 {h['H1']}  H2 {h['H2']}  SS worse than Presidio-default {h['ss_leak_worse_than_presidio_default']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
