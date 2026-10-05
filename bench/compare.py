#!/usr/bin/env python3
# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""SurrogateShield vs default-config Presidio on the same data — one table.

    python bench/compare.py                         # tuning view: synth dev + realworld dev
    python bench/compare.py --realworld dev2
    python bench/compare.py --final                 # synth test + realworld test (once)
    python bench/compare.py --json bench/results/compare_dev.json

Both systems see the same messages. SurrogateShield runs its single send path
(``json_tester.prepare_send``, seeded per message); Presidio runs
``presidio.detect`` (default config, threshold ``presidio.engine.SCORE_THRESHOLD``,
overlaps resolved by longest span) and replaces every entity it returns.
No provider calls, no network.

Synthetic set (``experiment/synth_<split>_key.json``, gold spans with offsets):

* **any type** — a gold span is protected if any edit overlaps it; an edit
  is correct if it overlaps any gold span. Micro P/R/F1 and recall per gold
  type. This is what a reader of the sent text sees.
* **shared universe, typed** — ``eval_metrics.SHARED_TYPES`` only (types both
  systems can emit), a prediction counts only with the right type; gold of
  other types is neutral for both systems. Micro and macro P/R/F1.
* **rnr** — SurrogateShield's recognised-but-not-replaced spans. "without rnr"
  (protection) is the headline; "with rnr" (recognition) is reported beside it.
  Presidio replaces everything it finds, so both columns are the same for it.
* **negatives** — share of PII-free messages that got any edit.

Real-world corpus (``bench/realworld/<split>.jsonl``): leaked must-protect
values and spurious edits, scored by ``bench/realworld.score_message`` for
both systems (gate J2 applies to SurrogateShield only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import eval_metrics as em  # noqa: E402

SEED = 20260101              # offline_eval.SEED: message i uses MimicGen(seed=SEED + i)


def _presidio():
    from presidio.detect import detect
    from presidio.engine import baseline_config

    def found(text):
        ents = detect(text)
        if ents is None:
            raise RuntimeError("Presidio is not installed (pip install presidio-analyzer, "
                               "python -m spacy download en_core_web_lg)")
        return ents
    return found, baseline_config()


def _any(spans):
    return [em.Span(s.start, s.end, "ANY", s.value) for s in spans]


def score_synth(split: str, limit: int | None = None) -> dict:
    from generation.logic import MimicGen
    from json_tester import prepare_send

    rows = json.loads((ROOT / f"experiment/synth_{split}_key.json").read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    presidio_found, presidio_cfg = _presidio()

    t = {k: em.Tally() for k in ("ss_any", "ss_any_rnr", "prs_any",
                                 "ss_typed", "ss_typed_rnr", "prs_typed")}
    by_type = defaultdict(lambda: Counter())
    ms = {"ss": [], "presidio": []}
    for i, r in enumerate(rows):
        text = r["Question"]
        gold, _ = em.entry_gold(text, r)

        t0 = time.perf_counter()
        spans = prepare_send(text, MimicGen(seed=SEED + i)).spans()
        ms["ss"].append((time.perf_counter() - t0) * 1000)
        ss_all = [em.Span(s["start"], s["end"], em.normalize_type(s["type"]), s["text"]) for s in spans]
        ss_rep = [p for p, s in zip(ss_all, spans) if s["replaced"]]

        t0 = time.perf_counter()
        ents = presidio_found(text)
        ms["presidio"].append((time.perf_counter() - t0) * 1000)
        prs_all = [em.Span(e.start, e.end, e.entity_type, e.text) for e in ents]

        g_any = _any(gold)
        hits = {}
        for name, pred in (("ss_any", ss_rep), ("ss_any_rnr", ss_all), ("prs_any", prs_all)):
            g_hit, _ = t[name].add(g_any, _any(pred))
            hits[name] = g_hit
        for g, a, b, c in zip(gold, hits["ss_any"], hits["ss_any_rnr"], hits["prs_any"]):
            by_type[g.type].update(gold=1, ss=a, ss_rnr=b, presidio=c)

        shared_gold, neutral = em.split_shared(gold)
        for name, pred in (("ss_typed", ss_rep), ("ss_typed_rnr", ss_all)):
            ss = [em.Span(p.start, p.end, em.ss_shared_type(p.type), p.value)
                  for p in pred if em.ss_shared_type(p.type)]
            t[name].add(shared_gold, em.drop_neutral(ss, neutral, shared_gold), em.same_type)
        prs = [em.Span(p.start, p.end, em.presidio_shared_type(p.type, p.value), p.value)
               for p in prs_all if em.presidio_shared_type(p.type, p.value)]
        t["prs_typed"].add(shared_gold, em.drop_neutral(prs, neutral, shared_gold), em.same_type)

    def p50(v):
        return round(sorted(v)[len(v) // 2], 1) if v else None

    return {
        "dataset": f"experiment/synth_{split}_key.json",
        "messages": len(rows),
        "negatives": sum(1 for r in rows if not r["Spans"]),
        "gold_spans": sum(len(r["Spans"]) for r in rows),
        "seed": SEED,
        "presidio": presidio_cfg,
        "any_type": {"ss": t["ss_any"].summary(), "ss_rnr": t["ss_any_rnr"].summary(),
                     "prs": t["prs_any"].summary()},
        "shared_typed": {k: t[k].summary() for k in ("ss_typed", "ss_typed_rnr", "prs_typed")},
        "recall_by_gold_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "latency_ms_p50": {k: p50(v) for k, v in ms.items()},
    }


def score_realworld(split: str) -> dict:
    import realworld

    presidio_found, presidio_cfg = _presidio()

    def presidio_prepare(text, _i):
        ents = presidio_found(text)
        edits = [(e.start, e.end, e.text, f"[{e.entity_type}]") for e in ents]
        from presidio.redact import redact
        return SimpleNamespace(edits=edits, sanitized=redact(text, ents))

    keys = ("leak_rate", "leaked", "policy", "protect_values", "spurious_rate", "spurious",
            "edits", "negatives_untouched", "negatives", "leak_by_type", "latency_ms_p50")
    ss = realworld.run(split, show=False)
    prs = realworld.run(split, show=False, prepare=presidio_prepare)
    return {"dataset": f"bench/realworld/{split}.jsonl", "messages": ss["corpus"]["messages"],
            "presidio": presidio_cfg, "J2": ss["J2"],
            "ss": {k: ss[k] for k in keys}, "presidio_result": {k: prs[k] for k in keys}}


# ── printing ─────────────────────────────────────────────────────────────────

def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def print_synth(r: dict) -> None:
    print(f"\n== {r['dataset']}  ({r['messages']} messages, {r['gold_spans']} gold spans, "
          f"{r['negatives']} PII-free) ==")
    a, s = r["any_type"], r["shared_typed"]
    head = f"{'':34}{'SS':>8}{'SS+rnr':>8}{'Presidio':>10}"
    print(head)
    rows = [
        ("any type   micro P", "micro", "precision"), ("any type   micro R", "micro", "recall"),
        ("any type   micro F1", "micro", "f1"),
    ]
    for label, block, m in rows:
        print(f"{label:34}{_f(a['ss'][block][m]):>8}{_f(a['ss_rnr'][block][m]):>8}"
              f"{_f(a['prs'][block][m]):>10}")
    for label, block, m in [
        ("shared typed micro P", "micro", "precision"), ("shared typed micro R", "micro", "recall"),
        ("shared typed micro F1", "micro", "f1"), ("shared typed macro P", "macro", "precision"),
        ("shared typed macro R", "macro", "recall"), ("shared typed macro F1", "macro", "f1"),
    ]:
        print(f"{label:34}{_f(s['ss_typed'][block][m]):>8}{_f(s['ss_typed_rnr'][block][m]):>8}"
              f"{_f(s['prs_typed'][block][m]):>10}")
    for label, k in (("negatives with any edit", "negative_question_fp_rate"),
                     ("edits on negatives", "fp_spans_on_negatives")):
        print(f"{label:34}{_f(a['ss']['negatives'][k]):>8}{_f(a['ss_rnr']['negatives'][k]):>8}"
              f"{_f(a['prs']['negatives'][k]):>10}")
    lat = r["latency_ms_p50"]
    print(f"{'latency p50 ms':34}{_f(lat['ss']):>8}{'':>8}{_f(lat['presidio']):>10}")
    print(f"\n  recall by gold type (any edit overlapping){'':4}{'n':>6}{'SS':>8}{'SS+rnr':>8}"
          f"{'Presidio':>10}")
    for typ, c in r["recall_by_gold_type"].items():
        n = c["gold"]
        mark = "  ← SS below Presidio" if c["ss"] < c["presidio"] else ""
        print(f"  {typ:44}{n:>6}{c['ss'] / n:>8.3f}{c['ss_rnr'] / n:>8.3f}"
              f"{c['presidio'] / n:>10.3f}{mark}")
    print(f"\n  shared universe per type (typed F1){'':11}{'SS':>8}{'Presidio':>10}")
    pt_s, pt_p = s["ss_typed"]["per_type"], s["prs_typed"]["per_type"]
    for typ in sorted(set(pt_s) | set(pt_p)):
        fs, fp = pt_s.get(typ, {}).get("f1", 0.0), pt_p.get(typ, {}).get("f1", 0.0)
        mark = "  ← SS below Presidio" if fs < fp else ""
        print(f"  {typ:44}{fs:>8.3f}{fp:>10.3f}{mark}")


def print_realworld(r: dict) -> None:
    s, p = r["ss"], r["presidio_result"]
    print(f"\n== {r['dataset']}  ({r['messages']} messages) ==")
    print(f"{'':34}{'SS':>8}{'Presidio':>10}")
    print(f"{'leaked must-protect values':34}{s['leak_rate']:>8.3f}{p['leak_rate']:>10.3f}"
          f"   ({s['leaked']} / {p['leaked']} of {s['protect_values'] - s['policy']})")
    print(f"{'spurious edits':34}{s['spurious_rate']:>8.3f}{p['spurious_rate']:>10.3f}"
          f"   ({s['spurious']}/{s['edits']} vs {p['spurious']}/{p['edits']})")
    print(f"{'negatives untouched':34}{s['negatives_untouched']:>8}{p['negatives_untouched']:>10}"
          f"   (of {s['negatives']})")
    print(f"{'service-query policy (SS, not leaked)':34}{s['policy']:>8}")
    print(f"{'latency p50 ms':34}{_f(s['latency_ms_p50']):>8}{_f(p['latency_ms_p50']):>10}")
    print(f"J2 (SS): {r['J2']}")
    print(f"\n  leaked by type{'':20}{'n':>6}{'SS':>8}{'Presidio':>10}")
    for typ, c in s["leak_by_type"].items():
        pl = p["leak_by_type"].get(typ, {}).get("leaked", 0)
        mark = "  ← SS leaks more" if c["leaked"] > pl else ""
        print(f"  {typ:32}{c['values']:>6}{c['leaked']:>8}{pl:>10}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synth", choices=("dev", "test", "none"), default="dev")
    ap.add_argument("--realworld", choices=("dev", "dev2", "test", "none"), default="dev")
    ap.add_argument("--final", action="store_true",
                    help="the held-out test splits of both sets (run once, never tune on it)")
    ap.add_argument("--limit", type=int, help="first N synthetic messages (smoke runs)")
    ap.add_argument("--json", type=Path, help="also write every number here")
    a = ap.parse_args()
    if a.final:
        a.synth, a.realworld = "test", "test"

    out = {"command": "python bench/compare.py " + " ".join(sys.argv[1:])}
    if a.synth != "none":
        out["synth"] = score_synth(a.synth, a.limit)
        print_synth(out["synth"])
    if a.realworld != "none":
        out["realworld"] = score_realworld(a.realworld)
        print_realworld(out["realworld"])
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
