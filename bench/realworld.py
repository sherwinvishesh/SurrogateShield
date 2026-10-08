"""Gate J2 — real-world robustness of the text actually sent to the provider.

Runs every message of ``bench/realworld/<split>.jsonl`` through
``json_tester.prepare_send`` (the single detect → generate → substitute path
the chat, the runner and ``offline_eval.py --protection`` use) and scores the
planned edits against the gold lists described in ``bench/realworld/GUIDE.md``.

    python bench/realworld.py --lint bench/realworld/test.jsonl
    python bench/realworld.py --split dev  [--show]      # tune on this
    python bench/realworld.py --split test               # J2; never tune on it
    python bench/realworld.py --file PATH [--id-prefix P] # any corpus in this format

Definitions (character spans in the original message):

* leaked     — a ``protect`` value with any letter or digit of one of its
               occurrences outside every edit, or copied verbatim into the
               surrogate that covers it. Rate = leaked values / protect values.
* policy     — the same, for ADDRESS / LOCATION values of a gold
               ``service_query`` message (documented service-query policy).
               Reported, not counted as leaked.
* spurious   — an edit that overlaps no ``protect`` / ``sensitive`` /
               ``optional`` value. Rate = spurious edits / all edits.
               ``keep hits`` are the spurious edits that touch a ``keep`` value.
* sensitive  — ``sensitive`` values leaked; reported separately (not in J2).

J2 passes when leaked ≤ 2 % and spurious ≤ 3 %. Offline: no provider calls,
no geocoder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "bench" / "realworld"

PROTECT_TYPES = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "ORG", "DATE_OF_BIRTH",
                 "AGE", "ID", "NETWORK", "URL", "HANDLE", "CREDENTIAL"}
SENSITIVE_TYPES = {"HEALTH", "RELIGION", "ETHNICITY", "ORIENTATION", "POLITICAL"}
POLICY_TYPES = {"ADDRESS", "LOCATION"}
LISTS = ("protect", "sensitive", "optional", "keep")
ID_RE = re.compile(r"^(rw-(dev[234]?|test)|rd-[a-z0-9]+-(dev|test[23]?))-\d{4}$")   # rd-: the real-data benchmark
SPLITS = ("dev", "dev2", "dev3", "dev4", "test")  # dev–dev3 tune; dev4 checks a fix; test is J2

LEAK_GATE = 0.02
SPURIOUS_GATE = 0.03


# ── corpus ───────────────────────────────────────────────────────────────────

def _wordchar(c: str) -> bool:
    """A letter/digit of a script that separates words with spaces. Han, kana
    and Thai run words together, so they never glue a value to its context."""
    if not c.isalnum():
        return False
    cp = ord(c)
    return not (0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
                or 0xF900 <= cp <= 0xFAFF or 0x0E00 <= cp <= 0x0E7F or 0x20000 <= cp <= 0x2FA1F)


def occurrences(text: str, value: str):
    """Start/end of every occurrence of *value* not glued to a letter/digit."""
    out, i = [], text.find(value)
    while i != -1:
        j = i + len(value)
        before = text[i - 1] if i else " "
        after = text[j] if j < len(text) else " "
        if not (_wordchar(before) and _wordchar(value[0])) and not (_wordchar(after) and _wordchar(value[-1])):
            out.append((i, j))
        i = text.find(value, i + 1)
    return out


def _entries(rec, name):
    for item in rec.get(name, []):
        if name == "keep":
            yield (item if isinstance(item, str) else item.get("value")), None
        else:
            yield item.get("value"), item.get("type")


def id_pattern(prefix: str):
    """Ids ``<prefix><anything>-NNNN`` for a corpus outside the known families."""
    return re.compile(rf"^{re.escape(prefix)}[A-Za-z0-9-]*-\d{{4}}$")


def lint(records, id_re=ID_RE) -> list:
    """Return ``(id, problem)`` pairs. Never echoes message text."""
    errors, seen = [], set()
    for n, rec in enumerate(records):
        rid = rec.get("id", f"<line {n + 1}>")
        if not isinstance(rid, str) or not id_re.match(rid):
            errors.append((rid, "bad id"))
        if rid in seen:
            errors.append((rid, "duplicate id"))
        seen.add(rid)
        if not isinstance(rec.get("text"), str) or not rec["text"].strip():
            errors.append((rid, "empty text")); continue
        if not isinstance(rec.get("service_query"), bool):
            errors.append((rid, "service_query not bool"))
        if not rec.get("category") or not rec.get("lang"):
            errors.append((rid, "missing category/lang"))
        spans = {}
        for name in LISTS:
            if not isinstance(rec.get(name, []), list):
                errors.append((rid, f"{name} not a list")); continue
            for k, (value, typ) in enumerate(_entries(rec, name)):
                where = f"{name}[{k}]"
                if not isinstance(value, str) or not value.strip():
                    errors.append((rid, f"{where} empty value")); continue
                if name == "protect" and typ not in PROTECT_TYPES:
                    errors.append((rid, f"{where} bad type")); continue
                if name == "sensitive" and typ not in SENSITIVE_TYPES:
                    errors.append((rid, f"{where} bad type")); continue
                occ = occurrences(rec["text"], value)
                if not occ:
                    errors.append((rid, f"{where} value not a whole-word substring"))
                spans[(name, k)] = occ
        # a keep value must not overlap a protect value
        for (n1, k1), o1 in spans.items():
            for (n2, k2), o2 in spans.items():
                if n1 == "protect" and n2 == "keep" and any(
                        a < d and c < b for a, b in o1 for c, d in o2):
                    errors.append((rid, f"protect[{k1}] overlaps keep[{k2}]"))
    return errors


def load(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    sys.exit(f"{path}:{n}: invalid JSON ({exc.msg})")
    return records


def stats(records) -> dict:
    types, cats, langs = Counter(), Counter(), Counter()
    for r in records:
        cats[r["category"]] += 1
        langs[r["lang"]] += 1
        for name in ("protect", "sensitive"):
            for item in r.get(name, []):
                types[item["type"]] += 1
    neg = sum(1 for r in records if not r.get("protect"))
    return {"messages": len(records), "negatives": neg,
            "negative_share": round(neg / max(len(records), 1), 3),
            "service_query": sum(1 for r in records if r["service_query"]),
            "types": dict(sorted(types.items())), "langs": dict(langs.most_common()),
            "categories": len(cats)}


# ── scoring ──────────────────────────────────────────────────────────────────

def _covered(s, e, text, edits):
    """True if every letter/digit in text[s:e] lies inside some edit."""
    return all(any(a <= i < b for a, b, _, _ in edits)
               for i in range(s, e) if text[i].isalnum())


def score_message(rec, prepared) -> dict:
    text, edits = rec["text"], prepared.edits
    out = {"id": rec["id"], "leaked": [], "policy": [], "sensitive_leaked": [],
           "spurious": [], "keep_hits": [], "edits": len(edits)}

    def leaked(value):
        for s, e in occurrences(text, value):
            if not _covered(s, e, text, edits):
                return True
            for a, b, _, sur in edits:
                if a < e and s < b and len(value) >= 4 and value in sur:
                    return True
        return False

    for item in rec.get("protect", []):
        if leaked(item["value"]):
            policy = rec["service_query"] and item["type"] in POLICY_TYPES
            out["policy" if policy else "leaked"].append(item)
    for item in rec.get("sensitive", []):
        if leaked(item["value"]):
            out["sensitive_leaked"].append(item)

    gold = [sp for name in ("protect", "sensitive", "optional")
            for item in rec.get(name, []) for sp in occurrences(text, item["value"])]
    keep = [sp for v, _ in _entries(rec, "keep") for sp in occurrences(text, v)]
    for a, b, orig, sur in edits:
        if not any(a < e and s < b for s, e in gold):
            out["spurious"].append(orig)
            if any(a < e and s < b for s, e in keep):
                out["keep_hits"].append(orig)
    return out


def ss_prepare():
    """SurrogateShield's send path: ``prepare(text, i)`` → object with
    ``edits`` [(start, end, original, surrogate)] and ``sanitized``."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    from generation.logic import MimicGen
    from json_tester import prepare_send

    return lambda text, i: prepare_send(text, MimicGen(seed=i))     # reproducible run


def run(split: str | None, show: bool, prepare=None, path: Path | None = None, id_re=ID_RE) -> dict:
    """Score *split* (or the corpus at *path*) with *prepare* (default:
    SurrogateShield, ``ss_prepare``). A file is scored exactly as a split is;
    its summary is labelled with the file's stem."""
    prepare = prepare or ss_prepare()
    path = path or CORPUS / f"{split}.jsonl"
    split = split or path.stem
    records = load(path)
    errors = lint(records, id_re)
    if errors:
        for rid, problem in errors:
            print(f"  {rid}: {problem}")
        sys.exit(f"{path}: {len(errors)} lint errors — fix the corpus first")

    random.seed(0)
    results, ms = [], []
    for i, rec in enumerate(records):
        t = time.perf_counter()
        prepared = prepare(rec["text"], i)
        ms.append((time.perf_counter() - t) * 1000)
        res = score_message(rec, prepared)
        res.update(category=rec["category"], lang=rec["lang"])
        results.append(res)
        if show and (res["leaked"] or res["spurious"] or res["sensitive_leaked"]):
            print(f"\n[{rec['id']}] {rec['category']} ({rec['lang']})")
            print(f"  text: {rec['text']!r}")
            print(f"  sent: {prepared.sanitized!r}")
            for item in res["leaked"]:
                print(f"  LEAK {item['type']}: {item['value']!r}")
            for item in res["sensitive_leaked"]:
                print(f"  sensitive {item['type']}: {item['value']!r}")
            for orig in res["spurious"]:
                print(f"  SPURIOUS: {orig!r}" + ("  (keep)" if orig in res["keep_hits"] else ""))

    n_protect = sum(len(r.get("protect", [])) for r in records)
    n_policy = sum(len(r["policy"]) for r in results)
    n_leak = sum(len(r["leaked"]) for r in results)
    n_sens = sum(len(r.get("sensitive", [])) for r in records)
    n_edits = sum(r["edits"] for r in results)
    n_spur = sum(len(r["spurious"]) for r in results)
    negatives = [r for rec, r in zip(records, results) if not rec.get("protect")]

    by_type = defaultdict(lambda: [0, 0])
    for rec, r in zip(records, results):
        for item in rec.get("protect", []):
            by_type[item["type"]][0] += 1
        for item in r["leaked"]:
            by_type[item["type"]][1] += 1
    by_lang = defaultdict(lambda: [0, 0])
    for rec, r in zip(records, results):
        by_lang[rec["lang"]][0] += len(rec.get("protect", []))
        by_lang[rec["lang"]][1] += len(r["leaked"])

    leak_rate = n_leak / max(n_protect - n_policy, 1)
    spur_rate = n_spur / max(n_edits, 1)
    summary = {
        "split": split, "corpus": stats(records),
        "protect_values": n_protect, "leaked": n_leak, "policy": n_policy,
        "leak_rate": round(leak_rate, 4),
        "edits": n_edits, "spurious": n_spur, "spurious_rate": round(spur_rate, 4),
        "keep_hits": sum(len(r["keep_hits"]) for r in results),
        "messages_with_spurious": sum(1 for r in results if r["spurious"]),
        "negatives_untouched": sum(1 for r in negatives if not r["edits"]),
        "negatives": len(negatives),
        "sensitive_values": n_sens,
        "sensitive_leaked": sum(len(r["sensitive_leaked"]) for r in results),
        "leak_by_type": {t: {"values": v[0], "leaked": v[1]} for t, v in sorted(by_type.items())},
        "leak_by_lang": {k: {"values": v[0], "leaked": v[1]} for k, v in sorted(by_lang.items())},
        "latency_ms_p50": round(sorted(ms)[len(ms) // 2], 1) if ms else None,
        "J2": "PASS" if leak_rate <= LEAK_GATE and spur_rate <= SPURIOUS_GATE else "FAIL",
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--file", type=Path, help="score this corpus file instead of a split")
    ap.add_argument("--id-prefix", help="accept ids <prefix>…-NNNN (default: the rw-/rd- families)")
    ap.add_argument("--lint", type=Path, help="validate a corpus file and print its stats")
    ap.add_argument("--show", action="store_true", help="print every failing message (dev only)")
    ap.add_argument("--json", type=Path, help="also write the summary here")
    args = ap.parse_args()

    id_re = id_pattern(args.id_prefix) if args.id_prefix else ID_RE
    if args.lint:
        records = load(args.lint)
        errors = lint(records, id_re)
        print(json.dumps(stats(records) if records else {}, indent=2))
        for rid, problem in errors:
            print(f"  {rid}: {problem}")
        print(f"{len(errors)} lint errors")
        return 1 if errors else 0
    if not args.split and not args.file:
        ap.error("--split, --file or --lint is required")
    if args.split and args.file:
        ap.error("--split and --file are exclusive")
    if args.show and args.split in ("test", "dev4"):
        ap.error("--show is for dev; test and dev4 are not inspected while tuning")
    if args.show and args.file:
        ap.error("--show prints message text; a --file corpus may hold real text, so it is never shown")

    summary = run(args.split, args.show, path=args.file, id_re=id_re)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["J2"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
