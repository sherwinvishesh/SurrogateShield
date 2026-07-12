# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
offline_eval.py — Detection-only evaluation with NO LLM calls.

Runs the full SentinelLayer cascade over every question in a key file
(experiment/test_key.json format: [{"Question": …, "Answer-Key": {…}}, …])
and scores detected values against the ground truth. Local models only
(spaCy + HuggingFace) — nothing leaves the machine and no API key is needed.

Matching modes:
  default (span-aware) — a key value counts as found when it equals a
      detected value OR is contained in one (case-insensitive). v2 detects
      addresses as FULL spans ("6720 Palm Dr, Phoenix, AZ") while the ground
      truth annotates components separately ("6720 Palm Dr" + gpe
      ["Phoenix", "AZ"]); containment scores that correctly instead of
      punishing the wider, more-protective span.
  --strict — exact (lowercased) string equality only; comparable to the v1
      evaluator.py numbers.

Usage:
    python offline_eval.py --key experiment/test_key.json
    python offline_eval.py --key experiment/test_key.json --json out.json
    python offline_eval.py --key experiment/test_key.json --types address --limit 200
    python offline_eval.py --key experiment/experiment_key.json --lint-key
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from evaluator import KEY_TYPE_MAP, NORMALIZE_TYPE, parse_key_entry


# ─────────────────────────────────────────────
# Key linting (no models needed)
# ─────────────────────────────────────────────

def lint_key(entries: list) -> int:
    """Flag suspicious ground-truth values. Returns the number of findings."""
    from detection import address_parser

    findings = 0
    for i, entry in enumerate(entries):
        question = entry.get("Question", "")
        raw_key = entry.get("Answer-Key") or {}
        if not isinstance(raw_key, dict):
            continue
        for label, val in raw_key.items():
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                v = str(v).strip()
                if not v:
                    continue
                if v.lower() not in question.lower():
                    findings += 1
                    print(f"[lint] entry {i}: {label}={v!r} not found verbatim in question")
                if KEY_TYPE_MAP.get(label) == "address" and address_parser.parse(v) is None:
                    findings += 1
                    print(f"[lint] entry {i}: {label}={v!r} does not parse as an address")
    print(f"\n{findings} finding(s) across {len(entries)} entries")
    return findings


# ─────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────

def _matches(key_value: str, detected_values: set, strict: bool) -> bool:
    if key_value in detected_values:
        return True
    if strict:
        return False
    # wider detected span covers the annotated value, or the detected value
    # covers most of it (e.g. "Alexei Volkov" for annotated "Dr. Alexei Volkov")
    return any(
        key_value in d or (d in key_value and len(d) >= 0.6 * len(key_value))
        for d in detected_values
    )


def _detected_matches_key(detected: str, key_values: set, strict: bool) -> bool:
    if detected in key_values:
        return True
    if strict:
        return False
    # detected span contains an annotated value (wider span — correct), or is
    # a component of one (narrower span — still a legitimate protection hit)
    return any(k in detected or detected in k for k in key_values)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def run_offline_eval(
    key_path: Path,
    limit: int | None = None,
    strict: bool = False,
    type_filter: set | None = None,
) -> dict:
    from detection.logic import run_cascade, deduplicate
    from detection.service_query import is_service_query
    from config import SERVICE_QUERY_DETECTION_ENABLED

    entries = json.loads(key_path.read_text())
    if limit:
        entries = entries[:limit]

    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    misses: list = []
    false_positives: list = []

    start = time.time()
    for i, entry in enumerate(entries):
        question = entry.get("Question", "")
        key_flat, key_typed = parse_key_entry(entry.get("Answer-Key"))

        is_svc = SERVICE_QUERY_DETECTION_ENABLED and is_service_query(question)
        confirmed, _ = run_cascade(question, skip_location_entities=is_svc)
        confirmed = deduplicate(confirmed)
        skipped = getattr(confirmed, "_skipped_entities", [])

        # detection = confirmed (will be surrogated) + recognized-not-replaced
        detected = {e.text.strip().lower(): NORMALIZE_TYPE.get(e.type, e.type)
                    for e in list(confirmed) + list(skipped)}
        detected_values = set(detected)
        key_values = {v.strip().lower() for v in key_flat}

        for internal_type, vals in key_typed.items():
            norm_type = NORMALIZE_TYPE.get(internal_type, internal_type)
            for v in vals:
                v_low = v.strip().lower()
                if _matches(v_low, detected_values, strict):
                    overall["tp"] += 1
                    per_type[norm_type]["tp"] += 1
                else:
                    overall["fn"] += 1
                    per_type[norm_type]["fn"] += 1
                    misses.append({"entry": i, "type": norm_type, "value": v,
                                   "question": question[:120]})

        for d_low, d_type in detected.items():
            if not _detected_matches_key(d_low, key_values, strict):
                overall["fp"] += 1
                per_type[d_type]["fp"] += 1
                false_positives.append({"entry": i, "type": d_type, "value": d_low,
                                        "question": question[:120]})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  … {i + 1}/{len(entries)} questions ({elapsed:.0f}s)", flush=True)

    def _prf(stats):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    result = {
        "key_file": str(key_path),
        "questions": len(entries),
        "matching": "strict" if strict else "span-aware",
        "overall": _prf(overall),
        "per_type": {
            t: _prf(s)
            for t, s in sorted(per_type.items())
            if not type_filter or t in type_filter
        },
        "misses": misses,
        "false_positives": false_positives,
    }
    return result


def _print_report(result: dict) -> None:
    print(f"\n═══ Offline detection evaluation ({result['matching']} matching) ═══")
    print(f"key file:  {result['key_file']}")
    print(f"questions: {result['questions']}")
    o = result["overall"]
    print(f"\nOVERALL   P={o['precision']:.4f}  R={o['recall']:.4f}  "
          f"F1={o['f1']:.4f}  (tp={o['tp']} fp={o['fp']} fn={o['fn']})")
    print("\nPer type:")
    print(f"  {'type':<20}{'P':>8}{'R':>8}{'F1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for t, s in result["per_type"].items():
        print(f"  {t:<20}{s['precision']:>8.4f}{s['recall']:>8.4f}"
              f"{s['f1']:>8.4f}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}")
    if result["misses"]:
        print(f"\nFirst misses ({min(10, len(result['misses']))} of {len(result['misses'])}):")
        for m in result["misses"][:10]:
            print(f"  [{m['entry']}] {m['type']}: {m['value']!r}")
    if result["false_positives"]:
        print(f"\nFirst false positives ({min(10, len(result['false_positives']))} "
              f"of {len(result['false_positives'])}):")
        for m in result["false_positives"][:10]:
            print(f"  [{m['entry']}] {m['type']}: {m['value']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--key", type=Path, default=Path("experiment/test_key.json"),
                        help="key file: [{Question, Answer-Key}, …]")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    parser.add_argument("--types", type=str, default=None,
                        help="comma-separated type filter for the per-type report")
    parser.add_argument("--json", type=Path, default=None,
                        help="write full results (incl. misses) to this JSON file")
    parser.add_argument("--strict", action="store_true",
                        help="exact string matching (v1-comparable numbers)")
    parser.add_argument("--lint-key", action="store_true",
                        help="only lint the key file for corrupt values (no models)")
    args = parser.parse_args()

    if not args.key.exists():
        print(f"key file not found: {args.key}", file=sys.stderr)
        return 2

    if args.lint_key:
        entries = json.loads(args.key.read_text())
        return 1 if lint_key(entries) else 0

    type_filter = set(args.types.split(",")) if args.types else None
    result = run_offline_eval(args.key, args.limit, args.strict, type_filter)
    _print_report(result)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nfull results → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
