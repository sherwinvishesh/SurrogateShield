# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
offline_eval.py — Detection and protection evaluation with NO LLM calls.

Runs the exact send path of the batch runner (``json_tester.prepare_send``:
cascade → seeded surrogates → span substitution) over every question in a key
file (``[{"Question": …, "Answer-Key": {…}}, …]``) and scores it with the
shared definitions in ``eval_metrics.py``. Local models only; nothing leaves
the machine and no API key is needed.

Two scores, never merged:
  protection   — spans that were actually replaced before sending.
  recognition  — protection spans plus spans recognised but deliberately left
                 in place (service-query locations, topical geo). Reported for
                 diagnosis only; it is not a privacy number.

Matching is span overlap (see eval_metrics). Micro and macro, exact-boundary
counts, and precision on negative questions are always reported.

``--protection`` (gate J1) also checks the sent text itself: every gold value
found verbatim in it is a leak, split into deliberate (left in place under one
of the documented ``POLICY_REASONS``) and unintended. Shift-mode addresses must
appear in the sent text exactly as their shifted surrogate. Exit status 1 if
any unintended leak or shift mismatch exists.

Usage:
    python offline_eval.py --key experiment/test_key.json
    python offline_eval.py --key experiment/test_key.json --protection --json out.json
    python offline_eval.py --key experiment/test_key.json --types address --limit 200
    python offline_eval.py --key experiment/test_key.json --lint-key
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import eval_metrics as em

SEED = 20260101

# Documented policies under which a gold value may reach the provider verbatim
# (gate J1). Anything else in the sent text is an unintended leak.
#   service_query_address_shift      — non-sensitive service query, address
#                                      mode "shift": only the house number
#                                      changes; street/city/state/ZIP stay.
#   service_query_location_suppressed — standalone city/state in a service
#                                      query ("coffee near Tempe").
#   topical_geo_filtered             — a place that is only the topic of a
#                                      question ("Japan's GDP") in a message
#                                      with no direct identifier of a person.
POLICY_REASONS = frozenset({
    "service_query_address_shift",
    "service_query_location_suppressed",
    "topical_geo_filtered",
})


# ─────────────────────────────────────────────
# Key linting (no models needed)
# ─────────────────────────────────────────────

def lint_key(entries: list) -> int:
    """Flag suspicious ground-truth values. Returns the number of findings."""
    from detection import address_parser

    findings = 0
    for i, entry in enumerate(entries):
        question = entry.get("Question", "")
        for t, v in em.key_values(entry.get("Answer-Key")):
            if not em.find_occurrences(question, v):
                findings += 1
                print(f"[lint] entry {i}: {t}={v!r} not found as a whole token in question")
            if t == "address" and address_parser.parse(v) is None:
                findings += 1
                print(f"[lint] entry {i}: {t}={v!r} does not parse as an address")
    print(f"\n{findings} finding(s) across {len(entries)} entries")
    return findings


# ─────────────────────────────────────────────
# Per-question scoring
# ─────────────────────────────────────────────

def _pred_spans(spans: list, replaced_only: bool) -> list:
    return [
        em.Span(s["start"], s["end"], em.normalize_type(s["type"]), s["text"])
        for s in spans
        if s["replaced"] or not replaced_only
    ]


def check_sent_text(question: str, answer_key, prep) -> dict:
    """J1 check on one prepared send. Returns leaks and shift mismatches."""
    rnr = [s for s in prep.spans() if not s["replaced"]]
    # Service-query shift policy: only the house number changes, so street,
    # city, state and ZIP inside a shifted address go out verbatim by design.
    address_texts = {e.text for e in prep.confirmed if e.type == "address"}
    shifted = [
        (b, e) for b, e, orig, _sur in prep.edits
        if prep.address_mode == "shift" and orig in address_texts
    ]
    leaks = []
    for t, v in em.key_values(answer_key):
        if not em.find_occurrences(question, v):
            continue  # cannot leak what is not in the input
        if not em.contains_value(prep.sanitized, v):
            continue
        occ = em.find_occurrences(question, v)
        reasons = {
            s.get("reason") or "no_reason_recorded" for s in rnr
            if any(s["start"] < e and b < s["end"] for b, e in occ)
        }
        if any(sb <= b and e <= se for b, e in occ for sb, se in shifted):
            reasons.add("service_query_address_shift")
        reasons = sorted(reasons)
        leaks.append({
            "type": t,
            "value": v,
            "deliberate": bool(reasons) and set(reasons) <= POLICY_REASONS,
            "reasons": reasons,
        })
    shift_mismatch = []
    if prep.address_mode == "shift":
        for ent in prep.confirmed:
            if ent.type == "address" and ent.text in prep.surrogate_map:
                if prep.surrogate_map[ent.text] not in prep.sanitized:
                    shift_mismatch.append(ent.text)
    return {"leaks": leaks, "shift_mismatch": shift_mismatch}


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def run_offline_eval(
    key_path: Path,
    limit: int | None = None,
    type_filter: set | None = None,
    protection: bool = False,
    seed: int = SEED,
) -> dict:
    from generation.logic import MimicGen
    from json_tester import prepare_send

    entries = json.loads(key_path.read_text())
    if limit:
        entries = entries[:limit]

    prot = em.Tally()
    recog = em.Tally()
    misses: list = []
    false_positives: list = []
    gold_not_in_text: list = []
    leak_rows: list = []
    shift_rows: list = []
    total_gold_values = 0

    start = time.time()
    for i, entry in enumerate(entries):
        question = entry.get("Question", "")
        key = entry.get("Answer-Key")
        gold, missing = em.gold_spans(question, key)
        for t, v in missing:
            gold_not_in_text.append({"entry": i, "type": t, "value": v})

        prep = prepare_send(question, MimicGen(seed=seed + i))
        spans = prep.spans()

        prot_pred = _pred_spans(spans, replaced_only=True)
        g_hit, p_hit = prot.add(gold, prot_pred)
        recog.add(gold, _pred_spans(spans, replaced_only=False))

        for g, hit in zip(gold, g_hit):
            if not hit:
                misses.append({"entry": i, "type": g.type, "value": g.value,
                               "question": question[:160]})
        for p, hit in zip(prot_pred, p_hit):
            if not hit:
                false_positives.append({"entry": i, "type": p.type, "value": p.value,
                                        "question": question[:160]})

        if protection:
            total_gold_values += sum(
                1 for _t, v in em.key_values(key) if em.find_occurrences(question, v)
            )
            chk = check_sent_text(question, key, prep)
            for leak in chk["leaks"]:
                leak_rows.append({"entry": i, **leak, "sent": prep.sanitized[:200]})
            for addr in chk["shift_mismatch"]:
                shift_rows.append({"entry": i, "address": addr})

        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(entries)} questions ({time.time() - start:.0f}s)",
                  flush=True)

    def _filter(per_type: dict) -> dict:
        return {t: s for t, s in per_type.items() if not type_filter or t in type_filter}

    p_sum, r_sum = prot.summary(), recog.summary()
    p_sum["per_type"] = _filter(p_sum["per_type"])
    r_sum["per_type"] = _filter(r_sum["per_type"])

    result = {
        "key_file": str(key_path),
        "questions": len(entries),
        "seed": seed,
        "matching": "span overlap (eval_metrics.match); exact-boundary counts alongside",
        "protection": p_sum,
        "recognition": r_sum,
        "gold_not_in_text": gold_not_in_text,
        "misses": misses,
        "false_positives": false_positives,
    }
    if protection:
        unintended = [r for r in leak_rows if not r["deliberate"]]
        deliberate = [r for r in leak_rows if r["deliberate"]]
        result["sent_text_check"] = {
            "gold_values_in_input": total_gold_values,
            "leaked_values": len(leak_rows),
            "unintended_leaks": len(unintended),
            "deliberate_leaks": len(deliberate),
            "deliberate_by_reason": _count_reasons(deliberate),
            "unintended_leak_rate": round(len(unintended) / total_gold_values, 6)
            if total_gold_values else 0.0,
            "shift_mismatches": len(shift_rows),
            "passed": not unintended and not shift_rows,
            "leaks": leak_rows,
            "shift_mismatch_rows": shift_rows,
        }
    return result


def _count_reasons(rows: list) -> dict:
    out: dict = {}
    for r in rows:
        for reason in r["reasons"]:
            out[reason] = out.get(reason, 0) + 1
    return out


def _print_block(name: str, summ: dict) -> None:
    mi, ma, neg = summ["micro"], summ["macro"], summ["negatives"]
    print(f"\n{name.upper():<12} micro P={mi['precision']:.4f} R={mi['recall']:.4f} "
          f"F1={mi['f1']:.4f}  (gold={mi['gold']} tp={mi['tp']} fn={mi['fn']} "
          f"pred={mi['pred']} fp={mi['fp']} exact={mi['exact_boundary_matches']})")
    print(f"{'':<12} macro P={ma['precision']:.4f} R={ma['recall']:.4f} "
          f"F1={ma['f1']:.4f}  over {ma['n_types']} types")
    print(f"{'':<12} negatives: {neg['negative_questions_with_any_prediction']}/"
          f"{neg['negative_questions']} negative questions had a prediction "
          f"({neg['fp_spans_on_negatives']} spans)")


def _print_report(result: dict) -> None:
    print("\n═══ Offline evaluation (span-overlap matching) ═══")
    print(f"key file:  {result['key_file']}")
    print(f"questions: {result['questions']}   seed: {result['seed']}")
    print(f"gold values not in their question (excluded): {len(result['gold_not_in_text'])}")
    _print_block("protection", result["protection"])
    _print_block("recognition", result["recognition"])
    print("\nProtection per type:")
    print(f"  {'type':<20}{'P':>8}{'R':>8}{'F1':>8}{'gold':>6}{'tp':>6}{'fn':>6}{'pred':>6}{'fp':>6}")
    for t, s in result["protection"]["per_type"].items():
        print(f"  {t:<20}{s['precision']:>8.4f}{s['recall']:>8.4f}{s['f1']:>8.4f}"
              f"{s['gold']:>6}{s['tp']:>6}{s['fn']:>6}{s['pred']:>6}{s['fp']:>6}")
    for label, rows in (("misses", result["misses"]),
                        ("false positives", result["false_positives"])):
        if rows:
            print(f"\nFirst {label} ({min(10, len(rows))} of {len(rows)}):")
            for m in rows[:10]:
                print(f"  [{m['entry']}] {m['type']}: {m['value']!r}")
    chk = result.get("sent_text_check")
    if chk:
        print("\nSent-text check (J1):")
        print(f"  gold values in input:  {chk['gold_values_in_input']}")
        print(f"  unintended leaks:      {chk['unintended_leaks']}  "
              f"(rate {chk['unintended_leak_rate']:.6f})")
        print(f"  deliberate leaks:      {chk['deliberate_leaks']}  {chk['deliberate_by_reason']}")
        print(f"  shift mismatches:      {chk['shift_mismatches']}")
        print(f"  J1: {'PASS' if chk['passed'] else 'FAIL'}")
        for r in [r for r in chk["leaks"] if not r["deliberate"]][:10]:
            print(f"  [{r['entry']}] {r['type']}: {r['value']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline detection/protection evaluation (no LLM calls)")
    parser.add_argument("--key", type=Path, default=Path("experiment/test_key.json"),
                        help="key file: [{Question, Answer-Key}, …]")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    parser.add_argument("--types", type=str, default=None,
                        help="comma-separated type filter for the per-type report")
    parser.add_argument("--json", type=Path, default=None,
                        help="write full results (incl. misses) to this JSON file")
    parser.add_argument("--protection", action="store_true",
                        help="also check the sent text for verbatim gold values (gate J1)")
    parser.add_argument("--seed", type=int, default=SEED, help="base surrogate seed")
    parser.add_argument("--lint-key", action="store_true",
                        help="only lint the key file (no models)")
    args = parser.parse_args()

    if not args.key.exists():
        print(f"key file not found: {args.key}", file=sys.stderr)
        return 2

    if args.lint_key:
        entries = json.loads(args.key.read_text())
        return 1 if lint_key(entries) else 0

    import logging
    logging.getLogger().setLevel(logging.WARNING)

    type_filter = set(args.types.split(",")) if args.types else None
    result = run_offline_eval(args.key, args.limit, type_filter, args.protection, args.seed)
    _print_report(result)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nfull results → {args.json}")

    if args.protection and not result["sent_text_check"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
