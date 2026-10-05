# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
evaluator.py — score a batch-runner answers file (``json_tester.py`` output).

No UI, no LLM calls, no models. Every definition comes from ``eval_metrics.py``
(shared with ``offline_eval.py`` and ``bench/``):

* **detection** — protection (replaced spans) and recognition (replaced +
  recognised-not-replaced) reported separately; span overlap; micro and macro;
  exact-boundary counts; precision on negative questions. Empty sets score 0.
* **sanitization** — every gold value found verbatim in the text that was sent,
  split into deliberate (a documented ``POLICY_REASONS`` entry) and
  unintended (A6).
* **restoration** — the stored ``final_output`` from the shipped
  ``ResolvePass`` is scored: surrogates left, over-restored originals,
  collateral edits (A7).
* **presidio_comparison** — one shared type universe applied to both systems,
  the same rows for both, untyped headline plus typed table, a question-level
  bootstrap CI, and a separate table for types outside the universe (A4, A12).
* **bertscore** — rescaled scores only flagged as such; input fidelity and
  output utility kept apart; identical-text rows excluded from the pairs (A8).
* **ablation** — not computable from stored answers; points to
  ``offline_eval.py --ablation``, which re-runs the cascade (A9).

Rows the runner stored with an ``error`` are counted and excluded from every
metric (I32). Files that do not line up (different lengths, a stored question
that differs from the questions file) raise ``ValueError``; nothing is
swallowed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import eval_metrics as em

# Re-exported for callers that used the old module-level tables.
KEY_TYPE_MAP = em.KEY_TYPE_MAP
NORMALIZE_TYPE = em.NORMALIZE_TYPE

EXPERIMENT_DIR = Path(__file__).parent / "experiment"

EVAL_FIELDS = [
    ("answers",             "Answers  (answered / empty / error rows)"),
    ("surrogate_counts",    "Surrogate counts  (replaced values vs gold values)"),
    ("detection",           "Detection  (protection and recognition · micro + macro · per type · negatives)"),
    ("sanitization",        "Sanitization  (gold values in the text sent to the LLM · deliberate vs unintended)"),
    ("restoration",         "Restoration  (final_output: surrogates left · over-restored · collateral edits)"),
    ("presidio_comparison", "Presidio comparison  (shared type universe · same rows · bootstrap CI)"),
    ("bertscore",           "BERTScore  (rescaled · input fidelity and output utility · paired statistics)"),
    ("ablation",            "Ablation  (needs a cascade re-run — offline_eval.py --ablation)"),
    ("timing",              "Stage timings  (mean / p50 / p95 ms)"),
]

ABLATION_COMMAND = "python offline_eval.py --key <key file> --ablation"

_BERTSCORE_KINDS = {
    # kind: (ss field, presidio field, note)
    "input_fidelity": (
        "bertscore_ss", "bertscore_presidio",
        "sanitised prompt vs original prompt — rewards minimal edits; not a utility measure (A8)",
    ),
    "output_utility": (
        "bertscore_ss_output", "bertscore_presidio_output",
        "each arm's answer vs the answer to the unsanitised prompt",
    ),
}


def parse_key_entry(answer_key) -> tuple[list[str], dict[str, list[str]]]:
    """Flat list of key values and ``{canonical_type: [values]}``.

    Dict keys use ``eval_metrics.key_values`` (non-string values are
    stringified, I32). The legacy comma-separated string format returns an
    untyped flat list.
    """
    if isinstance(answer_key, str):
        flat = [t.strip().strip('"').strip("'").strip() for t in answer_key.split(",")]
        return [t for t in flat if t], {}
    typed: dict[str, list[str]] = {}
    flat: list[str] = []
    for t, v in em.key_values(answer_key):
        typed.setdefault(t, []).append(v)
        flat.append(v)
    return flat, typed


# ─────────────────────────────────────────────
# Per-row helpers
# ─────────────────────────────────────────────

def row_spans(row: dict, question: str) -> tuple[list[dict], str]:
    """Predicted spans for a row, in question coordinates.

    Uses the runner's ``pii_spans`` when stored. Older answers files have
    none; their spans are located from ``surrogate_map`` and
    ``recognized_not_replaced`` by whole-token search and the source is
    reported as ``located``.
    """
    if row.get("pii_spans") is not None:
        return row["pii_spans"], "pii_spans"
    detail = row.get("pii_detail") or {}
    spans: list[dict] = []
    for orig, sur in (row.get("surrogate_map") or {}).items():
        d = detail.get(orig)
        typ = d.get("type", "unknown") if isinstance(d, dict) else "unknown"
        for s, e in em.find_occurrences(question, orig):
            spans.append({"text": question[s:e], "start": s, "end": e,
                          "type": typ, "replaced": sur != orig})
    for r in row.get("recognized_not_replaced") or []:
        if not isinstance(r, dict) or not r.get("value"):
            continue
        if r.get("start") is not None and r.get("end") is not None:
            occ = [(r["start"], r["end"])]
        else:
            occ = em.find_occurrences(question, r["value"])
        for s, e in occ:
            spans.append({"text": question[s:e], "start": s, "end": e,
                          "type": r.get("type", "unknown"), "replaced": False,
                          "reason": r.get("reason")})
    spans.sort(key=lambda r: (r["start"], r["end"]))
    return spans, "located"


def pred_spans(spans: list[dict], replaced_only: bool) -> list[em.Span]:
    return [
        em.Span(s["start"], s["end"], em.normalize_type(s["type"]), s["text"])
        for s in spans
        if s["replaced"] or not replaced_only
    ]


def presidio_spans(found: list, question: str) -> tuple[list[em.Span], Counter]:
    """Presidio entities mapped into the shared universe, plus counts of the
    ones outside it (by Presidio type)."""
    out: list[em.Span] = []
    outside: Counter = Counter()
    for e in found:
        if not isinstance(e, dict) or not e.get("value"):
            continue
        shared = em.presidio_shared_type(e.get("type", ""), e["value"])
        if shared is None:
            t = e.get("type", "")
            outside["DATE_TIME (not date-shaped)" if t == "DATE_TIME" else t] += 1
            continue
        if e.get("start") is not None and e.get("end") is not None:
            occ = [(e["start"], e["end"])]
        else:
            occ = em.find_occurrences(question, e["value"])
        out.extend(em.Span(s, en, shared, question[s:en]) for s, en in occ)
    return out, outside


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _load(base: Path, name: str):
    return json.loads((base / name).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def run_evaluation(
    questions_filename: str,
    answers_filename: str,
    key_filename: str,
    fields: dict[str, bool],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    experiment_dir: Optional[Path] = None,
) -> dict:
    """Score an answers file against its questions and key.

    Args:
        questions_filename: ``[{"input": …}, …]`` inside *experiment_dir*.
        answers_filename:   runner output, same length and order.
        key_filename:       ``[{"Question": …, "Answer-Key": {…}}, …]``.
        fields:             ``EVAL_FIELDS`` key → bool.
        progress_cb:        called as ``(index, total, "ok" | "error")``.
        experiment_dir:     defaults to ``experiment/``.

    Returns a dict with the enabled sections plus ``questions``,
    ``rows_scored``, ``error_rows`` and the run metadata when the runner's
    ``.meta.json`` sits next to the answers file.

    Raises:
        ValueError: the files do not line up, or a row is malformed (the
            message names the row).
    """
    base = Path(experiment_dir) if experiment_dir is not None else EXPERIMENT_DIR
    questions = _load(base, questions_filename)
    answers = _load(base, answers_filename)
    keys = _load(base, key_filename)

    if not (len(questions) == len(answers) == len(keys)):
        raise ValueError(
            f"File length mismatch: questions={len(questions)}, "
            f"answers={len(answers)}, keys={len(keys)}. "
            "All three files must have the same number of entries."
        )

    want = {k: bool(fields.get(k)) for k, _ in EVAL_FIELDS}
    total = len(questions)

    error_rows: list[dict] = []
    answered = empty = 0
    span_sources: Counter = Counter()

    replaced_values = gold_values_in_text = gold_not_in_text = 0

    prot, recog = em.Tally(), em.Tally()

    san = {"rows": 0, "rows_without_sent_text": 0, "gold_values": 0,
           "leaks": [], "shift_mismatch": []}

    res = {"rows": 0, "rows_with_replacements": 0, "recomputed": 0,
           "surrogates_left": [], "over_restored": [], "collateral": [],
           "rows_with_surrogate_left": 0, "rows_with_collateral": 0}

    cmp_rows = cmp_missing = 0
    cmp = {k: em.Tally() for k in ("ss_untyped", "prs_untyped", "ss_typed", "prs_typed")}
    cmp_counts = {"ss": [], "presidio": []}
    outside_universe: dict = defaultdict(lambda: {"gold": 0, "ss_found": 0, "presidio_found": 0})
    presidio_only: Counter = Counter()
    ss_outside: Counter = Counter()
    sent_cmp = {"gold_values": 0, "ss": 0, "ss_unintended": 0, "presidio": 0,
                "rows_with_presidio_text": 0}

    bs = {kind: {"ss": [], "presidio": [], "pairs": [], "excluded_identical": 0}
          for kind in _BERTSCORE_KINDS}
    bs_rescaled: set = set()
    bs_errors = 0

    timing_vals: dict = defaultdict(list)

    for i in range(total):
        q_entry, row, k_entry = questions[i], answers[i], keys[i]
        question = q_entry.get("input", q_entry.get("Question", "")) if isinstance(q_entry, dict) else ""
        if isinstance(k_entry, dict) and k_entry.get("Question") not in (None, question):
            raise ValueError(f"row {i}: key file question differs from the questions file")
        if not isinstance(row, dict):
            raise ValueError(f"row {i}: answers entry is not an object")
        if row.get("question") is not None and row["question"] != question:
            raise ValueError(f"row {i}: answers file question differs from the questions file")

        if "error" in row:
            error_rows.append({"row": i, "error_type": row.get("error_type", "unknown"),
                               "error": str(row["error"])[:200]})
            if progress_cb:
                progress_cb(i, total, "error")
            continue

        try:
            key = k_entry.get("Answer-Key") if isinstance(k_entry, dict) else None
            smap = row.get("surrogate_map") or {}
            llm_response = row.get("llm_response") or ""
            if llm_response:
                answered += 1
            else:
                empty += 1

            gold, missing = em.entry_gold(question, k_entry)
            gold_not_in_text += len(missing)
            spans, src = row_spans(row, question)
            span_sources[src] += 1

            if want["surrogate_counts"]:
                replaced_values += sum(1 for k, v in smap.items() if v and v != k)
                gold_values_in_text += sum(
                    1 for _t, v in em.key_values(key) if em.find_occurrences(question, v))

            prot_pred = pred_spans(spans, replaced_only=True)
            if want["detection"]:
                prot.add(gold, prot_pred)
                recog.add(gold, pred_spans(spans, replaced_only=False))

            if want["sanitization"]:
                sent = row.get("sanitized_input")
                if sent is None:
                    san["rows_without_sent_text"] += 1
                else:
                    san["rows"] += 1
                    chk = em.classify_sent_leaks(question, key, sent, spans,
                                                 row.get("address_mode"), smap)
                    san["gold_values"] += chk["gold_values"]
                    san["leaks"].extend({"row": i, **lk} for lk in chk["leaks"])
                    san["shift_mismatch"].extend({"row": i, "address": a}
                                                 for a in chk["shift_mismatch"])

            if want["restoration"] and llm_response:
                final = row.get("final_output")
                if final is None:
                    from reconstruction.logic import ResolvePass
                    final = ResolvePass().resolve(
                        llm_response, {v: k for k, v in smap.items()}) if smap else llm_response
                    res["recomputed"] += 1
                r = em.score_restoration(llm_response, final, smap, question)
                res["rows"] += 1
                res["rows_with_replacements"] += any(v and v != k for k, v in smap.items())
                if r["surrogates_left"]:
                    res["rows_with_surrogate_left"] += 1
                if r["collateral_edits"]:
                    res["rows_with_collateral"] += 1
                res["surrogates_left"].extend({"row": i, "surrogate": s} for s in r["surrogates_left"])
                res["over_restored"].extend({"row": i, **o} for o in r["over_restored"])
                res["collateral"].extend({"row": i, **c} for c in r["collateral_edits"])

            if want["presidio_comparison"]:
                found = row.get("presidio_found_piis")
                if found is None:
                    cmp_missing += 1
                else:
                    cmp_rows += 1
                    shared_gold, neutral = em.split_shared(gold)
                    ss = []
                    for p in prot_pred:
                        t = em.ss_shared_type(p.type)
                        if t is None:
                            ss_outside[p.type] += 1
                        else:
                            ss.append(em.Span(p.start, p.end, t, p.value))
                    prs, outside = presidio_spans(found, question)
                    presidio_only.update(outside)
                    ss = em.drop_neutral(ss, neutral, shared_gold)
                    prs = em.drop_neutral(prs, neutral, shared_gold)

                    g_hit, p_hit = cmp["ss_untyped"].add(shared_gold, ss)
                    cmp_counts["ss"].append(em.question_counts(shared_gold, ss, g_hit, p_hit))
                    g_hit, p_hit = cmp["prs_untyped"].add(shared_gold, prs)
                    cmp_counts["presidio"].append(em.question_counts(shared_gold, prs, g_hit, p_hit))
                    cmp["ss_typed"].add(shared_gold, ss, em.same_type)
                    cmp["prs_typed"].add(shared_gold, prs, em.same_type)

                    all_prs = [
                        em.Span(s, en, e.get("type", ""), e["value"])
                        for e in found if isinstance(e, dict) and e.get("value")
                        for s, en in ([(e["start"], e["end"])]
                                      if e.get("start") is not None and e.get("end") is not None
                                      else em.find_occurrences(question, e["value"]))
                    ]
                    for g in neutral:
                        o = outside_universe[g.type]
                        o["gold"] += 1
                        o["ss_found"] += any(g.overlaps(p) for p in prot_pred)
                        o["presidio_found"] += any(g.overlaps(p) for p in all_prs)

                    sent = row.get("sanitized_input")
                    p_sent = row.get("presidio_sanitized_input")
                    if sent is not None and p_sent is not None:
                        chk = em.classify_sent_leaks(question, key, sent, spans,
                                                     row.get("address_mode"), smap)
                        sent_cmp["rows_with_presidio_text"] += 1
                        sent_cmp["gold_values"] += chk["gold_values"]
                        sent_cmp["ss"] += len(chk["leaks"])
                        sent_cmp["ss_unintended"] += sum(not lk["deliberate"] for lk in chk["leaks"])
                        sent_cmp["presidio"] += sum(
                            1 for _t, v in em.key_values(key)
                            if em.find_occurrences(question, v) and em.contains_value(p_sent, v))

            if want["bertscore"]:
                if row.get("bertscore_error"):
                    bs_errors += 1
                identical = (row.get("sanitized_input") == question
                             or row.get("presidio_sanitized_input") == question)
                for kind, (ss_f, prs_f, _note) in _BERTSCORE_KINDS.items():
                    a, b = row.get(ss_f), row.get(prs_f)
                    for d in (a, b):
                        if isinstance(d, dict):
                            bs_rescaled.add(bool(d.get("rescaled")))
                    if isinstance(a, dict):
                        bs[kind]["ss"].append(a)
                    if isinstance(b, dict):
                        bs[kind]["presidio"].append(b)
                    if isinstance(a, dict) and isinstance(b, dict):
                        if identical:
                            bs[kind]["excluded_identical"] += 1
                        else:
                            bs[kind]["pairs"].append((a["f1"], b["f1"]))

            if want["timing"]:
                for k, v in (row.get("stage_timings_ms") or {}).items():
                    if isinstance(v, (int, float)):
                        timing_vals[k].append(float(v))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"row {i}: malformed answers entry ({type(exc).__name__}: {exc})") from exc

        if progress_cb:
            progress_cb(i, total, "ok")

    scored = total - len(error_rows)
    result: dict = {
        "questions": total,
        "rows_scored": scored,
        "error_rows": len(error_rows),
        "error_types": dict(Counter(e["error_type"] for e in error_rows)),
        "error_examples": error_rows[:20],
        "span_source": dict(span_sources),
        "files": {"questions": questions_filename, "answers": answers_filename,
                  "key": key_filename},
    }
    meta_path = (base / answers_filename).with_suffix(".meta.json")
    if meta_path.exists():
        result["run_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    if want["answers"]:
        result["answers"] = {
            "answered": answered,
            "empty": empty,
            "errors": len(error_rows),
            "answer_rate": round(answered / total, 4) if total else 0.0,
            "answer_rate_excluding_errors": round(answered / scored, 4) if scored else 0.0,
        }

    if want["surrogate_counts"]:
        result["surrogate_counts"] = {
            "replaced_values": replaced_values,
            "gold_values_in_text": gold_values_in_text,
            "gold_values_not_in_text": gold_not_in_text,
            "avg_replaced_per_question": round(replaced_values / scored, 4) if scored else 0.0,
            "avg_gold_per_question": round(gold_values_in_text / scored, 4) if scored else 0.0,
        }

    if want["detection"]:
        result["detection"] = {
            "matching": "span overlap (eval_metrics.match); exact-boundary counts alongside",
            "headline": "protection",
            "protection": prot.summary(),
            "recognition": recog.summary(),
            "gold_values_not_in_text": gold_not_in_text,
        }

    if want["sanitization"]:
        leaks = san["leaks"]
        unintended = [lk for lk in leaks if not lk["deliberate"]]
        deliberate = [lk for lk in leaks if lk["deliberate"]]
        by_reason: Counter = Counter(r for lk in deliberate for r in lk["reasons"])
        n = san["gold_values"]
        result["sanitization"] = {
            "rows": san["rows"],
            "rows_without_sent_text": san["rows_without_sent_text"],
            "gold_values_in_input": n,
            "leaked_values": len(leaks),
            "deliberate_leaks": len(deliberate),
            "unintended_leaks": len(unintended),
            "deliberate_by_reason": dict(by_reason),
            "leak_rate": round(len(leaks) / n, 6) if n else 0.0,
            "unintended_leak_rate": round(len(unintended) / n, 6) if n else 0.0,
            "rows_with_unintended_leak": len({lk["row"] for lk in unintended}),
            "shift_mismatches": len(san["shift_mismatch"]),
            "policy_reasons": sorted(em.POLICY_REASONS),
            "unintended_examples": unintended[:20],
            "shift_mismatch_examples": san["shift_mismatch"][:20],
        }

    if want["restoration"]:
        result["restoration"] = {
            "source": "final_output stored by the runner (ResolvePass)",
            "rows_recomputed_with_resolvepass": res["recomputed"],
            "rows": res["rows"],
            "rows_with_replacements": res["rows_with_replacements"],
            "surrogates_left": len(res["surrogates_left"]),
            "rows_with_surrogate_left": res["rows_with_surrogate_left"],
            "over_restored_values": len(res["over_restored"]),
            "collateral_edits": len(res["collateral"]),
            "rows_with_collateral_edit": res["rows_with_collateral"],
            "examples": {
                "surrogates_left": res["surrogates_left"][:20],
                "over_restored": res["over_restored"][:20],
                "collateral_edits": res["collateral"][:20],
            },
        }

    if want["presidio_comparison"]:
        if cmp_rows == 0:
            result["presidio_comparison"] = {
                "available": False,
                "reason": "no row has presidio_found_piis (run the runner with the "
                          "Presidio comparison setting on)",
                "rows_without_presidio_data": cmp_missing,
            }
        else:
            def _sum(t: em.Tally) -> dict:
                s = t.summary()
                return {"micro": s["micro"], "macro": s["macro"],
                        "per_type": s["per_type"], "negatives": s["negatives"]}

            result["presidio_comparison"] = {
                "available": True,
                "rows": cmp_rows,
                "rows_without_presidio_data": cmp_missing,
                "data_status": "full" if cmp_missing == 0 else "partial",
                "universe": sorted(em.SHARED_TYPES),
                "matching": "span overlap; SS scored on protection (replaced spans); "
                            "predictions that overlap only out-of-universe gold are "
                            "dropped for both systems",
                "presidio_config": (result.get("run_meta") or {}).get("presidio"),
                "untyped": {"ss": _sum(cmp["ss_untyped"]), "presidio": _sum(cmp["prs_untyped"])},
                "typed": {"ss": _sum(cmp["ss_typed"]), "presidio": _sum(cmp["prs_typed"])},
                "bootstrap_untyped_micro_f1_ss_minus_presidio":
                    em.bootstrap_micro_f1_diff(cmp_counts["ss"], cmp_counts["presidio"]),
                "outside_universe": {
                    t: {**c, "reason": em.SS_ONLY_REASONS.get(t, "outside the shared type universe")}
                    for t, c in sorted(outside_universe.items())
                },
                "ss_predictions_outside_universe": dict(ss_outside),
                "presidio_only_counts": dict(presidio_only),
                "gold_values_verbatim_in_sent_text": sent_cmp,
            }

    if want["bertscore"]:
        def _means(rows: list) -> dict:
            n = len(rows)
            if not n:
                return {"n": 0, "precision": None, "recall": None, "f1": None}
            return {"n": n, **{m: round(sum(r[m] for r in rows) / n, 4)
                               for m in ("precision", "recall", "f1")}}

        out = {}
        for kind, (_a, _b, note) in _BERTSCORE_KINDS.items():
            d = bs[kind]
            pairs = d["pairs"]
            out[kind] = {
                "note": note,
                "ss": _means(d["ss"]),
                "presidio": _means(d["presidio"]),
                "excluded_identical_text": d["excluded_identical"],
                "paired": em.paired_stats([p[0] for p in pairs], [p[1] for p in pairs]),
            }
        rescaled = (None if not bs_rescaled
                    else True if bs_rescaled == {True}
                    else False if bs_rescaled == {False} else "mixed")
        out["rescaled"] = rescaled
        out["rows_with_error"] = bs_errors
        if rescaled not in (True, None):
            out["warning"] = "contains raw (unrescaled) scores — regenerate with the current runner (A8)"
        result["bertscore"] = out

    if want["ablation"]:
        result["ablation"] = {
            "available": False,
            "reason": "stored answers only allow a post-hoc split of detected "
                      "entities by stage, which cannot measure stage interactions "
                      "(audit A9); the ablation re-runs the cascade per configuration",
            "command": ABLATION_COMMAND,
        }

    if want["timing"]:
        tim = {}
        for k, vals in sorted(timing_vals.items()):
            s = sorted(vals)
            tim[k] = {"n": len(s), "mean": round(sum(s) / len(s), 3),
                      "p50": round(_percentile(s, 0.5), 3),
                      "p95": round(_percentile(s, 0.95), 3),
                      "max": round(s[-1], 3)}
        result["timing"] = tim

    return result
