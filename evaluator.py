"""
evaluator.py — Pure computation logic for SurrogateShield pipeline evaluation.

No UI, no Rich, no LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

EXPERIMENT_DIR = Path(__file__).parent / "experiment"

EVAL_FIELDS = [
    ("no_of_questions",      "No. of questions"),
    ("no_of_answers",        "No. of answers (non-empty LLM responses)"),
    ("no_of_answers_empty",  "No. of empty answers (errors/failures)"),
    ("answer_rate",          "Answer rate  (non-empty / total)"),
    ("surrogate_counts",     "Surrogate counts  (found vs key totals + averages)"),
    ("surrogate_quality",    "Surrogate quality  (precision / recall / F1 / accuracy / error)"),
    ("timing",               "Stage timings  (avg ms per stage)"),
    ("resolve_quality",      "ResolvePass quality  (surrogate leak rate + accuracy)"),
    ("sanitization_quality", "Sanitization quality  (PII leak to LLM rate + accuracy)"),
]


def parse_key_entry(entry: str) -> list[str]:
    """Parse a raw Answer-Key string into a list of clean PII values.

    Handles values wrapped in quotes, values without quotes, extra whitespace,
    and empty strings gracefully.
    """
    if not entry or not entry.strip():
        return []
    result = []
    for token in entry.split(","):
        cleaned = token.strip().strip('"').strip("'").strip()
        if cleaned:
            result.append(cleaned)
    return result


def run_evaluation(
    questions_filename: str,
    answers_filename: str,
    key_filename: str,
    fields: dict[str, bool],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Compute evaluation metrics for a pipeline run.

    Args:
        questions_filename: Filename inside experiment/ for the questions.
        answers_filename:   Filename inside experiment/ for the pipeline answers.
        key_filename:       Filename inside experiment/ for the ground-truth keys.
        fields:             Dict of EVAL_FIELDS key → bool controlling output.
        progress_cb:        Called as (index, total, status) per question.

    Returns:
        Dict of metric_key → value (only enabled fields are included).
        All floats are rounded to 4 decimal places.

    Raises:
        ValueError: If the three files have different lengths.
    """
    questions = json.loads((EXPERIMENT_DIR / questions_filename).read_text(encoding="utf-8"))
    answers   = json.loads((EXPERIMENT_DIR / answers_filename).read_text(encoding="utf-8"))
    keys      = json.loads((EXPERIMENT_DIR / key_filename).read_text(encoding="utf-8"))

    if not (len(questions) == len(answers) == len(keys)):
        raise ValueError(
            f"File length mismatch: questions={len(questions)}, "
            f"answers={len(answers)}, keys={len(keys)}. "
            "All three files must have the same number of entries."
        )

    total = len(questions)

    need_answers = any(fields.get(k) for k in ("no_of_answers", "no_of_answers_empty", "answer_rate"))
    need_counts  = fields.get("surrogate_counts", False)
    need_quality = fields.get("surrogate_quality", False)
    need_timing  = fields.get("timing", False)
    need_resolve = fields.get("resolve_quality", False)
    need_sanit   = fields.get("sanitization_quality", False)

    no_answers       = 0
    no_answers_empty = 0

    total_surrogates_found  = 0
    total_surrogates_in_key = 0

    precisions = []
    recalls    = []
    f1s        = []
    accuracies = []
    q_errors   = []

    timing_sums  = {k: 0.0 for k in ("pattern_scan_ms", "entity_trace_ms", "context_guard_ms", "surrogate_gen_ms")}
    timing_count = 0

    total_resolve_leaks            = 0
    total_individual_resolve_leaks = 0

    total_pii_leaks_to_llm     = 0
    total_individual_pii_leaks = 0
    leakage_accuracies = []
    leakage_errors     = []

    for i in range(total):
        status = "ok"
        try:
            a_entry = answers[i]
            k_entry = keys[i]

            surrogate_map   = a_entry.get("surrogate_map") or {}
            sanitized_input = a_entry.get("sanitized_input") or ""
            llm_response    = a_entry.get("llm_response") or ""
            stage_timings   = a_entry.get("stage_timings_ms")

            raw_key      = k_entry.get("Answer-Key", "")
            key_pii_list = parse_key_entry(raw_key) if raw_key else []

            if need_answers:
                if llm_response:
                    no_answers += 1
                else:
                    no_answers_empty += 1

            if need_counts:
                total_surrogates_found  += len(surrogate_map)
                total_surrogates_in_key += len(key_pii_list)

            if need_quality:
                found_set = {k.lower() for k in surrogate_map}
                key_set   = {v.lower() for v in key_pii_list}
                tp = len(found_set & key_set)
                fp = len(found_set - key_set)
                fn = len(key_set - found_set)
                p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                precisions.append(p)
                recalls.append(r)
                f1s.append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
                accuracies.append(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0)
                q_errors.append(  fn / (tp + fn) if (tp + fn) > 0 else 0.0)

            if need_timing and stage_timings:
                timing_count += 1
                for k in timing_sums:
                    timing_sums[k] += stage_timings.get(k, 0.0)

            if need_resolve:
                # Simulate ResolvePass to find genuine leaks (surrogates that
                # ResolvePass would have failed to restore).
                #
                # Q1: surrogate_map={"Revanth":"Victoria Mitchell","544-87-2944":"348-67-6360"}
                #   llm_response says "Hi Victoria!" — exact match finds neither full
                #   surrogate. Component pass: "Victoria" is found and replaced with
                #   "Revanth"; "348-67-6360" is absent. restored has no surrogates.
                #   leaked=[]. Correct — 0 leaks.
                #
                # Q2: surrogate_map={"revanth@gmail.com":"laurabennett@example.org",
                #                    "480-555-1234":"+1-141-020-9475"}
                #   Both surrogates appear verbatim. Exact pass restores both.
                #   restored has no surrogates. leaked=[]. Correct — 0 leaks.

                # Step 1 — inverted map: surrogate_value → original_pii
                inv = {v: k for k, v in surrogate_map.items() if v}

                # Step 2 — exact-match ResolvePass (longest surrogate first)
                restored = llm_response
                for surrogate in sorted(inv, key=len, reverse=True):
                    restored = restored.replace(surrogate, inv[surrogate])

                # Step 3 — component pass for multi-word surrogates whose full
                # form was not found (e.g. LLM used "Victoria" from "Victoria Mitchell")
                import re as _re
                for surrogate, original in inv.items():
                    if " " not in surrogate:
                        continue
                    if surrogate in llm_response:
                        continue  # already handled by exact pass
                    sur_words  = surrogate.split()
                    orig_words = original.split()
                    if len(sur_words) != len(orig_words):
                        continue
                    for sw, ow in zip(sur_words, orig_words):
                        restored = _re.sub(
                            r'\b' + _re.escape(sw) + r'\b', ow, restored
                        )

                # Step 4 — any surrogate still present in restored is a real leak
                leaked = [v for v in surrogate_map.values() if v and v in restored]
                total_resolve_leaks            += 1 if leaked else 0
                total_individual_resolve_leaks += len(leaked)

            if need_sanit and key_pii_list:
                si_lower   = sanitized_input.lower()
                leaked_pii = [v for v in key_pii_list if v.lower() in si_lower]
                total_pii_leaks_to_llm     += 1 if leaked_pii else 0
                total_individual_pii_leaks += len(leaked_pii)
                le = len(leaked_pii) / len(key_pii_list)
                leakage_errors.append(le)
                leakage_accuracies.append(1.0 - le)

        except Exception:
            status = "error"

        if progress_cb:
            progress_cb(i, total, status)

    def _r(x: float) -> float:
        return round(x, 4)

    result: dict = {}

    if fields.get("no_of_questions"):
        result["no_of_questions"] = total

    if fields.get("no_of_answers"):
        result["no_of_answers"] = no_answers

    if fields.get("no_of_answers_empty"):
        result["no_of_answers_empty"] = no_answers_empty

    if fields.get("answer_rate"):
        result["answer_rate"] = _r(no_answers / total) if total > 0 else 0.0

    if need_counts:
        result["no_surrogates_found"]                = total_surrogates_found
        result["no_surrogates_in_key"]               = total_surrogates_in_key
        result["avg_surrogates_per_question_found"]  = _r(total_surrogates_found  / total) if total > 0 else 0.0
        result["avg_surrogates_per_question_in_key"] = _r(total_surrogates_in_key / total) if total > 0 else 0.0

    if need_quality:
        n = len(precisions) or 1
        result["precision_surrogates"] = _r(sum(precisions) / n)
        result["recall_surrogates"]    = _r(sum(recalls)    / n)
        result["f1_surrogates"]        = _r(sum(f1s)        / n)
        result["accuracy_surrogates"]  = _r(sum(accuracies) / n)
        result["error_surrogates"]     = _r(sum(q_errors)   / n)

    if need_timing:
        tc = timing_count or 1
        result["avg_pattern_scan_ms"]  = _r(timing_sums["pattern_scan_ms"]  / tc)
        result["avg_entity_trace_ms"]  = _r(timing_sums["entity_trace_ms"]  / tc)
        result["avg_context_guard_ms"] = _r(timing_sums["context_guard_ms"] / tc)
        result["avg_surrogate_gen_ms"] = _r(timing_sums["surrogate_gen_ms"] / tc)

    if need_resolve:
        result["total_resolve_leaks"]            = total_resolve_leaks
        result["total_individual_resolve_leaks"] = total_individual_resolve_leaks
        result["resolve_leak_rate"]              = _r(total_resolve_leaks / total) if total > 0 else 0.0
        result["precision_resolve"]              = _r(1.0 - (total_resolve_leaks / total)) if total > 0 else 1.0
        result["error_resolve"]                  = _r(total_resolve_leaks / total) if total > 0 else 0.0

    if need_sanit:
        n = len(leakage_errors) or 1
        result["total_pii_leaks_to_llm"]     = total_pii_leaks_to_llm
        result["total_individual_pii_leaks"] = total_individual_pii_leaks
        result["pii_leak_rate"]              = _r(total_pii_leaks_to_llm / total) if total > 0 else 0.0
        result["accuracy_sanitization"]      = _r(sum(leakage_accuracies) / n)
        result["error_sanitization"]         = _r(sum(leakage_errors)     / n)

    return result
