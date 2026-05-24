"""
evaluator.py — Pure computation logic for SurrogateShield pipeline evaluation.

No UI, no Rich, no LLM calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

KEY_TYPE_MAP = {
    "name":           "PERSON",
    "person":         "PERSON",
    "PERSON":         "PERSON",
    "email":          "email",
    "phone":          "phone",
    "phone_us":       "phone",
    "phone_uk":       "phone",
    "phone_intl":     "phone",
    "ssn":            "ssn",
    "address":        "address",
    "dob":            "dob",
    "date_of_birth":  "dob",
    "org":            "ORG",
    "ORG":            "ORG",
    "organization":   "ORG",
    "gpe":            "GPE",
    "GPE":            "GPE",
    "location":       "GPE",
    "loc":            "LOC",
    "LOC":            "LOC",
    "credit_card":    "credit_card",
    "api_key":        "api_key",
    "ip_address":     "ip_address",
    "ip":             "ip_address",
    "zip":            "postal_code",
    "zip_us":         "postal_code",
    "postcode":       "postal_code",
    "postcode_uk":    "postal_code",
    "gender":         "gender_indicator",
    "fac":              "FAC",
    "FAC":              "FAC",
    "crypto":           "crypto",
    "bitcoin":          "crypto",
    "ethereum":         "crypto",
    "wallet":           "crypto",
    "bank_number":      "us_bank_number",
    "bank_account":     "us_bank_number",
    "us_bank_number":   "us_bank_number",
    "routing_number":   "us_bank_number",
    "routing":          "us_bank_number",
    "driver_license":   "us_driver_license",
    "us_driver_license": "us_driver_license",
    "drivers_license":  "us_driver_license",
    "dl":               "us_driver_license",
    "license":          "us_driver_license",
}

NORMALIZE_TYPE = {
    "phone_us":    "phone",
    "phone_uk":    "phone",
    "phone_intl":  "phone",
    "zip_us":      "postal_code",
    "postcode_uk": "postal_code",
}

# Maps Presidio's native entity type strings to SurrogateShield's
# comparable consolidated types for fair side-by-side comparison.
# Types not in this map are Presidio-only (no SS equivalent).
PRESIDIO_TO_COMPARABLE = {
    "PERSON":        "PERSON",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER":  "phone",
    "US_SSN":        "ssn",
    "CREDIT_CARD":   "credit_card",
    "IP_ADDRESS":    "ip_address",
    "DATE_TIME":         "dob",    # approximate: Presidio detects all
                                   # datetime; SS focuses on DOB only
    "LOCATION":          "GPE",    # approximate: both NER-based location
    "CRYPTO":            "crypto",
    "US_BANK_NUMBER":    "us_bank_number",
    "US_DRIVER_LICENSE": "us_driver_license",
}

# Types SS detects that Presidio cannot — shown separately in table
SS_ONLY_TYPES = [
    "api_key", "address", "postal_code", "gender_indicator",
    "ORG", "FAC",
]

EXPERIMENT_DIR = Path(__file__).parent / "experiment"

EVAL_FIELDS = [
    ("no_of_questions",      "No. of questions"),
    ("no_of_answers",        "No. of answers (non-empty LLM responses)"),
    ("no_of_answers_empty",  "No. of empty answers (errors/failures)"),
    ("answer_rate",          "Answer rate  (non-empty / total)"),
    ("surrogate_counts",     "Surrogate counts  (found vs key totals + averages)"),
    ("surrogate_quality",    "Surrogate quality  (precision / recall / F1 / accuracy / error)"),
    ("per_entity_type",      "Per-entity-type breakdown  (F1 / precision / recall per PII type)"),
    ("presidio_comparison",
     "Presidio comparison  (SS vs Presidio — side-by-side Table 1 for paper)"),
    ("bertscore_comparison",
     "BERTScore comparison  (utility preservation — Table 2 for paper)"),
    ("ablation_study",
     "Ablation study  (per-stage contribution — Table 4 for paper)"),
    ("timing",               "Stage timings  (avg ms per stage)"),
    ("resolve_quality",      "ResolvePass quality  (surrogate leak rate + accuracy)"),
    ("sanitization_quality", "Sanitization quality  (PII leak to LLM rate + accuracy)"),
]


def parse_key_entry(answer_key) -> tuple[list[str], dict[str, list[str]]]:
    """
    Parse an Answer-Key value into:
      - flat list of all PII values (strings, for overall metrics)
      - typed dict mapping internal_system_type -> list of values

    Handles:
      1. New dict format: {"name": "Revanth", "ssn": "544-87-2944"}
      2. New dict format with list values: {"name": ["John", "Jane"], "email": "j@x.com"}
      3. Old string format: '"Revanth", "544-87-2944"' (backward compat, returns empty typed dict)
      4. None or empty input: returns ([], {})
    """
    if not answer_key:
        return [], {}

    if isinstance(answer_key, dict):
        typed: dict[str, list[str]] = {}
        flat: list[str] = []
        for label, val in answer_key.items():
            internal_type = KEY_TYPE_MAP.get(label, "other")
            vals = val if isinstance(val, list) else [val]
            vals = [v.strip() for v in vals if v and str(v).strip()]
            if vals:
                typed.setdefault(internal_type, []).extend(vals)
                flat.extend(vals)
        return flat, typed

    if isinstance(answer_key, str):
        flat = []
        for token in answer_key.split(","):
            cleaned = token.strip().strip('"').strip("'").strip()
            if cleaned:
                flat.append(cleaned)
        return flat, {}

    return [], {}


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
    need_sanit    = fields.get("sanitization_quality", False)
    need_per_type      = fields.get("per_entity_type", False)
    need_presidio_cmp  = fields.get("presidio_comparison", False)
    need_bertscore_cmp = fields.get("bertscore_comparison", False)
    need_ablation      = fields.get("ablation_study", False)

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

    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Per-type stats for Presidio (comparable types only)
    presidio_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    presidio_only_counts = defaultdict(int)  # types with no SS equivalent
    presidio_questions_with_data = 0         # questions that had presidio_found_piis

    # SS comparable-type stats (computed independently for comparison,
    # regardless of whether per_entity_type toggle is on)
    ss_cmp_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    ss_bs_precisions  = []
    ss_bs_recalls     = []
    ss_bs_f1s         = []

    prs_bs_precisions = []
    prs_bs_recalls    = []
    prs_bs_f1s        = []

    from collections import defaultdict as _dd

    abl_configs = {
        "ps_only": _dd(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "ps_et":   _dd(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "ps_cg":   _dd(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "full":    _dd(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }

    abl_overall = {
        "ps_only": {"tp": 0, "fp": 0, "fn": 0},
        "ps_et":   {"tp": 0, "fp": 0, "fn": 0},
        "ps_cg":   {"tp": 0, "fp": 0, "fn": 0},
        "full":    {"tp": 0, "fp": 0, "fn": 0},
    }

    stage_counts = {"pattern": 0, "ner": 0, "slm": 0}

    questions_needing_et    = 0
    questions_needing_cg    = 0
    abl_questions_with_data = 0

    for i in range(total):
        status = "ok"
        try:
            a_entry = answers[i]
            k_entry = keys[i]

            surrogate_map   = a_entry.get("surrogate_map") or {}
            sanitized_input = a_entry.get("sanitized_input") or ""
            llm_response    = a_entry.get("llm_response") or ""
            stage_timings   = a_entry.get("stage_timings_ms")

            raw_key = k_entry.get("Answer-Key", "")
            key_pii_list, key_typed = parse_key_entry(raw_key) if raw_key else ([], {})

            if need_answers:
                if llm_response:
                    no_answers += 1
                else:
                    no_answers_empty += 1

            if need_counts:
                total_surrogates_found  += len(surrogate_map)
                total_surrogates_in_key += len(key_pii_list)

            if need_quality:
                rnr_set = {
                    r["value"].lower()
                    for r in (a_entry.get("recognized_not_replaced") or [])
                    if isinstance(r, dict) and r.get("value")
                }
                found_set = {k.lower() for k in surrogate_map} | rnr_set
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

                # Step 4 — any surrogate still present in restored is a real leak.
                # Skip self-mapped entries (original == surrogate): the pipeline
                # detected the PII but could not generate a different surrogate,
                # so the value was intentionally kept unchanged — not a leak.
                leaked = [
                    v for k, v in surrogate_map.items()
                    if v and k != v and v in restored
                ]
                total_resolve_leaks            += 1 if leaked else 0
                total_individual_resolve_leaks += len(leaked)

            if need_sanit and key_pii_list:
                import re as _re
                rnr_set = {
                    r["value"].lower()
                    for r in (a_entry.get("recognized_not_replaced") or [])
                    if isinstance(r, dict) and r.get("value")
                }
                si_lower = sanitized_input.lower()
                # Use word-boundary matching: a PII value only counts as leaked
                # if it appears as a distinct token, not as a substring embedded
                # inside another word (e.g. 'il' inside 'vehicle', 'sti' inside
                # 'testing').  _re.escape handles special chars (+, ., etc.).
                def _pii_in_text(val: str) -> bool:
                    pat = r'(?<![a-zA-Z0-9])' + _re.escape(val) + r'(?![a-zA-Z0-9])'
                    return bool(_re.search(pat, si_lower))

                leaked_pii = [
                    v for v in key_pii_list
                    if v.lower() not in rnr_set and _pii_in_text(v.lower())
                ]
                total_pii_leaks_to_llm     += 1 if leaked_pii else 0
                total_individual_pii_leaks += len(leaked_pii)
                le = len(leaked_pii) / len(key_pii_list)
                leakage_errors.append(le)
                leakage_accuracies.append(1.0 - le)

            if need_per_type and key_typed:
                rnr_set = {
                    r["value"].lower()
                    for r in (a_entry.get("recognized_not_replaced") or [])
                    if isinstance(r, dict) and r.get("value")
                }
                found_lower   = {k.lower() for k in surrogate_map.keys()} | rnr_set
                key_all_lower = {v.lower() for v in key_pii_list}

                for internal_type, vals in key_typed.items():
                    for val in vals:
                        if val.lower() in found_lower:
                            type_stats[internal_type]["tp"] += 1
                        else:
                            type_stats[internal_type]["fn"] += 1

                pii_detail = a_entry.get("pii_detail") or {}
                for detected_val, detail in pii_detail.items():
                    if detected_val.lower() not in key_all_lower:
                        if detected_val.lower() in rnr_set:
                            continue
                        raw_type = detail.get("type", "other") if isinstance(detail, dict) else "other"
                        normalized_type = NORMALIZE_TYPE.get(raw_type, raw_type)
                        type_stats[normalized_type]["fp"] += 1

            if need_presidio_cmp:
                p_found = a_entry.get("presidio_found_piis")

                # ── SS comparable-type stats (always computed for comparison) ──
                if key_typed:
                    rnr_set = {
                        r["value"].lower()
                        for r in (a_entry.get("recognized_not_replaced") or [])
                        if isinstance(r, dict) and r.get("value")
                    }
                    found_lower   = {k.lower() for k in surrogate_map.keys()} | rnr_set
                    key_all_lower = {v.lower() for v in key_pii_list}

                    for internal_type, vals in key_typed.items():
                        cmp_type = NORMALIZE_TYPE.get(internal_type, internal_type)
                        for val in vals:
                            if val.lower() in found_lower:
                                ss_cmp_type_stats[cmp_type]["tp"] += 1
                            else:
                                ss_cmp_type_stats[cmp_type]["fn"] += 1

                    pii_detail = a_entry.get("pii_detail") or {}
                    for detected_val, detail in pii_detail.items():
                        if detected_val.lower() not in key_all_lower:
                            if detected_val.lower() in rnr_set:
                                continue
                            raw_type = detail.get("type", "other") if isinstance(detail, dict) else "other"
                            cmp_type = NORMALIZE_TYPE.get(raw_type, raw_type)
                            ss_cmp_type_stats[cmp_type]["fp"] += 1

                # ── Presidio stats ─────────────────────────────────────────────
                if p_found is None:
                    pass  # Presidio was unavailable for this question — skip
                else:
                    presidio_questions_with_data += 1

                    if key_typed:
                        key_all_lower = {v.lower() for v in key_pii_list}
                        presidio_detected_lower = {
                            e["value"].lower()
                            for e in p_found
                            if isinstance(e, dict) and e.get("value")
                        }

                        # TPs and FNs — iterate over typed key
                        for internal_type, vals in key_typed.items():
                            cmp_type = NORMALIZE_TYPE.get(internal_type, internal_type)
                            # Only count types that are in PRESIDIO_TO_COMPARABLE
                            # (types SS has but Presidio cannot detect are SS-only)
                            if cmp_type not in PRESIDIO_TO_COMPARABLE.values():
                                continue
                            for val in vals:
                                if val.lower() in presidio_detected_lower:
                                    presidio_type_stats[cmp_type]["tp"] += 1
                                else:
                                    presidio_type_stats[cmp_type]["fn"] += 1

                        # FPs — Presidio detected but not in key
                        for e in p_found:
                            if not isinstance(e, dict):
                                continue
                            val   = e.get("value", "")
                            etype = e.get("type", "")
                            comparable = PRESIDIO_TO_COMPARABLE.get(etype)
                            if comparable is None:
                                # Presidio-only type (URL, CRYPTO, etc.)
                                presidio_only_counts[etype] += 1
                                continue
                            if val.lower() not in key_all_lower:
                                presidio_type_stats[comparable]["fp"] += 1

            if need_bertscore_cmp:
                bs_ss  = a_entry.get("bertscore_ss")
                bs_prs = a_entry.get("bertscore_presidio")

                if isinstance(bs_ss, dict):
                    ss_bs_precisions.append(bs_ss.get("precision", 0))
                    ss_bs_recalls.append(   bs_ss.get("recall",    0))
                    ss_bs_f1s.append(       bs_ss.get("f1",        0))

                if isinstance(bs_prs, dict):
                    prs_bs_precisions.append(bs_prs.get("precision", 0))
                    prs_bs_recalls.append(   bs_prs.get("recall",    0))
                    prs_bs_f1s.append(       bs_prs.get("f1",        0))

            if need_ablation:
                ps_pii  = [v.lower() for v in (a_entry.get("pattern_scan_pii")  or [])]
                et_pii  = [v.lower() for v in (a_entry.get("entity_trace_pii")  or [])]
                cg_pii  = [v.lower() for v in (a_entry.get("context_guard_pii") or [])]
                all_pii = [v.lower() for v in (a_entry.get("confirmed_pii")     or [])]

                has_stage_data = (
                    a_entry.get("pattern_scan_pii") is not None or
                    a_entry.get("entity_trace_pii") is not None
                )
                if not has_stage_data or not key_pii_list:
                    pass
                else:
                    abl_questions_with_data += 1
                    key_lower = {v.lower() for v in key_pii_list}

                    stage_counts["pattern"] += len(ps_pii)
                    stage_counts["ner"]     += len(et_pii)
                    stage_counts["slm"]     += len(cg_pii)

                    detected = {
                        "ps_only": set(ps_pii),
                        "ps_et":   set(ps_pii) | set(et_pii),
                        "ps_cg":   set(ps_pii) | set(cg_pii),
                        "full":    set(all_pii),
                    }

                    for cfg_name, det_set in detected.items():
                        tp = len(det_set & key_lower)
                        fp = len(det_set - key_lower)
                        fn = len(key_lower - det_set)
                        abl_overall[cfg_name]["tp"] += tp
                        abl_overall[cfg_name]["fp"] += fp
                        abl_overall[cfg_name]["fn"] += fn

                    pii_detail = a_entry.get("pii_detail") or {}

                    val_to_type: dict = {}
                    for val, detail in pii_detail.items():
                        if isinstance(detail, dict):
                            raw_type = detail.get("type", "other")
                            normalized = NORMALIZE_TYPE.get(raw_type, raw_type)
                            val_to_type[val.lower()] = normalized

                    key_val_to_type: dict = {}
                    for etype, vals in key_typed.items():
                        for v in vals:
                            key_val_to_type[v.lower()] = etype

                    for cfg_name, det_set in detected.items():
                        for kval in key_lower:
                            ktype = key_val_to_type.get(kval, "other")
                            if kval in det_set:
                                abl_configs[cfg_name][ktype]["tp"] += 1
                            else:
                                abl_configs[cfg_name][ktype]["fn"] += 1
                        for dval in det_set:
                            if dval not in key_lower:
                                dtype = val_to_type.get(dval, "other")
                                abl_configs[cfg_name][dtype]["fp"] += 1

                    if set(et_pii) & key_lower:
                        if set(ps_pii) & key_lower != key_lower:
                            questions_needing_et += 1
                    if set(cg_pii) & key_lower:
                        if (set(ps_pii) | set(et_pii)) & key_lower != key_lower:
                            questions_needing_cg += 1

        except Exception:
            status = "error"

        if progress_cb:
            progress_cb(i, total, status)

    per_type_result: dict = {}
    if need_per_type:
        for etype, counts in type_stats.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_type_result[etype] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

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
        result["avg_pattern_scan_ms"]  = round(timing_sums["pattern_scan_ms"]  / tc, 6)
        result["avg_entity_trace_ms"]  = round(timing_sums["entity_trace_ms"]  / tc, 6)
        result["avg_context_guard_ms"] = round(timing_sums["context_guard_ms"] / tc, 6)
        result["avg_surrogate_gen_ms"] = round(timing_sums["surrogate_gen_ms"] / tc, 6)

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

    if need_per_type:
        result["per_entity_type"] = per_type_result

    if need_presidio_cmp:
        def _compute_metrics(stats_dict):
            """Compute precision/recall/F1 from a type_stats defaultdict."""
            result_types = {}
            total_tp = total_fp = total_fn = 0
            for etype, counts in stats_dict.items():
                tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
                p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                result_types[etype] = {
                    "precision": _r(p),
                    "recall":    _r(r),
                    "f1":        _r(f1),
                    "tp": tp, "fp": fp, "fn": fn,
                }
                total_tp += tp
                total_fp += fp
                total_fn += fn
            overall_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
            overall_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
            overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0
            overall = {
                "precision": _r(overall_p),
                "recall":    _r(overall_r),
                "f1":        _r(overall_f1),
            }
            return result_types, overall

        ss_per_type, ss_overall   = _compute_metrics(ss_cmp_type_stats)
        prs_per_type, prs_overall = _compute_metrics(presidio_type_stats)

        result["presidio_comparison"] = {
            "presidio_questions_with_data": presidio_questions_with_data,
            "data_status": (
                "no_data"  if presidio_questions_with_data == 0
                else "partial" if presidio_questions_with_data < total
                else "full"
            ),
            "data_count":  presidio_questions_with_data,
            "total_count": total,
            "ss_overall":       ss_overall,
            "presidio_overall": prs_overall,
            "per_type": {
                etype: {
                    "ss":      ss_per_type.get(etype),
                    "presidio": prs_per_type.get(etype),
                }
                for etype in sorted(
                    set(ss_per_type.keys()) | set(prs_per_type.keys())
                )
            },
            "ss_only_types": {
                etype: ss_per_type[etype]
                for etype in SS_ONLY_TYPES
                if etype in ss_per_type
            },
            "presidio_only_counts": dict(presidio_only_counts),
            "approximate_note": "DATE_TIME/dob and LOCATION/GPE are approximate comparisons",
        }

    if need_bertscore_cmp:

        def _mean(lst):
            return _r(sum(lst) / len(lst)) if lst else None

        def _bs_status(count, total):
            if count == 0:     return "no_data"
            if count < total:  return "partial"
            return "full"

        ss_count  = len(ss_bs_f1s)
        prs_count = len(prs_bs_f1s)

        result["bertscore_comparison"] = {
            "total_questions": total,
            "ss": {
                "precision":   _mean(ss_bs_precisions),
                "recall":      _mean(ss_bs_recalls),
                "f1":          _mean(ss_bs_f1s),
                "data_count":  ss_count,
                "data_status": _bs_status(ss_count, total),
            },
            "presidio": {
                "precision":   _mean(prs_bs_precisions),
                "recall":      _mean(prs_bs_recalls),
                "f1":          _mean(prs_bs_f1s),
                "data_count":  prs_count,
                "data_status": _bs_status(prs_count, total),
            },
        }

    if need_ablation:

        def _abl_metrics(tp, fp, fn):
            p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
            return {"precision": _r(p), "recall": _r(r), "f1": _r(f1),
                    "tp": tp, "fp": fp, "fn": fn}

        CONFIG_LABELS = {
            "ps_only": "PatternScan only",
            "ps_et":   "PatternScan + EntityTrace",
            "ps_cg":   "PatternScan + ContextGuard",
            "full":    "Full cascade (all three)",
        }

        overall_metrics = {}
        for cfg, counts in abl_overall.items():
            overall_metrics[cfg] = {
                "label": CONFIG_LABELS[cfg],
                **_abl_metrics(counts["tp"], counts["fp"], counts["fn"])
            }

        per_type_by_config = {}
        all_types_seen = set()
        for cfg, type_dict in abl_configs.items():
            per_type_by_config[cfg] = {}
            for etype, counts in type_dict.items():
                per_type_by_config[cfg][etype] = _abl_metrics(
                    counts["tp"], counts["fp"], counts["fn"]
                )
                all_types_seen.add(etype)

        result["ablation_study"] = {
            "questions_with_data":  abl_questions_with_data,
            "total_questions":      total,
            "stage_entity_counts": {
                "pattern_scan":  stage_counts["pattern"],
                "entity_trace":  stage_counts["ner"],
                "context_guard": stage_counts["slm"],
                "total":         sum(stage_counts.values()),
            },
            "stage_necessity": {
                "questions_needing_entity_trace":  questions_needing_et,
                "questions_needing_context_guard": questions_needing_cg,
                "pct_needing_entity_trace":  _r(questions_needing_et  / max(abl_questions_with_data, 1)),
                "pct_needing_context_guard": _r(questions_needing_cg  / max(abl_questions_with_data, 1)),
            },
            "configurations":    overall_metrics,
            "per_type":          per_type_by_config,
            "entity_types_seen": sorted(all_types_seen),
        }

    return result
