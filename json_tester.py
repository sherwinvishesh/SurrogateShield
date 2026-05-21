"""
json_tester.py — Batch JSON testing for SurrogateShield.

Input:  experiment/<name>.json   — list of {"input": "..."} objects
Output: experiment/<name>_answers.json — list of result objects

Progress is flushed to disk every 25 questions so a run can be safely
interrupted and resumed without losing work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

EXPERIMENT_DIR = Path(__file__).parent / "experiment"

# ── Output field registry ─────────────────────────────────────────────────────
# (key, display label)
OUTPUT_FIELDS: List[Tuple[str, str]] = [
    ("question",          "Question asked"),
    ("pattern_scan_pii",  "PatternScan PIIs       (regex stage)"),
    ("entity_trace_pii",  "EntityTrace PIIs        (spaCy NER stage)"),
    ("context_guard_pii", "ContextGuard PIIs       (distilbert stage)"),
    ("confirmed_pii",     "Confirmed PIIs          (final combined list)"),
    ("pii_detail",        "PII detail              (type, score, source per entity)"),
    ("quasi_id_risks",    "Quasi-ID risks          (combination re-identification risks)"),
    ("surrogate_map",     "Surrogate map           (original → replacement)"),
    ("sanitized_input",   "Sanitized input         (text sent to LLM)"),
    ("llm_response",      "LLM response            (raw text received)"),
    ("stage_timings_ms",  "Stage timings (ms)      (PatternScan / EntityTrace / ContextGuard / surrogate gen / LLM)"),
    ("presidio_sanitized_input",
     "Presidio sanitized  (Presidio [TYPE] redaction — baseline for BERTScore)"),
    ("presidio_found_piis",
     "Presidio found PIIs  (raw detected entities — type, value, score)"),
    ("recognized_not_replaced",
     "Recognized not replaced  (detected as PII but intentionally skipped)"),
    ("bertscore_ss",
     "BERTScore SS       (original vs SS sanitized — utility score)"),
    ("bertscore_presidio",
     "BERTScore Presidio (original vs Presidio sanitized — baseline)"),
]

DEFAULT_FIELDS: Dict[str, bool] = {key: True for key, _ in OUTPUT_FIELDS}
DEFAULT_FIELDS["presidio_sanitized_input"] = False
DEFAULT_FIELDS["presidio_found_piis"] = False
DEFAULT_FIELDS["recognized_not_replaced"] = True
DEFAULT_FIELDS["bertscore_ss"]       = False
DEFAULT_FIELDS["bertscore_presidio"] = False


# ── Core per-question processor ───────────────────────────────────────────────

def _ms(t0: float) -> float:
    """Elapsed milliseconds since t0, rounded to 2 d.p."""
    return round((time.time() - t0) * 1000, 2)


def _process_one(
    question: str,
    chat,
    fields: Dict[str, bool],
) -> dict:
    """Run the full detection + optional LLM call for one question."""
    from detection.logic import run_cascade, deduplicate
    from generation.logic import MimicGen

    want_timings = fields.get("stage_timings_ms", False)
    timings: Dict[str, float] = {}
    t_total = time.time()

    answer: dict = {}

    if fields.get("question"):
        answer["question"] = question

    # ── Per-stage timing (runs each stage individually, then discards results) ─
    # run_cascade() applies post-processing passes (A–D) that individual stage
    # calls skip, so we still call run_cascade() below for the real entity list.
    # The overhead is small: PatternScan is regex, EntityTrace/ContextGuard use
    # cached models and are fast after the first question in a batch.
    if want_timings:
        from detection import pattern_scan as _ps, entity_trace as _et, context_guard as _cg
        from util import mask_spans
        from config import CONTEXT_GUARD_ENABLED

        t = time.time()
        _ps_results = _ps.scan(question)
        timings["pattern_scan_ms"] = _ms(t)

        _remaining = mask_spans(question, _ps_results)
        t = time.time()
        _et_conf, _et_border = _et.trace(_remaining, existing_entities=_ps_results)
        timings["entity_trace_ms"] = _ms(t)

        timings["context_guard_ms"] = 0.0
        if CONTEXT_GUARD_ENABLED and _et_border:
            _remaining2 = mask_spans(_remaining, _et_conf)
            t = time.time()
            _cg.guard(_remaining2, _et_border)
            timings["context_guard_ms"] = _ms(t)

    # ── Detection (authoritative results with all post-processing passes) ──────
    from detection.service_query import is_service_query, fuzz_addresses
    from config import SERVICE_QUERY_DETECTION_ENABLED

    is_svc = SERVICE_QUERY_DETECTION_ENABLED and is_service_query(question)
    fuzzed_addresses: dict = {}
    working_question = question

    if is_svc:
        from config import SERVICE_QUERY_VERIFY_ADDRESSES
        working_question, fuzzed_addresses = fuzz_addresses(
            question, verify=SERVICE_QUERY_VERIFY_ADDRESSES
        )

    confirmed, _ = run_cascade(
        working_question,
        skip_location_entities=is_svc,
    )
    confirmed = deduplicate(confirmed)

    # Collect recognized-but-not-replaced entities
    skipped_entities = getattr(confirmed, '_skipped_entities', [])

    # Combine for display — confirmed (will get surrogates) + skipped (detected, no surrogate)
    all_detected_for_display = list(confirmed) + list(skipped_entities)

    # Per-stage breakdowns (filter by source set by each detector)
    if fields.get("pattern_scan_pii"):
        answer["pattern_scan_pii"] = [
            e.text for e in all_detected_for_display if e.source == "pattern"
        ]

    if fields.get("entity_trace_pii"):
        answer["entity_trace_pii"] = [
            e.text for e in all_detected_for_display if e.source == "ner"
        ]

    if fields.get("context_guard_pii"):
        answer["context_guard_pii"] = [
            e.text for e in all_detected_for_display if e.source == "slm"
        ]

    if fields.get("confirmed_pii"):
        answer["confirmed_pii"] = [e.text for e in all_detected_for_display]

    if fields.get("pii_detail"):
        detail: dict = {
            e.text: {
                "type":   e.type,
                "score":  round(e.score, 4),
                "source": e.source,
            }
            for e in confirmed
        }
        _skip_reason = (
            "service_query_location_suppressed" if is_svc else "topical_geo_filtered"
        )
        for e in skipped_entities:
            detail[e.text] = {
                "type":             e.type,
                "score":            round(e.score, 4),
                "source":           e.source,
                "surrogate_status": "skipped",
                "skip_reason":      _skip_reason,
            }
        for original_addr, fuzzed_addr in fuzzed_addresses.items():
            detail[original_addr] = {
                "type":             "address",
                "score":            1.0,
                "source":           "pattern",
                "surrogate_status": "fuzzed",
                "skip_reason":      "service_query_fuzzed",
                "fuzzed_to":        fuzzed_addr,
            }
        answer["pii_detail"] = detail

    if fields.get("quasi_id_risks"):
        qi_matches = getattr(confirmed, "_qi_matches", [])
        answer["quasi_id_risks"] = [
            {
                "combination":        m.combination_name,
                "matched_fields":     m.matched_fields,
                "risk_level":         m.risk_level,
                "all_fields_matched": m.all_fields_matched,
                "reference":          m.reference if m.all_fields_matched else m.partial_reference,
            }
            for m in qi_matches
        ]

    # ── Surrogate generation ──────────────────────────────────────────────────
    need_surrogates = any(
        fields.get(k) for k in ("surrogate_map", "sanitized_input", "llm_response")
    )
    surrogate_map: Dict[str, str] = {}
    if need_surrogates:
        t = time.time()
        mimic = MimicGen()
        surrogate_map = mimic.generate_all(confirmed) if confirmed else {}
        if want_timings:
            timings["surrogate_gen_ms"] = _ms(t)
    elif want_timings:
        timings["surrogate_gen_ms"] = 0.0

    if fields.get("surrogate_map"):
        answer["surrogate_map"] = surrogate_map

    # ── Sanitized text ────────────────────────────────────────────────────────
    sanitized = question
    for orig in sorted(surrogate_map, key=len, reverse=True):
        sanitized = sanitized.replace(orig, surrogate_map[orig])

    if fields.get("sanitized_input"):
        answer["sanitized_input"] = sanitized

    # ── Recognized-but-not-replaced entities ──────────────────────────────────
    if fields.get("recognized_not_replaced"):
        rnr_list = []

        for ent in skipped_entities:
            rnr_list.append({
                "value":  ent.text,
                "type":   ent.type,
                "reason": (
                    "service_query_location_suppressed"
                    if is_svc
                    else "topical_geo_filtered"
                ),
            })

        for original_addr, fuzzed_addr in fuzzed_addresses.items():
            rnr_list.append({
                "value":     original_addr,
                "type":      "address",
                "reason":    "service_query_fuzzed",
                "fuzzed_to": fuzzed_addr,
            })

        answer["recognized_not_replaced"] = rnr_list

    # ── LLM call (single-turn, no conversation history) ───────────────────────
    if want_timings:
        timings["llm_call_ms"] = 0.0

    if fields.get("llm_response"):
        t = time.time()
        raw = chat._send_to_api([{"role": "user", "content": sanitized}])
        if want_timings:
            timings["llm_call_ms"] = _ms(t)
        answer["llm_response"] = raw

    # ── Assemble timings ──────────────────────────────────────────────────────
    if want_timings:
        timings["total_ms"] = _ms(t_total)
        answer["stage_timings_ms"] = timings

    # ── Presidio detection (shared for both presidio fields) ─────────────────
    need_presidio = (
        fields.get("presidio_sanitized_input") or
        fields.get("presidio_found_piis")
    )

    if need_presidio:
        try:
            from presidio.detect import detect as _presidio_detect
            from presidio.redact import redact as _presidio_redact

            p_entities = _presidio_detect(question)

            if p_entities is None:
                # Presidio unavailable — store None for both fields
                if fields.get("presidio_sanitized_input"):
                    answer["presidio_sanitized_input"] = None
                if fields.get("presidio_found_piis"):
                    answer["presidio_found_piis"] = None

            else:
                # presidio_sanitized_input
                if fields.get("presidio_sanitized_input"):
                    if p_entities:
                        answer["presidio_sanitized_input"] = _presidio_redact(
                            question, p_entities
                        )
                    else:
                        # No entities found — original text is unchanged
                        answer["presidio_sanitized_input"] = question

                # presidio_found_piis
                if fields.get("presidio_found_piis"):
                    answer["presidio_found_piis"] = [
                        {
                            "value": ent.text,
                            "type":  ent.entity_type,
                            "score": ent.score,
                        }
                        for ent in p_entities
                    ]

        except Exception:
            if fields.get("presidio_sanitized_input"):
                answer["presidio_sanitized_input"] = None
            if fields.get("presidio_found_piis"):
                answer["presidio_found_piis"] = None

    return answer


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    filename: str,
    fields: Dict[str, bool],
    progress_cb: Optional[Callable[[int, int, str, str, float], None]] = None,
) -> str:
    """
    Process all questions in experiment/<filename> and write results.

    Args:
        filename:    Filename inside experiment/ (e.g. "questions.json").
        fields:      Dict of field_key → bool controlling output columns.
        progress_cb: Called as (index, total, question, status, elapsed_s).
                     status is "running", "ok", or "error".

    Returns:
        Absolute path to the output file.
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    in_path  = EXPERIMENT_DIR / filename
    stem     = in_path.stem
    out_path = EXPERIMENT_DIR / f"{stem}_answers.json"

    questions: List[dict] = json.loads(in_path.read_text(encoding="utf-8"))
    total = len(questions)

    # Resume: load existing answers and skip already-processed questions
    answers: List[dict] = []
    if out_path.exists():
        try:
            answers = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            answers = []

    start_idx = len(answers)

    # Only initialise the LLM client if the llm_response field is requested.
    # This lets detection-only runs work without any API key configured.
    chat = None
    if fields.get("llm_response"):
        from chatbot.chat import ClaudeChat
        chat = ClaudeChat()

    # Suppress presidio_sanitized_input when Presidio Comparison is off in
    # settings — even if the user toggled it on in the field-select screen.
    from settings_manager import load_settings as _load_settings
    if not _load_settings().get("presidio_comparison", False):
        fields = {**fields, "presidio_sanitized_input": False}

    # Pre-load Presidio engine if the field is enabled.
    # The engine.py singleton caches it — this just triggers the 3-5s
    # spaCy model load ONCE before the loop instead of mid-first-question.
    if fields.get("presidio_sanitized_input") or fields.get("presidio_found_piis"):
        try:
            from presidio.engine import get_analyzer
            get_analyzer()
        except Exception:
            pass   # graceful — _process_one handles unavailability per question

    def _flush():
        out_path.write_text(
            json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for i, item in enumerate(questions[start_idx:], start=start_idx):
        question = item.get("input", "")
        t0 = time.time()

        if progress_cb:
            progress_cb(i, total, question, "running", 0.0)

        try:
            result  = _process_one(question, chat, fields)
            answers.append(result)
            elapsed = time.time() - t0
            status  = "ok"
        except Exception as exc:
            answers.append({"question": question, "error": str(exc)})
            elapsed = time.time() - t0
            status  = "error"

        # Flush every 25 new answers and on the final question — before the
        # progress callback fires so the 💾 indicator reflects the real state.
        processed = i + 1 - start_idx
        if processed % 25 == 0 or (i + 1) == total:
            _flush()

        if progress_cb:
            progress_cb(i, total, question, status, elapsed)

    # ── Post-loop: BERTScore batch computation ────────────────────────
    need_bs_ss  = fields.get("bertscore_ss",       False)
    need_bs_prs = fields.get("bertscore_presidio", False)

    if need_bs_ss or need_bs_prs:
        _run_bertscore_batch(
            answers=answers,
            questions=questions,
            need_ss=need_bs_ss,
            need_presidio=need_bs_prs,
            progress_cb=progress_cb,
            total=total,
        )
        # BERTScore done — _run_bertscore_batch updates answers in-place
        # The final _flush() below saves everything including scores

    _flush()
    return str(out_path)


def _run_bertscore_batch(
    answers: list,
    questions: list,
    need_ss: bool,
    need_presidio: bool,
    progress_cb,
    total: int,
) -> None:
    """
    Compute BERTScore for all questions in one batch pass.
    Updates answers in-place. Never raises — errors stored as None.

    Signals to progress_cb using sentinel index -1 and special statuses:
        "bertscore_start"    — batch computation beginning
        "bertscore_done"     — batch computation complete
        "bertscore_skipped"  — bert-score not installed
        "bertscore_warn"     — partial data warning
    """
    import time as _time

    # ── Check bert-score is installed ────────────────────────────────
    try:
        import logging
        import warnings

        # Suppress transformers load report and HuggingFace Hub warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

        from bert_score import score as _bs_score
    except ImportError:
        if progress_cb:
            progress_cb(-1, total, "", "bertscore_skipped", 0.0)
        for ans in answers:
            if need_ss:
                ans["bertscore_ss"] = None
            if need_presidio:
                ans["bertscore_presidio"] = None
        return

    if progress_cb:
        progress_cb(-1, total, "", "bertscore_start", 0.0)

    t0 = _time.time()

    # ── Collect SS pairs ──────────────────────────────────────────────
    if need_ss:
        ss_indices    = []
        ss_candidates = []   # sanitized_input values
        ss_references = []   # original questions

        for i, (ans, q_entry) in enumerate(zip(answers, questions)):
            original   = q_entry.get("input", "")
            sanitized  = ans.get("sanitized_input")
            if original and sanitized:
                ss_indices.append(i)
                ss_candidates.append(sanitized)
                ss_references.append(original)

        if ss_indices:
            try:
                P, R, F1 = _bs_score(
                    cands=ss_candidates,
                    refs=ss_references,
                    lang="en",
                    model_type="roberta-large",
                    batch_size=16,
                    verbose=False,
                )
                for rank, idx in enumerate(ss_indices):
                    answers[idx]["bertscore_ss"] = {
                        "precision": round(P[rank].item(),  4),
                        "recall":    round(R[rank].item(),  4),
                        "f1":        round(F1[rank].item(), 4),
                    }
            except Exception:
                for idx in ss_indices:
                    answers[idx]["bertscore_ss"] = None

        # Mark questions that had no sanitized_input as None
        for i, ans in enumerate(answers):
            if need_ss and "bertscore_ss" not in ans:
                ans["bertscore_ss"] = None

    # ── Collect Presidio pairs ────────────────────────────────────────
    if need_presidio:
        prs_indices    = []
        prs_candidates = []   # presidio_sanitized_input values
        prs_references = []   # original questions
        null_count     = 0

        for i, (ans, q_entry) in enumerate(zip(answers, questions)):
            original           = q_entry.get("input", "")
            presidio_sanitized = ans.get("presidio_sanitized_input")

            if presidio_sanitized is None:
                null_count += 1
                continue
            if original and presidio_sanitized:
                prs_indices.append(i)
                prs_candidates.append(presidio_sanitized)
                prs_references.append(original)

        # Warn if presidio_sanitized_input was missing for some questions
        if null_count > 0 and progress_cb:
            progress_cb(-1, total, "", "bertscore_warn", float(null_count))

        if prs_indices:
            try:
                P, R, F1 = _bs_score(
                    cands=prs_candidates,
                    refs=prs_references,
                    lang="en",
                    model_type="roberta-large",
                    batch_size=16,
                    verbose=False,
                )
                for rank, idx in enumerate(prs_indices):
                    answers[idx]["bertscore_presidio"] = {
                        "precision": round(P[rank].item(),  4),
                        "recall":    round(R[rank].item(),  4),
                        "f1":        round(F1[rank].item(), 4),
                    }
            except Exception:
                for idx in prs_indices:
                    answers[idx]["bertscore_presidio"] = None

        # Mark remaining questions as None
        for i, ans in enumerate(answers):
            if need_presidio and "bertscore_presidio" not in ans:
                ans["bertscore_presidio"] = None

    elapsed = _time.time() - t0
    if progress_cb:
        progress_cb(-1, total, "", "bertscore_done", elapsed)
