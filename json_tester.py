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
]

DEFAULT_FIELDS: Dict[str, bool] = {key: True for key, _ in OUTPUT_FIELDS}
DEFAULT_FIELDS["presidio_sanitized_input"] = False


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
    confirmed, _ = run_cascade(question)
    confirmed = deduplicate(confirmed)

    # Per-stage breakdowns (filter by source set by each detector)
    if fields.get("pattern_scan_pii"):
        answer["pattern_scan_pii"] = [
            e.text for e in confirmed if e.source == "pattern"
        ]

    if fields.get("entity_trace_pii"):
        answer["entity_trace_pii"] = [
            e.text for e in confirmed if e.source == "ner"
        ]

    if fields.get("context_guard_pii"):
        answer["context_guard_pii"] = [
            e.text for e in confirmed if e.source == "slm"
        ]

    if fields.get("confirmed_pii"):
        answer["confirmed_pii"] = [e.text for e in confirmed]

    if fields.get("pii_detail"):
        answer["pii_detail"] = {
            e.text: {
                "type":   e.type,
                "score":  round(e.score, 4),
                "source": e.source,
            }
            for e in confirmed
        }

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

    # ── Presidio sanitized input ──────────────────────────────────────────────
    if fields.get("presidio_sanitized_input"):
        try:
            from presidio.detect import detect as _presidio_detect
            from presidio.redact import redact as _presidio_redact

            p_entities = _presidio_detect(question)

            if p_entities is None:
                # Presidio unavailable — store null so BERTScore
                # code can skip this entry rather than use bad data
                answer["presidio_sanitized_input"] = None
            elif not p_entities:
                # Presidio found nothing — original text is the
                # "sanitized" version (no redaction applied)
                answer["presidio_sanitized_input"] = question
            else:
                answer["presidio_sanitized_input"] = _presidio_redact(
                    question, p_entities
                )
        except Exception:
            answer["presidio_sanitized_input"] = None

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
    if fields.get("presidio_sanitized_input"):
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

    _flush()
    return str(out_path)
