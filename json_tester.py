# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
json_tester.py — Batch JSON testing for SurrogateShield.

Input:  experiment/<name>.json              — list of {"input": "..."} objects
Output: experiment/<name>_answers.json      — one result object per question,
                                              same order and length as the input
        experiment/<name>_answers.meta.json — run metadata: generator commit,
                                              timestamp, base seed, Presidio
                                              baseline config, BERTScore setup

Guarantees (TECHNICAL_AUDIT I22, I31, A8):
  * The text sent to the provider is exactly ``sanitized_input`` in every
    address mode; ``_send_checked`` raises otherwise.
  * ``stage_timings_ms`` come from the real detection pass, not a throwaway
    pre-pass.
  * Surrogates are generated with a per-question seed (base seed + index), so
    a resumed run regenerates the same row.
  * Results are written atomically (temp file + fsync + replace) every 25
    questions, on Ctrl-C and at the end. Resume re-processes error rows.
  * BERTScore uses rescale_with_baseline=True; a failure is stored per row in
    ``bertscore_error``, never silently as None.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from surrogateshield.core.errors import DetectorUnavailable

EXPERIMENT_DIR = Path(__file__).parent / "experiment"
FLUSH_EVERY = 25
DEFAULT_SEED = 20260101

# ── Output field registry ─────────────────────────────────────────────────────
# (key, display label)
OUTPUT_FIELDS: List[Tuple[str, str]] = [
    ("question",          "Question asked"),
    ("pattern_scan_pii",  "PatternScan PIIs       (regex stage)"),
    ("entity_trace_pii",  "EntityTrace PIIs        (spaCy NER stage)"),
    ("context_guard_pii", "ContextGuard PIIs       (distilbert stage)"),
    ("confirmed_pii",     "Confirmed PIIs          (final combined list)"),
    ("pii_detail",        "PII detail              (type, score, source per entity)"),
    ("pii_spans",         "PII spans               (offsets, type, replaced — scored by the evaluator)"),
    ("quasi_id_risks",    "Quasi-ID risks          (combination re-identification risks)"),
    ("surrogate_map",     "Surrogate map           (original → replacement)"),
    ("sanitized_input",   "Sanitized input         (text sent to LLM)"),
    ("llm_response",      "LLM response            (raw text received)"),
    ("final_output",      "Final output            (llm_response with surrogates restored to real values)"),
    ("stage_timings_ms",  "Stage timings (ms)      (PatternScan / EntityTrace / ContextGuard / surrogate gen / LLM)"),
    ("presidio_sanitized_input",
     "Presidio sanitized  (Presidio [TYPE] redaction — baseline)"),
    ("presidio_found_piis",
     "Presidio found PIIs  (raw detected entities — type, value, offsets, score)"),
    ("recognized_not_replaced",
     "Recognized not replaced  (detected as PII but intentionally skipped)"),
    ("presidio_llm_response",
     "Presidio LLM response  (Presidio-redacted text sent to the LLM — extra API call)"),
    ("clean_llm_response",
     "Clean LLM response  (ORIGINAL text sent to the LLM — synthetic data only, extra API call)"),
    ("bertscore_ss",
     "BERTScore SS       (rescaled; input fidelity + output vs clean answer when available)"),
    ("bertscore_presidio",
     "BERTScore Presidio (rescaled; same pairs for the Presidio arm)"),
]

DEFAULT_FIELDS: Dict[str, bool] = {key: True for key, _ in OUTPUT_FIELDS}
DEFAULT_FIELDS["presidio_sanitized_input"] = False
DEFAULT_FIELDS["presidio_found_piis"] = False
DEFAULT_FIELDS["presidio_llm_response"] = False
DEFAULT_FIELDS["clean_llm_response"] = False
DEFAULT_FIELDS["bertscore_ss"] = False
DEFAULT_FIELDS["bertscore_presidio"] = False

_PRESIDIO_FIELDS = ("presidio_sanitized_input", "presidio_found_piis", "presidio_llm_response")


class SendMismatch(RuntimeError):
    """The text about to be sent differs from the recorded sanitized text."""


# ── Shared send preparation (runner + offline protection check) ──────────────

@dataclass
class Prepared:
    """Everything the runner decides before a provider call."""

    question: str
    is_service_query: bool
    address_mode: str
    confirmed: list
    skipped: list
    skip_reason: str
    surrogate_map: Dict[str, str]
    edits: List[Tuple[int, int, str, str]]
    sanitized: str
    timings: Dict[str, float] = field(default_factory=dict)
    qi_matches: list = field(default_factory=list)
    # per-entity reason (keyed by (start, end)) that overrides skip_reason
    skip_reasons: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def reason_for(self, ent) -> str:
        return self.skip_reasons.get((ent.start, ent.end), self.skip_reason)

    def spans(self) -> List[dict]:
        """Every predicted span in question coordinates.

        ``replaced`` is True when the span's text was substituted before
        sending; recognised-not-replaced spans carry ``replaced: False`` and
        their reason.
        """
        by_text = {e.text: e for e in self.confirmed}
        out: List[dict] = []
        for start, end, original, surrogate in self.edits:
            ent = by_text.get(original)
            out.append({
                "text": original,
                "start": start,
                "end": end,
                "type": ent.type if ent is not None else "unknown",
                "source": ent.source if ent is not None else "repeat",
                "replaced": surrogate != original,
            })
            if getattr(ent, "view", None):
                out[-1]["view"] = ent.view          # found on a canonical view
        claimed = [(s["start"], s["end"]) for s in out]
        for ent in list(self.confirmed) + list(self.skipped):
            if any(not (ent.end <= s or ent.start >= e) for s, e in claimed):
                continue
            row = {
                "text": ent.text,
                "start": ent.start,
                "end": ent.end,
                "type": ent.type,
                "source": ent.source,
                "replaced": False,
            }
            if ent in self.skipped:
                row["reason"] = self.reason_for(ent)
            else:
                row["reason"] = "no_surrogate"
            out.append(row)
            claimed.append((ent.start, ent.end))
        out.sort(key=lambda r: (r["start"], r["end"]))
        return out


def prepare_send(question: str, mimic, cascade_options: Optional[dict] = None) -> Prepared:
    """Detect, generate surrogates and build the exact text to send.

    This is the single code path used by the runner and by
    ``offline_eval.py --protection`` / ``--ablation``. *cascade_options* are
    passed to ``run_cascade`` (stage switches for the ablation).
    """
    from config import (
        ADDRESS_MODE,
        ADDRESS_SHIFT_RANGE,
        SERVICE_QUERY_DETECTION_ENABLED,
    )
    from detection.logic import deduplicate, run_cascade
    from detection.service_query import resolve as resolve_service
    from util import plan_substitutions, splice

    # Same address-mode resolution as pipeline.process_turn.
    is_svc, address_mode = resolve_service(question, ADDRESS_MODE, SERVICE_QUERY_DETECTION_ENABLED)

    timings: Dict[str, float] = {}
    confirmed, _ = run_cascade(question, skip_location_entities=is_svc, timings=timings,
                               **(cascade_options or {}))
    confirmed = deduplicate(confirmed)
    skipped = list(getattr(confirmed, "_skipped_entities", []))

    t = time.perf_counter()
    surrogate_map = (
        mimic.generate_all(
            confirmed,
            address_mode=address_mode,
            address_shift_range=ADDRESS_SHIFT_RANGE,
            text=question,
        )
        if confirmed
        else {}
    )
    timings["surrogate_gen_ms"] = round((time.perf_counter() - t) * 1000, 3)

    edits = plan_substitutions(question, confirmed, surrogate_map)
    return Prepared(
        question=question,
        is_service_query=is_svc,
        address_mode=address_mode,
        confirmed=list(confirmed),
        skipped=skipped,
        skip_reason="service_query_location_suppressed" if is_svc else "topical_geo_filtered",
        surrogate_map=surrogate_map,
        edits=edits,
        sanitized=splice(question, edits),
        timings=timings,
        qi_matches=list(getattr(confirmed, "_qi_matches", [])),
        skip_reasons=dict(getattr(confirmed, "_skip_reasons", {})),
    )


def _send(chat, text: str) -> str:
    return chat._send_to_api([{"role": "user", "content": text}])


def _send_checked(chat, text: str, prep: Prepared) -> str:
    """Send *text* only if it is the planned sanitisation of the question (I22).

    The expected text is rebuilt from the original question and the planned
    edits, independently of ``prep.sanitized``; every replaced span must hold
    its surrogate (a shifted address byte-for-byte) and no replaced original
    may survive at its planned position.
    """
    from util import splice

    expected = splice(prep.question, prep.edits)
    if text != expected or text != prep.sanitized:
        raise SendMismatch("runner tried to send text that differs from sanitized_input")
    shift = 0
    for start, end, original, surrogate in sorted(prep.edits):
        s = start + shift
        if text[s:s + len(surrogate)] != surrogate:
            raise SendMismatch(f"surrogate missing at offset {s}")
        shift += len(surrogate) - (end - start)
    return _send(chat, text)


# ── Core per-question processor ───────────────────────────────────────────────

def _process_one(
    question: str,
    chat,
    fields: Dict[str, bool],
    seed: Optional[int] = None,
) -> dict:
    """Run detection, surrogate generation and the optional LLM calls for one question."""
    from generation.logic import MimicGen

    t_total = time.perf_counter()
    answer: dict = {}

    if fields.get("question"):
        answer["question"] = question

    prep = prepare_send(question, MimicGen(seed=seed))
    confirmed, skipped = prep.confirmed, prep.skipped
    all_detected = confirmed + skipped

    if fields.get("pattern_scan_pii"):
        answer["pattern_scan_pii"] = [e.text for e in all_detected if e.source == "pattern"]
    if fields.get("entity_trace_pii"):
        answer["entity_trace_pii"] = [e.text for e in all_detected if e.source == "ner"]
    if fields.get("context_guard_pii"):
        answer["context_guard_pii"] = [e.text for e in all_detected if e.source == "slm"]
    if fields.get("confirmed_pii"):
        answer["confirmed_pii"] = [e.text for e in all_detected]

    if fields.get("pii_detail"):
        detail: dict = {
            e.text: {"type": e.type, "score": round(e.score, 4), "source": e.source}
            for e in confirmed
        }
        for e in skipped:
            detail[e.text] = {
                "type": e.type,
                "score": round(e.score, 4),
                "source": e.source,
                "surrogate_status": "skipped",
                "skip_reason": prep.reason_for(e),
            }
        answer["pii_detail"] = detail

    if fields.get("pii_spans"):
        answer["pii_spans"] = prep.spans()

    if fields.get("quasi_id_risks"):
        answer["quasi_id_risks"] = [
            {
                "combination": m.combination_name,
                "matched_fields": m.matched_fields,
                "risk_level": m.risk_level,
                "all_fields_matched": m.all_fields_matched,
                "reference": m.reference if m.all_fields_matched else m.partial_reference,
            }
            for m in prep.qi_matches
        ]

    if fields.get("surrogate_map"):
        answer["surrogate_map"] = prep.surrogate_map
    if fields.get("sanitized_input"):
        answer["sanitized_input"] = prep.sanitized
    answer["address_mode"] = prep.address_mode

    if fields.get("recognized_not_replaced"):
        answer["recognized_not_replaced"] = [
            {"value": e.text, "type": e.type, "start": e.start, "end": e.end,
             "reason": prep.reason_for(e)}
            for e in skipped
        ]

    timings = dict(prep.timings)

    # ── LLM call (single-turn, no conversation history) ───────────────────────
    timings["llm_call_ms"] = 0.0
    if fields.get("llm_response"):
        t = time.perf_counter()
        raw = _send_checked(chat, prep.sanitized, prep)
        timings["llm_call_ms"] = round((time.perf_counter() - t) * 1000, 3)
        answer["llm_response"] = raw

    if fields.get("final_output"):
        raw_response = answer.get("llm_response")
        if raw_response is not None and prep.surrogate_map:
            from reconstruction.logic import ResolvePass
            shadow_map = {sur: orig for orig, sur in prep.surrogate_map.items()}
            answer["final_output"] = ResolvePass().resolve(raw_response, shadow_map)
        else:
            answer["final_output"] = raw_response

    if fields.get("stage_timings_ms"):
        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)
        answer["stage_timings_ms"] = timings

    # ── Presidio baseline (exceptions propagate → error row) ──────────────────
    if any(fields.get(k) for k in _PRESIDIO_FIELDS):
        from presidio.detect import detect as _presidio_detect
        from presidio.redact import redact as _presidio_redact

        p_entities = _presidio_detect(question)
        if p_entities is None:
            from presidio.engine import unavailability_reason
            raise RuntimeError(f"Presidio unavailable: {unavailability_reason()}")
        p_sanitized = _presidio_redact(question, p_entities) if p_entities else question
        if fields.get("presidio_sanitized_input"):
            answer["presidio_sanitized_input"] = p_sanitized
        if fields.get("presidio_found_piis"):
            answer["presidio_found_piis"] = [
                {"value": e.text, "type": e.entity_type, "start": e.start,
                 "end": e.end, "score": e.score}
                for e in p_entities
            ]
        if fields.get("presidio_llm_response"):
            answer["presidio_llm_response"] = _send(chat, p_sanitized)

    if fields.get("clean_llm_response"):
        answer["clean_llm_response"] = _send(chat, question)

    return answer


# ── Run metadata, persistence ─────────────────────────────────────────────────

def _git_commit() -> dict:
    root = Path(__file__).parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _load_resumable(out_path: Path, questions: List[dict]) -> List[dict]:
    """Existing answers, validated against the input. Raises on a stale file."""
    if not out_path.exists():
        return []
    answers = json.loads(out_path.read_text(encoding="utf-8"))
    if not isinstance(answers, list) or len(answers) > len(questions):
        raise ValueError(
            f"{out_path.name} does not match the input file (length {len(answers)} "
            f"> {len(questions)}); move it away to start a fresh run"
        )
    for i, ans in enumerate(answers):
        q = questions[i].get("input", "")
        if ans.get("question", q) != q:
            raise ValueError(
                f"{out_path.name} row {i} answers a different question; "
                "move it away to start a fresh run"
            )
    return answers


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    filename: str,
    fields: Dict[str, bool],
    progress_cb: Optional[Callable[[int, int, str, str, float], None]] = None,
    chat=None,
    seed: int = DEFAULT_SEED,
    experiment_dir: Optional[Path] = None,
) -> str:
    """
    Process every question in experiment/<filename> and write results.

    Args:
        filename:    Filename inside the experiment directory.
        fields:      Dict of field_key → bool controlling output columns.
        progress_cb: Called as (index, total, question, status, elapsed_s);
                     status is "running", "ok", "error" or a sentinel with
                     index -1 for BERTScore progress.
        chat:        Provider client (tests inject a fake). Created on demand.
        seed:        Base seed; question i uses ``seed + i``.

    Returns:
        Path to the answers file.
    """
    exp_dir = experiment_dir or EXPERIMENT_DIR
    exp_dir.mkdir(parents=True, exist_ok=True)

    in_path = exp_dir / filename
    out_path = exp_dir / f"{in_path.stem}_answers.json"
    meta_path = exp_dir / f"{in_path.stem}_answers.meta.json"

    questions: List[dict] = json.loads(in_path.read_text(encoding="utf-8"))
    total = len(questions)

    from settings_manager import load_settings
    settings = load_settings()
    if not settings.get("presidio_comparison", False):
        # The setting gates every Presidio field, not just one (I31).
        fields = {**fields, **{k: False for k in _PRESIDIO_FIELDS}}
    fields = {**fields}
    if fields.get("bertscore_presidio") and not fields.get("presidio_sanitized_input"):
        fields["bertscore_presidio"] = False

    answers = _load_resumable(out_path, questions)

    needs_chat = any(fields.get(k) for k in
                     ("llm_response", "presidio_llm_response", "clean_llm_response"))
    if needs_chat and chat is None:
        from chatbot.chat import ClaudeChat
        chat = ClaudeChat()

    presidio_meta = None
    if any(fields.get(k) for k in _PRESIDIO_FIELDS):
        from presidio.engine import baseline_config, get_analyzer, unavailability_reason
        if get_analyzer() is None:
            raise RuntimeError(f"Presidio fields requested but unavailable: {unavailability_reason()}")
        presidio_meta = baseline_config()

    meta = {
        "generator": _git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": filename,
        "base_seed": seed,
        "fields": fields,
        "provider": settings.get("llm_provider"),
        "presidio": presidio_meta,
    }

    def _flush() -> None:
        _atomic_write_json(out_path, answers)

    todo = [i for i in range(total) if i >= len(answers) or "error" in answers[i]]
    try:
        for n, i in enumerate(todo, 1):
            question = questions[i].get("input", "")
            t0 = time.perf_counter()
            if progress_cb:
                progress_cb(i, total, question, "running", 0.0)
            try:
                result = _process_one(question, chat, fields, seed=seed + i)
                status = "ok"
            except (SendMismatch, DetectorUnavailable):
                raise  # a missing model is not a per-row error (audit I17)
            except Exception as exc:  # recorded per row, counted by the evaluator
                result = {"question": question, "error": str(exc),
                          "error_type": type(exc).__name__}
                status = "error"
            if i < len(answers):
                answers[i] = result
            else:
                answers.append(result)
            if n % FLUSH_EVERY == 0:
                _flush()
            if progress_cb:
                progress_cb(i, total, question, status, time.perf_counter() - t0)
    except KeyboardInterrupt:
        _flush()
        raise
    _flush()

    if fields.get("bertscore_ss") or fields.get("bertscore_presidio"):
        meta["bertscore"] = _run_bertscore_batch(
            answers=answers,
            questions=questions,
            need_ss=fields.get("bertscore_ss", False),
            need_presidio=fields.get("bertscore_presidio", False),
            progress_cb=progress_cb,
            total=total,
        )
        _flush()

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_write_json(meta_path, meta)
    return str(out_path)


# ── BERTScore ─────────────────────────────────────────────────────────────────

BERTSCORE_MODEL = "roberta-large"

# (answer field, candidate field, reference field or None for the question)
_BERTSCORE_PAIRS = {
    "ss": [
        ("bertscore_ss", "sanitized_input", None),
        ("bertscore_ss_output", "final_output", "clean_llm_response"),
    ],
    "presidio": [
        ("bertscore_presidio", "presidio_sanitized_input", None),
        ("bertscore_presidio_output", "presidio_llm_response", "clean_llm_response"),
    ],
}


def _run_bertscore_batch(
    answers: list,
    questions: list,
    need_ss: bool,
    need_presidio: bool,
    progress_cb,
    total: int,
    scorer: Optional[Callable] = None,
) -> dict:
    """
    Compute rescaled BERTScore for every available pair, in place.

    Pairs:
      * ``bertscore_ss`` / ``bertscore_presidio`` — sanitized input vs the
        original question (input fidelity).
      * ``bertscore_ss_output`` / ``bertscore_presidio_output`` — the arm's
        answer vs the answer to the unsanitized question (utility), only when
        ``clean_llm_response`` was collected.

    A row whose pair is missing gets None; a scoring failure stores the
    message in ``bertscore_error``. Returns the configuration for the run
    metadata.
    """
    config = {
        "model": BERTSCORE_MODEL,
        "rescale_with_baseline": True,
        "lang": "en",
        "pairs": {k: [(f, c, r or "question") for f, c, r in v]
                  for k, v in _BERTSCORE_PAIRS.items()},
    }
    if scorer is None:
        try:
            import logging
            import warnings

            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
            from bert_score import score as _bs_score
        except ImportError:
            if progress_cb:
                progress_cb(-1, total, "", "bertscore_skipped", 0.0)
            for ans in answers:
                ans["bertscore_error"] = "bert-score not installed"
            config["available"] = False
            return config

        def scorer(cands, refs):
            return _bs_score(
                cands=cands, refs=refs, lang="en", model_type=BERTSCORE_MODEL,
                rescale_with_baseline=True, batch_size=16, verbose=False,
            )

    if progress_cb:
        progress_cb(-1, total, "", "bertscore_start", 0.0)
    t0 = time.perf_counter()

    arms = (["ss"] if need_ss else []) + (["presidio"] if need_presidio else [])
    for arm in arms:
        for out_field, cand_field, ref_field in _BERTSCORE_PAIRS[arm]:
            idx, cands, refs = [], [], []
            for i, (ans, q) in enumerate(zip(answers, questions)):
                cand = ans.get(cand_field)
                ref = q.get("input", "") if ref_field is None else ans.get(ref_field)
                if "error" in ans or not cand or not ref:
                    ans[out_field] = None
                    continue
                idx.append(i)
                cands.append(cand)
                refs.append(ref)
            if not idx:
                continue
            try:
                P, R, F1 = scorer(cands, refs)
            except Exception as exc:  # stored, not swallowed
                for i in idx:
                    answers[i][out_field] = None
                    answers[i]["bertscore_error"] = f"{out_field}: {type(exc).__name__}: {exc}"
                continue
            for rank, i in enumerate(idx):
                answers[i][out_field] = {
                    "precision": round(float(P[rank]), 4),
                    "recall": round(float(R[rank]), 4),
                    "f1": round(float(F1[rank]), 4),
                    "rescaled": True,
                }

    if progress_cb:
        progress_cb(-1, total, "", "bertscore_done", time.perf_counter() - t0)
    config["available"] = True
    return config
