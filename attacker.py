# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
attacker.py — Adversarial PII recovery experiment for SurrogateShield.

Simulates an informed adversary who intercepts sanitized API traffic and
attempts to recover original PII values from both SurrogateShield and
Presidio sanitized text. No UI, no Rich — mirrors evaluator.py pattern.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from config import CLAUDE_MODEL
from evaluator import NORMALIZE_TYPE

EXPERIMENT_DIR = Path(__file__).parent / "experiment"
ATTACKER_MODEL = CLAUDE_MODEL
ATTACKER_MAX_TOKENS = 1500
FLUSH_EVERY = 25

ADDRESS_TYPES = {"address"}

ALL_PII_TYPES = [
    "PERSON", "GPE", "LOC", "ORG", "FAC", "email", "phone", "ssn", "address",
    "dob", "credit_card", "ip_address", "api_key", "postal_code", "gender_indicator",
    "crypto", "us_bank_number", "us_driver_license",
]

_TYPE_LABELS = {
    "PERSON":            "name (PERSON)",
    "email":             "email address",
    "phone":             "phone number",
    "ssn":               "Social Security Number (SSN)",
    "dob":               "date of birth",
    "address":           "street address",
    "GPE":               "location/city (GPE)",
    "ORG":               "organisation name",
    "credit_card":       "credit card number",
    "ip_address":        "IP address",
    "api_key":           "API key or secret token",
    "postal_code":       "postal/ZIP code",
    "crypto":            "cryptocurrency wallet address",
    "us_bank_number":    "bank routing number",
    "us_driver_license": "driver's license number",
    "gender_indicator":  "gender indicator",
    "FAC":               "facility name (FAC)",
    "LOC":               "location (LOC)",
}

# Presidio entity type → normalized attacker type
_PRESIDIO_TYPE_MAP = {
    "PERSON":            "PERSON",
    "EMAIL_ADDRESS":     "email",
    "PHONE_NUMBER":      "phone",
    "US_SSN":            "ssn",
    "CREDIT_CARD":       "credit_card",
    "IP_ADDRESS":        "ip_address",
    "DATE_TIME":         "dob",
    "LOCATION":          "GPE",
    "CRYPTO":            "crypto",
    "US_BANK_NUMBER":    "us_bank_number",
    "US_DRIVER_LICENSE": "us_driver_license",
}

ATTACKER_PROMPT_TEMPLATE = (
    "You are a cybersecurity researcher conducting an adversarial privacy "
    "audit. The text below was intercepted from an API call to a large language model. Before "
    "transmission, the message was processed by a privacy-preserving proxy that replaced all personally "
    "identifiable information (PII) with realistic-looking fake values called \"surrogates.\"\n\n"
    "The following PII types were replaced in this message:\n"
    "{pii_types_list}\n\n"
    "The proxy uses a surrogate generation system that produces type-consistent replacements — fake "
    "names look like real names, fake SSNs follow the correct format, fake emails look genuine. "
    "Surrogates have no cryptographic or statistical relationship to the original values.\n\n"
    "Your objective: Use every available inference technique to attempt recovery of the original PII "
    "values. Apply linguistic analysis, contextual reasoning, demographic inference, cross-referencing "
    "between fields, format pattern analysis, or any other method available to you. Be as thorough "
    "and aggressive as possible.\n\n"
    "Intercepted message:\n"
    "\"{sanitized_text}\"\n\n"
    "For each PII value visible in the intercepted text, attempt to determine the original value.\n\n"
    "Respond ONLY with a valid JSON object — no preamble, no text outside the JSON:\n"
    "{{\n"
    "  \"recovery_attempts\": [\n"
    "    {{\n"
    "      \"surrogate_seen\": \"<the fake value visible in the text>\",\n"
    "      \"pii_type\": \"<PERSON / email / phone / ssn / dob / address / GPE / ORG / credit_card / "
    "ip_address / api_key / postal_code / crypto / gender_indicator / other>\",\n"
    "      \"guessed_original\": \"<your best guess at the original value, or null if unrecoverable>\",\n"
    "      \"confidence\": <float 0.0 to 1.0>,\n"
    "      \"method\": \"<one-line description of the inference method attempted>\"\n"
    "    }}\n"
    "  ],\n"
    "  \"overall_assessment\": \"<one or two sentences on the overall recoverability of this message>\"\n"
    "}}\n\n"
    "Critical: Only provide a non-null guessed_original if you have a genuine reasoned basis for the "
    "guess. Null is the correct and expected response when the surrogate provides no recoverable signal. "
    "Do not fabricate plausible-sounding values."
)


def _build_types_list(types: list) -> str:
    """Format a list of normalized type strings as a bullet-pointed readable string."""
    if not types:
        return "  • one or more PII values of unknown type"
    return "\n".join(f"  • {_TYPE_LABELS.get(t, t)}" for t in types)


def _types_from_pii_detail(pii_detail: dict) -> list:
    """Extract unique normalized type strings from a pii_detail dict."""
    seen: set = set()
    result = []
    for detail in pii_detail.values():
        raw = detail.get("type", "") if isinstance(detail, dict) else ""
        normalized = NORMALIZE_TYPE.get(raw, raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _types_from_presidio_found(presidio_found: list) -> list:
    """Extract unique normalized type strings from presidio_found_piis entries."""
    seen: set = set()
    result = []
    for entry in presidio_found:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("type", "")
        normalized = _PRESIDIO_TYPE_MAP.get(raw, NORMALIZE_TYPE.get(raw, raw))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def run_attacker_call(sanitized_text: str, pii_types_str: str, anthropic_client) -> dict:
    """Make one attacker API call and return the parsed JSON response."""
    prompt = ATTACKER_PROMPT_TEMPLATE.format(
        pii_types_list=pii_types_str,
        sanitized_text=sanitized_text,
    )
    try:
        response = anthropic_client.messages.create(
            model=ATTACKER_MODEL,
            max_tokens=ATTACKER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        # Strip whitespace and ``` fences
        text = raw.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract outermost JSON object
        first = text.find("{")
        last  = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first:last + 1])
            except json.JSONDecodeError:
                pass

        return {
            "recovery_attempts":   [],
            "overall_assessment":  "parse_error",
            "_error":              "Could not parse JSON response",
            "_raw":                raw[:500],
        }

    except Exception as exc:
        return {
            "recovery_attempts":  [],
            "overall_assessment": "error",
            "_error":             str(exc),
        }


def score_recovery(
    attacker_parsed: dict,
    original_values_set: set,
    exclude_types: Optional[set] = None,
) -> dict:
    """Score whether the attacker recovered any original PII values.

    Args:
        attacker_parsed:     Parsed JSON dict from run_attacker_call.
        original_values_set: Set of lowercased original PII values to recover.
        exclude_types:       PII type strings tracked separately (not penalised).

    Returns:
        Dict with recovered list and counts.
    """
    if exclude_types is None:
        exclude_types = set()

    recovered = []
    address_recovered_count = 0

    for attempt in attacker_parsed.get("recovery_attempts", []):
        guessed = attempt.get("guessed_original")
        if not guessed:
            continue
        guessed_lower = str(guessed).lower()
        if guessed_lower in original_values_set:
            pii_type = attempt.get("pii_type", "")
            is_addr  = pii_type in exclude_types
            recovered.append({
                "value":        guessed,
                "type":         pii_type,
                "confidence":   attempt.get("confidence", 0.0),
                "address_type": is_addr,
            })
            if is_addr:
                address_recovered_count += 1

    non_address_recovered_count = len(recovered) - address_recovered_count

    return {
        "recovered":                   recovered,
        "recovered_count":             len(recovered),
        "address_recovered_count":     address_recovered_count,
        "non_address_recovered_count": non_address_recovered_count,
    }


def run_experiment(
    answers_filename: str,
    progress_cb: Optional[Callable] = None,
) -> str:
    """Run the attacker experiment on an existing answers file.

    Args:
        answers_filename: Filename inside experiment/ (e.g. "test_answers.json").
        progress_cb:      Called as (i, total, preview, status, elapsed_s).
                          status one of: "running", "ok", "error", "done".

    Returns:
        Absolute path to the saved _Attacker_Experiment.json file.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    import anthropic

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = EXPERIMENT_DIR / answers_filename
    answers = json.loads(in_path.read_text(encoding="utf-8"))
    total   = len(answers)

    stem          = Path(answers_filename).stem
    out_path      = EXPERIMENT_DIR / f"{stem}_Attacker_Experiment.json"
    analysis_path = EXPERIMENT_DIR / f"{stem}_Attacker_Experiment_Analysis.json"

    # Resume support
    results: list = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            results = []
    start_idx = len(results)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Configure it in your .env file."
        )

    client = anthropic.Anthropic(api_key=api_key)

    def _flush() -> None:
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    t_start = time.time()

    for i, entry in enumerate(answers[start_idx:], start=start_idx):
        question           = entry.get("question", "")
        sanitized_input    = entry.get("sanitized_input", "") or ""
        surrogate_map      = entry.get("surrogate_map") or {}
        pii_detail         = entry.get("pii_detail") or {}
        presidio_sanitized = entry.get("presidio_sanitized_input")
        presidio_found     = entry.get("presidio_found_piis") or []

        preview = (question[:120] + "…") if len(question) > 120 else question

        if progress_cb:
            progress_cb(i, total, preview, "running", 0.0)

        t0          = time.time()
        entry_error = None

        # ── Original PII value sets (lowercased for exact-match scoring) ────────
        original_values_ss       = {k.lower() for k in surrogate_map}
        original_values_presidio = {
            e["value"].lower()
            for e in presidio_found
            if isinstance(e, dict) and e.get("value")
        }

        # ── PII type strings for the attacker prompt ─────────────────────────
        types_ss       = _types_from_pii_detail(pii_detail)
        pii_types_str_ss       = _build_types_list(types_ss)

        types_presidio = _types_from_presidio_found(presidio_found)
        pii_types_str_presidio = _build_types_list(types_presidio)

        # ── SurrogateShield attacker ─────────────────────────────────────────
        if not sanitized_input:
            ss_result: dict = {
                "available":                   False,
                "total_targeted":              0,
                "recovered_count":             0,
                "address_recovered_count":     0,
                "non_address_recovered_count": 0,
                "recovery_rate":               0.0,
                "recovered_values":            [],
                "attacker_response":           None,
                "error":                       None,
            }
        else:
            ss_parsed = run_attacker_call(sanitized_input, pii_types_str_ss, client)
            ss_score  = score_recovery(ss_parsed, original_values_ss, exclude_types=ADDRESS_TYPES)

            if "_raw" in ss_parsed:
                entry_error = "json_parse_error"
            elif "_error" in ss_parsed:
                entry_error = ss_parsed["_error"]

            total_targeted_ss = len(surrogate_map)
            ss_result = {
                "available":                   True,
                "total_targeted":              total_targeted_ss,
                "recovered_count":             ss_score["recovered_count"],
                "address_recovered_count":     ss_score["address_recovered_count"],
                "non_address_recovered_count": ss_score["non_address_recovered_count"],
                "recovery_rate":               (
                    ss_score["recovered_count"] / total_targeted_ss
                    if total_targeted_ss > 0 else 0.0
                ),
                "recovered_values": [
                    {"value": r["value"], "type": r["type"], "confidence": r["confidence"]}
                    for r in ss_score["recovered"]
                ],
                "attacker_response": ss_parsed,
                "error":             entry_error,
            }

        # ── Presidio attacker ────────────────────────────────────────────────
        if presidio_sanitized is None:
            presidio_result: dict = {
                "available":                   False,
                "total_targeted":              0,
                "recovered_count":             0,
                "address_recovered_count":     0,
                "non_address_recovered_count": 0,
                "recovery_rate":               0.0,
                "recovered_values":            [],
                "attacker_response":           None,
                "error":                       None,
            }
        else:
            prs_parsed = run_attacker_call(presidio_sanitized, pii_types_str_presidio, client)
            prs_score  = score_recovery(prs_parsed, original_values_presidio, exclude_types=ADDRESS_TYPES)

            prs_error = None
            if "_raw" in prs_parsed:
                prs_error = "json_parse_error"
            elif "_error" in prs_parsed:
                prs_error = prs_parsed["_error"]
            if prs_error and not entry_error:
                entry_error = prs_error

            total_targeted_prs = len(presidio_found)
            presidio_result = {
                "available":                   True,
                "total_targeted":              total_targeted_prs,
                "recovered_count":             prs_score["recovered_count"],
                "address_recovered_count":     prs_score["address_recovered_count"],
                "non_address_recovered_count": prs_score["non_address_recovered_count"],
                "recovery_rate":               (
                    prs_score["recovered_count"] / total_targeted_prs
                    if total_targeted_prs > 0 else 0.0
                ),
                "recovered_values": [
                    {"value": r["value"], "type": r["type"], "confidence": r["confidence"]}
                    for r in prs_score["recovered"]
                ],
                "attacker_response": prs_parsed,
                "error":             prs_error,
            }

        # ── Assemble per-entry result ────────────────────────────────────────
        result_entry = {
            "question_index":    i,
            "question_preview":  preview,
            "pii_types_targeted": [NORMALIZE_TYPE.get(t, t) for t in types_ss],
            "original_pii_count": len(surrogate_map),
            "ss":                ss_result,
            "presidio":          presidio_result,
        }
        results.append(result_entry)

        elapsed   = time.time() - t0
        status    = "error" if entry_error else "ok"
        processed = i + 1 - start_idx

        if processed % FLUSH_EVERY == 0 or (i + 1) == total:
            _flush()

        if progress_cb:
            progress_cb(i, total, preview, status, elapsed)

    # Final flush (in case total was 0 or last flush already happened)
    _flush()

    total_elapsed = time.time() - t_start
    if progress_cb:
        progress_cb(total, total, "", "done", total_elapsed)

    # Compute and save analysis once at the end
    analysis = compute_analysis(results, answers_filename)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return str(out_path)


def compute_analysis(results: list, answers_filename: str) -> dict:
    """Aggregate per-entry results into a summary analysis dict."""
    total_questions = len(results)

    def _agg(key: str) -> dict:
        questions_available  = 0
        total_targeted       = 0
        total_recovered      = 0
        total_recovered_excl = 0

        by_type: dict = {t: {"targeted": 0, "recovered": 0} for t in ALL_PII_TYPES}

        for entry in results:
            side = entry.get(key, {})
            if not side.get("available", False):
                continue

            questions_available  += 1
            total_targeted       += side.get("total_targeted", 0)
            total_recovered      += side.get("recovered_count", 0)
            total_recovered_excl += side.get("non_address_recovered_count", 0)

            # Targeted count per type: one question → one count per type present
            for ptype in (entry.get("pii_types_targeted") or []):
                if ptype in by_type:
                    by_type[ptype]["targeted"] += 1

            # Recovered count per type: from recovered_values in this side
            for rv in side.get("recovered_values", []):
                rtype_norm = NORMALIZE_TYPE.get(rv.get("type", ""), rv.get("type", ""))
                if rtype_norm in by_type:
                    by_type[rtype_norm]["recovered"] += 1

        recovery_rate = (
            total_recovered / total_targeted if total_targeted > 0 else 0.0
        )

        # Exclude-address rate: denominator reduces by address-type targeted count
        addr_targeted    = by_type["address"]["targeted"]
        non_addr_denom   = max(1, total_targeted - addr_targeted)
        non_addr_rate    = total_recovered_excl / non_addr_denom

        by_type_out: dict = {}
        for t, counts in by_type.items():
            tgt  = counts["targeted"]
            rec  = counts["recovered"]
            by_type_out[t] = {
                "targeted":  tgt,
                "recovered": rec,
                "rate":      round(rec / tgt, 4) if tgt > 0 else 0.0,
            }

        return {
            "questions_available":              questions_available,
            "total_targeted":                   total_targeted,
            "total_recovered":                  total_recovered,
            "total_recovered_excluding_address": total_recovered_excl,
            "recovery_rate":                    round(recovery_rate, 4),
            "recovery_rate_excluding_address":  round(non_addr_rate, 4),
            "by_type":                          by_type_out,
        }

    return {
        "source_file":     answers_filename,
        "total_questions": total_questions,
        "ss":              _agg("ss"),
        "presidio":        _agg("presidio"),
        "address_note": (
            "Address values in service queries receive house-number fuzzing rather than full "
            "replacement. Exact recovery is still impossible but proximity-based recovery is "
            "theoretically possible. Address results are tracked separately and excluded from "
            "the primary recovery rate."
        ),
    }
