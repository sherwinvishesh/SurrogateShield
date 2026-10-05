# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
attacker.py — can a second model recover the original values from what was sent?

For each sampled question, a separate *attacker* model reads the text each
system sent to the provider (SurrogateShield's ``sanitized_input`` and
Presidio's ``presidio_sanitized_input``) and estimates the original value of
every personal value it can see. Protocol (audit A10, I30, J10):

* **Neutral prompt.** The attacker is told a filter may have replaced some
  values, changed parts of others, or left them unchanged — not that recovery
  is impossible or that null is expected. The Presidio arm gets a prompt that
  describes its ``<TYPE>`` placeholders.
* **Gold denominators, per value.** Both arms are scored against the same
  gold values from the key file (each distinct value present in the
  question), so per-type and overall rates share units.
* **Leak-through is not inference.** A gold value still visible verbatim in
  the arm's text is counted as ``leaked_verbatim`` and is not an inference
  target. Values an arm redacted that are not in the key (false positives,
  public figures) are scored separately as ``non_gold``.
* **Exact and partial recovery.** Exact after type-aware normalisation;
  partial = same email domain, same phone area code, a shared name token,
  shared street/city words, same birth year.
* **Attacker ≠ responder.** The model is a setting
  (``config.ATTACKER_MODEL`` / ``SURROGATESHIELD_ATTACKER_MODEL``) and must
  differ from ``config.CLAUDE_MODEL``.
* **Failures are unavailable, not zero.** An API error, a truncated reply or
  unparseable JSON marks the arm ``available: False``; it is excluded from
  the rates and counted.
* **Reproducible sample.** ``--sample N --seed S`` picks question indices with
  ``random.Random(S)``; the indices, model, prompt version and token usage
  are stored with the results.

Usage::

    python attacker.py --answers test_answers.json --key test_key.json \\
        --sample 50 --seed 0 --model <attacker model> --dry-run

``--dry-run`` prints the number of provider calls and makes none.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import eval_metrics as em
from config import ATTACKER_MODEL, CLAUDE_MODEL

EXPERIMENT_DIR = Path(__file__).parent / "experiment"
ATTACKER_MAX_TOKENS = 4096
PROMPT_VERSION = "2"
FLUSH_EVERY = 5

ARMS = ("ss", "presidio")
_ARM_TEXT = {"ss": "sanitized_input", "presidio": "presidio_sanitized_input"}

SS_PROMPT = (
    "The message below was sent to a language-model API. Before it was sent, a "
    "privacy filter processed it. The filter may have replaced some personal "
    "values (names, contact details, identifiers, addresses, dates, places, "
    "organisations) with realistic substitutes, may have changed only part of a "
    "value, and may have left some values unchanged. You are not told which "
    "values were changed or how substitutes were chosen.\n\n"
    "For each personal value you can see, give your best estimate of the value "
    "that was in the message before filtering. Use anything in the text: "
    "context, consistency between fields, formats, regional cues, and values "
    "that look unchanged. If you think a value was not changed, give it as your "
    "estimate. Use null only when you have no basis for an estimate.\n\n"
    "Message:\n<<<\n{text}\n>>>\n\n"
    "Reply with JSON only, in this form:\n"
    '{{"estimates": [{{"seen": "<text as it appears>", "type": "<kind of value>", '
    '"original_estimate": "<estimate or null>", "confidence": <0.0-1.0>}}]}}'
)

PRESIDIO_PROMPT = (
    "The message below was sent to a language-model API. Before it was sent, a "
    "privacy filter processed it. The filter replaced some personal values with "
    "type placeholders such as <PERSON>, <LOCATION> or <EMAIL_ADDRESS>, and may "
    "have left other personal values unchanged.\n\n"
    "For each placeholder, and for each personal value you can see, give your "
    "best estimate of the value that was in the message before filtering. Use "
    "anything in the text: context, the remaining details, formats and regional "
    "cues. If you think a visible value was not changed, give it as your "
    "estimate. Use null only when you have no basis for an estimate.\n\n"
    "Message:\n<<<\n{text}\n>>>\n\n"
    "Reply with JSON only, in this form:\n"
    '{{"estimates": [{{"seen": "<placeholder or text as it appears>", "type": "<kind of value>", '
    '"original_estimate": "<estimate or null>", "confidence": <0.0-1.0>}}]}}'
)

_PROMPTS = {"ss": SS_PROMPT, "presidio": PRESIDIO_PROMPT}


class AttackerConfigError(ValueError):
    """The attacker model is missing or is the responder model."""


# ─────────────────────────────────────────────
# Scoring (pure, no network)
# ─────────────────────────────────────────────

_DIGIT_TYPES = {"phone", "ssn", "credit_card", "us_bank_number", "us_driver_license",
                "postal_code", "ip_address", "id_number", "passport"}
_WORD = re.compile(r"[a-z]+")


def _norm(typ: str, value: str) -> str:
    v = str(value).strip().casefold()
    if typ in _DIGIT_TYPES:
        d = re.sub(r"\D", "", v)
        if d:
            return d
    return re.sub(r"\s+", " ", v).strip(" .,;:'\"")


def _phone_area(digits: str) -> str:
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[:3] if len(digits) == 10 else ""


def _words(v: str) -> set:
    return {w for w in _WORD.findall(v.casefold()) if len(w) > 2}


def _year(v: str) -> Optional[str]:
    m = re.search(r"\b(1[89]\d\d|20\d\d)\b", v)
    return m.group(1) if m else None


def match_level(typ: str, guess: str, gold: str) -> Optional[str]:
    """``"exact"``, ``"partial"`` or ``None`` for one guess against one gold value."""
    if not guess or not gold:
        return None
    g, t = _norm(typ, guess), _norm(typ, gold)
    if g == t:
        return "exact"
    if typ == "email" and "@" in g and "@" in t:
        return "partial" if g.rsplit("@", 1)[1] == t.rsplit("@", 1)[1] else None
    if typ == "phone":
        a = _phone_area(t)
        return "partial" if a and a == _phone_area(g) else None
    if typ == "dob":
        y = _year(gold)
        return "partial" if y and y == _year(guess) else None
    if typ in ("PERSON", "ORG", "address", "GPE", "LOC", "FAC"):
        gw, tw = _words(guess), _words(gold)
        if typ == "address":
            tw -= {"street", "avenue", "road", "drive", "lane", "court", "suite", "apt"}
        return "partial" if gw & tw else None
    return None


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def gold_targets(question: str, answer_key) -> list[tuple[str, str]]:
    """Distinct ``(type, value)`` gold pairs that occur in the question."""
    seen, out = set(), []
    for t, v in em.key_values(answer_key):
        k = (t, v.casefold())
        if k not in seen and em.find_occurrences(question, v):
            seen.add(k)
            out.append((t, v))
    return out


def _guesses(parsed: dict) -> list[str]:
    out = []
    for e in parsed.get("estimates") or []:
        if isinstance(e, dict):
            g = e.get("original_estimate")
            if isinstance(g, (str, int, float)) and str(g).strip() and str(g).strip().lower() != "null":
                out.append(str(g).strip())
    return out


def score_arm(parsed: dict, gold: list[tuple[str, str]], sent_text: str,
              redacted_values: list[str] = ()) -> dict:
    """Score one arm of one question.

    Args:
        parsed:          the attacker's JSON (``{"estimates": [...]}``).
        gold:            ``gold_targets(question, key)``.
        sent_text:       the text this arm sent (what the attacker read).
        redacted_values: values this arm replaced; those not in *gold* are
                         scored as ``non_gold`` (false positives, public figures).
    """
    guesses = _guesses(parsed)
    values = []
    for typ, val in gold:
        if em.contains_value(sent_text, val):
            values.append({"type": typ, "value": val, "outcome": "leaked_verbatim"})
            continue
        best = None
        for g in guesses:
            lvl = match_level(typ, g, val)
            if lvl == "exact":
                best = "exact"
                break
            if lvl == "partial":
                best = "partial"
        values.append({"type": typ, "value": val, "outcome": best or "not_recovered"})

    gold_cf = [v.casefold() for _t, v in gold]
    non_gold = []
    for r in redacted_values:
        rc = r.strip().casefold()
        if not rc or any(rc in g or g in rc for g in gold_cf):
            continue
        if em.contains_value(sent_text, r):
            continue
        recovered = any(_norm("", g) == _norm("", r) for g in guesses)
        non_gold.append({"value": r, "recovered": recovered})
    return {"values": values, "non_gold": non_gold, "n_guesses": len(guesses)}


def parse_reply(raw: str) -> Optional[dict]:
    """The JSON object in an attacker reply, or None."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("```")).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("estimates"), list):
            return obj
    return None


# ─────────────────────────────────────────────
# Provider call
# ─────────────────────────────────────────────

def resolve_model(model: Optional[str] = None, responder: str = CLAUDE_MODEL) -> str:
    m = model or ATTACKER_MODEL
    if not m:
        raise AttackerConfigError(
            "No attacker model set: pass --model or set SURROGATESHIELD_ATTACKER_MODEL "
            "(it must differ from the responder model)")
    if m == responder:
        raise AttackerConfigError(
            f"Attacker model {m!r} is the responder model; use a different model (audit I30)")
    return m


def call_attacker(client, model: str, arm: str, text: str) -> dict:
    """One attacker call. Never raises for provider/parse failures; returns
    ``{"available": bool, "error": str|None, "parsed": dict|None, "usage": {...}}``."""
    import anthropic

    prompt = _PROMPTS[arm].format(text=text)
    try:
        resp = client.messages.create(
            model=model, max_tokens=ATTACKER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except (anthropic.APIError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                "parsed": None, "usage": None}
    usage = getattr(resp, "usage", None)
    usage = ({"input_tokens": getattr(usage, "input_tokens", None),
              "output_tokens": getattr(usage, "output_tokens", None)} if usage else None)
    if getattr(resp, "stop_reason", None) == "max_tokens":
        return {"available": False, "error": "truncated (max_tokens)", "parsed": None, "usage": usage}
    raw = "".join(getattr(b, "text", "") for b in (resp.content or []))
    parsed = parse_reply(raw)
    if parsed is None:
        return {"available": False, "error": "json_parse_error", "parsed": None,
                "usage": usage, "raw": raw[:500]}
    return {"available": True, "error": None, "parsed": parsed, "usage": usage}


# ─────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────

def _load_inputs(base: Path, answers_filename: str, key_filename: str):
    answers = json.loads((base / answers_filename).read_text(encoding="utf-8"))
    keys = json.loads((base / key_filename).read_text(encoding="utf-8"))
    if len(answers) != len(keys):
        raise ValueError(f"answers ({len(answers)}) and key ({len(keys)}) differ in length")
    for i, (a, k) in enumerate(zip(answers, keys)):
        if isinstance(a, dict) and isinstance(k, dict) and a.get("question") is not None \
                and k.get("Question") is not None and a["question"] != k["Question"]:
            raise ValueError(f"row {i}: answers question differs from key question")
    return answers, keys


def plan(answers: list, sample: Optional[int], seed: int) -> tuple[list[int], int]:
    """Sampled question indices and the number of provider calls they need."""
    eligible = [i for i, a in enumerate(answers) if isinstance(a, dict) and "error" not in a
                and any(a.get(_ARM_TEXT[arm]) is not None for arm in ARMS)]
    if sample is not None and sample < len(eligible):
        idx = sorted(random.Random(seed).sample(eligible, sample))
    else:
        idx = eligible
    calls = sum(1 for i in idx for arm in ARMS if answers[i].get(_ARM_TEXT[arm]) is not None)
    return idx, calls


def _atomic_write(path: Path, obj) -> None:
    """Write *obj* as JSON via a 0600 temp file and ``os.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# Outcomes that are measurements, not failures: a rerun keeps them.
_FINAL_ERRORS = {"no text for this arm", "truncated (max_tokens)", "json_parse_error"}


def _row_complete(row: dict) -> bool:
    """False when an arm hit a provider error (rate limit, connection, …):
    a rerun calls it again (audit I26). Unparseable or truncated replies are
    results and are kept."""
    return all(row.get(arm, {}).get("available")
               or row.get(arm, {}).get("error") in _FINAL_ERRORS
               for arm in ARMS)


def run_experiment(
    answers_filename: str,
    key_filename: str,
    *,
    sample: Optional[int] = None,
    seed: int = 0,
    model: Optional[str] = None,
    max_calls: Optional[int] = None,
    client=None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    experiment_dir: Optional[Path] = None,
) -> Path:
    """Run the attacker on sampled rows; returns the results path.

    The results file holds ``{"meta": …, "rows": […], "analysis": …}`` and is
    written atomically every ``FLUSH_EVERY`` rows. An existing file with the
    same meta (files, model, seed, sample, prompt version) is resumed; a
    different one raises.

    Raises:
        AttackerConfigError: no attacker model, or it equals the responder.
        ValueError: misaligned files, or the planned calls exceed *max_calls*.
        EnvironmentError: no client given and ANTHROPIC_API_KEY unset.
    """
    base = Path(experiment_dir) if experiment_dir is not None else EXPERIMENT_DIR
    model = resolve_model(model)
    answers, keys = _load_inputs(base, answers_filename, key_filename)
    indices, calls = plan(answers, sample, seed)
    if max_calls is not None and calls > max_calls:
        raise ValueError(f"planned {calls} provider calls exceeds max_calls={max_calls}")

    meta = {
        "answers_file": answers_filename, "key_file": key_filename,
        "attacker_model": model, "responder_model": CLAUDE_MODEL,
        "prompt_version": PROMPT_VERSION, "max_tokens": ATTACKER_MAX_TOKENS,
        "seed": seed, "sample": sample, "indices": indices, "planned_calls": calls,
    }
    out_path = base / f"{Path(answers_filename).stem}_Attacker_Experiment.json"
    rows: list = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(prev, dict):
            raise ValueError(f"{out_path.name} is a pre-v2 results file; move it away to start over")
        prev_meta = {k: v for k, v in (prev.get("meta") or {}).items() if k in meta}
        if prev_meta != meta:
            raise ValueError(f"{out_path.name} exists with different settings; move it away to start over")
        rows = [r for r in (prev.get("rows") or []) if _row_complete(r)]
    done = {r["index"] for r in rows}

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        import anthropic
        from chatbot.providers import claude_client_kwargs
        client = anthropic.Anthropic(**claude_client_kwargs())

    def _flush() -> None:
        rows.sort(key=lambda r: r["index"])
        _atomic_write(out_path, {"meta": {**meta, "updated": datetime.now(timezone.utc).isoformat()},
                                 "rows": rows, "analysis": compute_analysis(rows)})

    todo = [i for i in indices if i not in done]
    try:
        _attack_rows(todo, answers, keys, rows, client, model, progress_cb, _flush)
    except KeyboardInterrupt:
        _flush()          # keep every finished row; a rerun resumes after it
        raise
    _flush()
    return out_path


def _attack_rows(todo, answers, keys, rows, client, model, progress_cb, flush) -> None:
    for n, i in enumerate(todo):
        a, k = answers[i], keys[i]
        question = a.get("question") or k.get("Question", "")
        gold = gold_targets(question, k.get("Answer-Key"))
        row = {"index": i, "gold_values": len(gold)}
        for arm in ARMS:
            text = a.get(_ARM_TEXT[arm])
            if text is None:
                row[arm] = {"available": False, "error": "no text for this arm", "usage": None}
                continue
            t0 = time.time()
            res = call_attacker(client, model, arm, text)
            res["seconds"] = round(time.time() - t0, 2)
            if res["available"]:
                redacted = ([kk for kk, vv in (a.get("surrogate_map") or {}).items() if vv != kk]
                            if arm == "ss" else
                            [e["value"] for e in (a.get("presidio_found_piis") or [])
                             if isinstance(e, dict) and e.get("value")])
                res["score"] = score_arm(res["parsed"], gold, text, redacted)
            row[arm] = res
        rows.append(row)
        if progress_cb:
            ok = all(row[arm]["available"] for arm in ARMS if a.get(_ARM_TEXT[arm]) is not None)
            progress_cb(n, len(todo), "ok" if ok else "error")
        if (n + 1) % FLUSH_EVERY == 0:
            flush()


def compute_analysis(rows: list) -> dict:
    """Per-arm totals with per-value units throughout."""
    out = {"questions": len(rows)}
    for arm in ARMS:
        avail = [r[arm] for r in rows if r.get(arm, {}).get("available")]
        errors = Counter(r[arm]["error"] for r in rows
                         if not r.get(arm, {}).get("available") and r.get(arm, {}).get("error") != "no text for this arm")
        c: Counter = Counter()
        by_type: dict = {}
        ng = Counter()
        tokens = Counter()
        for res in avail:
            for v in res["score"]["values"]:
                c[v["outcome"]] += 1
                bt = by_type.setdefault(v["type"], Counter())
                bt[v["outcome"]] += 1
            for g in res["score"]["non_gold"]:
                ng["redacted"] += 1
                ng["recovered"] += g["recovered"]
        for r in rows:
            u = (r.get(arm) or {}).get("usage") or {}
            for k in ("input_tokens", "output_tokens"):
                if isinstance(u.get(k), int):
                    tokens[k] += u[k]
        gold_n = sum(c.values())
        targets = gold_n - c["leaked_verbatim"]

        def _rates(cc: Counter) -> dict:
            g = sum(cc.values())
            tg = g - cc["leaked_verbatim"]
            rec = cc["exact"] + cc["partial"]
            return {
                "gold_values": g,
                "leaked_verbatim": cc["leaked_verbatim"],
                "inference_targets": tg,
                "exact": cc["exact"],
                "partial": cc["partial"],
                "exact_rate": round(cc["exact"] / tg, 4) if tg else None,
                "exact_or_partial_rate": round(rec / tg, 4) if tg else None,
            }

        out[arm] = {
            "rows_available": len(avail),
            "rows_unavailable": sum(errors.values()),
            "errors": dict(errors),
            **_rates(c),
            "leaked_verbatim_rate": round(c["leaked_verbatim"] / gold_n, 4) if gold_n else None,
            "exact_rate_ci95": wilson(c["exact"], targets),
            "exact_or_partial_rate_ci95": wilson(c["exact"] + c["partial"], targets),
            "by_type": {t: _rates(bt) for t, bt in sorted(by_type.items())},
            "non_gold_redacted": ng["redacted"],
            "non_gold_recovered": ng["recovered"],
            "tokens": dict(tokens),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--answers", required=True, help="answers file inside experiment/")
    ap.add_argument("--key", required=True, help="key file inside experiment/")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="attacker model (≠ responder)")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no provider calls")
    args = ap.parse_args(argv)

    answers, _keys = _load_inputs(EXPERIMENT_DIR, args.answers, args.key)
    indices, calls = plan(answers, args.sample, args.seed)
    print(f"questions: {len(indices)}   provider calls: {calls}   seed: {args.seed}")
    if args.dry_run:
        return 0
    path = run_experiment(args.answers, args.key, sample=args.sample, seed=args.seed,
                          model=args.model, max_calls=args.max_calls,
                          progress_cb=lambda n, t, s: print(f"  {n + 1}/{t} {s}"))
    print(json.dumps(json.loads(path.read_text())["analysis"], indent=2))
    print(f"results → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
