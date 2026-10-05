"""Phase 8 — live re-run: answers, BERTScore on outputs, attacker (A8, A10, J10).

1. Sample ``--sample`` questions that carry gold PII from the synthetic test
   split with ``random.Random(--seed)``. The sample and its key rows are
   written in the same order to ``experiment/phase8/`` (git-ignored, 0600).
2. ``json_tester.run_batch`` on the sample. It makes three responder calls
   per question: the SurrogateShield text, the Presidio text and the
   original text (synthetic data only). Then it computes rescaled BERTScore
   locally. The utility pair is each arm's answer against the answer to
   the original question (A8).
3. ``attacker.run_experiment`` on every answered row, with ``--attacker``,
   two calls per row.
4. A counts-only summary goes to ``--json``.

Every HTTP attempt, retries included, passes through one counter. A call
that would cross ``--max-calls`` raises before it is sent. ``--dry-run``
prints the plan and makes no call. Keys are read from the environment,
or from ``.env`` through python-dotenv, and are never printed.

    python bench/phase8.py --sample 50 --seed 0 --attacker claude-opus-5-5 --dry-run
    python bench/phase8.py --sample 50 --seed 0 --attacker claude-opus-5-5 \\
        --max-calls 300 --json bench/results/phase8.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-library"))

SYNTH_QUESTIONS = ROOT / "experiment" / "synth_test.json"
SYNTH_KEY = ROOT / "experiment" / "synth_test_key.json"
OUT_DIR = ROOT / "experiment" / "phase8"
SAMPLE_FILE = "phase8_sample.json"
SAMPLE_KEY = "phase8_sample_key.json"
RESPONDER_CALLS_PER_QUESTION = 3
ATTACKER_CALLS_PER_QUESTION = 2

FIELDS = {
    "question": True, "pii_spans": True, "surrogate_map": True, "sanitized_input": True,
    "llm_response": True, "final_output": True, "stage_timings_ms": True,
    "presidio_sanitized_input": True, "presidio_found_piis": True,
    "presidio_llm_response": True, "clean_llm_response": True,
    "bertscore_ss": True, "bertscore_presidio": True,
}


class BudgetExceeded(RuntimeError):
    """A provider call would cross the run's call cap."""


class CallCounter:
    def __init__(self, cap: int):
        self.cap = cap
        self.calls = {"responder": 0, "attacker": 0}

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def take(self, kind: str) -> None:
        if self.total >= self.cap:
            raise BudgetExceeded(f"call cap {self.cap} reached")
        self.calls[kind] += 1


class CountedAdapter:
    """Wraps a ``chatbot.providers`` adapter; one count per attempt."""

    def __init__(self, inner, counter: CallCounter):
        self._inner, self._counter = inner, counter
        self.provider = getattr(inner, "provider", "")
        self.model = getattr(inner, "model", None)

    def __call__(self, payload, system):
        self._counter.take("responder")
        return self._inner(payload, system)


class CountedAnthropic:
    """Exposes ``messages.create`` of an Anthropic client, counted."""

    def __init__(self, client, counter: CallCounter):
        self._client, self._counter = client, counter
        self.messages = self

    def create(self, **kw):
        self._counter.take("attacker")
        return self._client.messages.create(**kw)


def sample_rows(n: int, seed: int) -> list[int]:
    """Indices of *n* synthetic-test questions with at least one gold value."""
    key = json.loads(SYNTH_KEY.read_text(encoding="utf-8"))
    eligible = [i for i, k in enumerate(key) if k.get("Answer-Key")]
    return sorted(random.Random(seed).sample(eligible, n))


def _write_private(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def write_sample(indices: list[int]) -> None:
    questions = json.loads(SYNTH_QUESTIONS.read_text(encoding="utf-8"))
    key = json.loads(SYNTH_KEY.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    _write_private(OUT_DIR / SAMPLE_FILE, [questions[i] for i in indices])
    _write_private(OUT_DIR / SAMPLE_KEY, [key[i] for i in indices])


def _bootstrap_mean_diff(pairs: list[tuple[float, float]], seed: int, n_boot: int = 2000):
    if not pairs:
        return None
    rng = random.Random(seed)
    diffs = [a - b for a, b in pairs]
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {"n": len(diffs), "mean_diff": round(statistics.fmean(diffs), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)]}


def summarise_bertscore(answers: list, seed: int) -> dict:
    out: dict = {}
    for name in ("bertscore_ss", "bertscore_presidio", "bertscore_ss_output", "bertscore_presidio_output"):
        vals = [a[name]["f1"] for a in answers if isinstance(a.get(name), dict)]
        out[name] = {"n": len(vals),
                     "mean_f1": round(statistics.fmean(vals), 4) if vals else None,
                     "median_f1": round(statistics.median(vals), 4) if vals else None}
    for kind, (a_name, b_name) in {"input": ("bertscore_ss", "bertscore_presidio"),
                                   "output": ("bertscore_ss_output", "bertscore_presidio_output")}.items():
        pairs = [(a[a_name]["f1"], a[b_name]["f1"]) for a in answers
                 if isinstance(a.get(a_name), dict) and isinstance(a.get(b_name), dict)]
        out[f"paired_{kind}_ss_minus_presidio"] = _bootstrap_mean_diff(pairs, seed)
    out["rows_with_error"] = sum(1 for a in answers if "error" in a)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacker", required=True, help="attacker model (≠ responder)")
    ap.add_argument("--max-calls", type=int, default=300)
    ap.add_argument("--json", type=Path, default=None, help="counts-only summary")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import config
    from attacker import resolve_model

    attacker_model = resolve_model(args.attacker, responder=config.CLAUDE_MODEL)
    indices = sample_rows(args.sample, args.seed)
    planned = {"responder": RESPONDER_CALLS_PER_QUESTION * len(indices),
               "attacker": ATTACKER_CALLS_PER_QUESTION * len(indices)}
    print(f"questions: {len(indices)} (seed {args.seed}) responder: {config.CLAUDE_MODEL} "
          f"attacker: {attacker_model}")
    print(f"planned calls: responder {planned['responder']} + attacker {planned['attacker']} "
          f"= {sum(planned.values())}; cap {args.max_calls} (retries count against it)")
    if sum(planned.values()) > args.max_calls:
        print("planned calls exceed the cap; nothing sent", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    import anthropic
    import attacker
    import json_tester
    from chatbot.chat import ClaudeChat

    write_sample(indices)
    counter = CallCounter(args.max_calls)
    chat = ClaudeChat()
    chat._client = CountedAdapter(chat._client, counter)

    def progress(i, total, _q, status, secs):
        if i >= 0 and status != "running":
            print(f"  answer {i + 1}/{total} {status} ({secs:.1f}s) calls={counter.total}", flush=True)
        elif i < 0:
            print(f"  bertscore {status}", flush=True)

    answers_path = Path(json_tester.run_batch(SAMPLE_FILE, FIELDS, progress_cb=progress, chat=chat,
                                              experiment_dir=OUT_DIR))
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    meta = json.loads((OUT_DIR / "phase8_sample_answers.meta.json").read_text(encoding="utf-8"))

    client = CountedAnthropic(anthropic.Anthropic(max_retries=0), counter)
    attack_path = attacker.run_experiment(
        answers_path.name, SAMPLE_KEY, seed=args.seed, model=attacker_model,
        max_calls=args.max_calls - counter.total, client=client, experiment_dir=OUT_DIR,
        progress_cb=lambda n, t, s: print(f"  attack {n + 1}/{t} {s} calls={counter.total}", flush=True))
    attack = json.loads(Path(attack_path).read_text(encoding="utf-8"))

    summary = {
        "command": "python bench/phase8.py " + " ".join(sys.argv[1:] if argv is None else argv),
        "generator": meta.get("generator"),
        "split": "experiment/synth_test_key.json (seen; questions with gold PII only)",
        "sample": len(indices), "seed": args.seed, "indices": indices,
        "responder_model": config.CLAUDE_MODEL, "attacker_model": attacker_model,
        "attacker_prompt_version": attack["meta"].get("prompt_version"),
        "provider_calls": {**counter.calls, "total": counter.total, "cap": args.max_calls},
        "answer_rows_with_error": sum(1 for a in answers if "error" in a),
        "bertscore": {"model": json_tester.BERTSCORE_MODEL, "rescaled": True,
                      **summarise_bertscore(answers, args.seed)},
        "attacker": attack["analysis"],
    }
    print(json.dumps({k: summary[k] for k in ("provider_calls", "bertscore", "attacker")}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"summary → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
