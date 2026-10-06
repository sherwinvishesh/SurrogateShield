"""Phase 6: the live runs of the real-data benchmark, on the test split.

    .venv/bin/python -m bench.realdata.live estimate
    .venv/bin/python -m bench.realdata.live utility   --pilot 10
    .venv/bin/python -m bench.realdata.live utility   --out bench/results/utility_realdata.json
    .venv/bin/python -m bench.realdata.live multiturn --out bench/results/multiturn_realdata.json
    .venv/bin/python -m bench.realdata.live attacker  --out bench/results/attacker_realdata.json

``utility`` is E5b (injected slice) and the natural-slice utility cost;
``multiturn`` is E7b; ``attacker`` is E8, whose conversation condition reads
E7b's SurrogateShield histories, so it runs last. ``--pilot N`` runs the first
N rows of the sample (datasets interleaved) and writes no result; its replies
are cached, so the main run reuses them.

Roles. Responder: Claude Sonnet 4.6 (``config.CLAUDE_MODEL``) with the app's
system prompt (``chatbot.chat.SYSTEM_PROMPT``) and ``max_tokens`` for every
arm, the original included, as in Phase 8. Attacker and judge: Claude Opus
5.5, never the responder. Every request goes through the Message Batches API
and is taken from the run's call ledger (``provider.Ledger``, cap 4,000)
before it is sent.

Cache. Every reply is stored under ``experiment/realdata/<run>/`` (git-ignored,
0700 directory, 0600 files), keyed by the hash of its slot and request
parameters, so a rerun sends only what is missing. A failed request is stored
as an error row and is not resent unless ``--retry-errors`` is given (each
retry is counted). Arm texts come from the frozen test spans the scorer wrote
(``bench/realdata/build/spans/<arm>/test-<dataset>.jsonl``); the multi-turn
replay runs the app's own ``Pipeline``.

Samples (ids are stored in each result):

* E5b: 100 injected single-turn test prompts per dataset (``single`` and
  ``shift``), ``derive_seed("live-utility", dataset)``; arms original, ``ss``,
  ``presidio_default``, ``presidio_faker``, one responder call each.
* natural: 50 first user turns per dataset whose silver labels hold nothing
  to protect, ``derive_seed("live-natural", dataset)``; the untouched prompt
  is answered twice (``orig`` and ``rerun``, the noise floor), and ``ss`` and
  ``presidio_default`` are sent only when they edited the prompt; an unedited
  prompt takes the ``rerun`` answer.
* E7b: 30 injected multi-turn conversations per dataset,
  ``derive_seed("live-multiturn", dataset)``, arms ``ss`` and
  ``presidio_faker``.
* E8: the first 60 prompts of the E5b sample per dataset, arms ``ss``,
  ``presidio_default``, ``presidio_faker``, ``llm_guard``; and, for ``ss``
  only, the E7b conversations as the provider saw them.

Metrics.

* BERTScore F1 (roberta-large, rescaled) of each arm's displayed answer against
  the answer to the original. SurrogateShield's answer is restored with the
  app's ``ResolvePass`` from its own edits; the other arms have nothing to
  restore. Truncated answers are kept and counted.
* Pairwise judge (E5b): ``ss`` against ``presidio_default``; Opus sees the
  original prompt and the two displayed answers, positions drawn per prompt
  with ``derive_seed("live-judge", id)``, arm names hidden; A, B or tie.
  Score +1 / −1 / 0 for SurrogateShield; agreement with BERTScore is the
  share of non-tie verdicts whose winner has the higher BERTScore.
* E7b: (i) restoration: a turn whose raw answer contains one of the arm's
  replacement values is restored when every such value is gone from the
  displayed answer and its original is there (``bench/echo.py``: no
  surrogate left in, the original back); low-entropy values and values that
  are also originals are exempt. (ii) I2: no value the arm replaced appears in
  an assistant message of a later request; gold values in any request are
  reported apart (they include detection misses). (iii) consistency: a gold
  value in two or more user turns gets one replacement in every one. (iv)
  judge: Opus sees the original user messages and the final displayed answer
  and grades its use of the user's details: correct (1), partly (0.5),
  incorrect (0) or not_needed (left out).
* E8: protocol v2 of ``attacker.py`` (its prompt for realistic-substitute
  arms; for ``presidio_default``, whose placeholders are ``[TYPE]``, the same
  prompt naming ``[PERSON]`` instead of ``<PERSON>``). Gold: the distinct
  protect values of each message, the same for every arm. A value the arm
  left in (the scorer's leak, or kept by the service-query policy) is
  counted apart; a reply that is not valid JSON makes its values
  unavailable. Exact and partial recovery with Wilson intervals; partials
  are classified as in Phase 8 (e-mail domain, phone area code, name token,
  address part, place word, organisation word, birth year) plus three
  classes Phase 8 had no type for (e-mail local part, IPv4 /24, URL host),
  each with a flag for whether the recovered part is visible in the text the
  attacker read.

Criteria, fixed here before any live result (PROMPT_FOR_OPUS_NEW §2):

* H3 holds on a dataset when (a) the mean BERTScore difference ss −
  presidio_default has a 95 % paired bootstrap interval above 0, (b) the mean
  judge score of ss against presidio_default has an interval above 0, and
  (c) the mean BERTScore difference ss − presidio_faker has a lower bound
  above −0.02 (not lower than Presidio+Faker within 0.02).
* H4 holds on a dataset when the judge score (iv) and the consistency rate
  (iii) of ss are both higher than presidio_faker's, each with a 95 %
  bootstrap interval over conversations above 0.
* H5 holds on a dataset when SurrogateShield has no exact recovery in the
  single-message condition; the conversation condition is reported as
  H5-conv beside it.

Sizes against the prompt's (call cap): E5b drops LLM Guard as a responder arm
(it stays in E8) and judges ss against presidio_default only (H3 compares
Presidio+Faker on BERTScore); E7b runs 30 conversations per dataset, not 50;
E8 runs 60 prompts per dataset, not 100. The reasons and the estimate are in
the run's progress log.

Real text: natural-slice prompts go to the responder only; the judge and the
attacker see injected-slice text only. Committed results hold counts, ids and
prompt templates, never message text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from bench.realdata import provider as pv
from bench.realdata.common import BUILD, DATASETS, ROOT, derive_seed, git_state, read_jsonl, sha256

RUNS = ROOT / "experiment" / "realdata"
TMP = BUILD / "live-tmp"
PROMPT_VERSION = "rd1"
RESPONDER_MAX_TOKENS = 4096          # chatbot.providers.MAX_TOKENS
JUDGE_MAX_TOKENS = 1024
ATTACKER_MAX_TOKENS = 4096           # attacker.ATTACKER_MAX_TOKENS
CAPTURED = "(captured)"
HISTORIES = "multiturn-ss-histories.jsonl"

UTILITY_PER_DS = 100
NATURAL_PER_DS = 50
MULTITURN_PER_DS = 30
ATTACK_PER_DS = 60

UTILITY_ARMS = ("original", "ss", "presidio_default", "presidio_faker")
ATTACK_ARMS = ("ss", "presidio_default", "presidio_faker", "llm_guard")
SPAN_ARMS = ("ss", "presidio_default", "presidio_faker", "llm_guard")
MULTITURN_ARMS = ("ss", "presidio_faker")
GRADES = {"correct": 1.0, "partly": 0.5, "incorrect": 0.0}

# ── prompts (versioned; their hashes go into every result) ───────────────────

JUDGE_PAIR_PROMPT = (
    "Below is a user's request to an AI assistant and two answers to it. Decide "
    "which answer better serves the user's request. Judge correctness, "
    "completeness and how well each answer does what the user asked, including "
    "whether it uses the user's own details (names, contact details, dates, "
    "places, identifiers) correctly where the request calls for them. Do not "
    "prefer an answer for its length or style alone. If neither is clearly "
    "better, choose tie.\n\n"
    "Request:\n<<<\n{prompt}\n>>>\n\n"
    "Answer A:\n<<<\n{a}\n>>>\n\n"
    "Answer B:\n<<<\n{b}\n>>>\n\n"
    "Give your verdict with the verdict tool."
)
VERDICT_TOOL = {
    "name": "verdict",
    "description": "Record which answer better serves the user's request.",
    "input_schema": {"type": "object", "properties": {
        "better": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string", "description": "One or two sentences."}},
        "required": ["better", "reason"]},
}

JUDGE_DETAILS_PROMPT = (
    "Below are the messages a user sent to an AI assistant, in order, and the "
    "assistant's final answer as the user saw it. Grade whether the final "
    "answer uses the user's own details correctly: names, contact details, "
    "dates, ages, identifiers, addresses, places and organisations given in "
    "the user's messages.\n\n"
    "- correct: every such detail the answer uses matches what the user gave;\n"
    "- partly: some match, and some are wrong, replaced by other values or "
    "placeholders, or missing where the answer needs them;\n"
    "- incorrect: the answer uses other values or placeholders where it needs "
    "the user's details;\n"
    "- not_needed: the answer has no reason to use any of the user's details.\n\n"
    "User messages:\n{turns}\n\n"
    "Final answer:\n<<<\n{answer}\n>>>\n\n"
    "Give your grade with the grade tool."
)
GRADE_TOOL = {
    "name": "grade",
    "description": "Record how the final answer uses the user's details.",
    "input_schema": {"type": "object", "properties": {
        "grade": {"type": "string", "enum": ["correct", "partly", "incorrect", "not_needed"]},
        "reason": {"type": "string", "description": "One or two sentences."}},
        "required": ["grade", "reason"]},
}

ATTACK_CONV_PROMPT = (
    "The conversation below, between a user and an AI assistant, was sent to a "
    "language-model API. Before each user message was sent, a privacy filter "
    "processed it. The filter may have replaced some personal values (names, "
    "contact details, identifiers, addresses, dates, places, organisations) "
    "with realistic substitutes, may have changed only part of a value, and may "
    "have left some values unchanged; the assistant's replies were written from "
    "the filtered messages. You are not told which values were changed or how "
    "substitutes were chosen.\n\n"
    "For each personal value of the user's that you can see, give your best "
    "estimate of the value that was in the user's messages before filtering. "
    "Use anything in the conversation: context, consistency between messages "
    "and fields, formats, regional cues, and values that look unchanged. If you "
    "think a value was not changed, give it as your estimate. Use null only "
    "when you have no basis for an estimate.\n\n"
    "Conversation:\n{text}\n\n"
    "Reply with JSON only, in this form:\n"
    '{{"estimates": [{{"seen": "<text as it appears>", "type": "<kind of value>", '
    '"original_estimate": "<estimate or null>", "confidence": <0.0-1.0>}}]}}'
)


def attack_prompts() -> Dict[str, str]:
    """Protocol v2's templates: realistic substitutes, and ``[TYPE]`` placeholders."""
    import attacker
    old = "<PERSON>, <LOCATION> or <EMAIL_ADDRESS>"
    if old not in attacker.PRESIDIO_PROMPT:
        raise RuntimeError("attacker.PRESIDIO_PROMPT changed; update the placeholder variant")
    return {"substitute": attacker.SS_PROMPT,
            "placeholder": attacker.PRESIDIO_PROMPT.replace(old, "[PERSON], [LOCATION] or [EMAIL_ADDRESS]"),
            "conversation": ATTACK_CONV_PROMPT}


def system_prompt() -> str:
    from chatbot.chat import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def prompt_record() -> dict:
    t = {"judge_pair": JUDGE_PAIR_PROMPT, "judge_details": JUDGE_DETAILS_PROMPT,
         **{f"attack_{k}": v for k, v in attack_prompts().items()}}
    return {"version": PROMPT_VERSION, "responder_system_sha256": sha256(system_prompt()),
            "templates": t, "sha256": {k: sha256(v) for k, v in t.items()},
            "tools": {"verdict": VERDICT_TOOL, "grade": GRADE_TOOL}}


def responder_params(messages: List[dict]) -> dict:
    return {"model": pv.SONNET, "max_tokens": RESPONDER_MAX_TOKENS, "system": system_prompt(),
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages]}


def judge_pair_params(prompt: str, a: str, b: str) -> dict:
    return {"model": pv.OPUS, "max_tokens": JUDGE_MAX_TOKENS, "tools": [VERDICT_TOOL],
            "tool_choice": {"type": "tool", "name": "verdict"},
            "messages": [{"role": "user", "content": JUDGE_PAIR_PROMPT.format(prompt=prompt, a=a, b=b)}]}


def judge_details_params(turns: Sequence[str], answer: str) -> dict:
    shown = "\n\n".join(f"Message {i + 1}:\n<<<\n{t}\n>>>" for i, t in enumerate(turns))
    return {"model": pv.OPUS, "max_tokens": JUDGE_MAX_TOKENS, "tools": [GRADE_TOOL],
            "tool_choice": {"type": "tool", "name": "grade"},
            "messages": [{"role": "user", "content": JUDGE_DETAILS_PROMPT.format(turns=shown, answer=answer)}]}


def attack_params(template: str, text: str) -> dict:
    return {"model": pv.OPUS, "max_tokens": ATTACKER_MAX_TOKENS,
            "messages": [{"role": "user", "content": template.format(text=text)}]}


def conversation_text(messages: Sequence[dict]) -> str:
    return "\n\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}:\n<<<\n{m['content']}\n>>>"
                       for m in messages)


def check_roles() -> None:
    import config
    if pv.SONNET != config.CLAUDE_MODEL:
        raise SystemExit(f"responder {pv.SONNET} is not config.CLAUDE_MODEL ({config.CLAUDE_MODEL})")
    if pv.OPUS == pv.SONNET:
        raise SystemExit("the attacker and judge must not be the responder")


# ── the run: cache, batches, call counter ────────────────────────────────────

def cache_key(slot: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"slot": slot, "params": params}, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _cid(key: str) -> str:
    return "k" + key[:40]


def reply_row(key: str, slot: str, stage: str, kind: str, attempt: int, res: Optional[dict]) -> dict:
    row = {"key": key, "slot": slot, "stage": stage, "kind": kind, "attempt": attempt,
           "status": "errored", "text": None, "tool": None, "stop_reason": None, "usage": None,
           "model": None, "error": None}
    if not res or res.get("status") != "succeeded":
        row["error"] = str((res or {}).get("error") or (res or {}).get("status") or "no result")[:500]
        return row
    msg = res["message"]
    row.update(text="".join(b.get("text", "") for b in msg.get("content") or [] if b.get("type") == "text"),
               tool=next((b.get("input") for b in msg.get("content") or [] if b.get("type") == "tool_use"), None),
               stop_reason=msg.get("stop_reason"), usage=msg.get("usage"), model=msg.get("model"),
               status="truncated" if msg.get("stop_reason") == "max_tokens" else "ok")
    return row


class Run:
    """One run directory: every provider reply, cached by request."""

    def __init__(self, name: str = "main", *, root: Path = RUNS, ledger: Optional[pv.Ledger] = None,
                 client=None, batch_runner: Callable = pv.run_batch, max_calls: Optional[int] = None,
                 retry_errors: bool = False, log: Callable[[str], None] = print):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            raise ValueError(f"run name {name!r}: letters, digits, _ and - only")
        self.name, self.dir = name, Path(root) / name
        self.ledger = ledger if ledger is not None else pv.Ledger(run=f"live-{name}")
        self._client, self.batch_runner = client, batch_runner
        self.max_calls, self.retry_errors, self.log = max_calls, retry_errors, log
        self.sent = 0
        self.retried: set = set()             # keys resent in this invocation
        self.cache: Dict[str, dict] = {}
        path = self.dir / "results.jsonl"
        if path.exists():
            for r in read_jsonl(path):
                self.cache[r["key"]] = r          # a later row (a retry) wins

    def client(self):
        if self._client is None:
            self._client = pv.client()
        return self._client

    def get(self, slot: str, params: dict) -> Optional[dict]:
        return self.cache.get(cache_key(slot, params))

    def wants(self, row: Optional[dict], slot: str, params: dict) -> bool:
        """Whether ``fetch`` would send this request."""
        return row is None or (self.retry_errors and row["status"] == "errored"
                               and cache_key(slot, params) not in self.retried)

    def _append(self, rows: List[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        fd = os.open(str(self.dir / "results.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        for r in rows:
            self.cache[r["key"]] = r

    def fetch(self, stage: str, phase: str, kind: str, items: Sequence[Tuple[str, dict]]) -> List[dict]:
        """The cached reply to every ``(slot, params)``, sending the missing ones
        as one batch first."""
        keys = [cache_key(s, p) for s, p in items]
        todo: Dict[str, Tuple[str, dict]] = {}
        for k, (s, p) in zip(keys, items):
            row = self.cache.get(k)
            if row is None or (self.retry_errors and row["status"] == "errored" and k not in self.retried):
                todo.setdefault(k, (s, p))
        if todo:
            if self.max_calls is not None and self.sent + len(todo) > self.max_calls:
                raise pv.BudgetExceeded(f"{stage}: {len(todo)} more call(s) would pass --max-calls "
                                        f"{self.max_calls} (this run has sent {self.sent})")
            attempt = 1 + max((self.cache[k]["attempt"] for k in todo if k in self.cache), default=0)
            ids = hashlib.sha256("".join(sorted(todo)).encode()).hexdigest()[:12]
            name = f"{stage}-{ids}" + (f"-a{attempt}" if attempt > 1 else "")
            requests = [{"custom_id": _cid(k), "params": p} for k, (_s, p) in sorted(todo.items())]
            before = self.ledger.total()
            client = self.client() if self.batch_runner is pv.run_batch else self._client
            res = self.batch_runner(client, self.ledger, name, phase, kind, requests,
                                    log=self.log, directory=self.dir / "batches")
            self.sent += self.ledger.total() - before
            self.retried.update(k for k in todo if k in self.cache)
            self._append([reply_row(k, s, stage, kind, attempt, res.get(_cid(k)))
                          for k, (s, _p) in sorted(todo.items())])
        return [self.cache[k] for k in keys]

    def write_private(self, name: str, rows: Iterable[dict]) -> Path:
        from bench.realdata.common import write_jsonl
        path = self.dir / name
        write_jsonl(path, rows, private=True)
        return path


# ── data: the frozen test split and each arm's text ──────────────────────────

def load_test(datasets: Sequence[str] = DATASETS, arms: Sequence[str] = SPAN_ARMS) -> Dict[str, dict]:
    """Per dataset: ``units`` by message id and each arm's checked span rows."""
    from bench.arms.run import PRIVATE
    from bench.realdata import score
    _hashes, loaded = score.load_split("test", datasets)
    out = {}
    for ds in datasets:
        units = loaded[ds]["units"]
        out[ds] = {"units": {u["mid"]: u for u in units},
                   "spans": {a: score.read_spans(PRIVATE / a / f"test-{ds}.jsonl", units, loaded[ds]["input_sha"])
                             for a in arms}}
    return out


def original(data: dict, ds: str, mid: str) -> str:
    return data[ds]["units"][mid]["gold"]["text"]


def edits_of(data: dict, ds: str, arm: str, mid: str) -> Optional[list]:
    row = data[ds]["spans"][arm][mid]
    return None if "refused" in row else row["edits"]


def arm_text(data: dict, ds: str, arm: str, mid: str) -> Optional[str]:
    """What *arm* sends for message *mid*; None when it refused to send."""
    from bench.arms.base import apply_edits
    if arm == "original":
        return original(data, ds, mid)
    edits = edits_of(data, ds, arm, mid)
    return None if edits is None else apply_edits(original(data, ds, mid), edits)


def restore_ss(raw: str, text: str, edits: list, sent: str) -> str:
    """The app's restoration of *raw* from SurrogateShield's own edits."""
    from reconstruction.logic import ResolvePass
    mapping = {rep: text[s:e] for s, e, _t, rep in edits if rep and rep != text[s:e]}
    if not mapping:
        return raw
    return ResolvePass().resolve(raw, mapping, current=set(mapping), sent=sent)


def turn_mids(data: dict, ds: str, conv: str) -> List[str]:
    return sorted((m for m, u in data[ds]["units"].items() if u["conv"] == conv),
                  key=lambda m: data[ds]["units"][m]["turn"])


# ── samples ──────────────────────────────────────────────────────────────────

def utility_sample(data: dict, ds: str, n: int = UTILITY_PER_DS) -> List[str]:
    mids = sorted(m for m, u in data[ds]["units"].items()
                  if "injected" in u["slices"] and "multi" not in u["slices"])
    return random.Random(derive_seed("live-utility", ds)).sample(mids, min(n, len(mids)))


def natural_sample(data: dict, ds: str, n: int = NATURAL_PER_DS) -> List[str]:
    mids = sorted(m for m, u in data[ds]["units"].items()
                  if "natural" in u["slices"] and u["turn"] == 0
                  and not any(u["gold"][k] for k in ("protect", "sensitive", "optional")))
    return random.Random(derive_seed("live-natural", ds)).sample(mids, min(n, len(mids)))


def multiturn_sample(data: dict, ds: str, n: int = MULTITURN_PER_DS) -> List[str]:
    convs = sorted({u["conv"] for u in data[ds]["units"].values() if "multi" in u["slices"]})
    return random.Random(derive_seed("live-multiturn", ds)).sample(convs, min(n, len(convs)))


def attack_sample(data: dict, ds: str, n: int = ATTACK_PER_DS) -> List[str]:
    return utility_sample(data, ds)[:n]


def pilot_of(sample: Dict[str, List[str]], pilot: Optional[int]) -> Dict[str, List[str]]:
    """The first *pilot* rows, datasets interleaved; everything when None."""
    if pilot is None:
        return sample
    order = [(ds, x) for i in range(max(map(len, sample.values()), default=0))
             for ds, xs in sample.items() if i < len(xs) for x in [xs[i]]][:pilot]
    return {ds: [x for d, x in order if d == ds] for ds in sample}


# ── statistics ───────────────────────────────────────────────────────────────

def wilson(k: int, n: int) -> dict:
    from bench.realdata.score import rate
    return rate(k, n)


def mean_diff(clusters: Sequence[str], a: Sequence[float], b: Sequence[float], seed: int) -> dict:
    """Paired cluster bootstrap of mean(a) − mean(b)."""
    from bench.realdata.score import bootstrap
    if not clusters:
        return {"diff": None, "ci95": None, "resamples": 0, "n": 0}
    out = bootstrap(clusters, [(x, 1) for x in a], [(y, 1) for y in b], seed)
    return {**out, "n": len(clusters)}


def mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def bertscore(pairs: Sequence[Tuple[str, str]], cache: Path, scorer: Optional[Callable] = None) -> List[float]:
    """Rescaled roberta-large F1 of each ``(candidate, reference)``, cached by
    the pair's hashes in a private file; None when either text is empty."""
    have: Dict[str, float] = {}
    if cache.exists():
        have = {r["k"]: r["f1"] for r in read_jsonl(cache)}
    keys = [sha256(c) + sha256(r) if c.strip() and r.strip() else None for c, r in pairs]
    todo = sorted({k: p for k, p in zip(keys, pairs) if k is not None and k not in have}.items())
    if todo:
        if scorer is None:
            def scorer(cands, refs):
                import bert_score
                _p, _r, f = bert_score.score(cands, refs, lang="en", model_type="roberta-large",
                                             rescale_with_baseline=True, batch_size=16, verbose=False)
                return [float(x) for x in f]
        f1 = scorer([c for _k, (c, _r) in todo], [r for _k, (_c, r) in todo])
        new = [{"k": k, "f1": round(float(x), 6)} for (k, _p), x in zip(todo, f1)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(cache), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, "".join(json.dumps(r) + "\n" for r in new).encode())
        finally:
            os.close(fd)
        have.update({r["k"]: r["f1"] for r in new})
    return [None if k is None else have[k] for k in keys]


def _groups(datasets: Sequence[str]) -> List[str]:
    return list(datasets) + (["all"] if len(datasets) > 1 else [])


def _in(group: str, ds: str) -> bool:
    return group == "all" or group == ds


# ── requests ─────────────────────────────────────────────────────────────────

def slot(ds: str, ident: str, what: str) -> str:
    """The cache slot of one request: every row of every arm is its own call,
    even when two datasets hold the same text."""
    return f"{ds}/{ident}/{what}"


def _user(text: str) -> dict:
    return responder_params([{"role": "user", "content": text}])


def utility_requests(data: dict, datasets: Sequence[str], sample: Dict[str, List[str]]) -> List[tuple]:
    return [((ds, mid, arm), (slot(ds, mid, arm), _user(text))) for ds in datasets for mid in sample[ds]
            for arm in UTILITY_ARMS for text in [arm_text(data, ds, arm, mid)] if text is not None]


def natural_requests(data: dict, datasets: Sequence[str], sample: Dict[str, List[str]]) -> Tuple[List[tuple], dict]:
    """The untouched prompt twice (``orig``, ``rerun``), and each arm's text
    when the arm changed it; and which arms changed which prompt."""
    out, edited = [], {}
    for ds in datasets:
        for mid in sample[ds]:
            text = original(data, ds, mid)
            out += [((ds, mid, w), (slot(ds, mid, w), _user(text))) for w in ("orig", "rerun")]
            for arm in ("ss", "presidio_default"):
                sent = arm_text(data, ds, arm, mid)
                edited[(ds, mid, arm)] = sent is not None and sent != text
                if edited[(ds, mid, arm)]:
                    out.append(((ds, mid, arm), (slot(ds, mid, arm), _user(sent))))
    return out, edited


def attack_requests(data: dict, datasets: Sequence[str], sample: Dict[str, List[str]]) -> List[tuple]:
    prompts = attack_prompts()
    return [((ds, mid, arm), (slot(ds, mid, f"attack-{arm}"),
                              attack_params(prompts["placeholder" if arm == "presidio_default" else "substitute"], text)))
            for ds in datasets for mid in sample[ds] for arm in ATTACK_ARMS
            for text in [arm_text(data, ds, arm, mid)] if text is not None]


def _fetch(run: Run, stage: str, phase: str, kind: str, reqs: List[tuple]) -> Dict[tuple, dict]:
    return dict(zip([k for k, _ in reqs], run.fetch(stage, phase, kind, [it for _, it in reqs])))


# ── E5b and the natural slice ────────────────────────────────────────────────

def utility(run: Run, data: dict, datasets: Sequence[str] = DATASETS, pilot: Optional[int] = None,
            scorer: Optional[Callable] = None) -> dict:
    sample = pilot_of({ds: utility_sample(data, ds) for ds in datasets}, pilot)
    replies = _fetch(run, "utility", "6-utility", "responder", utility_requests(data, datasets, sample))

    shown: Dict[tuple, str] = {}
    for (ds, mid, arm), row in replies.items():
        if row["status"] == "errored":
            continue
        if arm == "ss":
            shown[(ds, mid, arm)] = restore_ss(row["text"], original(data, ds, mid),
                                               edits_of(data, ds, "ss", mid), arm_text(data, ds, "ss", mid))
            shown[(ds, mid, "ss_raw")] = row["text"]
        else:
            shown[(ds, mid, arm)] = row["text"]

    jindex, jitems = [], []
    for ds in datasets:
        for mid in sample[ds]:
            a, b = shown.get((ds, mid, "ss")), shown.get((ds, mid, "presidio_default"))
            if a is None or b is None:
                continue
            ss_first = random.Random(derive_seed("live-judge", mid)).random() < 0.5
            jindex.append((ds, mid, ss_first))
            jitems.append((slot(ds, mid, "judge"), judge_pair_params(original(data, ds, mid), *((a, b) if ss_first else (b, a)))))
    verdicts = {}
    for (ds, mid, ss_first), row in zip(jindex, run.fetch("judge", "6-judge", "judge", jitems)):
        better = (row.get("tool") or {}).get("better") if row["status"] == "ok" else None
        if better not in ("A", "B", "tie"):
            verdicts[(ds, mid)] = {"score": None, "first": None}
            continue
        ss_won = better == ("A" if ss_first else "B")
        verdicts[(ds, mid)] = {"score": 0 if better == "tie" else (1 if ss_won else -1),
                               "first": None if better == "tie" else better == "A"}

    arms_scored = ("ss", "ss_raw", "presidio_default", "presidio_faker")
    pairs = [(ds, mid, arm) for ds in datasets for mid in sample[ds] for arm in arms_scored
             if (ds, mid, arm) in shown and (ds, mid, "original") in shown]
    f1 = {k: v for k, v in zip(pairs, bertscore([(shown[p], shown[(p[0], p[1], "original")]) for p in pairs],
                                                 run.dir / "bertscore.jsonl", scorer)) if v is not None}

    e5b = {}
    for g in _groups(datasets):
        mids = [(ds, mid) for ds in datasets if _in(g, ds) for mid in sample[ds]]
        res = {"n": len(mids),
               "errored": {a: sum(1 for ds, m in mids if (ds, m, a) in replies
                                  and replies[(ds, m, a)]["status"] == "errored") for a in UTILITY_ARMS},
               "refused": {a: sum(1 for ds, m in mids if (ds, m, a) not in replies) for a in UTILITY_ARMS},
               "truncated": {a: sum(1 for ds, m in mids if replies.get((ds, m, a), {}).get("status") == "truncated")
                             for a in UTILITY_ARMS},
               "bertscore": {a: {"n": len(v), "mean": mean(v)} for a in arms_scored
                             for v in [[f1[(ds, m, a)] for ds, m in mids if (ds, m, a) in f1]]}}
        res["bertscore_differences"] = {}
        for other in ("presidio_default", "presidio_faker"):
            both = [(ds, m) for ds, m in mids if (ds, m, "ss") in f1 and (ds, m, other) in f1]
            res["bertscore_differences"][f"ss-{other}"] = mean_diff(
                [f"{ds}/{m}" for ds, m in both], [f1[(ds, m, "ss")] for ds, m in both],
                [f1[(ds, m, other)] for ds, m in both], derive_seed("live-boot", "e5b", other, g))
        judged = [(ds, m) for ds, m in mids if verdicts.get((ds, m), {}).get("score") is not None]
        scores = [verdicts[k]["score"] for k in judged]
        decided = [k for k in judged if verdicts[k]["score"] != 0]
        both_f1 = [k for k in decided if (k[0], k[1], "ss") in f1 and (k[0], k[1], "presidio_default") in f1]
        agree = sum(1 for k in both_f1 if (verdicts[k]["score"] > 0) ==
                    (f1[(k[0], k[1], "ss")] > f1[(k[0], k[1], "presidio_default")]))
        res["judge"] = {"pair": "ss vs presidio_default", "n": len(judged),
                        "unavailable": sum(1 for k in mids if k in verdicts and verdicts[k]["score"] is None),
                        "ss_wins": scores.count(1), "presidio_default_wins": scores.count(-1), "ties": scores.count(0),
                        "ss_win_rate": wilson(scores.count(1), len(scores)),
                        "score": mean_diff([f"{ds}/{m}" for ds, m in judged], scores, [0] * len(scores),
                                           derive_seed("live-boot", "judge", g)),
                        "first_position_chosen": wilson(sum(1 for k in decided if verdicts[k]["first"]), len(decided)),
                        "agreement_with_bertscore": wilson(agree, len(both_f1))}
        d_pd, d_pf, j = (res["bertscore_differences"]["ss-presidio_default"],
                         res["bertscore_differences"]["ss-presidio_faker"], res["judge"]["score"])
        res["H3"] = {"bertscore_above_presidio_default": bool(d_pd["ci95"] and d_pd["ci95"][0] > 0),
                     "judge_above_presidio_default": bool(j["ci95"] and j["ci95"][0] > 0),
                     "not_below_presidio_faker": bool(d_pf["ci95"] and d_pf["ci95"][0] > -0.02)}
        res["H3"]["supported"] = all(res["H3"].values())
        e5b[g] = res
    rows = [{"dataset": ds, "id": mid, "arm": arm, "status": replies[(ds, mid, arm)]["status"],
             "bertscore": f1.get((ds, mid, arm)), "judge": verdicts.get((ds, mid), {}).get("score")}
            for ds in datasets for mid in sample[ds] for arm in UTILITY_ARMS if (ds, mid, arm) in replies]
    return {"sample": sample, "results": e5b, "rows": rows}


def natural(run: Run, data: dict, datasets: Sequence[str] = DATASETS, pilot: Optional[int] = None,
            scorer: Optional[Callable] = None) -> dict:
    """The cost of spurious edits on prompts with nothing to protect."""
    sample = pilot_of({ds: natural_sample(data, ds) for ds in datasets}, pilot)
    reqs, edited = natural_requests(data, datasets, sample)
    replies = _fetch(run, "natural", "6-natural", "responder", reqs)

    def shown(ds, mid, arm):
        if arm in ("ss", "presidio_default") and not edited[(ds, mid, arm)]:
            arm = "rerun"                  # nothing changed: the same prompt, answered again
        row = replies.get((ds, mid, arm))
        if row is None or row["status"] == "errored":
            return None
        if arm == "ss":
            return restore_ss(row["text"], original(data, ds, mid), edits_of(data, ds, "ss", mid),
                              arm_text(data, ds, "ss", mid))
        return row["text"]

    pairs = []
    for ds in datasets:
        for mid in sample[ds]:
            ref = shown(ds, mid, "orig")
            for arm in ("rerun", "ss", "presidio_default"):
                cand = shown(ds, mid, arm)
                if ref is not None and cand is not None:
                    pairs.append(((ds, mid, arm), (cand, ref)))
    f1 = {k: v for k, v in zip([k for k, _ in pairs], bertscore([p for _k, p in pairs], run.dir / "bertscore.jsonl",
                                                                  scorer)) if v is not None}
    out = {}
    for g in _groups(datasets):
        mids = [(ds, m) for ds in datasets if _in(g, ds) for m in sample[ds]]
        res = {"n": len(mids),
               "edited": {a: wilson(sum(1 for ds, m in mids if edited[(ds, m, a)]), len(mids))
                          for a in ("ss", "presidio_default")},
               "errored": sum(1 for r in (replies[k] for k in replies if _in(g, k[0])) if r["status"] == "errored"),
               "bertscore": {a: {"n": len(v), "mean": mean(v)} for a in ("rerun", "ss", "presidio_default")
                             for v in [[f1[(ds, m, a)] for ds, m in mids if (ds, m, a) in f1]]},
               "differences": {}}
        for a, b in (("ss", "rerun"), ("presidio_default", "rerun"), ("ss", "presidio_default")):
            both = [(ds, m) for ds, m in mids if (ds, m, a) in f1 and (ds, m, b) in f1]
            res["differences"][f"{a}-{b}"] = mean_diff([f"{ds}/{m}" for ds, m in both], [f1[(ds, m, a)] for ds, m in both],
                                                       [f1[(ds, m, b)] for ds, m in both],
                                                       derive_seed("live-boot", "natural", a, b, g))
        res["edited_only"] = {}                  # the cost per edited prompt, against the noise floor
        for a in ("ss", "presidio_default"):
            both = [(ds, m) for ds, m in mids if edited[(ds, m, a)] and (ds, m, a) in f1 and (ds, m, "rerun") in f1]
            res["edited_only"][a] = {"n": len(both), "mean": mean([f1[(ds, m, a)] for ds, m in both]),
                                     "rerun_mean": mean([f1[(ds, m, "rerun")] for ds, m in both]),
                                     "minus_rerun": mean_diff([f"{ds}/{m}" for ds, m in both],
                                                              [f1[(ds, m, a)] for ds, m in both],
                                                              [f1[(ds, m, "rerun")] for ds, m in both],
                                                              derive_seed("live-boot", "natural-edited", a, g))}
        out[g] = res
    return {"sample": sample, "results": out}


# ── E7b: multi-turn through the app ──────────────────────────────────────────

@contextmanager
def app_env(cascade: Optional[Callable] = None):
    """The app's modules with a provider adapter set by the caller, settings
    fixed, and storage under a temporary home (``bench/echo.py``)."""
    import config
    import pipeline
    import settings_manager
    import surrogateshield.core.storage.shadow_map as store
    from chatbot import chat as chat_mod, providers
    saved = (store._secret_cache, config.SHADOWMAP_DIR, config.DEVICE_KEY_PATH, pipeline.sentinel_layer.run_cascade,
             chat_mod.load_settings, settings_manager.load_settings, providers.build,
             os.environ.get("SURROGATESHIELD_HOME"))
    env = SimpleNamespace(adapter=None)
    config.SHADOWMAP_DIR = config.DEVICE_KEY_PATH = None
    if cascade is not None:
        pipeline.sentinel_layer.run_cascade = cascade
    chat_mod.load_settings = lambda: {"llm_provider": "claude"}
    settings_manager.load_settings = lambda: {"llm_provider": "claude", "detailed_view": False}
    providers.build = lambda provider: (lambda payload, system: env.adapter(payload, system))
    try:
        yield env
    finally:
        (store._secret_cache, config.SHADOWMAP_DIR, config.DEVICE_KEY_PATH, pipeline.sentinel_layer.run_cascade,
         chat_mod.load_settings, settings_manager.load_settings, providers.build, home) = saved
        if home is None:
            os.environ.pop("SURROGATESHIELD_HOME", None)
        else:
            os.environ["SURROGATESHIELD_HOME"] = home


def replay_ss(env, conv_id: str, texts: Sequence[str], answer: Callable[[List[dict]], Optional[str]],
              seed: int, tmp: Optional[Path] = None) -> List[dict]:
    """Run the conversation through ``Pipeline.process_turn`` in a fresh home;
    stop after the first turn whose reply *answer* does not have yet."""
    import pipeline
    import surrogateshield.core.storage.shadow_map as store
    from chatbot.chat import ClaudeChat
    from util import Conversation
    tmp = Path(tmp or TMP)
    tmp.mkdir(parents=True, exist_ok=True)
    os.chmod(tmp, 0o700)
    home = tempfile.mkdtemp(prefix="home-", dir=tmp)
    os.environ["SURROGATESHIELD_HOME"] = home
    store._secret_cache = {}
    calls: List[tuple] = []

    def adapter(payload, system):
        payload = [{"role": m["role"], "content": m["content"]} for m in payload]
        got = answer(payload)
        calls.append((payload, system, got))
        return CAPTURED if got is None else got
    env.adapter = adapter
    turns = []
    try:
        p = pipeline.Pipeline(chat=ClaudeChat(Conversation(id=conv_id)), seed=seed)
        for text in texts:
            restored, _confirmed, smap = p.process_turn(text, interactive=False)
            if len(calls) != len(turns) + 1:
                raise RuntimeError(f"{conv_id}: turn {len(turns)} made {len(calls) - len(turns)} provider call(s)")
            payload, system, got = calls[-1]
            if system != system_prompt():
                raise RuntimeError("the app sent a different system prompt")
            turns.append({"payload": payload, "raw": got, "shown": None if got is None else restored,
                          "map": dict(smap), "mappings": dict(p.shadow.all_mappings())})
            if got is None:
                break
    finally:
        shutil.rmtree(home)
    return turns


def replay_plain(texts: Sequence[str], answer: Callable[[List[dict]], Optional[str]]) -> List[dict]:
    """An arm with no mapping: its texts in order, its raw answers as history."""
    history, turns = [], []
    for text in texts:
        payload = history + [{"role": "user", "content": text}]
        got = answer(payload)
        turns.append({"payload": payload, "raw": got, "shown": got})
        if got is None:
            break
        history = payload + [{"role": "assistant", "content": got}]
    return turns


def converse(run: Run, data: dict, convs: List[Tuple[str, str]], cascade: Optional[Callable] = None,
             log: Callable[[str], None] = print) -> Dict[tuple, dict]:
    """Advance every conversation one turn per round, one batch per round,
    until each has a reply to every turn or a failed one."""
    done: Dict[tuple, dict] = {}
    progress: Dict[tuple, int] = {}
    with app_env(cascade) as env:
        for rnd in range(1 + max(len(turn_mids(data, ds, c)) for ds, c in convs)):
            pending = []
            for ds, conv in convs:
                mids = turn_mids(data, ds, conv)
                for arm in MULTITURN_ARMS:
                    if (ds, conv, arm) in done:
                        continue
                    state: dict = {}

                    def answer(payload, name=slot(ds, conv, arm), state=state):
                        params = responder_params(payload)
                        row = run.get(name, params)
                        if run.wants(row, name, params):
                            state["pending"] = payload
                            return None
                        if row["status"] == "errored":
                            state["failed"] = row["error"]
                            return None
                        return row["text"]
                    if arm == "ss":
                        turns = replay_ss(env, conv, [original(data, ds, m) for m in mids], answer,
                                          derive_seed("live-mimic", conv))
                    else:
                        texts = [arm_text(data, ds, arm, m) for m in mids]
                        if any(t is None for t in texts):
                            done[(ds, conv, arm)] = {"turns": [], "failed": "refused"}
                            continue
                        turns = replay_plain(texts, answer)
                    if "pending" in state:
                        k = len(turns)
                        if progress.get((ds, conv, arm), 0) >= k:
                            raise RuntimeError(f"{conv} {arm}: the replay did not advance; the app is not deterministic")
                        progress[(ds, conv, arm)] = k
                        pending.append((slot(ds, conv, arm), responder_params(state["pending"])))
                    else:
                        done[(ds, conv, arm)] = {"turns": turns, "failed": state.get("failed")}
            log(f"multiturn round {rnd}: {len(pending)} turn(s) to send")
            if not pending:
                return done
            run.fetch("multiturn", "6-multiturn", "responder", pending)
    raise RuntimeError("multi-turn replay did not finish")


def _present(text: Optional[str], value: str) -> bool:
    import eval_metrics as em
    return bool(text) and em.contains_value(text, value)


def replacements(data: dict, ds: str, arm: str, mids: Sequence[str], turns: List[dict]) -> List[Dict[str, str]]:
    """Per turn, original (casefold) → the value that replaced it."""
    out = []
    for k, t in enumerate(turns):
        if arm == "ss":
            out.append({o.casefold(): s for o, s in t["map"].items()})
        else:
            text = original(data, ds, mids[k])
            out.append({text[s:e].casefold(): rep for s, e, _t, rep in edits_of(data, ds, arm, mids[k]) or []})
    return out


def score_conversation(data: dict, ds: str, conv: str, arm: str, convo: dict) -> dict:
    from surrogateshield.core.consistency import is_low_entropy
    mids = turn_mids(data, ds, conv)
    turns = convo["turns"]
    gold = [[x["value"] for x in data[ds]["units"][m]["gold"]["protect"]] for m in mids]
    all_gold = {v.casefold(): v for vs in gold for v in vs}
    reps = replacements(data, ds, arm, mids, turns)
    out = {"turns": len(turns), "complete": convo["failed"] is None and len(turns) == len(mids),
           "restore_needed": 0, "restored": 0, "history_original": 0, "history_original_user": 0,
           "payload_gold": 0, "recurring": 0, "consistent": 0, "replaced_every_turn": 0}
    mapping: Dict[str, str] = {}                       # replacement → original, so far
    for k, t in enumerate(turns):
        if arm == "ss":
            mapping = dict(t["mappings"])
        else:
            mapping.update({rep: o for o, rep in reps[k].items()})
        originals_cf = {o.casefold() for o in mapping.values()} | set(all_gold)
        seen = [s for s in mapping if s and not is_low_entropy(s) and s.casefold() not in originals_cf
                and _present(t["raw"], s)]
        if seen:
            out["restore_needed"] += 1
            out["restored"] += all(not _present(t["shown"], s) and _present(t["shown"], mapping[s]) for s in seen)
        replaced = {o for o in mapping.values() if o and not is_low_entropy(o)}
        hist = [m["content"] for m in t["payload"][:-1]]
        out["history_original"] += any(_present(c, o) for c in hist[1::2] for o in replaced)
        out["history_original_user"] += any(_present(c, o) for c in hist[0::2] + [t["payload"][-1]["content"]]
                                             for o in replaced)
        out["payload_gold"] += any(_present(m["content"], v) for m in t["payload"] for v in all_gold.values())
    for v_cf in all_gold:
        where = [k for k in range(len(turns)) if any(v.casefold() == v_cf for v in gold[k])]
        if len(where) < 2:
            continue
        out["recurring"] += 1
        got = [reps[k].get(v_cf) for k in where]
        out["replaced_every_turn"] += all(g is not None for g in got)
        out["consistent"] += all(g is not None for g in got) and len(set(got)) == 1
    return out


def multiturn(run: Run, data: dict, datasets: Sequence[str] = DATASETS, pilot: Optional[int] = None,
              cascade: Optional[Callable] = None, log: Callable[[str], None] = print) -> dict:
    sample = pilot_of({ds: multiturn_sample(data, ds) for ds in datasets}, pilot)
    convs = [(ds, c) for ds in datasets for c in sample[ds]]
    done = converse(run, data, convs, cascade, log)
    scored = {k: score_conversation(data, k[0], k[1], k[2], v) for k, v in done.items()}

    jindex, jitems = [], []
    for (ds, conv, arm), v in sorted(done.items()):
        if scored[(ds, conv, arm)]["complete"]:
            jindex.append((ds, conv, arm))
            jitems.append((slot(ds, conv, f"judge-{arm}"), judge_details_params(
                [original(data, ds, m) for m in turn_mids(data, ds, conv)], v["turns"][-1]["shown"])))
    grades = {}
    for k, row in zip(jindex, run.fetch("judge-mt", "6-judge", "judge", jitems)):
        g = (row.get("tool") or {}).get("grade") if row["status"] == "ok" else None
        grades[k] = g if g in GRADES or g == "not_needed" else None

    out = {}
    for g in _groups(datasets):
        res = {}
        for arm in MULTITURN_ARMS:
            ks = [(ds, c, arm) for ds, c in convs if _in(g, ds)]
            s = [scored[k] for k in ks if k in scored]
            tot = Counter()
            for x in s:
                tot.update({k: int(v) for k, v in x.items()})
            res[arm] = {"conversations": len(ks), "complete": tot["complete"],
                        "restored": wilson(tot["restored"], tot["restore_needed"]),
                        "history_original": wilson(tot["history_original"], tot["turns"]),
                        "history_original_user": wilson(tot["history_original_user"], tot["turns"]),
                        "payload_gold": wilson(tot["payload_gold"], tot["turns"]),
                        "consistent": wilson(tot["consistent"], tot["recurring"]),
                        "replaced_every_turn": wilson(tot["replaced_every_turn"], tot["recurring"]),
                        "grades": dict(Counter(grades.get(k) or "unavailable" for k in ks if k in grades))}
        pairs = [(ds, c) for ds, c in convs if _in(g, ds)
                 and grades.get((ds, c, "ss")) in GRADES and grades.get((ds, c, "presidio_faker")) in GRADES]
        judge = mean_diff([f"{ds}/{c}" for ds, c in pairs], [GRADES[grades[(ds, c, "ss")]] for ds, c in pairs],
                          [GRADES[grades[(ds, c, "presidio_faker")]] for ds, c in pairs],
                          derive_seed("live-boot", "multiturn-judge", g))
        rec = [(ds, c) for ds, c in convs if _in(g, ds) and (ds, c, "ss") in scored
               and (ds, c, "presidio_faker") in scored and scored[(ds, c, "ss")]["recurring"]]
        from bench.realdata.score import bootstrap
        cons = bootstrap([f"{ds}/{c}" for ds, c in rec],
                         [(scored[(ds, c, "ss")]["consistent"], scored[(ds, c, "ss")]["recurring"]) for ds, c in rec],
                         [(scored[(ds, c, "presidio_faker")]["consistent"],
                           scored[(ds, c, "presidio_faker")]["recurring"]) for ds, c in rec],
                         derive_seed("live-boot", "multiturn-consistency", g)) if rec else {"diff": None, "ci95": None}
        res["differences"] = {"judge_ss-presidio_faker": judge, "consistency_ss-presidio_faker": cons}
        res["H4"] = {"judge_above_presidio_faker": bool(judge["ci95"] and judge["ci95"][0] > 0),
                     "consistency_above_presidio_faker": bool(cons.get("ci95") and cons["ci95"][0] > 0)}
        res["H4"]["supported"] = all(res["H4"].values())
        out[g] = res
    histories = {f"{ds}/{conv}": done[(ds, conv, "ss")]["turns"] for ds, conv in convs
                 if (ds, conv, "ss") in done and scored[(ds, conv, "ss")]["complete"]}
    run.write_private(HISTORIES,
                      [{"dataset": k.split("/")[0], "conv": k.split("/")[1],
                        "messages": v[-1]["payload"] + [{"role": "assistant", "content": v[-1]["raw"]}]}
                       for k, v in sorted(histories.items())])
    return {"sample": sample, "results": out,
            "rows": [{"dataset": k[0], "conv": k[1], "arm": k[2], **scored[k], "grade": grades.get(k)}
                     for k in sorted(scored)]}


# ── E8: the attacker ─────────────────────────────────────────────────────────

_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                   "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_STREET = {"street", "avenue", "road", "drive", "lane", "court", "suite", "apt", "boulevard", "way", "place"}
_STOP = {"the", "and", "for", "of"}
_WORD = re.compile(r"[a-z]+")


def _words(v: str) -> set:
    return {w for w in _WORD.findall(v.casefold()) if len(w) > 2 and w not in _STOP}


def _number(v: str) -> Optional[int]:
    v = v.casefold()
    d = re.findall(r"\d+", v)
    if d:
        return int(d[0])
    total, hit = 0, False
    for w in re.findall(r"[a-z]+", v):
        if w in _NUM:
            total, hit = total + _NUM[w], True
        elif w in _TENS:
            total, hit = total + _TENS[w], True
        elif w == "hundred" and hit:
            total *= 100
    return total if hit else None


def _digits(v: str) -> str:
    """Digits of a phone number written with digits or digit words
    ("double five" is 55)."""
    words = {**{k: str(i) for k, i in _NUM.items() if i < 10}, "oh": "0"}
    out, times = [], 1
    for tok in re.findall(r"\d|[a-z]+", v.casefold()):
        if tok in ("double", "triple"):
            times = 2 if tok == "double" else 3
            continue
        d = tok if tok.isdigit() else words.get(tok, "")
        if d:
            out.append(d * times)
            times = 1
    return "".join(out)


def _email(v: str) -> str:
    v = v.strip().casefold()
    if "@" not in v:
        v = re.sub(r"\s*[\[(]?\bat\b[\])]?\s*", "@", v, count=1)
        v = re.sub(r"\s*[\[(]?\bdot\b[\])]?\s*", ".", v)
    return re.sub(r"\s+", "", v).strip(".,;:'\"<>")


def _date(v: str) -> Optional[str]:
    from dateutil import parser
    try:
        a = parser.parse(v, fuzzy=True, default=datetime(1900, 1, 1))
        b = parser.parse(v, fuzzy=True, default=datetime(1904, 2, 2))
    except (ValueError, OverflowError):
        return None
    return a.date().isoformat() if a == b else None


def _ip(v: str) -> Optional[str]:
    import ipaddress
    try:
        return ipaddress.ip_address(v.strip().strip("[]")).compressed
    except ValueError:
        return None


def _url(v: str) -> str:
    v = re.sub(r"^[a-z]+://", "", v.strip().casefold())
    return re.sub(r"^www\.", "", v).rstrip("/")


def _plain(v: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", v.casefold())).strip()


def canon(typ: str, v: str):
    """The form two values of *typ* are compared in."""
    v = str(v).strip()
    if typ == "EMAIL":
        return _email(v)
    if typ == "PHONE":
        return _digits(v)
    if typ == "AGE":
        return _number(v)
    if typ == "DATE_OF_BIRTH":
        return _date(v) or _plain(v)
    if typ == "PERSON":
        return tuple(sorted(_WORD.findall(v.casefold())))
    if typ in ("ID",):
        return re.sub(r"[^0-9a-z]", "", v.casefold())
    if typ == "NETWORK":
        ip = _ip(v)
        if ip:
            return ip
        hexd = re.sub(r"[^0-9a-f]", "", v.casefold())
        return hexd if len(hexd) == 12 else re.sub(r"\s+", "", v.casefold())
    if typ == "URL":
        return _url(v)
    if typ == "HANDLE":
        return v.casefold().lstrip("@")
    if typ == "CREDENTIAL":
        return v
    return _plain(v)                          # ORG, LOCATION, ADDRESS


def _year(v: str) -> Optional[str]:
    m = re.search(r"\b(1[89]\d\d|20\d\d)\b", v)
    return m.group(1) if m else None


def _area(digits: str) -> str:
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[:3] if len(digits) == 10 else ""


def recovery(typ: str, guess: str, gold: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(level, class, part)`` of one guess against one gold value: level is
    ``"exact"``, ``"partial"`` or None; *part* is what a partial recovered."""
    if not str(guess).strip() or not gold:
        return None, None, None
    g, t = canon(typ, guess), canon(typ, gold)
    if t not in (None, "", ()) and g == t:
        return "exact", None, None
    if typ == "PHONE" and len(g) >= 9 and len(t) >= 9 and g[-9:] == t[-9:]:
        return "exact", None, None
    if typ == "EMAIL" and "@" in g and "@" in t:
        gl, gd = g.rsplit("@", 1)
        tl, td = t.rsplit("@", 1)
        if gd == td:
            return "partial", "email_domain", td
        if gl == tl:
            return "partial", "email_local", tl
    if typ == "PHONE":
        a = _area(t)
        if a and a == _area(g):
            return "partial", "phone_area", a
    if typ == "DATE_OF_BIRTH":
        y = _year(gold)
        if y and y == _year(guess):
            return "partial", "birth_year", y
    if typ == "NETWORK" and _ip(gold) and _ip(guess) and "." in t and "." in g:
        if t.rsplit(".", 1)[0] == g.rsplit(".", 1)[0]:
            return "partial", "ipv4_24", t.rsplit(".", 1)[0]
    if typ == "URL":
        th, gh = t.split("/", 1)[0], g.split("/", 1)[0]
        if th and th == gh:
            return "partial", "url_host", th
    words = {"PERSON": "name_token", "ORG": "org_word", "ADDRESS": "address_part", "LOCATION": "place_word"}
    if typ in words:
        shared = _words(guess) & (_words(gold) - (_STREET if typ == "ADDRESS" else set()))
        if shared:
            return "partial", words[typ], sorted(shared)[0]
    return None, None, None


def attack_outcomes(gold: Sequence[Tuple[str, str]], left_in: Dict[str, str], parsed: Optional[dict],
                    seen_text: str) -> List[dict]:
    """One outcome per gold value: ``leaked`` / ``policy`` (the arm left it in;
    counted apart), ``unavailable`` (no valid reply), ``exact``, ``partial``
    or ``not_recovered``."""
    import attacker
    guesses = attacker._guesses(parsed) if parsed is not None else []
    out = []
    for typ, val in gold:
        row = {"type": typ, "outcome": None, "class": None, "visible": None}
        if val in left_in:
            row["outcome"] = left_in[val]
        elif parsed is None:
            row["outcome"] = "unavailable"
        else:
            best = None
            for gs in guesses:
                lvl, cls, part = recovery(typ, gs, val)
                if lvl == "exact":
                    best = ("exact", None, None)
                    break
                if lvl == "partial" and best is None:
                    best = ("partial", cls, part)
            if best is None:
                row["outcome"] = "not_recovered"
            else:
                row["outcome"], row["class"] = best[0], best[1]
                if best[2] is not None:
                    row["visible"] = str(best[2]).casefold() in seen_text.casefold()
        out.append(row)
    return out


def message_gold(unit: dict) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for x in unit["gold"]["protect"]:
        if x["value"].casefold() not in seen:
            seen.add(x["value"].casefold())
            out.append((x["type"], x["value"]))
    return out


def left_in_by_arm(unit: dict, edits: list) -> Dict[str, str]:
    """Gold values the arm sent (the scorer's leak, or its service-query policy)."""
    from bench import realworld as rw
    text = unit["gold"]["text"]
    r = rw.score_message(unit["gold"], SimpleNamespace(edits=[(s, e, text[s:e], rep) for s, e, _t, rep in edits]))
    return {**{x["value"]: "policy" for x in r["policy"]}, **{x["value"]: "leaked" for x in r["leaked"]}}


def _parse(row: dict) -> Optional[dict]:
    import attacker
    return attacker.parse_reply(row["text"] or "") if row["status"] == "ok" else None


def summarise_attack(rows: Sequence[dict]) -> dict:
    c = Counter(r["outcome"] for r in rows)
    n = c["exact"] + c["partial"] + c["not_recovered"]
    classes = Counter(r["class"] for r in rows if r["outcome"] == "partial")
    visible = Counter(r["class"] for r in rows if r["outcome"] == "partial" and r["visible"])
    return {"values": len(rows), "leaked": c["leaked"], "policy": c["policy"], "unavailable": c["unavailable"],
            "refused": c["refused"], "attacked": n, "exact": wilson(c["exact"], n), "partial": wilson(c["partial"], n),
            "exposed": wilson(c["exact"] + c["partial"] + c["leaked"] + c["policy"],
                              len(rows) - c["unavailable"] - c["refused"]),
            "partial_classes": {k: {"n": v, "visible_in_text": visible[k]} for k, v in sorted(classes.items())},
            "exact_types": dict(sorted(Counter(r["type"] for r in rows if r["outcome"] == "exact").items()))}


def attack(run: Run, data: dict, datasets: Sequence[str] = DATASETS, pilot: Optional[int] = None,
           conversations: bool = True) -> dict:
    prompts = attack_prompts()
    sample = pilot_of({ds: attack_sample(data, ds) for ds in datasets}, pilot)
    replies = _fetch(run, "attack", "6-attacker", "attacker", attack_requests(data, datasets, sample))
    rows: List[dict] = []
    for ds in datasets:
        for mid in sample[ds]:
            unit = data[ds]["units"][mid]
            for arm in ATTACK_ARMS:
                if (ds, mid, arm) not in replies:
                    rows += [{"dataset": ds, "id": mid, "arm": arm, "type": t, "outcome": "refused", "class": None,
                              "visible": None} for t, _v in message_gold(unit)]
                    continue
                row = replies[(ds, mid, arm)]
                rows += [{"dataset": ds, "id": mid, "arm": arm, **o} for o in attack_outcomes(
                    message_gold(unit), left_in_by_arm(unit, edits_of(data, ds, arm, mid)), _parse(row),
                    arm_text(data, ds, arm, mid))]

    conv_rows: List[dict] = []
    hist_path = run.dir / HISTORIES
    if conversations and hist_path.exists():
        hist = [h for h in read_jsonl(hist_path) if h["dataset"] in datasets]
        if pilot is not None:
            hist = hist[:pilot]
        texts = [conversation_text(h["messages"]) for h in hist]
        creplies = run.fetch("attack-conv", "6-attacker", "attacker",
                             [(slot(h["dataset"], h["conv"], "attack-conv"), attack_params(prompts["conversation"], t))
                              for h, t in zip(hist, texts)])
        for h, text, row in zip(hist, texts, creplies):
            ds, conv = h["dataset"], h["conv"]
            units = [data[ds]["units"][m] for m in turn_mids(data, ds, conv)]
            gold, seen = [], set()
            for u in units:
                for t, v in message_gold(u):
                    if v.casefold() not in seen:
                        seen.add(v.casefold())
                        gold.append((t, v))
            sent = [m["content"] for m in h["messages"] if m["role"] == "user"]
            left = {v: "leaked" for _t, v in gold if any(_present(s, v) for s in sent)}
            conv_rows += [{"dataset": ds, "id": conv, "arm": "ss-conversation", **o}
                          for o in attack_outcomes(gold, left, _parse(row), text)]

    out = {}
    for g in _groups(datasets):
        res = {arm: summarise_attack([r for r in rows if r["arm"] == arm and _in(g, r["dataset"])])
               for arm in ATTACK_ARMS}
        if conv_rows:
            res["ss-conversation"] = summarise_attack([r for r in conv_rows if _in(g, r["dataset"])])
        res["H5"] = {"supported": res["ss"]["attacked"] > 0 and res["ss"]["exact"]["k"] == 0,
                     "exact": res["ss"]["exact"]}
        if conv_rows:
            res["H5-conv"] = {"supported": res["ss-conversation"]["exact"]["k"] == 0,
                              "exact": res["ss-conversation"]["exact"]}
        out[g] = res
    return {"sample": sample, "results": out, "rows": rows + conv_rows,
            "unparsed": {a: sum(1 for k, r in replies.items() if k[2] == a and _parse(r) is None) for a in ATTACK_ARMS}}


# ── estimate ─────────────────────────────────────────────────────────────────

def estimate(run: Run, data: dict, datasets: Sequence[str] = DATASETS) -> dict:
    """Calls each phase still needs at full size (requests whose text depends
    on earlier replies are counted by slot, as upper bounds), and tokens per
    call from the replies cached so far."""
    def missing(reqs):
        return sum(1 for _k, (s, p) in reqs if run.wants(run.get(s, p), s, p))
    have = Counter(r["slot"] for r in run.cache.values() if r["status"] != "errored" or not run.retry_errors)

    def lacking(slots):
        return sum(max(0, n - have[s]) for s, n in slots)
    util = {ds: utility_sample(data, ds) for ds in datasets}
    nat, _edited = natural_requests(data, datasets, {ds: natural_sample(data, ds) for ds in datasets})
    convs = [(ds, c) for ds in datasets for c in multiturn_sample(data, ds)]
    plan = {"utility responder": missing(utility_requests(data, datasets, util)),
            "utility judge": lacking((slot(ds, m, "judge"), 1) for ds in datasets for m in util[ds]),
            "natural responder": missing(nat),
            "multiturn responder (at most)": lacking((slot(ds, c, a), len(turn_mids(data, ds, c)))
                                                     for ds, c in convs for a in MULTITURN_ARMS),
            "multiturn judge": lacking((slot(ds, c, f"judge-{a}"), 1) for ds, c in convs for a in MULTITURN_ARMS),
            "attacker single": missing(attack_requests(data, datasets, {ds: attack_sample(data, ds)
                                                                          for ds in datasets})),
            "attacker conversation": lacking((slot(ds, c, "attack-conv"), 1) for ds, c in convs)}
    used = run.ledger.total()
    tokens: Dict[str, dict] = {}
    for r in run.cache.values():
        if r.get("usage"):
            t = tokens.setdefault(r["kind"], {"n": 0, "input_tokens": 0, "output_tokens": 0})
            t["n"] += 1
            t["input_tokens"] += r["usage"].get("input_tokens") or 0
            t["output_tokens"] += r["usage"].get("output_tokens") or 0
    per = {k: {"in": round(v["input_tokens"] / v["n"]), "out": round(v["output_tokens"] / v["n"]), "n": v["n"]}
           for k, v in tokens.items() if v["n"]}
    return {"ledger_total": used, "cap": run.ledger.cap, "remaining": run.ledger.cap - used, "plan": plan,
            "planned_total": sum(plan.values()), "after": used + sum(plan.values()), "tokens_per_call": per}


# ── output ───────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _calls(run: Run, phases: Sequence[str]) -> dict:
    s = run.ledger.summary()
    return {p: s[p] for p in phases if p in s}


def write_result(doc: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _bs(x: Optional[dict]) -> str:
    if not x or x.get("diff") is None:
        return "–"
    return f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def _r(x: Optional[dict]) -> str:
    if not x or not x.get("n"):
        return "–"
    lo, hi = x["wilson95"]
    return f"{x['k']}/{x['n']} ({x['rate']:.1%}, {lo:.1%}–{hi:.1%})"


def markdown(kind: str, doc: dict) -> str:
    lines = [f"# {kind} (real data, test)", "", f"Command: `{doc['command']}` at commit `{doc['git']['commit'][:12]}`"
             + (f" with {len(doc['git']['modified'])} modified tracked file(s)" if doc["git"]["modified"] else ""), ""]
    if kind == "utility":
        lines += ["| group | n | BERTScore ss | ss raw | presidio_default | presidio_faker | ss − pd | ss − pf | "
                  "judge ss/pd/tie | judge score | H3 |", "|" + "---|" * 11]
        for g, r in doc["e5b"].items():
            b = r["bertscore"]
            lines.append(f"| {g} | {r['n']} | {b['ss']['mean']} | {b['ss_raw']['mean']} | {b['presidio_default']['mean']} "
                         f"| {b['presidio_faker']['mean']} | {_bs(r['bertscore_differences']['ss-presidio_default'])} "
                         f"| {_bs(r['bertscore_differences']['ss-presidio_faker'])} | {r['judge']['ss_wins']}/"
                         f"{r['judge']['presidio_default_wins']}/{r['judge']['ties']} | {_bs(r['judge']['score'])} "
                         f"| {r['H3']['supported']} |")
        lines += ["", "Natural slice (nothing to protect): BERTScore against the answer to the untouched prompt.", "",
                  "| group | n | edited ss | edited pd | rerun | ss | pd | ss − rerun | pd − rerun | ss − pd |",
                  "|" + "---|" * 10]
        for g, r in doc["natural"].items():
            b, d = r["bertscore"], r["differences"]
            lines.append(f"| {g} | {r['n']} | {_r(r['edited']['ss'])} | {_r(r['edited']['presidio_default'])} "
                         f"| {b['rerun']['mean']} | {b['ss']['mean']} | {b['presidio_default']['mean']} "
                         f"| {_bs(d['ss-rerun'])} | {_bs(d['presidio_default-rerun'])} | {_bs(d['ss-presidio_default'])} |")
    elif kind == "multiturn":
        lines += ["| group | arm | complete | restored | history original | gold in request | consistent | "
                  "grades |", "|" + "---|" * 8]
        for g, r in doc["results"].items():
            for arm in MULTITURN_ARMS:
                a = r[arm]
                lines.append(f"| {g} | {arm} | {a['complete']}/{a['conversations']} | {_r(a['restored'])} "
                             f"| {_r(a['history_original'])} | {_r(a['payload_gold'])} | {_r(a['consistent'])} "
                             f"| {a['grades']} |")
            lines.append(f"| {g} | ss − pf | | | | | {_bs(r['differences']['consistency_ss-presidio_faker'])} "
                         f"| judge {_bs(r['differences']['judge_ss-presidio_faker'])}; H4 {r['H4']['supported']} |")
    else:
        lines += ["| group | arm | values | left in (leak / policy) | unavailable | exact | partial | exposed |",
                  "|" + "---|" * 8]
        for g, r in doc["results"].items():
            for arm in (*ATTACK_ARMS, "ss-conversation"):
                if arm in r:
                    a = r[arm]
                    lines.append(f"| {g} | {arm} | {a['values']} | {a['leaked']} / {a['policy']} | {a['unavailable']} "
                                 f"| {_r(a['exact'])} | {_r(a['partial'])} | {_r(a['exposed'])} |")
    return "\n".join(lines) + "\n"


def _publish(kind: str, doc: dict, out: Path) -> None:
    write_result(doc, out)
    out.with_suffix(".md").write_text(markdown(kind, doc))


def _command(sub: str, out: Path) -> str:
    return f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.live {sub} --out {_rel(out)}"


def main(argv=None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("PYTHONHASHSEED") != "0":            # the app's detection must replay identically
        from bench.arms.base import arm_env
        os.execve(sys.executable, [sys.executable, "-m", "bench.realdata.live", *args_in], arm_env())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for p in (str(ROOT), str(ROOT / "python-library")):
        if p not in sys.path:
            sys.path.insert(0, p)

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("estimate", "utility", "multiturn", "attacker"))
    ap.add_argument("--run", default="main")
    ap.add_argument("--pilot", type=int, help="first N rows only; writes no result")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-calls", type=int, help="stop before this invocation sends more than this")
    ap.add_argument("--retry-errors", action="store_true", help="resend requests stored as errors (counted)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args(args_in)
    check_roles()
    run = Run(args.run, max_calls=args.max_calls, retry_errors=args.retry_errors)
    data = load_test(args.datasets)

    if args.what == "estimate":
        print(json.dumps(estimate(run, data, args.datasets), indent=1))
        return 0
    if args.pilot is None and args.out is None:
        ap.error("--out is required without --pilot")
    common = {"git": git_state(), "models": {"responder": pv.SONNET, "judge": pv.OPUS, "attacker": pv.OPUS},
              "prompts": prompt_record(), "run": args.run, "datasets": args.datasets}
    if args.what == "utility":
        e5b = utility(run, data, args.datasets, args.pilot)
        nat = natural(run, data, args.datasets, args.pilot)
        run.write_private(f"utility-rows{'-pilot' if args.pilot else ''}.jsonl", e5b["rows"])
        doc = {"command": _command("utility", args.out or Path("pilot.json")), **common,
               "samples": {"e5b": e5b["sample"], "natural": nat["sample"]}, "e5b": e5b["results"],
               "natural": nat["results"], "bertscore": {"model": "roberta-large", "rescaled": True,
                                                        "reference": "the answer to the original prompt"},
               "calls": _calls(run, ("6-utility", "6-natural", "6-judge"))}
        kind = "utility"
    elif args.what == "multiturn":
        mt = multiturn(run, data, args.datasets, args.pilot)
        run.write_private(f"multiturn-rows{'-pilot' if args.pilot else ''}.jsonl", mt["rows"])
        doc = {"command": _command("multiturn", args.out or Path("pilot.json")), **common, "samples": mt["sample"],
               "results": mt["results"], "rows": mt["rows"], "calls": _calls(run, ("6-multiturn", "6-judge"))}
        kind = "multiturn"
    else:
        at = attack(run, data, args.datasets, args.pilot)
        run.write_private(f"attacker-rows{'-pilot' if args.pilot else ''}.jsonl", at["rows"])
        doc = {"command": _command("attacker", args.out or Path("pilot.json")), **common, "samples": at["sample"],
               "results": at["results"], "unparsed": at["unparsed"], "calls": _calls(run, ("6-attacker",))}
        kind = "attacker"
    doc["ledger_total"] = run.ledger.total()
    if args.pilot is not None:
        print(markdown(kind, doc))
        print(json.dumps(estimate(run, data, args.datasets), indent=1))
        return 0
    _publish(kind, doc, args.out)
    print(markdown(kind, doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
