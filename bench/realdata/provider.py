"""Provider access for the real-data run: a persistent call ledger with a hard
cap, and a resumable Message Batches runner.

Every request sent to the provider is one *call*, whether synchronous or one
request inside a batch, and is taken from the ledger **before** it is sent; a
call that would bring the run's total over the cap raises ``BudgetExceeded``
and nothing is sent. The ledger is a directory of append-only JSONL files
under the git-ignored ``experiment/realdata/ledger/``, one new file per
process (files are only ever added there), and the total is the sum over all
of them. Rows hold counts and token usage, never text.

A batch's state (its id, the hash of its request ids) is written to
``bench/realdata/build/batches/<name>.json`` as soon as it is created, so a
rerun after a crash or a timeout fetches the same batch instead of paying for
it twice. The API key comes from the environment or ``.env`` through
``chatbot.providers.claude_client_kwargs`` and is never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from bench.realdata.common import BUILD, ROOT

CAP = 4000                                  # PROMPT_FOR_OPUS_NEW §6 default (blanks left unfilled)
LEDGER = ROOT / "experiment" / "realdata" / "ledger"
BATCHES = BUILD / "batches"
SONNET = "claude-sonnet-4-6"                # annotator / placer / responder
OPUS = "claude-opus-5-5"                    # attacker / judge


class BudgetExceeded(RuntimeError):
    """A provider call would cross the run's call cap."""


class Ledger:
    def __init__(self, directory: Path = LEDGER, cap: int = CAP, run: str = "run"):
        self.dir, self.cap = Path(directory), cap
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.path = self.dir / f"{stamp}-{os.getpid()}-{run}.jsonl"

    def rows(self) -> List[dict]:
        out = []
        for f in sorted(self.dir.glob("*.jsonl")) if self.dir.exists() else []:
            with open(f, encoding="utf-8") as fh:
                out += [json.loads(l) for l in fh if l.strip()]
        return out

    def total(self) -> int:
        return sum(r.get("calls", 0) for r in self.rows())

    def _append(self, row: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def take(self, n: int, phase: str, kind: str, model: str) -> None:
        """Reserve *n* calls, or raise without reserving any."""
        if n <= 0:
            raise ValueError("take at least one call")
        total = self.total()
        if total + n > self.cap:
            raise BudgetExceeded(f"{n} more call(s) would bring the total to {total + n}, "
                                 f"over the cap of {self.cap} (already used {total})")
        self._append({"phase": phase, "kind": kind, "model": model, "calls": n})

    def usage(self, phase: str, kind: str, model: str, usage: dict) -> None:
        self._append({"phase": phase, "kind": kind, "model": model, "calls": 0,
                      **{k: int(usage.get(k) or 0) for k in USAGE_KEYS}})

    def summary(self) -> Dict[str, dict]:
        out: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for r in self.rows():
            s = out[r["phase"]]
            s["calls"] += r.get("calls", 0)
            for k in USAGE_KEYS:
                s[k] += r.get(k, 0)
        return {p: dict(v) for p, v in out.items()}


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def add_usage(acc: dict, usage: Optional[dict]) -> dict:
    for k in USAGE_KEYS:
        acc[k] = acc.get(k, 0) + int((usage or {}).get(k) or 0)
    return acc


def client():
    """An Anthropic client with the project's key handling; no SDK retries
    (each retry would be an uncounted call)."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from chatbot.providers import claude_client_kwargs
    return anthropic.Anthropic(**claude_client_kwargs(), max_retries=0, timeout=600)


def call(cl, ledger: Ledger, phase: str, kind: str, params: dict) -> dict:
    """One synchronous, counted call; returns the message as a dict."""
    ledger.take(1, phase, kind, params["model"])
    msg = cl.messages.create(**params).model_dump()
    ledger.usage(phase, kind, params["model"], msg.get("usage") or {})
    return msg


def _ids_hash(requests: List[dict]) -> str:
    return hashlib.sha256("\n".join(r["custom_id"] for r in requests).encode()).hexdigest()


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def run_batch(cl, ledger: Ledger, name: str, phase: str, kind: str, requests: List[dict],
              poll_s: float = 30.0, log: Callable[[str], None] = print,
              directory: Path = BATCHES, sleep: Callable[[float], None] = time.sleep) -> Dict[str, dict]:
    """Send *requests* (``{"custom_id", "params"}``) as one Message Batch and
    return ``custom_id -> {"status": "succeeded", "message": {...}}`` or
    ``{"status": "errored" | "expired" | "canceled", "error": "..."}``.

    Resumable by *name*: if ``<directory>/<name>.json`` exists, the batch it
    names is fetched (no new calls), or, once its results are saved, read from
    disk without a client; its request ids must be the same."""
    if not requests:
        return {}
    if len({r["custom_id"] for r in requests}) != len(requests):
        raise ValueError("custom_id values must be unique")
    state_path = directory / f"{name}.json"
    results_path = directory / f"{name}.results.jsonl"
    model = requests[0]["params"]["model"]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["ids_sha256"] != _ids_hash(requests):
            raise ValueError(f"{state_path} belongs to a different set of requests")
        if state.get("usage_recorded") and results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
            return {r.pop("custom_id"): r for r in rows}
        if cl is None:
            raise RuntimeError(f"{name}: batch {state['batch_id']} has no saved results; a client is needed")
        log(f"{name}: resuming batch {state['batch_id']}")
    else:
        if cl is None:
            raise RuntimeError(f"{name}: not sent yet; a client is needed")
        ledger.take(len(requests), phase, kind, model)
        batch = cl.messages.batches.create(requests=requests)
        state = {"batch_id": batch.id, "ids_sha256": _ids_hash(requests), "n": len(requests),
                 "phase": phase, "kind": kind, "model": model, "usage_recorded": False}
        _write_private(state_path, json.dumps(state, indent=1, sort_keys=True) + "\n")
        log(f"{name}: created batch {batch.id} with {len(requests)} requests")
    while True:
        b = cl.messages.batches.retrieve(state["batch_id"])
        if b.processing_status == "ended":
            break
        c = b.request_counts
        log(f"{name}: {b.processing_status} (processing {c.processing}, succeeded {c.succeeded}, "
            f"errored {c.errored})")
        sleep(poll_s)
    out: Dict[str, dict] = {}
    usage: dict = {}
    for r in cl.messages.batches.results(state["batch_id"]):
        res = r.result
        if res.type == "succeeded":
            msg = res.message.model_dump()
            out[r.custom_id] = {"status": "succeeded", "message": msg}
            add_usage(usage, msg.get("usage"))
        else:
            err = getattr(res, "error", None)
            out[r.custom_id] = {"status": res.type, "error": str(err.model_dump() if err else res.type)}
    missing = {q["custom_id"] for q in requests} - set(out)
    for cid in missing:
        out[cid] = {"status": "missing", "error": "no result returned"}
    _write_private(results_path, "".join(json.dumps({"custom_id": k, **v}, ensure_ascii=False) + "\n"
                                         for k, v in sorted(out.items())))
    if not state.get("usage_recorded"):
        ledger.usage(phase, kind, model, usage)
        state["usage_recorded"] = True
        _write_private(state_path, json.dumps(state, indent=1, sort_keys=True) + "\n")
    return out


def tool_input(message: dict, tool: str) -> Optional[dict]:
    """The input of the first ``tool_use`` block named *tool*, if any."""
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == tool:
            return block.get("input")
    return None


def requests_of(items: Iterable[tuple]) -> List[dict]:
    """``[(custom_id, params), ...]`` → the Batches API request list."""
    return [{"custom_id": cid, "params": params} for cid, params in items]
