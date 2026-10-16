#!/usr/bin/env python3
# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""Gate J15 — latency of ``Session.mask`` on a 200-token message.

    python bench/perf.py                              # table
    python bench/perf.py --json bench/results/j15_perf.json
    python bench/perf.py --messages 10 --rounds 1     # quicker

Warm: N distinct invented messages of 190–210 spaCy tokens (``spacy.blank``
tokenizer), built from a seeded template pool so each one carries a name,
contact details, a place, an employer and an ID. Each mask runs in a fresh
in-memory ``Session`` (so no surrogate is a cache hit) after one warm-up pass
that loads the models; p50/p95/max over N × rounds masks. Measured without
and with ContextGuard. Gate: p50 ≤ 50 ms without, ≤ 150 ms with.

Cold: one child process per mode times ``import surrogateshield``, then the
first ``mask`` (model loading included), and reports its own peak RSS.

No network: run with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 after the models
are cached (``surrogateshield doctor`` says whether they are).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python-library"))

GATE_MS = {False: 50.0, True: 150.0}

# invented people, places and numbers; none is a real person's
_FIRST = ["Dana", "Kofi", "Ines", "Mateo", "Priya", "Harriet", "Yusuf", "Lena", "Tomasz", "Aiko"]
_LAST = ["Whitfield", "Mensah", "Carvalho", "Ortega", "Raman", "Bligh", "Demir", "Kraus", "Nowak", "Sato"]
_TOWN = ["Tempe, AZ", "Boise, ID", "Dayton, OH", "Leeds", "Guelph, ON", "Ballarat", "Eugene, OR"]
_ORG = ["Harbor Point Logistics", "Ferncliff Dental", "Bluegate Analytics", "Ridgeway Middle School",
        "Copperline Credit Union", "Northstar Physio"]
_SENTENCES = [
    "Hi, my name is {name} and I work at {org} in {town}.",
    "You can reach me at {email} or on {phone} most afternoons.",
    "My manager asked me to write a short note explaining why the quarterly report slipped by two weeks.",
    "The main reason is that the vendor changed their invoicing system and every purchase order had to be re-entered by hand.",
    "I am {age} years old and this is my first job in operations, so I want the tone to be calm and professional.",
    "Our account number with the vendor is {acct} if that matters for the wording.",
    "Please keep it under two hundred words and avoid blaming anyone directly.",
    "I also need to mention that the new process should prevent the same delay next quarter.",
    "My home address is {street}, {town}, in case the HR form needs it.",
    "Could you also suggest a subject line that does not sound defensive?",
    "Last time I wrote one of these my director said it read like an apology letter.",
    "For context, the team is five people and we handle shipping for three warehouses.",
    "My employee ID is {emp} and my date of birth is {dob}.",
    "Thanks in advance, this has been a stressful week and I appreciate the help.",
]


def _fill(rng: random.Random) -> dict:
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    return {
        "name": f"{first} {last}", "org": rng.choice(_ORG), "town": rng.choice(_TOWN),
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "phone": f"(480) 555-{rng.randint(100, 199):04d}", "age": rng.randint(22, 64),
        "acct": f"{rng.randint(10**7, 10**8 - 1)}", "street": f"{rng.randint(10, 9999)} Juniper Lane",
        "emp": f"E-{rng.randint(10000, 99999)}",
        "dob": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1960, 2002)}",
    }


def messages(n: int, seed: int = 0, lo: int = 190, hi: int = 210) -> List[str]:
    """n distinct messages of lo..hi spaCy tokens; the first two sentences
    always carry a name, employer, town, email and phone."""
    import spacy
    tok = spacy.blank("en").tokenizer
    rng, out = random.Random(seed), []
    while len(out) < n:
        f = _fill(rng)
        body = _SENTENCES[2:]
        rng.shuffle(body)
        parts = [s.format(**f) for s in _SENTENCES[:2]]
        for s in body:
            if len(tok(" ".join(parts))) >= lo:
                break
            parts.append(s.format(**f))
        text = " ".join(parts)
        if lo <= len(tok(text)) <= hi and text not in out:
            out.append(text)
    return out


def warm(texts: List[str], *, context_guard: bool, rounds: int, seed: int) -> dict:
    from surrogateshield import Session
    from surrogateshield._state import cfg
    c = dataclasses.replace(cfg, context_guard_enabled=context_guard)
    with Session(config=c, seed=seed) as s:
        for t in texts:
            s.mask(t)                                   # warm-up: model loads
    ms: List[float] = []
    for r in range(rounds):
        for i, t in enumerate(texts):
            with Session(config=c, seed=seed + r * len(texts) + i) as s:
                t0 = time.perf_counter()
                s.mask(t)
                ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    p50 = statistics.median(ms)
    return {"context_guard": context_guard, "masks": len(ms),
            "p50_ms": round(p50, 1), "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1),
            "max_ms": round(ms[-1], 1), "gate_ms": GATE_MS[context_guard],
            "pass": p50 <= GATE_MS[context_guard]}


_COLD = r"""
import json, resource, sys, time, dataclasses
t0 = time.perf_counter()
import surrogateshield
from surrogateshield import Session
from surrogateshield._state import cfg
t1 = time.perf_counter()
c = dataclasses.replace(cfg, context_guard_enabled=sys.argv[1] == "1")
with Session(config=c, seed=0) as s:
    s.mask(sys.argv[2])
t2 = time.perf_counter()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10   # bytes vs KiB
print(json.dumps({"import_s": round(t1 - t0, 2), "first_mask_s": round(t2 - t1, 2),
                  "peak_rss_mb": round(rss_mb)}))
"""


def cold(text: str, *, context_guard: bool) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python-library"))
    p = subprocess.run([sys.executable, "-c", _COLD, "1" if context_guard else "0", text],
                       env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"cold-start child failed (rc={p.returncode}): {p.stderr.strip()[-400:]}")
    return {"context_guard": context_guard, **json.loads(p.stdout.strip().splitlines()[-1])}


def environment() -> dict:
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cpus": os.cpu_count(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--messages", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cold", action="store_true", help="skip the cold-start children")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    texts = messages(a.messages, a.seed)
    import spacy
    tok = spacy.blank("en").tokenizer
    lengths = [len(tok(t)) for t in texts]
    res = {"command": "python bench/perf.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "environment": environment(),
           "tokens": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
           "warm": [warm(texts, context_guard=m, rounds=a.rounds, seed=a.seed) for m in (False, True)],
           "cold": [] if a.no_cold else [cold(texts[0], context_guard=m) for m in (False, True)]}

    print(f"{a.messages} messages, {res['tokens']['min']}–{res['tokens']['max']} tokens, "
          f"{a.rounds} rounds, fresh session per mask")
    for w in res["warm"]:
        name = "with ContextGuard" if w["context_guard"] else "without ContextGuard"
        print(f"  warm {name:<22} p50 {w['p50_ms']:>7.1f} ms  p95 {w['p95_ms']:>7.1f}  "
              f"max {w['max_ms']:>7.1f}   gate ≤{w['gate_ms']:.0f}  {'PASS' if w['pass'] else 'FAIL'}")
    for c in res["cold"]:
        name = "with ContextGuard" if c["context_guard"] else "without ContextGuard"
        print(f"  cold {name:<22} import {c['import_s']:.2f} s  first mask {c['first_mask_s']:.2f} s"
              f"  peak RSS {c['peak_rss_mb']} MB")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0 if all(w["pass"] for w in res["warm"]) else 1


if __name__ == "__main__":
    sys.exit(main())
