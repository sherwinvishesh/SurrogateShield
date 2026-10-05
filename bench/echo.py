"""Gate J4 — multi-turn round trip with an echo LLM.

One ``Pipeline`` (one conversation, one shadow map) processes N messages. The
"LLM" returns the sanitised prompt verbatim, so the restored answer must
equal the user's message exactly. Anything else is a corrupted turn: a
surrogate left in, an original restored into the wrong place, or unrelated
text rewritten by an earlier turn's mapping (audit I5, E2, I15).

Also checked, per conversation:

* reuse   — an original seen in several turns gets one surrogate (I3).
  Low-entropy originals are exempt: when "male"→"female" and a later
  message says "female", "male" gets a fresh surrogate there (I5);
* overlap — no surrogate equals a real value of its own turn, nor (unless
  low-entropy: an age, a gender term, ≤ 2 characters, which recur and are
  restored only in their own turn) a real value of any turn.

    python bench/echo.py                       # 250 turns, dev sources
    python bench/echo.py --turns 250 --show    # print the diffs
    python bench/echo.py --source experiment/test.json --json out.json

Default sources are ``bench/realworld/dev.jsonl`` then
``experiment/experiment.json``, cycled to N turns (repeats exercise reuse).
Offline: no provider call, no geocoder, temporary storage home.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-library"))

DEFAULT_SOURCES = (ROOT / "bench" / "realworld" / "dev.jsonl", ROOT / "experiment" / "experiment.json")


def load_messages(paths) -> List[str]:
    out: List[str] = []
    for path in map(Path, paths):
        if path.suffix == ".jsonl":
            out += [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text())
            out += [row.get("input") or row.get("question") or row.get("text") for row in data]
    return [m for m in out if m]


def diff_spans(a: str, b: str) -> List[List[str]]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [[a[i1:i2], b[j1:j2]] for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def run(messages: List[str], turns: int, *, cascade: Optional[Callable] = None) -> dict:
    """Run *turns* messages through one Pipeline. *cascade* replaces
    ``run_cascade`` (tests use a model-free detector)."""
    home = tempfile.mkdtemp(prefix="ss-echo-")
    os.environ["SURROGATESHIELD_HOME"] = home
    import config
    import pipeline
    import settings_manager
    import surrogateshield.core.storage.shadow_map as store
    from chatbot import chat as chat_mod, providers
    from chatbot.chat import ClaudeChat
    from surrogateshield.core.consistency import is_low_entropy
    from util import Conversation

    saved = (store._secret_cache, config.SHADOWMAP_DIR, config.DEVICE_KEY_PATH,
             pipeline.sentinel_layer.run_cascade, chat_mod.load_settings,
             settings_manager.load_settings, providers.build)
    store._secret_cache = {}
    config.SHADOWMAP_DIR = config.DEVICE_KEY_PATH = None
    if cascade is not None:
        pipeline.sentinel_layer.run_cascade = cascade
    chat_mod.load_settings = lambda: {"llm_provider": "claude"}
    settings_manager.load_settings = lambda: {"llm_provider": "claude", "detailed_view": False}
    sent: List[str] = []

    def echo(payload, system):
        sent.append(payload[-1]["content"])
        return sent[-1]
    providers.build = lambda provider: echo
    try:
        p = pipeline.Pipeline(chat=ClaudeChat(Conversation(id="echo-bench")))
        corrupted, first, seen = [], None, defaultdict(set)
        same_turn_overlap = []
        known_originals: set = set()
        latencies = []
        for i in range(turns):
            msg = messages[i % len(messages)]
            t = time.perf_counter()
            restored, _, surrogate_map = p.process_turn(msg, interactive=False)
            latencies.append((time.perf_counter() - t) * 1000)
            for original, surrogate in surrogate_map.items():
                seen[original].add(surrogate)
                # a surrogate equal to a real value of this turn or, unless
                # low-entropy, of any earlier turn
                if (surrogate in surrogate_map or surrogate in known_originals
                        and not is_low_entropy(surrogate)):
                    same_turn_overlap.append(surrogate)
            known_originals.update(surrogate_map)
            if restored != msg:
                first = i if first is None else first
                corrupted.append({"turn": i, "map_size": len(p.shadow.all_mappings()),
                                  "diff": diff_spans(msg, restored), "message": msg,
                                  "sent": sent[-1] if sent else None, "restored": restored,
                                  "map": surrogate_map})
        mappings = p.shadow.all_mappings()
        originals = set(mappings.values())
        overlap = sorted(set(same_turn_overlap)
                         | {s for s in set(mappings) & originals if not is_low_entropy(s)})
        latencies.sort()
        return {
            "turns": turns,
            "messages": len(messages),
            "corrupted": len(corrupted),
            "first_corrupted_turn": first,
            "map_size": len(mappings),
            "originals_with_several_surrogates": sum(
                1 for o, s in seen.items() if len(s) > 1 and not is_low_entropy(o)),
            "low_entropy_with_several_surrogates": sum(
                1 for o, s in seen.items() if len(s) > 1 and is_low_entropy(o)),
            "surrogate_equals_original": overlap,
            "low_entropy_reused_as_value": sorted(
                s for s in set(mappings) & originals if is_low_entropy(s)),
            "latency_ms_p50": round(latencies[len(latencies) // 2], 1) if latencies else 0.0,
            "J4": "PASS" if not corrupted and not any(
                len(s) > 1 and not is_low_entropy(o) for o, s in seen.items())
                  and not overlap else "FAIL",
            "_corrupted": corrupted,
        }
    finally:
        (store._secret_cache, config.SHADOWMAP_DIR, config.DEVICE_KEY_PATH,
         pipeline.sentinel_layer.run_cascade, chat_mod.load_settings,
         settings_manager.load_settings, providers.build) = saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", type=int, default=250)
    ap.add_argument("--source", action="append", help="questions .json or corpus .jsonl (repeatable)")
    ap.add_argument("--show", action="store_true", help="print the corrupted turns' diffs")
    ap.add_argument("--verbose", action="store_true", help="with --show: message, sent text, map")
    ap.add_argument("--json", help="write the summary (counts only) here")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    result = run(load_messages(args.source or DEFAULT_SOURCES), args.turns)
    corrupted = result.pop("_corrupted")
    if args.show:
        for row in corrupted:
            print(f"turn {row['turn']:>3} (map {row['map_size']}): {row['diff']}", file=sys.stderr)
            if args.verbose:
                for k in ("message", "sent", "restored", "map"):
                    print(f"    {k:>8}: {row[k]!r}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["J4"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
