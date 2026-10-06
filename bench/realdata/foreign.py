"""Beyond English (PROMPT_FOR_OPUS_V3 §5.5): SurrogateShield on WildChat's non-English prompts.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.foreign \\
        --out bench/results/foreign_wildchat.json

Every test set is English (WildChat's own language label; ``pull.py``). This
draws, from the downloaded WildChat shards, a seeded sample of first user
turns per language label: the six the relation gate reads (German, French,
Spanish, Italian, Portuguese, Dutch), Russian (a large share it does not
read) and English as the control. Filters are the English pool's: not
toxic, the first occurrence of a conversation hash, 15–400 words; no
conversation any collection sampled. SurrogateShield's send path (the ``ss``
arm's config, traced as in ``attribute.py``) runs on each message.

Per language: messages, the gate's guess of the language
(``relation_gate.message_language``), messages edited, edits, entities by
type and source, the gate's drops of model spans by rule, and the drops that
``is_foreign_fragment`` names: the filter that keeps an English NER model
from redacting clauses of another language. There is no gold here, so leak
is not measured and the edits are an upper bound on over-redaction; the
English control gives the same counts on English text from the same shards.
Counts and hashes only: no text and no value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from bench.arms.base import msg_seed
from bench.realdata import score as S
from bench.realdata.common import COLLECTIONS, RAW, commit_note, derive_seed, git_state, read_jsonl, words

LANGS = ("English", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch", "Russian")
GATE = {"German": "de", "French": "fr", "Spanish": "es", "Italian": "it", "Portuguese": "pt", "Dutch": "nl"}
PER_LANG = 100
MIN_WORDS, MAX_WORDS = 15, 400
MODEL_SOURCES_EXCLUDED = ("pattern", "structural")     # is_foreign_fragment reads model spans only


def taken() -> set:
    """Conversation hashes any collection sampled from WildChat."""
    out = set()
    for coll in COLLECTIONS.values():
        pool = coll.rd / "wildchat" / "pool.jsonl"
        if pool.exists():
            out |= {r["source_id"] for r in read_jsonl(pool)}
    return out


def sample(rows: Iterable[dict], langs: Sequence[str] = LANGS, per: int = PER_LANG,
           exclude: Iterable[str] = ()) -> Dict[str, List[Tuple[str, str]]]:
    """``{language: [(conversation hash, first user turn)]}``, seeded per language."""
    exclude, seen, by = set(exclude), set(), defaultdict(list)
    for r in rows:
        h = r["conversation_hash"]
        if h in seen:
            continue
        seen.add(h)
        if r["language"] not in langs or r["toxic"] or h in exclude:
            continue
        msgs = r["conversation"] or []
        if not msgs or msgs[0]["role"] != "user":
            continue
        text = msgs[0]["content"] or ""
        if MIN_WORDS <= words(text) <= MAX_WORDS:
            by[r["language"]].append((h, text))
    out = {}
    for lang in langs:
        cands = sorted(by.get(lang, []))
        random.Random(derive_seed("foreign", lang)).shuffle(cands)
        out[lang] = cands[:per]
    return out


def gate_drops(text: str, trace: Sequence[dict]) -> List[Tuple[str, str, bool]]:
    """``(rule, source, is_foreign_fragment)`` for every gate drop, the entity
    taken from the last state before the gate."""
    from surrogateshield.core.detection.relation_gate import is_foreign_fragment
    from surrogateshield.core.entities import DetectedEntity
    before: Dict[Tuple[int, int], tuple] = {}
    out = []
    for t in trace:
        if "dropped" in t:
            for s, e, rule in t["dropped"]:
                x = before.get((s, e))
                if x is None:
                    out.append((rule, "?", False))
                    continue
                ent = DetectedEntity(text=text[s:e], start=s, end=e, type=x[2], source=x[3])
                out.append((rule, x[3], is_foreign_fragment(ent, text)))
        if "entities" in t:
            before = {(x[0], x[1]): tuple(x) for x in t["entities"]}
    return out


def measure(text: str, prepared, trace: Sequence[dict]) -> dict:
    from surrogateshield.core.detection.relation_gate import message_language
    drops = gate_drops(text, trace)
    model = [d for d in drops if d[1] not in MODEL_SOURCES_EXCLUDED]
    out = {"guess": message_language(text) or "none", "refused": prepared is None,
           "edits": 0, "types": Counter(), "sources": Counter(),
           "drops": Counter(rule for rule, _s, _f in model), "foreign": sum(f for _r, _s, f in model)}
    if prepared is not None:
        out["edits"] = len(prepared.edits)
        out["types"] = Counter(e.type for e in prepared.confirmed)
        out["sources"] = Counter(e.source for e in prepared.confirmed)
    return out


def aggregate(lang: str, rows: Sequence[dict]) -> dict:
    n = len(rows)
    total = lambda key: sum((r[key] for r in rows), Counter())
    return {"messages": n, "gate_reads": GATE.get(lang),
            "gate_guess": dict(Counter(r["guess"] for r in rows)),
            "gate_guess_right": S.rate(sum(r["guess"] == GATE.get(lang, "none") for r in rows), n),
            "refused": sum(r["refused"] for r in rows),
            "edited_messages": S.rate(sum(r["edits"] > 0 for r in rows), n),
            "edits": sum(r["edits"] for r in rows),
            "edits_per_message": round(sum(r["edits"] for r in rows) / n, 3) if n else None,
            "entities_by_type": dict(total("types")), "entities_by_source": dict(total("sources")),
            "model_drops_by_rule": dict(total("drops")),
            "foreign_fragment_drops": sum(r["foreign"] for r in rows),
            "messages_with_foreign_fragment_drop": S.rate(sum(r["foreign"] > 0 for r in rows), n)}


def run(samples: Dict[str, List[Tuple[str, str]]], runner: Optional[Callable] = None, log=print) -> Dict[str, dict]:
    if runner is None:
        from bench.realdata.attribute import traced_prepare
        runner = traced_prepare()
    out = {}
    for lang, items in samples.items():
        rows = []
        for h, text in items:
            prepared, trace = runner(text, msg_seed("ss", f"foreign-{h}"))
            rows.append(measure(text, prepared, trace))
        out[lang] = aggregate(lang, rows)
        log(f"{lang}: {len(rows)} messages, {out[lang]['edits']} edits, "
            f"{out[lang]['foreign_fragment_drops']} foreign-fragment drops")
    return out


def markdown(doc: dict) -> str:
    lines = [f"# SurrogateShield on WildChat's non-English prompts", "",
             f"Command: `{doc['command']}`" + commit_note(doc.get("git")), "",
             "No gold: edits bound over-redaction from above and leak is not measured. English is the control from "
             "the same shards. Rates with Wilson 95 % intervals.", "",
             "| language | gate reads | messages | gate guesses it | edited messages | edits | edits / message "
             "| foreign-fragment drops | other model drops |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lang, r in doc["results"].items():
        other = sum(r["model_drops_by_rule"].values()) - r["foreign_fragment_drops"]
        lines.append(f"| {lang} | {r['gate_reads'] or '–'} | {r['messages']} | {S._pct(r['gate_guess_right'])} "
                     f"| {S._pct(r['edited_messages'])} | {r['edits']} | {r['edits_per_message']} "
                     f"| {r['foreign_fragment_drops']} | {other} |")
    lines += ["", "For English and Russian the gate's guess is right when it reads none of its six languages.", ""]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None, rows: Optional[Iterable[dict]] = None,
         runner: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per", type=int, default=PER_LANG)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if rows is None:
        from bench.realdata.sources import _wildchat_rows
        rows = _wildchat_rows(a.raw)
    samples = sample(rows, per=a.per, exclude=taken())
    from bench.arms.ss import detection_config
    doc = {"command": "python -m bench.realdata.foreign " + " ".join(argv if argv is not None else sys.argv[1:]),
           "git": git_state(), "detection_config_hash": detection_config().config_hash(),
           "filters": {"words": [MIN_WORDS, MAX_WORDS], "toxic": False, "first_hash_only": True,
                       "excluded_collections": sorted(COLLECTIONS)},
           "sample_sha256": {lang: hashlib.sha256("\n".join(h for h, _t in items).encode()).hexdigest()
                             for lang, items in samples.items()},
           "results": run(samples, runner)}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    a.out.with_suffix(".md").write_text(markdown(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
