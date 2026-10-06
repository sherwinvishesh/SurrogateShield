"""Leak attribution (PROMPT_FOR_OPUS_V3 §3.1): one cause for every gold value.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute --split dev --out bench/results/attribution_dev.json
    ... --split devlarge --out bench/results/attribution_devlarge.json    # test-1's test split
    ... --split test2 --out bench/results/attribution_test2.json          # Phase 3 only, after FREEZE.json

SurrogateShield's send path (``json_tester.prepare_send``, ``MimicGen`` seeded
per message exactly as in the ``ss`` arm) runs on every message of the split
with the cascade's trace on (``run_cascade(trace=[...])``: the live
candidates after every pass, and the relation gate's rule for each drop).
Each protect value of the injected slice then gets one cause, decided with
the J2 scorer's own leak rule (``bench/realworld.py``; checked against
``score_message`` message by message):

``protected``  not leaked;
``policy``     leaked, but excused by the scorer (service query, ADDRESS / LOCATION);
``refused``    the system sent nothing (no surrogate), so nothing leaked;
``surrogate``  an edit over the value has a replacement that contains it;
``partial``    an edit overlaps the value and part of it reaches the output;
               detail: the parts left (ADDRESS house_number / unit / street /
               locality / postcode, PERSON first / middle / last / title,
               EMAIL local / domain, otherwise prefix / suffix / inner);
``unplanned``  a final entity covers it but no edit was planned (detail
               ``no_surrogate`` or ``overlapped``);
``gated``      a stage produced an overlapping candidate and a later pass
               removed it; detail: that pass (and the gate's rule), plus the
               source of the last candidate;
``missed``     no stage ever produced an overlapping candidate (detail
               ``in_url`` when the value lies in a span the NER stages never see).

On the natural slice every spurious edit (no gold overlap; the scorer's
rule) is attributed to the source and type of the entity that produced it.

``kept_parts`` (diagnostic, not a cause): an ADDRESS the scorer counts as
protected whose surrogate still carries its street name, locality or
postcode verbatim. One whole-address edit that only moves the house number
(``shift``) covers every character, yet sends the rest unchanged.

The output holds counts only (type × layout × format × slice × cause ×
detail): no text, no value. ``reproduced`` compares the traced run's edits
with the ``ss`` arm's private span file of the same input (when one exists),
so a trace that changed the system would show.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arms.base import msg_seed
from bench.arms.run import PRIVATE
from bench.realdata.common import COLLECTIONS, DATASETS, ROOT, commit_note, git_state, read_jsonl
from bench.realdata import score as S

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import realworld as rw  # noqa: E402

CAUSES = ("protected", "policy", "refused", "surrogate", "partial", "unplanned", "gated", "missed")
LEAK_CAUSES = ("surrogate", "partial", "unplanned", "gated", "missed")
FINAL_STAGES = ("needs_confirmation", "deduplicate")      # after run_cascade, in prepare_send


# ── one message ──────────────────────────────────────────────────────────────

def traced_prepare():
    """``(text, seed) -> (Prepared | None, trace)``; None when the system refused."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from generation.logic import MimicGen
    from json_tester import prepare_send
    from bench.arms.ss import detection_config
    cfg = detection_config()

    def fn(text: str, seed: int):
        trace: List[dict] = []
        try:
            return prepare_send(text, MimicGen(seed=seed), {"trace": trace}, config=cfg), trace
        except RuntimeError as exc:
            if str(exc).startswith("could not generate a surrogate"):
                return None, trace
            raise
    return fn


def states(trace: Sequence[dict], prepared) -> List[Tuple[str, List[tuple]]]:
    """The live candidates after every pass, then the two steps of
    ``prepare_send`` that can still drop one: only the confirmed bucket is
    replaced, and ``deduplicate`` merges what is left."""
    out = [(t["stage"], [tuple(e) for e in t["entities"]]) for t in trace if "entities" in t]
    last = out[-1][1] if out else []
    out.append(("needs_confirmation", [e for e in last if e[4] == "confirmed"]))
    out.append(("deduplicate", [(e.start, e.end, e.type, e.source, "confirmed") for e in prepared.confirmed]))
    return out


def gate_rules(trace: Sequence[dict]) -> Dict[Tuple[int, int], str]:
    return {(s, e): rule for t in trace for s, e, rule in t.get("dropped", [])}


def opaque(trace: Sequence[dict]) -> List[Tuple[int, int]]:
    return [tuple(sp) for t in trace for sp in t.get("opaque", [])]


def overlaps(a: int, b: int, c: int, d: int) -> bool:
    return a < d and c < b


def first_failure(text: str, value: str, edits) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """The scorer's ``leaked(value)``, but saying how: (None, None) when it
    does not leak, else ("surrogate" | "uncovered", the occurrence)."""
    for s, e in rw.occurrences(text, value):
        if not rw._covered(s, e, text, edits):
            return "uncovered", (s, e)
        for a, b, _o, sur in edits:
            if a < e and s < b and len(value) >= 4 and value in sur:
                return "surrogate", (s, e)
    return None, None


_TOKEN = re.compile(r"[^\W_]+(?:['’.\-][^\W_]+)*", re.U)
_UNIT = re.compile(r"^(flat|apt|apartment|unit|suite|ste|floor|fl|room|rm|block|bldg|building|piso|depto|dept)$", re.I)
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "mx", "herr", "frau", "sr", "sra"}
_AT = re.compile(r"\s*(?:@|[\(\[\{<]\s*at\s*[\)\]\}>]|\s+at\s+)\s*", re.I)


def _uncovered(text: str, s: int, e: int, edits) -> List[bool]:
    """Per character of text[s:e]: True when it is a letter/digit no edit covers."""
    return [text[i].isalnum() and not any(a <= i < b for a, b, _o, _r in edits) for i in range(s, e)]


def kept_parts(text: str, value: str, edits) -> str:
    """The street / locality / postcode of a covered ADDRESS that its
    surrogate repeats verbatim ("street+locality+postcode"), "" when none."""
    from surrogateshield.core.detection import address_assembly, address_parser
    p = address_assembly.parse(value) or address_parser.parse(value)
    if p is None:
        return ""
    if p.parts:
        role = {"name": "street", "city": "locality", "postcode": "postcode"}
        pieces = [(role[r], p.full_text[a:b]) for r, a, b in p.parts if r in role]
    else:
        pieces = [("street", p.street_name), ("locality", p.city), ("postcode", p.zip_code)]
    out = set()
    for s, e in rw.occurrences(text, value):
        sent = " ".join(r for a, b, _o, r in edits if overlaps(a, b, s, e))
        out |= {k for k, piece in pieces
                if piece and re.search(rf"(?<!\w){re.escape(piece)}(?!\w)", sent)}
    return "+".join(k for k in ("street", "locality", "postcode") if k in out)


def parts_left(typ: str, fmt: str, value: str, left: List[bool]) -> str:
    """Which part of a partially edited value reached the output (a coarse,
    documented heuristic; see the module docstring)."""
    def tokens(lo=0, hi=None):
        hi = len(value) if hi is None else hi
        return [(m.start() + lo, m.end() + lo, m.group()) for m in _TOKEN.finditer(value[lo:hi])]

    def hit(s, e):
        return any(left[s:e])

    parts = set()
    if typ == "ADDRESS":
        segs, pos = [], 0
        for m in re.finditer(r"[,\n;]", value):
            segs.append((pos, m.start()))
            pos = m.end()
        segs.append((pos, len(value)))
        for k, (lo, hi) in enumerate(segs):
            toks = tokens(lo, hi)
            for j, (s, e, t) in enumerate(toks):
                if not hit(s, e):
                    continue
                digit = any(c.isdigit() for c in t)
                if _UNIT.match(t) or (j and _UNIT.match(toks[j - 1][2])):
                    parts.add("unit")
                elif k == 0:
                    parts.add("house_number" if digit else "street")
                else:
                    parts.add("postcode" if digit else "locality")
    elif typ == "PERSON":
        toks = [x for x in tokens() if x[2].lower().rstrip(".") not in _TITLES]
        titles = [x for x in tokens() if x[2].lower().rstrip(".") in _TITLES]
        if any(hit(s, e) for s, e, _t in titles):
            parts.add("title")
        surname_first = "surname-first" in (fmt or "") or "," in value
        for j, (s, e, _t) in enumerate(toks):
            if not hit(s, e):
                continue
            if len(toks) == 1:
                parts.add("single")
            elif j == 0:
                parts.add("last" if surname_first else "first")
            elif j == len(toks) - 1:
                parts.add("first" if surname_first else "last")
            else:
                parts.add("middle")
    elif typ == "EMAIL" and _AT.search(value):
        m = _AT.search(value)
        if any(left[:m.start()]):
            parts.add("local")
        if any(left[m.end():]):
            parts.add("domain")
    if not parts:
        idx = [i for i, x in enumerate(left) if x]
        done = [i for i, c in enumerate(value) if c.isalnum() and not left[i]]
        if not done:
            parts.add("all")
        elif max(idx) < min(done):
            parts.add("prefix")
        elif min(idx) > max(done):
            parts.add("suffix")
        else:
            parts.add("inner")
    return "+".join(sorted(parts))


def attribute_value(text: str, item: dict, service_query: bool, prepared, trace) -> Tuple[str, str]:
    """(cause, detail) of one protect value."""
    if prepared is None:
        return "refused", ""
    edits = prepared.edits
    how, occ = first_failure(text, item["value"], edits)
    if how is None:
        return "protected", ""
    if service_query and item["type"] in rw.POLICY_TYPES:
        return "policy", ""
    if how == "surrogate":
        return "surrogate", ""
    s, e = occ
    if any(overlaps(a, b, s, e) for a, b, _o, _r in edits):
        return "partial", parts_left(item["type"], item.get("fmt", ""), text[s:e], _uncovered(text, s, e, edits))
    final = [x for x in prepared.confirmed if overlaps(x.start, x.end, s, e)]
    if final:
        return "unplanned", ("overlapped" if all(x.text in prepared.surrogate_map for x in final) else "no_surrogate")
    st = states(trace, prepared)
    seen = [i for i, (_n, ents) in enumerate(st) if any(overlaps(x[0], x[1], s, e) for x in ents)]
    if not seen:
        return "missed", ("in_url" if any(overlaps(a, b, s, e) for a, b in opaque(trace)) else "")
    i = seen[-1]
    if i + 1 >= len(st):          # still a candidate at the end, yet nothing overlaps in prepared.confirmed
        return "unplanned", "lost_after_cascade"
    stage = st[i + 1][0]
    last = [x for x in st[i][1] if overlaps(x[0], x[1], s, e)]
    if stage == "relation_gate":
        rules = gate_rules(trace)
        stage += ":" + ",".join(sorted({rules.get((x[0], x[1]), "?") for x in last}))
    return "gated", f"{stage} <- {','.join(sorted({x[3] for x in last}))}"


# ── a split ──────────────────────────────────────────────────────────────────

def layouts(rd: Path, ds: str, data_split: str) -> Dict[str, str]:
    return {r["id"]: r.get("layout") or "" for r in read_jsonl(rd / ds / f"{data_split}.jsonl")}


def spurious_sources(text: str, gold: dict, prepared) -> Counter:
    gold_spans = [sp for name in ("protect", "sensitive", "optional") for x in gold.get(name, [])
                  for sp in rw.occurrences(text, x["value"])]
    by_span = {(x.start, x.end): x for x in prepared.confirmed}
    out = Counter()
    for a, b, _o, _r in prepared.edits:
        if any(overlaps(a, b, s, e) for s, e in gold_spans):
            continue
        x = by_span.get((a, b))
        # a canonical-view hit counts under its view ("pattern:worded")
        src = x and (x.source + (f":{x.view}" if getattr(x, "view", None) else ""))
        out[(src, x.type) if x else ("repeat", "")] += 1
    return out


def run(split: str, datasets: Sequence[str] = DATASETS, prepare=None) -> dict:
    data_split, coll_name, _role = S.RUNS[split]
    coll = COLLECTIONS[coll_name]
    if split == "test2":
        S.check_freeze()
    prepare = prepare or traced_prepare()
    _hashes, loaded = S.load_split(data_split, datasets, coll.rd, coll.build, prefix=coll.prefix)
    rows: Counter = Counter()
    spurious: Counter = Counter()
    reproduced = Counter()
    natural = Counter()
    kept = Counter()
    for ds in datasets:
        units = loaded[ds]["units"]
        lay = layouts(coll.rd, ds, data_split)
        # the scorer's span name; devlarge reads test-1's test files, same input
        arm = next((f for f in (PRIVATE / "ss" / f"{n}-{ds}.jsonl" for n in dict.fromkeys((split, data_split)))
                    if f.exists() and _same_input(f, loaded[ds]["input_sha"])), None)
        arm_rows = {r["id"]: r for r in read_jsonl(arm)} if arm else {}
        for u in units:
            gold, text = u["gold"], u["gold"]["text"]
            prepared, trace = prepare(text, msg_seed("ss", u["mid"]))
            edits = [] if prepared is None else [(a, b, o, r) for a, b, o, r in prepared.edits]
            if arm_rows:
                want = [(s, e, rep) for s, e, _t, rep in arm_rows[u["mid"]]["edits"]]
                reproduced["equal" if want == [(a, b, r) for a, b, _o, r in edits] else "differ"] += 1
            if "natural" in u["slices"]:
                natural["messages"] += 1
                if prepared is not None:
                    c = spurious_sources(text, gold, prepared)
                    spurious.update({(ds,) + k: v for k, v in c.items()})
                    natural["untouched"] += not prepared.edits
                continue
            sl = next(x for x in u["slices"] if x != "injected")
            check = rw.score_message(gold, SimpleNamespace(edits=edits))
            mine = Counter()
            for item in gold["protect"]:
                cause, detail = attribute_value(text, item, gold["service_query"], prepared, trace)
                rows[(ds, item["type"], lay.get(u["conv"], ""), item.get("fmt") or "", sl, cause, detail)] += 1
                mine[cause] += 1
                if cause == "protected" and item["type"] == "ADDRESS":
                    kept[kept_parts(text, item["value"], edits) or "none"] += 1
            if prepared is not None and (sum(mine[c] for c in LEAK_CAUSES) != len(check["leaked"])
                                         or mine["policy"] != len(check["policy"])):
                raise RuntimeError(f"{u['mid']}: attribution disagrees with score_message")
    return summarise(split, data_split, coll_name, datasets, rows, spurious, reproduced, natural, kept)


def _same_input(span_file: Path, input_sha: str) -> bool:
    meta = Path(str(span_file) + ".meta.json")
    return meta.exists() and json.loads(meta.read_text()).get("input_sha256") == input_sha


def summarise(split, data_split, coll_name, datasets, rows: Counter, spurious: Counter,
              reproduced: Counter, natural: Counter, kept: Counter = None) -> dict:
    by_type: Dict[str, Counter] = defaultdict(Counter)
    by_cause: Counter = Counter()
    details: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_layout: Dict[str, Counter] = defaultdict(Counter)
    by_slice: Dict[str, Counter] = defaultdict(Counter)
    by_format: Dict[str, Counter] = defaultdict(Counter)
    for (ds, typ, lay, fmt, sl, cause, detail), n in rows.items():
        by_type[typ][cause] += n
        by_format[f"{typ}:{fmt or 'plain'}"][cause] += n
        by_cause[cause] += n
        by_layout[lay][cause] += n
        by_slice[sl][cause] += n
        if cause in ("partial", "gated", "unplanned", "missed", "surrogate"):
            details[typ][cause][detail or "-"] += n
    sp_by_source: Counter = Counter()
    sp_by_type: Counter = Counter()
    for (ds, source, typ), n in spurious.items():
        sp_by_source[source] += n
        sp_by_type[f"{source}:{typ}" if typ else source] += n

    def leak(c: Counter) -> dict:
        n = sum(c.values()) - c["policy"] - c["refused"]
        k = sum(c[x] for x in LEAK_CAUSES)
        return {"values": sum(c.values()), "leaked": k, "leak_rate": round(k / n, 4) if n else None}

    return {
        "command": "", "split": split, "data_split": data_split, "collection": coll_name,
        "datasets": list(datasets), "git": git_state(),
        "causes": list(CAUSES),
        "reproduced_vs_ss_arm": dict(reproduced),
        "injected": {"all": {**leak(by_cause), "by_cause": dict(by_cause)},
                     "by_type": {t: {**leak(c), "by_cause": dict(c)} for t, c in sorted(by_type.items())},
                     "by_layout": {l: {**leak(c), "by_cause": dict(c)} for l, c in sorted(by_layout.items())},
                     "by_slice": {s: {**leak(c), "by_cause": dict(c)} for s, c in sorted(by_slice.items())},
                     "by_format": {f: {**leak(c), "by_cause": dict(c)} for f, c in sorted(by_format.items())},
                     "details": {t: {c: dict(d.most_common()) for c, d in sorted(v.items())}
                                 for t, v in sorted(details.items())},
                     "address_kept_parts": dict((kept or Counter()).most_common()),
                     "rows": [dict(zip(("dataset", "type", "layout", "fmt", "slice", "cause", "detail"), k), n=n)
                              for k, n in sorted(rows.items())]},
        "natural": {**dict(natural), "spurious_edits": sum(spurious.values()),
                    "spurious_by_source": dict(sp_by_source.most_common()),
                    "spurious_by_source_type": dict(sp_by_type.most_common())},
    }


def markdown(doc: dict) -> str:
    inj = doc["injected"]
    causes = [c for c in CAUSES if any(v["by_cause"].get(c) for v in inj["by_type"].values())]
    lines = [f"# Leak attribution — `{doc['split']}` (data split `{doc['data_split']}`, collection `{doc['collection']}`)",
             "", f"Command: `{doc['command']}`{commit_note(doc.get('git'))}. Counts only.", "",
             f"Traced edits vs the `ss` arm's span file: {doc['reproduced_vs_ss_arm'] or 'no span file of this input'}.", "",
             "## Injected slice: cause by type", "",
             "| type | values | leak | " + " | ".join(causes) + " |",
             "|---|---|---|" + "---|" * len(causes)]
    for t, v in list(inj["by_type"].items()) + [("**all**", inj["all"])]:
        lines.append(f"| {t} | {v['values']} | {v['leak_rate']} | "
                     + " | ".join(str(v["by_cause"].get(c, 0)) for c in causes) + " |")
    for title, key in (("layout", "by_layout"), ("slice", "by_slice"), ("type:format", "by_format")):
        lines += ["", f"## Cause by {title}", "", f"| {title} | values | leak | " + " | ".join(causes) + " |",
                  "|---|---|---|" + "---|" * len(causes)]
        for k, v in inj[key].items():
            lines.append(f"| {k or '–'} | {v['values']} | {v['leak_rate']} | "
                         + " | ".join(str(v["by_cause"].get(c, 0)) for c in causes) + " |")
    lines += ["", "## Leak details (type → cause → detail: count)", ""]
    for t, cs in inj["details"].items():
        for c, d in cs.items():
            lines.append(f"- **{t}** {c}: " + ", ".join(f"`{k}` {n}" for k, n in d.items()))
    kept = inj.get("address_kept_parts", {})
    lines += ["", "## Protected ADDRESS values whose surrogate repeats a part verbatim", "",
              ", ".join(f"`{k}` {n}" for k, n in kept.items()) or "no protected ADDRESS value."]
    nat = doc["natural"]
    lines += ["", "## Natural slice: spurious edits by source", "",
              f"{nat.get('messages', 0)} messages, {nat.get('untouched', 0)} untouched, "
              f"{nat['spurious_edits']} spurious edits.", "", "| source:type | edits |", "|---|---|"]
    lines += [f"| {k} | {n} |" for k, n in nat["spurious_by_source_type"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=list(S.RUNS), required=True)
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":          # the arms run with it (bench.arms.base.arm_env)
        env = {**os.environ, "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-m", "bench.realdata.attribute", *(argv or sys.argv[1:])],
                               env=env, cwd=ROOT)
    doc = run(args.split, args.datasets)
    out = args.out.resolve()
    from bench.arms import ss
    from surrogateshield.core.detection.config import ENV_FILE
    # a config file merged on the benchmark config (a candidate stage, a
    # threshold) is part of what was attributed
    doc["config_file"] = ss.config_file()
    doc["config_hash"] = ss.effective_config().config_hash()
    env = f"{ENV_FILE}={doc['config_file']['path']} " if doc["config_file"] else ""
    doc["command"] = (f"{env}HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.attribute "
                      f"--split {args.split}"
                      + (f" --datasets {' '.join(args.datasets)}" if list(args.datasets) != list(DATASETS) else "")
                      + f" --out {S.rel(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    a = doc["injected"]["all"]
    print(f"{args.split}: {a['values']} values, leak {a['leak_rate']}, causes {a['by_cause']}; "
          f"address kept parts {doc['injected']['address_kept_parts']}; "
          f"reproduced {doc['reproduced_vs_ss_arm']} -> {S.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
