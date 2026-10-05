"""Silver labels for the natural slice (Phase 2, E1).

Claude Sonnet 4.6 (the annotator; never a system under test) labels every
user turn of the natural pools with the ``bench/realworld/GUIDE.md`` taxonomy,
quoted verbatim in the system prompt: ``protect`` / ``sensitive`` /
``optional`` / ``keep`` as exact substrings, ``service_query``, and a coarse
task label. Each message comes with *pooled candidates*: the union of the
spans the six arms (``bench/arms``) edited, without arm or type, which the
annotator adjudicates and extends (TREC-style pooling; REALDATA_PROGRESS D4).

    python -m bench.realdata.label --estimate            # groups and calls, no network
    python -m bench.realdata.label --pilot               # 1 call, 10 messages, token count
    python -m bench.realdata.label --run                 # Message Batches, resumable
    python -m bench.realdata.label --run --write [--out F] # labels, prevalence, PII-free sets
    python -m bench.realdata.label --human-check         # 100 rows for a person to correct
    python -m bench.realdata.label --agreement FILE      # Sonnet vs the corrected rows

Every value is validated with the J2 scorer's own lint (``bench/realworld.py``):
an exact, whole-word, case-sensitive substring, a valid type, no protect value
overlapping a keep value. A request whose answer fails is re-asked once with the
problems listed; a message still failing is ``label_status: "failed"`` and
excluded. Values stay private (``bench/realdata/build/labels/``, 0600); the
committed ``bench/realdata/<dataset>/labels.jsonl`` holds character offsets
and types only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from bench.arms.inputs import message_id, parse_id
from bench.realdata.common import BUILD, DATASETS, RD, ROOT, SEED, TASKS, read_jsonl, sha256, write_jsonl

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import realworld as rw  # noqa: E402  (the J2 scorer: lint, occurrences, type sets)

ARMS = ("ss", "presidio_default", "presidio_faker", "presidio_transformers", "llm_guard", "gliner_pii")
LITERAL_ARMS = ("ss", "presidio_default", "gliner_pii")       # the prompt's literal PII-free rule
SPANS = BUILD / "spans"
LABELS = BUILD / "labels"
GUIDE = ROOT / "bench" / "realworld" / "GUIDE.md"
PHASE = "2-labels"
TOOL_NAME = "record_labels"
PER_CALL = 10
CHAR_BUDGET = 16000            # message characters per request
MAX_CANDIDATE = 200
MAX_TOKENS = 8192
HUMAN_CHECK = RD / "human_check.jsonl"
PREVALENCE = ROOT / "bench" / "results" / "realdata_prevalence.json"


# ── prompt ───────────────────────────────────────────────────────────────────

def guide_sections(path: Path = GUIDE) -> str:
    """The parts of GUIDE.md that define a label, verbatim: the substring rule
    of "Record format", "The four lists" and "service_query"."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", text)
    sec = {p.split("\n", 1)[0].strip(): "## " + p for p in parts[1:]}
    rule = re.search(r"(?m)^Every `value` must be.*?\n\n", sec["Record format"], re.S).group(0)
    return ("## Record format (rule)\n\n" + rule + sec["The four lists"].rstrip() + "\n\n"
            + sec["service_query"].rstrip() + "\n")


INSTRUCTIONS = """\
You annotate messages that real users sent to AI chat assistants (public research \
datasets: OpenAssistant, ShareGPT, WildChat) for a benchmark of privacy filters. A \
privacy filter rewrites a message before it reaches the assistant's provider; your \
labels say what it must protect, what it may touch either way, and what it must leave \
alone.

Label every message with the annotation guide below. It is quoted verbatim from the \
benchmark; where it mentions an invented corpus, the rules still apply to these real \
messages.

<guide>
{guide}</guide>

How to apply the guide here:
1. Copy each value from the message character for character: same case, spelling and \
spacing, no surrounding quotes or punctuation, never normalised or completed. It must \
occur in the message as a whole word or value, not inside a longer word. List a value \
once even if it occurs several times, and in one list only.
2. Each message comes with candidates: substrings that automatic detectors flagged. \
They are hints and often wrong, boundaries included. Decide every candidate: record \
the value it holds, with its own exact boundaries, in protect, sensitive or optional, \
or put the candidate in keep when it holds no personal data. A candidate left in no \
list is an error. Then add every value the detectors missed.
3. A private person is the user or someone they know or deal with (family, colleague, \
patient, customer, landlord). Public figures, authors, companies, products, places in \
a general question and characters from published works go in keep. Names and details \
the user invents for a story, a role-play or a template, and obvious placeholders \
("John Doe", "example.com", "555-1234"), go in optional. Template slots such as \
"[Your Name]" are not values: leave them out, or put them in keep when they are \
candidates.
4. In pasted code, logs or configs: credentials, tokens, passwords and keys, personal \
e-mail addresses, and IP addresses or hostnames that identify a person's machine are \
protect. Programming identifiers go in keep only if a filter might plausibly touch \
them.
5. sensitive is only a special-category fact about a private person ("I was diagnosed \
with ...", "my wife is Muslim"). A general question about a condition or a religion is \
not sensitive.
6. task, the message's main request: writing (stories, essays, e-mails, rewriting a \
text), coding (code, debugging, data, technical setup), qa (a factual or explanatory \
question), advice (personal, health, legal, financial or practical advice about the \
user's own situation), translation, roleplay (the assistant plays a character or a \
game), business (marketing, sales, product, business plans or documents), other.
7. Turns of one conversation share a conversation id and come in order. Use earlier \
turns to decide what a value is, but label each turn's own text only, with its own \
task.

Return the annotation of every message, once, with the ids as given, by calling the \
record_labels tool."""


def system_prompt() -> str:
    return INSTRUCTIONS.format(guide=guide_sections())


def _items(types: Optional[Iterable[str]]) -> dict:
    t = {"type": "string"}
    if types is not None:
        t["enum"] = sorted(types)
    return {"type": "array", "items": {"type": "object", "properties": {"value": {"type": "string"}, "type": t},
                                       "required": ["value", "type"]}}


TOOL = {
    "name": TOOL_NAME,
    "description": "Record the annotation of every message in this request.",
    "input_schema": {"type": "object", "required": ["messages"], "properties": {"messages": {
        "type": "array", "items": {"type": "object",
                                   "required": ["id", "task", "service_query", "protect", "sensitive", "optional", "keep"],
                                   "properties": {
                                       "id": {"type": "string"},
                                       "task": {"type": "string", "enum": list(TASKS)},
                                       "service_query": {"type": "boolean"},
                                       "protect": _items(rw.PROTECT_TYPES),
                                       "sensitive": _items(rw.SENSITIVE_TYPES),
                                       "optional": _items(None),
                                       "keep": {"type": "array", "items": {"type": "string"}}}}}}},
}


def prompt_version() -> str:
    return sha256(system_prompt() + json.dumps(TOOL, sort_keys=True))[:16]


# ── inputs ───────────────────────────────────────────────────────────────────

def load_messages(ds: str, rd: Path = RD, build: Path = BUILD) -> Tuple[List[dict], Dict[str, dict]]:
    """Messages in pool order, ``{"id", "text", "conv", "turn", "kind"}``, after
    checking every text against the committed SHA-256; and the pool rows."""
    committed = {r["source_id"]: r for r in read_jsonl(rd / ds / "pool.jsonl")}
    msgs = []
    for r in read_jsonl(build / ds / "pool.jsonl"):
        c = committed[r["source_id"]]
        if [sha256(t) for t in r["turns"]] != c["sha256"]:
            raise SystemExit(f"{ds}/{r['source_id']}: text does not match the frozen hash; "
                             "run python -m bench.realdata.build")
        for k, t in enumerate(r["turns"]):
            msgs.append({"id": message_id(ds, r["source_id"], k), "text": t, "conv": r["source_id"],
                         "turn": k, "kind": c["kind"]})
    return msgs, committed


def merged_spans(text: str, spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Spans widened to whole words, then overlapping ones joined: arms draw
    different boundaries round one value ("Ana" / "Ana Ruiz" / "na Ruiz")."""
    wc = rw._wordchar
    snapped = []
    for s, e in spans:
        while s > 0 and wc(text[s - 1]) and wc(text[s]):
            s -= 1
        while e < len(text) and wc(text[e - 1]) and wc(text[e]):
            e += 1
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            snapped.append((s, e))
    out: List[Tuple[int, int]] = []
    for s, e in sorted(snapped):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def candidates(msgs: List[dict], name: str, spans: Path = SPANS, arms=ARMS) -> Dict[str, List[str]]:
    """Pooled candidates per message: the distinct substrings any arm edited
    (whole words, overlaps joined, ≤ MAX_CANDIDATE characters), in text order,
    without arm or type."""
    texts = {m["id"]: m["text"] for m in msgs}
    pooled: Dict[str, set] = defaultdict(set)
    for arm in arms:
        f = spans / arm / f"{name}.jsonl"
        rows = read_jsonl(f)
        if [r["id"] for r in rows] != [m["id"] for m in msgs]:
            raise SystemExit(f"{f}: messages differ from the pool; rerun python -m bench.arms.run --natural")
        for r in rows:
            for s, e, *_ in r["edits"]:
                pooled[r["id"]].add((s, e))
    out = {}
    for mid, t in texts.items():
        seen, cands = set(), []
        for s, e in merged_spans(t, pooled.get(mid, ())):
            v = t[s:e]
            if len(v) <= MAX_CANDIDATE and v not in seen:
                seen.add(v)
                cands.append(v)
        out[mid] = cands
    return out


def units(msgs: List[dict]) -> List[List[dict]]:
    """A single-turn message, or every turn of one conversation, in pool order."""
    out: List[List[dict]] = []
    for m in msgs:
        if out and out[-1][0]["conv"] == m["conv"] and m["turn"] > 0:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def groups(unit_list: List[List[dict]], per_call: int = PER_CALL, budget: int = CHAR_BUDGET) -> List[List[dict]]:
    """Pack whole units into requests of ≤ *per_call* messages and ≤ *budget*
    characters (a unit larger than either goes alone)."""
    out: List[List[dict]] = []
    cur: List[dict] = []
    size = 0
    for u in unit_list:
        n = sum(len(m["text"]) for m in u)
        if cur and (len(cur) + len(u) > per_call or size + n > budget):
            out.append(cur)
            cur, size = [], 0
        cur += u
        size += n
    if cur:
        out.append(cur)
    return out


def user_content(group: List[dict], cands: Dict[str, List[str]],
                 feedback: Optional[Dict[str, List[str]]] = None) -> Tuple[str, Dict[str, str]]:
    """The request text, and local id → message id. Local ids (m1, m2, ...)
    keep real ids out of the model's way."""
    local, parts, convs = {}, [], {}
    for i, m in enumerate(group, 1):
        lid = f"m{i}"
        local[lid] = m["id"]
        attrs = f'id="{lid}"'
        if m["kind"] == "multi":
            cid = convs.setdefault(m["conv"], f"c{len(convs) + 1}")
            attrs += f' conversation="{cid}" turn="{m["turn"] + 1}"'
        parts.append(f"<message {attrs}>\n{m['text']}\n</message>\n"
                     f"<candidates for=\"{lid}\">{json.dumps(cands.get(m['id'], []), ensure_ascii=False)}</candidates>")
    head = f"Annotate these {len(group)} messages.\n\n"
    if feedback:
        lines = [f"- {lid}: " + "; ".join(feedback[mid]) for lid, mid in local.items() if mid in feedback]
        head += ("A previous annotation of these messages broke the guide's rules:\n" + "\n".join(lines)
                 + "\nAnnotate every message again, fixing those problems.\n\n")
    return head + "\n\n".join(parts), local


def params(group: List[dict], cands: Dict[str, List[str]], model: str,
           feedback: Optional[Dict[str, List[str]]] = None) -> Tuple[dict, Dict[str, str]]:
    text, local = user_content(group, cands, feedback)
    return ({"model": model, "max_tokens": MAX_TOKENS, "temperature": 0,
             "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
             "tools": [TOOL], "tool_choice": {"type": "tool", "name": TOOL_NAME},
             "messages": [{"role": "user", "content": text}]}, local)


# ── validation ───────────────────────────────────────────────────────────────

FIELDS = ("task", "service_query", "protect", "sensitive", "optional", "keep")


def validate(item: dict, text: str, cands: Iterable[str] = ()) -> List[str]:
    """Problems with one message's annotation (never echoing its text): the
    J2 lint on a record built from it, the task label, and every candidate
    decided (overlapped by an entry of some list; D12)."""
    problems = []
    if item.get("task") not in TASKS:
        problems.append("task not in the list")
    for f in FIELDS:
        if f not in item:
            problems.append(f"{f} missing")
    if problems:
        return problems
    rec = {"id": "rw-dev-0000", "text": text, "category": item["task"], "lang": "en",
           **{f: item[f] for f in FIELDS if f != "task"}}
    for name in ("protect", "sensitive", "optional"):
        if not all(isinstance(x, dict) for x in rec[name]):
            problems.append(f"{name} entries must be objects with value and type")
    if not all(isinstance(x, str) for x in rec["keep"]):
        problems.append("keep entries must be strings")
    if problems:
        return problems
    problems = [p for _rid, p in rw.lint([rec])]
    if problems:
        return problems
    listed = [o for name in ("protect", "sensitive", "optional") for x in rec[name]
              for o in rw.occurrences(text, x["value"])] + [o for v in rec["keep"] for o in rw.occurrences(text, v)]
    for k, c in enumerate(cands):
        occ = rw.occurrences(text, c)
        if occ and not any(a < y and x < b for a, b in occ for x, y in listed):
            problems.append(f"candidate[{k}] in no list")
    return problems


def parse(message: dict, local: Dict[str, str], texts: Dict[str, str],
          cands: Optional[Dict[str, List[str]]] = None) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    """Valid annotations and problems, both keyed by message id."""
    from bench.realdata.provider import tool_input
    got = tool_input(message, TOOL_NAME)
    if got is None or not isinstance(got.get("messages"), list):
        return {}, {mid: ["no record_labels call in the answer"] for mid in local.values()}
    by_local: Dict[str, List[dict]] = defaultdict(list)
    for it in got["messages"]:
        if isinstance(it, dict):
            by_local[str(it.get("id"))].append(it)
    ok, bad = {}, {}
    for lid, mid in local.items():
        its = by_local.get(lid, [])
        if len(its) != 1:
            bad[mid] = ["no annotation for this id" if not its else "annotated more than once"]
            continue
        problems = validate(its[0], texts[mid], (cands or {}).get(mid, ()))
        if problems:
            bad[mid] = problems
        else:
            ok[mid] = {f: its[0][f] for f in FIELDS}
    return ok, bad


# ── outputs ──────────────────────────────────────────────────────────────────

def public_row(mid: str, text: str, lab: Optional[dict], status: str, rnd: int,
               problems: Optional[List[str]] = None) -> dict:
    """Offsets and types only: every value is ``text[s:e]`` of its occurrences."""
    row = {"id": mid, "sha256": sha256(text), "label_status": status, "round": rnd}
    if lab is None:
        row["problems"] = problems or []
        return row
    row["task"], row["service_query"] = lab["task"], lab["service_query"]
    for name in ("protect", "sensitive", "optional"):
        row[name] = [{"type": x["type"], "occ": [list(o) for o in rw.occurrences(text, x["value"])]} for x in lab[name]]
    row["keep"] = [{"occ": [list(o) for o in rw.occurrences(text, v)]} for v in lab["keep"]]
    return row


def from_public(row: dict, text: str) -> dict:
    """The value lists back from a committed label row and the rebuilt text."""
    lab = {"task": row["task"], "service_query": row["service_query"]}
    for name in ("protect", "sensitive", "optional"):
        lab[name] = [{"value": text[x["occ"][0][0]:x["occ"][0][1]], "type": x["type"]} for x in row[name]]
    lab["keep"] = [text[x["occ"][0][0]:x["occ"][0][1]] for x in row["keep"]]
    return lab


def has_pii(lab: Optional[dict]) -> bool:
    return bool(lab and (lab["protect"] or lab["sensitive"]))


def prevalence(labels: Dict[str, Dict[str, dict]], kinds: Dict[str, str]) -> dict:
    """Counts only: how many real messages carry personal data, by dataset,
    type, task and kind (single-turn prompt or conversation turn)."""
    out = {}
    for ds, labs in labels.items():
        ok = {m: l for m, l in labs.items() if l is not None}
        d = {"messages": len(labs), "labelled": len(ok), "failed": len(labs) - len(ok),
             "with_protect": sum(1 for l in ok.values() if l["protect"]),
             "with_sensitive": sum(1 for l in ok.values() if l["sensitive"]),
             "with_pii": sum(1 for l in ok.values() if has_pii(l)),
             "service_query": sum(1 for l in ok.values() if l["service_query"])}
        by_type: Counter = Counter()
        values: Counter = Counter()
        for l in ok.values():
            for name in ("protect", "sensitive"):
                for t in {x["type"] for x in l[name]}:
                    by_type[t] += 1
                for x in l[name]:
                    values[x["type"]] += 1
        d["messages_with_type"] = dict(sorted(by_type.items()))
        d["values_by_type"] = dict(sorted(values.items()))
        tasks: Dict[str, Counter] = defaultdict(Counter)
        for m, l in ok.items():
            tasks[l["task"]]["messages"] += 1
            tasks[l["task"]]["with_pii"] += has_pii(l)
        d["by_task"] = {t: dict(c) for t, c in sorted(tasks.items())}
        kind: Dict[str, Counter] = defaultdict(Counter)
        for m, l in ok.items():
            kind[kinds[m]]["messages"] += 1
            kind[kinds[m]]["with_pii"] += has_pii(l)
        d["by_kind"] = {k: dict(c) for k, c in sorted(kind.items())}
        out[ds] = d
    return out


def pii_free(msgs: List[dict], labs: Dict[str, Optional[dict]], spans: Path, name: str) -> dict:
    """Source ids whose every turn is PII-free, under the adjudicated rule
    (D4: labelled, empty protect and sensitive) and the prompt's literal rule
    (SS, Presidio-default and GLiNER-PII make no edit)."""
    edited = set()
    for arm in LITERAL_ARMS:
        for r in read_jsonl(spans / arm / f"{name}.jsonl"):
            if r["edits"]:
                edited.add(r["id"])
    out = {"single": {"adjudicated": [], "literal": []}, "multi": {"adjudicated": [], "literal": []}}
    for u in units(msgs):
        kind, conv = u[0]["kind"], u[0]["conv"]
        if all(labs.get(m["id"]) is not None and not has_pii(labs[m["id"]]) for m in u):
            out[kind]["adjudicated"].append(conv)
        if all(m["id"] not in edited for m in u):
            out[kind]["literal"].append(conv)
    return out


# ── human check ──────────────────────────────────────────────────────────────

def draw_human_check(labels: Dict[str, Dict[str, Optional[dict]]], n: int = 100, seed: int = SEED) -> List[str]:
    """*n* message ids, half with Sonnet-labelled PII and half without, spread
    evenly over the datasets (seeded)."""
    rnd = random.Random(seed)
    per_ds = {ds: n // len(labels) + (i < n % len(labels)) for i, ds in enumerate(sorted(labels))}
    out = []
    for ds in sorted(labels):
        ok = sorted(m for m, l in labels[ds].items() if l is not None)
        pos = [m for m in ok if has_pii(labels[ds][m])]
        neg = [m for m in ok if not has_pii(labels[ds][m])]
        k_pos = min(len(pos), per_ds[ds] // 2 + per_ds[ds] % 2)
        out += rnd.sample(pos, k_pos) + rnd.sample(neg, min(len(neg), per_ds[ds] - k_pos))
    return out


def agreement(rows: List[dict]) -> dict:
    """Sonnet (``sonnet``) vs the checked labels, over rows marked ``checked``:
    per protect / sensitive type, value precision and recall of Sonnet's labels
    taking the person's as truth (exact value and type), and message-level
    agreement on "carries personal data" with Cohen's kappa."""
    rows = [r for r in rows if r.get("checked")]
    per: Dict[str, Counter] = defaultdict(Counter)
    a = b = both = neither = 0
    for r in rows:
        s, h = r["sonnet"], r
        for name in ("protect", "sensitive"):
            sv = {(x["value"], x["type"]) for x in s[name]}
            hv = {(x["value"], x["type"]) for x in h[name]}
            for t in {x[1] for x in sv | hv}:
                st = {v for v in sv if v[1] == t}
                ht = {v for v in hv if v[1] == t}
                per[t]["tp"] += len(st & ht)
                per[t]["fp"] += len(st - ht)
                per[t]["fn"] += len(ht - st)
        sp, hp = has_pii(s), has_pii(h)
        a += sp and not hp
        b += hp and not sp
        both += sp and hp
        neither += not sp and not hp
    n = len(rows)
    po = (both + neither) / n if n else 0.0
    pe = (((both + a) * (both + b)) + ((neither + b) * (neither + a))) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else 1.0 if n else 0.0
    types = {}
    for t, c in sorted(per.items()):
        types[t] = {**c, "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
                    "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None}
    return {"rows": n, "message_level": {"both": both, "neither": neither, "sonnet_only": a, "human_only": b,
                                         "agreement": po, "kappa": kappa}, "types": types}


# ── driver ───────────────────────────────────────────────────────────────────

def plan(datasets=DATASETS) -> Tuple[List[Tuple[str, List[dict]]], Dict[str, str], Dict[str, List[str]], Dict[str, dict]]:
    """Round-1 requests as (dataset, group), plus texts, candidates and messages by id."""
    out, texts, cands, by_id = [], {}, {}, {}
    for ds in datasets:
        msgs, _ = load_messages(ds)
        cands.update(candidates(msgs, f"natural-{ds}"))
        for m in msgs:
            texts[m["id"]], by_id[m["id"]] = m["text"], m
        out += [(ds, g) for g in groups(units(msgs))]
    return out, texts, cands, by_id


def pilot_ids(texts: Dict[str, str], n: int = 10, seed: int = SEED) -> List[str]:
    singles = sorted(m for m in texts if m.endswith("#t0"))
    return sorted(random.Random(seed).sample(singles, n))


def _collect(res: Dict[str, dict], locals_: Dict[str, Dict[str, str]], texts: Dict[str, str],
             cands: Dict[str, List[str]]):
    ok, bad = {}, {}
    for cid, local in locals_.items():
        r = res.get(cid, {"status": "missing"})
        if r["status"] != "succeeded":
            bad.update({mid: [f"request {r['status']}"] for mid in local.values()})
            continue
        if r["message"].get("stop_reason") == "max_tokens":
            bad.update({mid: ["answer cut at max_tokens"] for mid in local.values()})
            continue
        o, b = parse(r["message"], local, texts, cands)
        ok.update(o)
        bad.update(b)
    return ok, bad


def run(datasets=DATASETS, model: Optional[str] = None, cl=None, ledger=None, log=print) -> dict:
    """Both rounds (Message Batches, resumable); returns id → (label | None,
    round, problems)."""
    from bench.realdata.provider import SONNET, run_batch, requests_of
    model = model or SONNET
    planned, texts, cands, by_id = plan(datasets)
    reqs, locals_ = [], {}
    for i, (_ds, g) in enumerate(planned):
        p, local = params(g, cands, model)
        reqs.append((f"r1-{i:04d}", p))
        locals_[f"r1-{i:04d}"] = local
    res = run_batch(cl, ledger, f"labels-{prompt_version()}-r1", PHASE, "label", requests_of(reqs), log=log)
    ok1, bad1 = _collect(res, locals_, texts, cands)
    out = {mid: (lab, 1, []) for mid, lab in ok1.items()}
    # round 2: every unit with a failing message, asked again with the problems
    redo = [u for u in units([by_id[m] for m in texts]) if any(m["id"] in bad1 for m in u)]
    reqs, locals_ = [], {}
    for i, g in enumerate(groups(redo)):
        p, local = params(g, cands, model, feedback={m["id"]: bad1[m["id"]] for m in g if m["id"] in bad1})
        reqs.append((f"r2-{i:04d}", p))
        locals_[f"r2-{i:04d}"] = local
    if reqs:
        res = run_batch(cl, ledger, f"labels-{prompt_version()}-r2", PHASE, "label-reask", requests_of(reqs), log=log)
        ok2, bad2 = _collect(res, locals_, texts, cands)
        for mid in bad1:
            out[mid] = (ok2[mid], 2, []) if mid in ok2 else (None, 2, bad2.get(mid, ["not re-asked"]))
    log(f"round 1: {len(ok1)} valid, {len(bad1)} re-asked; final: "
        f"{sum(1 for v in out.values() if v[0] is not None)} valid, "
        f"{sum(1 for v in out.values() if v[0] is None)} failed")
    return out


def write(result: dict, datasets=DATASETS, model: Optional[str] = None, log=print, out: Path = PREVALENCE) -> dict:
    from bench.realdata import manifest
    from bench.realdata.provider import SONNET
    model = model or SONNET
    labels, kinds, summary = {}, {}, {}
    m = manifest.load()
    for ds in datasets:
        msgs, _ = load_messages(ds)
        priv, pub, labs = [], [], {}
        for msg in msgs:
            lab, rnd, problems = result[msg["id"]]
            status = "ok" if lab is not None else "failed"
            priv.append({"id": msg["id"], "label_status": status, "round": rnd,
                         **({"label": lab} if lab else {"problems": problems})})
            pub.append(public_row(msg["id"], msg["text"], lab, status, rnd, problems))
            labs[msg["id"]] = lab
            kinds[msg["id"]] = msg["kind"]
        write_jsonl(LABELS / f"{ds}.jsonl", priv, private=True)
        write_jsonl(RD / ds / "labels.jsonl", pub)
        free = pii_free(msgs, labs, SPANS, f"natural-{ds}")
        (RD / ds / "pii_free.json").write_text(json.dumps(free, indent=1, sort_keys=True) + "\n")
        labels[ds] = labs
        for f in ("labels.jsonl", "pii_free.json"):
            m["frozen"][f"{ds}/{f}"] = manifest.file_hash(RD / ds / f)
        summary[ds] = {k: {r: len(v) for r, v in d.items()} for k, d in free.items()}
    prev = prevalence(labels, kinds)
    for ds in datasets:
        prev[ds]["pii_free_sources"] = summary[ds]
    doc = {"command": f"python -m bench.realdata.label --run --write --out {out.relative_to(ROOT)}", "annotator": model,
           "prompt_version": prompt_version(), "per_call": PER_CALL, "datasets": prev}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    m.setdefault("labels", {}).update({"annotator": model, "prompt_version": prompt_version(),
                                       "pooled_arms": list(ARMS), "literal_rule_arms": list(LITERAL_ARMS)})
    manifest.save(m)
    for ds in datasets:
        d = prev[ds]
        log(f"{ds}: {d['labelled']}/{d['messages']} labelled, {d['with_pii']} with PII, "
            f"PII-free sources {summary[ds]}")
    return doc


def write_human_check(labels_dir: Path = LABELS, out: Path = HUMAN_CHECK, n: int = 100) -> List[str]:
    labels, texts, by_id = {}, {}, {}
    for ds in DATASETS:
        msgs, _ = load_messages(ds)
        rows = {r["id"]: r.get("label") for r in read_jsonl(labels_dir / f"{ds}.jsonl")}
        labels[ds] = rows
        for msg in msgs:
            texts[msg["id"]], by_id[msg["id"]] = msg["text"], msg
    ids = draw_human_check(labels, n)
    rows = []
    for mid in ids:
        ds = mid.split("/", 1)[0]
        lab, msg = labels[ds][mid], by_id[mid]
        prev = [texts[message_id(ds, msg["conv"], k)] for k in range(msg["turn"])]
        rows.append({"id": mid, "dataset": ds, "earlier_turns": prev, "text": msg["text"],
                     **{f: lab[f] for f in FIELDS}, "sonnet": lab, "checked": False, "notes": ""})
    write_jsonl(out, rows, private=True)
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--human-check", action="store_true")
    g.add_argument("--agreement", type=Path)
    ap.add_argument("--write", action="store_true", help="with --run: write labels, prevalence, PII-free sets")
    ap.add_argument("--out", type=Path, default=PREVALENCE, help="with --write: the prevalence counts file")
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    args = ap.parse_args(argv)
    if args.estimate:
        planned, texts, cands, _ = plan(args.datasets)
        chars = [sum(len(m["text"]) for m in g) for _ds, g in planned]
        print(f"prompt {prompt_version()}: {len(texts)} messages in {len(planned)} round-1 requests "
              f"(≤ {PER_CALL} messages, ≤ {CHAR_BUDGET} characters); message characters per request: "
              f"median {sorted(chars)[len(chars) // 2]}, max {max(chars)}; "
              f"candidates {sum(len(v) for v in cands.values())}")
        return 0
    if args.pilot:
        from bench.realdata.provider import SONNET, Ledger, call, client
        planned, texts, cands, by_id = plan(DATASETS)
        group = [by_id[m] for m in pilot_ids(texts)]
        p, local = params(group, cands, SONNET)
        msg = call(client(), Ledger(run="label-pilot"), PHASE, "label-pilot", p)
        write_jsonl(LABELS / f"pilot-{prompt_version()}.jsonl", [{"message": msg, "local": local}], private=True)
        ok, bad = parse(msg, local, texts, cands)
        u = msg["usage"]
        print(f"pilot: model {msg['model']}, stop {msg['stop_reason']}, {len(ok)}/10 valid; "
              f"input {u['input_tokens']} (+ cache write {u.get('cache_creation_input_tokens')}, "
              f"read {u.get('cache_read_input_tokens')}), output {u['output_tokens']} tokens; "
              f"message characters {sum(len(m['text']) for m in group)}")
        for mid, probs in bad.items():
            print(f"  {mid.split('#')[0][:24]}…: {probs}")
        return 0
    if args.run:
        from bench.realdata.provider import Ledger, client
        result = run(args.datasets, cl=client(), ledger=Ledger(run="label"))
        if args.write:
            write(result, args.datasets, out=args.out.resolve())
        return 0
    if args.human_check:
        ids = write_human_check()
        print(f"{len(ids)} rows -> {HUMAN_CHECK.relative_to(ROOT)} (0600, git-ignored)")
        return 0
    rows = read_jsonl(args.agreement)
    res = agreement(rows)
    out = ROOT / "bench" / "results" / "realdata_label_agreement.json"
    out.write_text(json.dumps({"command": f"python -m bench.realdata.label --agreement {args.agreement}",
                               **res}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res["message_level"]), f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
