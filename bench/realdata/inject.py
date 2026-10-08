"""Injected slice (Phase 3, E2/E3): our fake values placed into PII-free real prompts.

Per dataset, 250 single-turn prompts (40 of them format-shift) and 50
multi-turn conversations whose every turn Sonnet adjudicated PII-free (D4,
``<dataset>/pii_free.json``) each get one fake identity from
``identities.py`` (types drawn from task priors, topped up to ≥ 50 prompts per
type). Claude Sonnet 4.6 *places* the given values and never invents one. It
answers with span edits: ``find`` is an exact substring of the original turn
that occurs once in it, ``replace`` is its new text. So everything outside
the edits is the user's text byte for byte, and the gold is exact: each
value's occurrences in the result.

    python -m bench.realdata.inject --plan            # bases, types, identities: counts only, no network
    python -m bench.realdata.inject --estimate        # requests and characters, no network
    python -m bench.realdata.inject --pilot           # 2 calls, 10 items, verified, token counts
    python -m bench.realdata.inject --run [--write]   # Message Batches, two rounds, resumable
    python -m bench.realdata.inject --collection test2 --plan | --estimate | --pilot | --run [--write]

Verification (``check`` and ``detector_problems``) covers every rule of the
prompt's Phase 3 step 2:

* every value is in turn 1 verbatim (case-sensitive, not glued to a letter
  or digit: the scorer's ``occurrences``);
* the text stays the user's: at least 65 % of the original words survive in
  place (``word_change``: words outside the edits, plus the longest common
  subsequence of old and new words inside each edited region), and the
  edits add at most 8 words per value + 4 besides the values (15 in a later
  turn);
* nothing else personal is added: no e-mail, link or number of 5+ digits
  outside the values, no provider token shape, and no span from an arm of
  the literal PII-free rule (SS, Presidio-default, GLiNER-PII) that touches
  added text but no placed value, flagged by at least two of the three
  (the prompt's "re-run the PII-free check with the injected values
  masked out"; one arm alone tags frame words such as "I" or "email");
* multi-turn: every value is in turn 1, and each later turn is edited to
  refer back while repeating no value, no part of the name and no group of
  3+ digits from one;
* ``json`` layout (a third of the format-shift prompts): the name sits
  inside a ``{...}`` object;
* each turn's record passes the J2 lint.

A failing item is re-asked once with its problems listed, then dropped. The
acceptance rate and the kinds of drop reason go to
``bench/results/realdata_injection.json``; the records go to
``bench/realdata/<dataset>/{dev,test}.jsonl`` (REALDATA_PROGRESS D13, D14).

``--collection test2`` builds the sealed second test's injected slice the same
way at 1.5× the test share (300 single-turn, 48 of them format-shift, and 60
conversations per dataset; ≥ 60 prompts per type), with its own seeds, the
**evaluation** half of the identity pools (``identities.py``, PROMPT_FOR_OPUS_V3
§5.3), batch names ``test2-inject-…``, and records ``rd-<dataset>-test2-NNNN``
in ``bench/realdata/test2/<dataset>/test2.jsonl``.

``--collection test3`` (PROMPT_FOR_OPUS_V4, ``HYPOTHESES_TEST3.md`` §5) keeps
test2's prompt counts and pool half, with test3's own seeds, batch names and
files, and one change: single-turn prompts are topped up to **100 per
dataset for AGE** (``TYPE_TARGETS``) and 60 for every other type.
"""

from __future__ import annotations

import argparse
import difflib
from importlib.metadata import version
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bench.realdata import identities as I
from bench.realdata.common import (BUILD, COLLECTIONS, DATASETS, ROOT, TEST1, Collection, derive_seed,
                                   read_jsonl, sha256, write_jsonl)
from bench.realdata.label import LITERAL_ARMS, labels_dir, load_messages

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import realworld as rw  # noqa: E402  (the J2 scorer: lint, occurrences)

PHASE = "3-inject"
TOOL_NAME = "place_values"
N_SINGLE = {"dev": 50, "test": 200}      # 20 % / 80 %: the split is the base prompt's (fixed in Phase 1)
N_SHIFT = {"dev": 8, "test": 32}
N_MULTI = {"dev": 10, "test": 40}
JSON_SHARE = 1 / 3                       # of the format-shift prompts: the name inside a JSON object
LAYOUTS = ("inline", "intro", "signature", "form")
PER_CALL = 5
CHAR_BUDGET = 12000
MAX_TOKENS = 8192
KEEP_MIN = 0.65                          # share of the user's words kept in place
FRAME_PER_VALUE, FRAME_SLACK, LATER_FRAME = 8, 4, 15
INJECT = BUILD / "inject"
SUMMARY = ROOT / "bench" / "results" / "realdata_injection.json"
# test2 (1.5x test1's test share): single, shift and multi by split, and single-turn prompts per type
N_SINGLE_TEST2, N_SHIFT_TEST2, N_MULTI_TEST2, TARGET_TEST2 = {"test2": 300}, {"test2": 48}, {"test2": 60}, 60
TYPE_TARGETS = {"test3": {"AGE": 100}}   # single-turn prompts per type above the target (HYPOTHESES_TEST3 §5)


def sizes(coll: Collection) -> tuple:
    """Single, shift and multi counts by split, and the per-type target; a
    sealed collection has test2's sizes under its own split name."""
    if coll.prefix:
        (split,) = coll.splits
        return ({split: N_SINGLE_TEST2["test2"]}, {split: N_SHIFT_TEST2["test2"]},
                {split: N_MULTI_TEST2["test2"]}, TARGET_TEST2)
    return N_SINGLE, N_SHIFT, N_MULTI, I.TARGET


def pool_of(coll: Collection) -> Optional[str]:
    """Identity pool half: test2 draws only evaluation tokens; test1 predates the split."""
    return "eval" if coll.prefix else None


def summary_file(coll: Collection = TEST1) -> Path:
    return SUMMARY.with_name(f"realdata_injection_{coll.name}.json") if coll.prefix else SUMMARY


WORD = re.compile(r"\w+")
WORD_CHAR = re.compile(r"[^\W_]")
SHAPE = re.compile(r"[\w.+-]+@[\w-]+\.\w|https?://|www\.|\d{5,}")
MASK = "\x00"                            # one word standing for a placed value
QUORUM = 2                               # literal-rule arms that must agree on added text (D14)

MEANS = {"PERSON": "the user's name", "EMAIL": "the user's e-mail address", "PHONE": "the user's phone number",
         "ADDRESS": "the user's home address", "LOCATION": "the city the user lives in",
         "ORG": "the user's employer or school", "DATE_OF_BIRTH": "the user's date of birth",
         "AGE": "the user's age", "URL": "the user's own link (a profile, site or shared file)",
         "HANDLE": "the user's social-media handle", "ID": "an ID number of the user's",
         "NETWORK": "an address of the user's machine", "CREDENTIAL": "the user's secret"}
FMT_MEANS = {"ssn": "the user's US Social Security number", "ssn-nodash": "the user's US Social Security number",
             "card": "the user's payment card number", "card-spaced": "the user's payment card number",
             "iban": "the user's bank account number (IBAN)", "iban-spaced": "the user's bank account number (IBAN)",
             "passport": "the user's passport number", "nino": "the user's UK National Insurance number",
             "nhs": "the user's NHS number", "policy": "the user's insurance policy number",
             "license": "the user's driving licence number", "student": "the user's student or staff ID",
             "ipv4": "the IP address of the user's server or machine",
             "ipv6": "the IP address of the user's server or machine", "mac": "the MAC address of the user's device",
             "hex": "the user's API key", "zqk": "the user's API token", "secret": "the user's secret key",
             "password": "the user's password"}


def means(v: dict) -> str:
    return FMT_MEANS.get(v["fmt"]) or MEANS[v["type"]]


# ── plan: bases, types, identities (local, seeded) ───────────────────────────

def base_units(ds: str, coll: Collection = TEST1) -> Dict[str, dict]:
    """source_id → the base's split, kind, words per turn, turn texts and
    per-turn labels (private; texts checked against the frozen hashes)."""
    msgs, committed = load_messages(ds, coll.rd, coll.build)
    labs = {r["id"]: r.get("label") for r in read_jsonl(labels_dir(coll) / f"{ds}.jsonl")}
    out: Dict[str, dict] = {}
    for m in msgs:
        sid = m["conv"]
        u = out.setdefault(sid, {"source_id": sid, "split": committed[sid]["split"], "kind": m["kind"],
                                 "words": committed[sid]["words"], "turns": [], "labels": []})
        u["turns"].append(m["text"])
        u["labels"].append(labs.get(m["id"]))
    return out


def draw(ids: Iterable[str], splits: Dict[str, str], need: Dict[str, int], rng: random.Random) -> List[str]:
    out = []
    for split in sorted(need):
        pool = sorted(i for i in ids if splits[i] == split)
        if len(pool) < need[split]:
            raise SystemExit(f"only {len(pool)} PII-free {split} sources, {need[split]} needed")
        out += rng.sample(pool, need[split])
    return sorted(out)


def avoid_values(labels: Sequence[dict]) -> List[str]:
    """The base's ``keep`` / ``optional`` values a placed value must stay
    clear of; one of under two letters or digits (a ``/``, a ``C``) is left
    to ``turn_record``, which drops it from a turn where a value covers it."""
    vals = {x["value"] for lab in labels for x in lab["optional"]} | {v for lab in labels for v in lab["keep"]}
    return sorted(v for v in vals if len(WORD_CHAR.findall(v)) >= 2)


def plan(ds: str, units: Optional[Dict[str, dict]] = None, free: Optional[dict] = None,
         coll: Collection = TEST1) -> List[dict]:
    """One row per base: dataset, source_id, split, kind, task, shift, layout,
    types and identity. Deterministic from the seed and the frozen files."""
    n_single, n_shift, n_multi, target = sizes(coll)
    units = base_units(ds, coll) if units is None else units
    free = json.loads((coll.rd / ds / "pii_free.json").read_text()) if free is None else free
    rng = random.Random(coll.seed("inject", ds))
    splits = {s: u["split"] for s, u in units.items()}
    singles = draw(free["single"]["adjudicated"], splits, n_single, rng)
    shift = set()
    for split in sorted(n_shift):
        shift |= set(rng.sample([s for s in singles if splits[s] == split], n_shift[split]))
    multis = draw(free["multi"]["adjudicated"], splits, n_multi, rng)
    rows = [{"key": f"{ds}/{sid}", "dataset": ds, "source_id": sid, "split": units[sid]["split"], "kind": kind,
             "task": units[sid]["labels"][0]["task"], "shift": sid in shift, "words": units[sid]["words"][0]}
            for kind, ids in (("single", singles), ("multi", multis)) for sid in ids]
    for kind, tgt, per in (("single", target, TYPE_TARGETS.get(coll.name)), ("multi", 0, None)):
        idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
        for i, types in zip(idx, I.assign_types([rows[i] for i in idx], coll.seed("inject-types", ds, kind), tgt,
                                                per)):
            rows[i]["types"] = types
    named = sorted(r["source_id"] for r in rows if r["shift"] and "PERSON" in r["types"])
    json_ids = set(rng.sample(named, min(len(named), round(sum(n_shift.values()) * JSON_SHARE))))
    for r in rows:
        u = units[r["source_id"]]
        r_rng = random.Random(coll.seed("inject-identity", ds, r["source_id"]))
        r["identity"] = I.identity(r_rng, r["types"], r["task"], r["shift"], text="\n".join(u["turns"]),
                                   avoid=avoid_values(u["labels"]), pool=pool_of(coll))
        r["layout"] = "json" if r["source_id"] in json_ids else r_rng.choice(LAYOUTS)
    return rows


# ── the placement request ────────────────────────────────────────────────────

INSTRUCTIONS = """\
You help build a privacy benchmark. Each item is a real message a user sent to \
a chatbot (or the user's turns of one conversation; the assistant's replies are \
not shown), plus a list of personal details. The details are fake and belong to \
the user. Your job is to place them in the user's text so it reads as if the \
user had typed them, changing as little else as possible. Do not answer or act \
on the message.

Rules:
1. Use every value exactly as given, character for character: same letters, \
case, spacing and punctuation. Do not reformat, shorten, translate or complete \
it, and do not put a letter or digit directly before or after it.
2. Keep the user's text. Change as few of the user's words as you can. Do not \
fix typos, grammar or style, do not rephrase, and keep the user's tone. Add only \
short connecting words around the values ("my email is", "I'm", "Thanks,").
3. Make the values fit the request where you can: an API key or password in the \
user's config or code, an IP address of their server, an address in a tenancy \
question, an IBAN in a banking question. If one does not fit, still place it \
plainly, for example in a sign-off or as context the user gives.
4. Add no other personal or identifying information: no names, places, \
organisations, numbers, dates, e-mail addresses, links, handles or IDs other \
than the given values.
5. Layout (use it unless it reads oddly for this message, then pick another; \
json is required when given):
   - inline: work the values into the user's own sentences or code.
   - intro: a short self-introduction before the request.
   - signature: a sign-off at the end with the details.
   - form: a short pasted block of "Field: value" lines.
   - json: a small JSON object the user pasted holding the values, e.g. \
{"name": ..., "phone": ...}; the name must be inside it.
6. An item with several turns is one conversation. Put every value in turn 1. \
Then edit each later turn so it refers back to those details without repeating \
any of them or any part of them ("use the address I gave you", "sign it with my \
name", "my number from before"). Every later turn needs at least one such edit. \
A later turn must not contain any value or any piece of one: no name, no \
sign-off, no digits from a number. Refer to the details only indirectly.

Answer with edits, not the whole text. Each edit has:
- turn: the turn number (1 for a single message);
- find: an exact, contiguous substring of that turn's original text that occurs \
exactly once in it. Copy it character for character, including punctuation and \
line breaks. A few words are enough; add more until it is unique;
- replace: the text that takes its place, usually find plus the added words.
To add text at the very start or end, use the first or last few words as find. \
Edits in one turn must not overlap. Call the place_values tool once with the \
edits for every item, using the ids as given."""

TOOL = {
    "name": TOOL_NAME,
    "description": "Record the edits for every item in this request.",
    "input_schema": {"type": "object", "required": ["items"], "properties": {"items": {
        "type": "array", "items": {"type": "object", "required": ["id", "edits"], "properties": {
            "id": {"type": "string"},
            "edits": {"type": "array", "items": {"type": "object", "required": ["turn", "find", "replace"],
                                                 "properties": {"turn": {"type": "integer", "minimum": 1},
                                                                "find": {"type": "string"},
                                                                "replace": {"type": "string"}}}}}}}}},
}


def prompt_version() -> str:
    return sha256(INSTRUCTIONS + json.dumps(TOOL, sort_keys=True))[:16]


def item_text(lid: str, row: dict, turns: Sequence[str]) -> str:
    vals = [{"value": v["value"], "means": means(v)} for v in row["identity"]["values"]]
    body = "\n".join(f'<turn n="{k}">\n{t}\n</turn>' for k, t in enumerate(turns, 1))
    return (f'<item id="{lid}" layout="{row["layout"]}" turns="{len(turns)}">\n{body}\n'
            f"<values>{json.dumps(vals, ensure_ascii=False)}</values>\n</item>")


def groups(rows: Sequence[dict], texts: Dict[str, List[str]], per_call: int = PER_CALL,
           budget: int = CHAR_BUDGET) -> List[List[dict]]:
    out: List[List[dict]] = []
    cur: List[dict] = []
    size = 0
    for r in rows:
        n = sum(len(t) for t in texts[r["key"]])
        if cur and (len(cur) + 1 > per_call or size + n > budget):
            out.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        out.append(cur)
    return out


def user_content(group: Sequence[dict], texts: Dict[str, List[str]],
                 feedback: Optional[Dict[str, List[str]]] = None) -> Tuple[str, Dict[str, str]]:
    local, parts = {}, []
    for i, r in enumerate(group, 1):
        lid = f"p{i}"
        local[lid] = r["key"]
        parts.append(item_text(lid, r, texts[r["key"]]))
    head = f"Place the values in these {len(group)} items.\n\n"
    if feedback:
        lines = [f"- {lid}: " + "; ".join(feedback[key]) for lid, key in local.items() if key in feedback]
        head += ("An earlier answer for these items broke the rules:\n" + "\n".join(lines)
                 + "\nAnswer every item again, fixing those problems.\n\n")
    return head + "\n\n".join(parts), local


def params(group: Sequence[dict], texts: Dict[str, List[str]], model: str,
           feedback: Optional[Dict[str, List[str]]] = None) -> Tuple[dict, Dict[str, str]]:
    text, local = user_content(group, texts, feedback)
    return ({"model": model, "max_tokens": MAX_TOKENS, "temperature": 0,
             "system": [{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
             "tools": [TOOL], "tool_choice": {"type": "tool", "name": TOOL_NAME},
             "messages": [{"role": "user", "content": text}]}, local)


def parse(message: dict, local: Dict[str, str]) -> Tuple[Dict[str, List[dict]], Dict[str, List[str]]]:
    """Edits and problems, both keyed by row key."""
    from bench.realdata.provider import tool_input
    got = tool_input(message, TOOL_NAME)
    if got is None or not isinstance(got.get("items"), list):
        return {}, {key: ["no place_values call in the answer"] for key in local.values()}
    by_local: Dict[str, List[dict]] = defaultdict(list)
    for it in got["items"]:
        if isinstance(it, dict):
            by_local[str(it.get("id"))].append(it)
    ok, bad = {}, {}
    for lid, key in local.items():
        its = by_local.get(lid, [])
        if len(its) != 1:
            bad[key] = ["no answer for this item" if not its else "answered more than once"]
            continue
        edits = its[0].get("edits")
        if not isinstance(edits, list) or not all(
                isinstance(e, dict) and isinstance(e.get("turn"), int) and isinstance(e.get("find"), str)
                and isinstance(e.get("replace"), str) for e in edits):
            bad[key] = ["edits must be objects with turn, find and replace"]
            continue
        ok[key] = [{"turn": e["turn"], "find": e["find"], "replace": e["replace"]} for e in edits]
    return ok, bad


# ── verification (pure) ──────────────────────────────────────────────────────

Region = Tuple[int, int, int, int]       # old start, old end, new start, new end


def apply_edits(text: str, edits: Sequence[dict]) -> Tuple[Optional[str], List[Region], List[str]]:
    """The new text and its edited regions, or ``None`` and the problems."""
    problems, placed = [], []
    for k, e in enumerate(edits, 1):
        f = e["find"]
        i = text.find(f) if f else -1
        if not f:
            problems.append(f"edit {k}: find is empty")
        elif i < 0:
            problems.append(f"edit {k}: find is not an exact substring of the turn")
        elif text.find(f, i + 1) >= 0:
            problems.append(f"edit {k}: find occurs more than once in the turn")
        else:
            placed.append((i, i + len(f), e["replace"], k))
    placed.sort()
    for a, b in zip(placed, placed[1:]):
        if b[0] < a[1]:
            problems.append(f"edits {a[3]} and {b[3]} overlap")
    if problems:
        return None, [], problems
    out, regions, pos, delta = [], [], 0, 0
    for s, e, rep, _k in placed:
        out += [text[pos:s], rep]
        regions.append((s, e, s + delta, s + delta + len(rep)))
        delta += len(rep) - (e - s)
        pos = e
    out.append(text[pos:])
    return "".join(out), regions, []


def inserted(text: str, new: str, regions: Sequence[Region]) -> List[Tuple[int, int]]:
    """Character ranges of *new* that the edits added (not matched to the
    text they replaced)."""
    out = []
    for s, e, ns, ne in regions:
        j = 0
        for _a, b, size in difflib.SequenceMatcher(None, text[s:e], new[ns:ne], autojunk=False).get_matching_blocks():
            if b > j:
                out.append((ns + j, ns + b))
            j = b + size
    return out


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _widen(text: str, regions: Sequence[Region]) -> List[Region]:
    """Regions widened to whole words (the text around an edit is the same
    before and after, so both sides move together), touching ones joined."""
    wc = rw._wordchar
    out: List[Region] = []
    for s, e, ns, ne in sorted(regions):
        while s > 0 and wc(text[s - 1]):
            s, ns = s - 1, ns - 1
        while e < len(text) and wc(text[e]):
            e, ne = e + 1, ne + 1
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e, out[-1][2], ne)
        else:
            out.append((s, e, ns, ne))
    return out


def _words(new: str, ns: int, ne: int, vspans: Sequence[Tuple[int, int]]) -> List[str]:
    """Words of ``new[ns:ne]``; the words of one value occurrence are one MASK."""
    out, seen = [], set()
    for m in WORD.finditer(new, ns, ne):
        a, b = m.span()
        hit = next((k for k, (s, e) in enumerate(vspans) if s < b and a < e), None)
        if hit is None:
            out.append(m.group())
        elif hit not in seen:
            seen.add(hit)
            out.append(MASK)
    return out


def word_change(text: str, new: str, regions: Sequence[Region],
                vspans: Sequence[Tuple[int, int]] = ()) -> Tuple[float, int]:
    """(share of the original words kept in place, words added besides the values)."""
    total = len(WORD.findall(text))
    kept, added = total, 0
    for s, e, ns, ne in _widen(text, regions):
        old = WORD.findall(text, s, e)
        cur = _words(new, ns, ne, vspans)
        common = _lcs(old, cur)
        kept -= len(old) - common
        added += sum(1 for w in cur if w != MASK) - common
    return (kept / total if total else 1.0), added


def _spans(text: str, values: Iterable[str]) -> List[Tuple[int, int]]:
    return sorted(o for v in values for o in rw.occurrences(text, v))


def added_text(new: str, ins: Sequence[Tuple[int, int]], vspans: Sequence[Tuple[int, int]]) -> str:
    """The inserted characters with every value occurrence blanked, pieces
    joined by a line break."""
    chars = list(new)
    for s, e in vspans:
        chars[s:e] = " " * (e - s)
    return "\n".join("".join(chars[s:e]) for s, e in ins)


def in_json(text: str, value: str) -> bool:
    """Some occurrence of *value* lies between a ``{`` and its ``}``."""
    for s, e in rw.occurrences(text, value):
        if text.rfind("{", 0, s) > text.rfind("}", 0, s):
            close, opn = text.find("}", e), text.find("{", e)
            if close >= 0 and (opn < 0 or close < opn):
                return True
    return False


def name_parts(values: Sequence[dict]) -> List[str]:
    """What a later turn may not repeat: each word of the name (2+ letters)
    and each group of 3+ digits of any value."""
    parts = set()
    for v in values:
        if v["type"] == "PERSON":
            parts |= {w for w in re.findall(r"[^\W\d_]+", v["value"]) if len(w) >= 2}
        parts |= set(re.findall(r"\d{3,}", v["value"]))
    return sorted(parts)


def turn_record(text: str, label: dict, values: Sequence[dict]) -> dict:
    """A GUIDE record for one turn: the placed values it holds are ``protect``;
    the base's ``keep`` / ``optional`` values that still occur and touch no
    placed value carry over; ``sensitive`` is empty (the base is PII-free)."""
    protect = [{"value": v["value"], "type": v["type"], "fmt": v["fmt"]} for v in values
               if rw.occurrences(text, v["value"])]
    occ = _spans(text, [v["value"] for v in protect])

    def clear(value: str) -> bool:
        o = rw.occurrences(text, value)
        return bool(o) and not any(a < d and c < b for a, b in o for c, d in occ)
    return {"text": text, "service_query": label["service_query"], "protect": protect, "sensitive": [],
            "optional": [x for x in label["optional"] if clear(x["value"])],
            "keep": [v for v in label["keep"] if clear(v)]}


def check(row: dict, turns: Sequence[str], labels: Sequence[dict],
          edits: Sequence[dict]) -> Tuple[Optional[List[str]], dict, List[str]]:
    """Apply *edits* to the base *turns* and check every rule but the
    detectors'. Returns the new turns (``None`` if the edits do not apply),
    measurements (``kept``, ``added``, ``inserted`` per turn) and problems."""
    n, values = len(turns), row["identity"]["values"]
    by_turn: Dict[int, List[dict]] = defaultdict(list)
    for e in edits:
        if not 1 <= e["turn"] <= n:
            return None, {}, [f"an edit names turn {e['turn']}; this item has {n}"]
        by_turn[e["turn"]].append(e)
    problems: List[str] = []
    new_turns: List[Optional[str]] = []
    info: dict = {"kept": [], "added": [], "inserted": []}
    for k in range(1, n + 1):
        new, regions, probs = apply_edits(turns[k - 1], by_turn.get(k, []))
        if new is None:
            problems += [f"turn {k}, {p}" for p in probs]
            new_turns.append(None)
            continue
        vs = _spans(new, [v["value"] for v in values]) if k == 1 else []
        kept, added = word_change(turns[k - 1], new, regions, vs)
        ins = inserted(turns[k - 1], new, regions)
        info["kept"].append(round(kept, 4))
        info["added"].append(added)
        info["inserted"].append(ins)
        if kept < KEEP_MIN:
            problems.append(f"turn {k}: only {kept:.0%} of the user's words are kept (at least {KEEP_MIN:.0%}); "
                            "change less of the user's text")
        limit = FRAME_PER_VALUE * len(values) + FRAME_SLACK if k == 1 else LATER_FRAME
        if added > limit:
            problems.append(f"turn {k}: {added} words added besides the values (at most {limit})")
        extra = added_text(new, ins, vs)              # a later turn may hold no value, so nothing is blanked
        if I.FORBIDDEN.search(extra):
            problems.append(f"turn {k}: added text has the shape of a real provider's token")
        if SHAPE.search(extra):
            problems.append(f"turn {k}: added text holds an e-mail, link or long number that is not a given value")
        if k > 1:
            if not by_turn.get(k):
                problems.append(f"turn {k} has no edit referring back to the details in turn 1")
            low = extra.lower()
            for v in values:
                if rw.occurrences(low, v["value"].lower()):
                    problems.append(f"turn {k} repeats {json.dumps(v['value'], ensure_ascii=False)}")
            for part in name_parts(values):
                if rw.occurrences(low, part.lower()):
                    problems.append(f"turn {k} repeats part of a detail ({json.dumps(part, ensure_ascii=False)})")
        new_turns.append(new)
    if any(t is None for t in new_turns):
        return None, info, problems
    first = new_turns[0]
    for v in values:
        if not rw.occurrences(first, v["value"]):
            problems.append(f"{json.dumps(v['value'], ensure_ascii=False)} is not in turn 1 exactly as given")
    if row["layout"] == "json":
        name = next((v["value"] for v in values if v["type"] == "PERSON"), values[0]["value"])
        if not in_json(first, name):
            problems.append("layout json: the name is not inside a JSON object")
    for k, (t, lab) in enumerate(zip(new_turns, labels), 1):
        rec = {"id": "rw-test-0001", "category": row["task"], "lang": "en",
               **turn_record(t, lab, values if k == 1 else [])}
        problems += [f"turn {k}: lint: {p}" for _rid, p in rw.lint([rec])]
    return new_turns, info, problems


def detector_problems(new: str, ins: Sequence[Tuple[int, int]], vspans: Sequence[Tuple[int, int]],
                      spans: Dict[str, Sequence[Sequence]], quorum: int = QUORUM) -> List[str]:
    """Added text that reads as personal data: a span that touches added
    text and no placed value, flagged by at least *quorum* of the arms
    (overlapping spans of different arms agree). One arm alone is not
    enough: in the pilot GLiNER alone tagged the frame words "I" (person),
    "email" and "address", and SS and Presidio flagged nothing extra."""
    flagged = [(s, e, typ, arm) for arm in sorted(spans) for s, e, typ, *_ in spans[arm]
               if any(s < b and a < e for a, b in ins) and not any(s < b and a < e for a, b in vspans)]
    out, seen = [], set()
    for s, e, _typ, _arm in flagged:
        agree = sorted({(a2, t2) for s2, e2, t2, a2 in flagged if s2 < e and s < e2})
        arms = sorted({a for a, _t in agree})
        if len(arms) < quorum:
            continue
        lo = min(s2 for s2, e2, _t, _a in flagged if s2 < e and s < e2)
        hi = max(e2 for s2, e2, _t, _a in flagged if s2 < e and s < e2)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        out.append(f"added text {json.dumps(new[lo:hi], ensure_ascii=False)} reads as personal data to "
                   f"{len(arms)} detectors ({', '.join(f'{a}: {t}' for a, t in agree)}); "
                   "add nothing but the given values")
    return out


def problem_kind(p: str) -> str:
    """A problem without quoted text or numbers, for the committed counts."""
    p = re.sub(r'"(?:[^"\\]|\\.)*"', "<text>", p)
    return re.sub(r"\d+%?", "N", p)


# ── detectors (the literal PII-free arms, offline subprocesses) ──────────────

def detect(texts: Dict[str, str], name: str, arms: Sequence[str] = LITERAL_ARMS) -> Dict[str, Dict[str, list]]:
    """message id → arm → spans ``[start, end, type, replacement]``."""
    from bench.arms.run import command
    d = INJECT / name
    src = d / "texts.jsonl"
    write_jsonl(src, [{"id": i, "text": t} for i, t in sorted(texts.items())], private=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    out: Dict[str, Dict[str, list]] = defaultdict(dict)
    for arm in arms:
        dst = d / f"{arm}.jsonl"
        proc = subprocess.run(command(arm, src, dst), cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise SystemExit(f"{arm} failed on {src} (exit {proc.returncode}):\n{tail}")
        for f in (dst, Path(str(dst) + ".meta.json")):
            os.chmod(f, 0o600)
        for r in read_jsonl(dst):
            out[r["id"]][arm] = r["edits"]
    return out


def verify(rows: Sequence[dict], texts: Dict[str, List[str]], labels: Dict[str, List[dict]],
           answers: Dict[str, List[dict]], name: str, detector=detect) -> Dict[str, dict]:
    """key → ``{"edits", "turns", "info", "problems"}`` for every answered row;
    the detectors run on the rows that pass every other check."""
    out = {}
    for r in rows:
        if r["key"] in answers:
            turns, info, probs = check(r, texts[r["key"]], labels[r["key"]], answers[r["key"]])
            out[r["key"]] = {"edits": answers[r["key"]], "turns": turns, "info": info, "problems": probs}
    todo = {f"{key}#t{k}": t for key, v in out.items() if not v["problems"] for k, t in enumerate(v["turns"])}
    spans = detector(todo, name) if todo else {}
    by_key = {r["key"]: r for r in rows}
    for key, v in out.items():
        if v["problems"]:
            continue
        values = [x["value"] for x in by_key[key]["identity"]["values"]]
        for k, t in enumerate(v["turns"]):
            vs = _spans(t, values) if k == 0 else []
            v["problems"] += [f"turn {k + 1}: {p}" for p in
                              detector_problems(t, v["info"]["inserted"][k], vs, spans.get(f"{key}#t{k}", {}))]
    return out


# ── rounds ───────────────────────────────────────────────────────────────────

def load_all(datasets=DATASETS, coll: Collection = TEST1
             ) -> Tuple[List[dict], Dict[str, List[str]], Dict[str, List[dict]]]:
    rows, texts, labels = [], {}, {}
    for ds in datasets:
        units = base_units(ds, coll)
        for r in plan(ds, units, coll=coll):
            rows.append(r)
            texts[r["key"]] = units[r["source_id"]]["turns"]
            labels[r["key"]] = units[r["source_id"]]["labels"]
    return rows, texts, labels


def _requests(rows, texts, model, tag, feedback=None):
    reqs, locals_ = [], {}
    for i, g in enumerate(groups(rows, texts)):
        p, local = params(g, texts, model, {k: feedback[k] for k in local_keys(g)} if feedback else None)
        reqs.append((f"{tag}-{i:04d}", p))
        locals_[f"{tag}-{i:04d}"] = local
    return reqs, locals_


def local_keys(group: Sequence[dict]) -> List[str]:
    return [r["key"] for r in group]


def collect(res: Dict[str, dict], locals_: Dict[str, Dict[str, str]]):
    ok, bad = {}, {}
    for cid, local in locals_.items():
        r = res.get(cid, {"status": "missing"})
        if r["status"] != "succeeded":
            bad.update({key: [f"request {r['status']}"] for key in local.values()})
            continue
        if r["message"].get("stop_reason") == "max_tokens":
            bad.update({key: ["answer cut at max_tokens"] for key in local.values()})
            continue
        o, b = parse(r["message"], local)
        ok.update(o)
        bad.update(b)
    return ok, bad


def problems_of(key: str, bad: Dict[str, List[str]], ver: Dict[str, dict]) -> List[str]:
    """Why *key*'s answer fails (empty when it passes): the answer's own
    problems, else the verifier's."""
    if key in bad:
        return bad[key]
    if key not in ver:
        return ["no answer for this item"]
    return ver[key]["problems"]


def run(datasets=DATASETS, model: Optional[str] = None, cl=None, ledger=None, log=print,
        detector=detect, coll: Collection = TEST1) -> Dict[str, dict]:
    """Both rounds (Message Batches, resumable) and verification; key →
    ``{"status": "accepted" | "dropped", "round", "turns", "info", "problems", "first_problems"}``."""
    from bench.realdata.provider import SONNET, requests_of, run_batch
    model = model or SONNET
    pv = prompt_version()
    rows, texts, labels = load_all(datasets, coll)
    reqs, locals_ = _requests(rows, texts, model, "r1")
    ans, bad = collect(run_batch(cl, ledger, coll.tag(f"inject-{pv}-r1"), PHASE, "place", requests_of(reqs), log=log),
                       locals_)
    ver = verify(rows, texts, labels, ans, coll.tag(f"{pv}-r1"), detector)
    final, redo = {}, {}
    for r in rows:
        k = r["key"]
        probs = problems_of(k, bad, ver)
        if probs:
            redo[k] = probs
        else:
            final[k] = {**ver[k], "status": "accepted", "round": 1, "first_problems": []}
    log(f"round 1: {len(final)} accepted, {len(redo)} re-asked")
    if redo:
        again = [r for r in rows if r["key"] in redo]
        reqs, locals_ = _requests(again, texts, model, "r2", feedback=redo)
        ans, bad = collect(run_batch(cl, ledger, coll.tag(f"inject-{pv}-r2"), PHASE, "place-reask", requests_of(reqs),
                                     log=log), locals_)
        ver = verify(again, texts, labels, ans, coll.tag(f"{pv}-r2"), detector)
        for k in redo:
            probs = problems_of(k, bad, ver)
            if probs:
                final[k] = {"status": "dropped", "round": 2, "turns": None, "info": ver.get(k, {}).get("info", {}),
                            "problems": probs, "first_problems": redo[k]}
            else:
                final[k] = {**ver[k], "status": "accepted", "round": 2, "first_problems": redo[k]}
    write_jsonl(INJECT / f"{coll.tag(f'results-{pv}')}.jsonl",
                [{"key": k, **{f: v for f, v in final[k].items() if f != "info"},
                  "kept": final[k].get("info", {}).get("kept"), "added": final[k].get("info", {}).get("added")}
                 for k in sorted(final)], private=True)
    n_ok = sum(1 for v in final.values() if v["status"] == "accepted")
    log(f"final: {n_ok} accepted, {len(final) - n_ok} dropped")
    return final


# ── outputs ──────────────────────────────────────────────────────────────────

def records(rows: Sequence[dict], labels: Dict[str, List[dict]], final: Dict[str, dict]) -> Dict[Tuple[str, str], List[dict]]:
    """GUIDE records of the accepted rows, by (dataset, split); single-turn
    first, then multi-turn, each in source-id order, numbered from 0001."""
    out: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    order = sorted((r for r in rows if final.get(r["key"], {}).get("status") == "accepted"),
                   key=lambda r: (r["dataset"], r["split"], r["kind"] != "single", r["source_id"]))
    for r in order:
        f, labs = final[r["key"]], labels[r["key"]]
        values = r["identity"]["values"]
        turns = [turn_record(t, lab, values if k == 0 else []) for k, (t, lab) in enumerate(zip(f["turns"], labs))]
        bucket = out[(r["dataset"], r["split"])]
        rec = {"id": f"rd-{r['dataset']}-{r['split']}-{len(bucket) + 1:04d}", "category": r["task"], "lang": "en",
               **turns[0], "dataset": r["dataset"], "source_id": r["source_id"], "task": r["task"],
               "slice": "injected", "shift": r["shift"], "layout": r["layout"], "round": f["round"],
               "identity": {"locale": r["identity"]["locale"], "country": r["identity"]["country"],
                            "values": [{"value": v["value"], "type": v["type"], "fmt": v["fmt"]} for v in values]},
               "turns": turns if r["kind"] == "multi" else None}
        bucket.append(rec)
    return out


def summary(rows: Sequence[dict], final: Dict[str, dict]) -> dict:
    """Counts only: acceptance by round and kind, drop reasons, type and
    format counts, kept-word shares."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        mine = [r for r in rows if r["dataset"] == ds]
        d: dict = {}
        for kind in ("single", "shift", "multi"):
            sel = [r for r in mine if (r["kind"] == "multi") == (kind == "multi") and (kind != "shift" or r["shift"])]
            st = [final[r["key"]] for r in sel]
            d[kind] = {"planned": len(sel),
                       "accepted_round1": sum(1 for v in st if v["status"] == "accepted" and v["round"] == 1),
                       "accepted_round2": sum(1 for v in st if v["status"] == "accepted" and v["round"] == 2),
                       "dropped": sum(1 for v in st if v["status"] == "dropped")}
            d[kind]["acceptance"] = round((d[kind]["accepted_round1"] + d[kind]["accepted_round2"]) / max(len(sel), 1), 4)
        acc = [r for r in mine if final[r["key"]]["status"] == "accepted"]
        d["types_single"] = I.type_counts(r["identity"] for r in acc if r["kind"] == "single")
        d["types_multi"] = I.type_counts(r["identity"] for r in acc if r["kind"] == "multi")
        d["formats_shift"] = dict(sorted(Counter(v["fmt"] for r in acc if r["shift"]
                                                 for v in r["identity"]["values"]).items()))
        d["layouts"] = dict(sorted(Counter(r["layout"] for r in acc).items()))
        d["split"] = dict(sorted(Counter(r["split"] for r in acc).items()))
        d["round1_problem_kinds"] = dict(Counter(problem_kind(p) for r in mine
                                                 for p in final[r["key"]]["first_problems"]).most_common())
        d["drop_problem_kinds"] = dict(Counter(problem_kind(p) for r in mine if final[r["key"]]["status"] == "dropped"
                                               for p in final[r["key"]]["problems"]).most_common())
        kept = sorted(x for r in acc for x in final[r["key"]]["info"]["kept"][:1])
        d["turn1_kept_share"] = {"min": kept[0], "median": kept[len(kept) // 2]} if kept else {}
        out[ds] = d
    return out


def write(final: Dict[str, dict], datasets=DATASETS, model: Optional[str] = None, out: Optional[Path] = None,
          log=print, coll: Collection = TEST1) -> dict:
    from bench.realdata import manifest
    from bench.realdata.provider import SONNET
    out = out or summary_file(coll)
    rows, _texts, labels = load_all(datasets, coll)
    recs = records(rows, labels, final)
    m = manifest.load()
    for ds in datasets:
        for split in coll.splits:
            path = coll.rd / ds / f"{split}.jsonl"
            got = recs.get((ds, split), [])
            errors = rw.lint(got) + [(r["id"], f"turn {k + 1}: {p}") for r in got for k, t in enumerate(r["turns"] or [])
                                     for _i, p in rw.lint([{"id": r["id"], "category": r["category"], "lang": "en", **t}])]
            if errors:
                raise SystemExit(f"{ds}/{split}: {len(errors)} lint errors, first {errors[:5]}")
            write_jsonl(path, got)
            m["frozen"][coll.key(f"{ds}/{split}.jsonl")] = manifest.file_hash(path)
    flag = f"--collection {coll.name} " if coll.prefix else ""
    doc = {"command": f"python -m bench.realdata.inject {flag}--run --write --out {out.relative_to(ROOT)}",
           "placer": model or SONNET, "prompt_version": prompt_version(),
           "rules": {"keep_min": KEEP_MIN, "frame_per_value": FRAME_PER_VALUE, "frame_slack": FRAME_SLACK,
                     "later_frame": LATER_FRAME, "detector_arms": list(LITERAL_ARMS), "detector_quorum": QUORUM},
           "datasets": summary(rows, final)}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (m.setdefault("injection", {}) if not coll.prefix else m.setdefault(coll.name, {}).setdefault("injection", {})
     ).update({"placer": model or SONNET, "prompt_version": prompt_version(), "faker": version("faker"),
               **({"identity_pool": pool_of(coll), "pool_key": I.POOL_KEY} if coll.prefix else {})})
    manifest.save(m)
    for ds in datasets:
        d = doc["datasets"][ds]
        log(f"{ds}: " + ", ".join(f"{k} {d[k]['accepted_round1'] + d[k]['accepted_round2']}/{d[k]['planned']}"
                                  for k in ("single", "shift", "multi"))
            + f"; min rows per type (single) {min(d['types_single']['rows'].values())}")
    return doc


# ── CLI ──────────────────────────────────────────────────────────────────────

def pilot_rows(rows: Sequence[dict], seed: int = 0) -> List[List[dict]]:
    """Two requests of five: single-turn prompts (one json, one other shift,
    three plain, across datasets) and five conversations."""
    rng = random.Random(derive_seed("inject-pilot", seed))
    single = [r for r in rows if r["kind"] == "single"]
    a = rng.sample([r for r in single if r["layout"] == "json"], 1)
    a += rng.sample([r for r in single if r["shift"] and r["layout"] != "json"], 1)
    a += rng.sample([r for r in single if not r["shift"]], 3)
    b = rng.sample([r for r in rows if r["kind"] == "multi"], 5)
    return [a, b]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--write", action="store_true", help="with --run: write records and counts")
    ap.add_argument("--out", type=Path, help="with --write: the counts file "
                    "(default bench/results/realdata_injection[_<collection>].json)")
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--collection", choices=list(COLLECTIONS), default="test1")
    args = ap.parse_args(argv)
    coll = COLLECTIONS[args.collection]
    if args.plan or args.estimate:
        rows, texts, _ = load_all(args.datasets, coll)
        for ds in args.datasets:
            mine = [r for r in rows if r["dataset"] == ds]
            single = [r for r in mine if r["kind"] == "single"]
            tc = I.type_counts(r["identity"] for r in single)["rows"]
            print(f"{ds}: single {len(single)} (shift {sum(r['shift'] for r in single)}, json "
                  f"{sum(r['layout'] == 'json' for r in single)}), multi {len(mine) - len(single)}; "
                  f"split {dict(sorted(Counter(r['split'] for r in mine).items()))}; "
                  f"values per item median {sorted(len(r['identity']['values']) for r in mine)[len(mine) // 2]}; "
                  f"rows per type (single) min {min(tc.values())} {dict(sorted(tc.items()))}")
        if args.estimate:
            gs = groups(rows, texts)
            chars = [sum(len(t) for r in grp for t in texts[r["key"]]) for grp in gs]
            print(f"prompt {prompt_version()}: {len(rows)} items in {len(gs)} round-1 requests (≤ {PER_CALL} items, "
                  f"≤ {CHAR_BUDGET} characters); characters per request median {sorted(chars)[len(chars) // 2]}, "
                  f"max {max(chars)}")
        return 0
    if args.pilot:
        from bench.realdata.provider import SONNET, Ledger, call, client
        rows, texts, labels = load_all(DATASETS, coll)
        saved = INJECT / f"{coll.tag(f'pilot-{prompt_version()}')}.jsonl"   # a rerun re-verifies without new calls
        msgs = read_jsonl(saved) if saved.exists() else []
        cl, ledger = (None, None) if msgs else (client(), Ledger(run=coll.tag("inject-pilot")))
        answers, bad = {}, {}
        for i, grp in enumerate(pilot_rows(rows)):
            p, local = params(grp, texts, SONNET)
            if i == len(msgs):
                msgs.append({"message": call(cl, ledger, PHASE, "place-pilot", p), "local": local})
            msg = msgs[i]["message"]
            o, b = parse(msg, local)
            answers.update(o)
            bad.update(b)
            u = msg["usage"]
            print(f"pilot request: stop {msg['stop_reason']}, input {u['input_tokens']} (+ cache write "
                  f"{u.get('cache_creation_input_tokens')}, read {u.get('cache_read_input_tokens')}), output "
                  f"{u['output_tokens']} tokens; characters {sum(len(t) for r in grp for t in texts[r['key']])}")
        write_jsonl(saved, msgs, private=True)
        flat = [r for grp in pilot_rows(rows) for r in grp]
        ver = verify(flat, texts, labels, answers, coll.tag(f"pilot-{prompt_version()}"))
        for r in flat:
            v = ver.get(r["key"])
            probs = problems_of(r["key"], bad, ver)
            print(f"  {r['kind']:6} {r['layout']:9} shift={int(r['shift'])} values {len(r['identity']['values'])}: "
                  + ("ok" if not probs else "; ".join(problem_kind(p) for p in probs))
                  + (f"  kept {v['info']['kept']} added {v['info']['added']}" if v and v["info"] else ""))
        return 0
    from bench.realdata.provider import Ledger, client
    final = run(args.datasets, cl=client(), ledger=Ledger(run=coll.tag("inject")), coll=coll)
    if args.write:
        write(final, args.datasets, out=args.out.resolve() if args.out else None, coll=coll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
