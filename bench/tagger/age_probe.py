"""V4 §3.2: the AGE probe, a synthetic diagnostic of where ages are read.

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.tagger.age_probe \\
        --ss frozen --out bench/results/age_probe_dev.json
    ... -m bench.tagger.age_probe --ss v4 --out bench/results/age_probe_dev.json   # one more variant beside the others

About 600 messages are built here, all synthetic: ``bank.py`` carriers
(``data.body``) and ages, names and cities from the **training** half of the
identity pools (``identities.Ctx(rng, False, "train")``; never the ``eval``
half that test-3 is injected from, never ``pool=None``). Half put the age in
the layouts the tagger's training bank has (``data.closed_sentence`` over
``bank.OWN["AGE"]``, ``form_block``, ``json_block``, ``table_block``); half in
layouts it lacks: third-person ``Name Name, NN`` closing a line (``eol``) or
the message (``signoff``), mid-sentence ``Name Name, NN,`` (``mid``),
``Name (NN)`` (``bracket``), ``Name, NN, from City`` (``from_city``), a
contact block with ``Age: NN`` after a name (``contact``), Reddit ``(NNF)`` /
``NNM here`` (``reddit``) and third-person worded ages (``worded``). A share of
the names in the new layouts have letters beyond ASCII (the training half of
27 Faker locales: Turkish, Romanian, Polish …); they are counted apart.

An age is caught when every letter and digit of it lies inside an edit and,
for a value of length ≥ 4, it is not inside its own surrogate: the scorer's
leak rule (``bench/realworld.py``) applied at the age's known span, since
``occurrences`` skips a number glued to a letter (``29F``). Scored per layout:

* the tagger alone (the default's installed tagger, decoded as
  ``evaluate.candidates`` does), at thresholds 0.3–0.95, with every candidate
  an edit (``any``) and with its AGE candidates only (``as_age``: what routing
  AGE to the tagger can add, §3.1 A);
* SurrogateShield (``bench.arms.ss``: the benchmark config at this checkout,
  with ``--config`` a partial config file on it), once per ``--ss`` variant. A
  variant is named for the code it ran (``frozen``: the test-2 detector; then
  the fixes) and kept beside the others with its code stamp and config hash.

Counts only: no text goes into the result. The messages and spans stay under
the git-ignored ``bench/tagger/build/probe/``. D's trigger (§3.1) is read here:
the tagger's ``as_age`` recall below 0.95 on the sign-off layouts (``signoff``,
``eol``) at the chosen threshold, after A–C.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bench.realdata import identities as ids
from bench.realdata.common import ROOT, TASKS, derive_seed, git_state, read_jsonl, sha256, write_jsonl
from bench.tagger import bank, data

TRAINED = {"sentence": 75, "form": 75, "json": 75, "table": 75}
NEW = {"eol": 38, "signoff": 38, "mid": 38, "bracket": 38, "from_city": 38, "contact": 38, "reddit": 37, "worded": 37}
LAYOUTS = (*TRAINED, *NEW)
SIGN_OFF = ("signoff", "eol")
NON_ASCII = .4                 # share of the new layouts' names drawn with letters beyond ASCII
WORDED = .3                    # share of the trained layouts' ages in words (the training data's SHIFT)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
VERSION = 1
BUILD = ROOT / "bench" / "tagger" / "build" / "probe"
OUT = ROOT / "bench" / "results" / "age_probe_dev.json"

# the layouts the training bank lacks, written for this probe: {n} full name, {f} first name, {c} city, {v} age
TEMPLATES = {
    "eol": ("Contact: {n}, {v}", "Applicant: {n}, {v}", "Submitted by {n}, {v}", "Team lead: {n}, {v}",
            "Tenant - {n}, {v}", "Reference: {n}, {v}"),
    "mid": ("My neighbour {n}, {v}, asked me to help with this.", "{n}, {v}, is a new client of ours.",
            "I'm writing for my aunt {n}, {v}, who can't type well.", "Our new hire {n}, {v}, starts on Monday.",
            "The patient, {n}, {v}, reported mild symptoms.", "My roommate {n}, {v}, wants to split the rent."),
    "bracket": ("My son {f} ({v}) wants to learn the piano.", "{n} ({v}) and I are planning a trip.",
                "Our daughter {f} ({v}) keeps waking up at night.", "Applicant {n} ({v}) asked about the deadline.",
                "My grandfather {f} ({v}) needs a simpler phone."),
    "from_city": ("{n}, {v}, from {c}, wrote in about this.", "Meet {n}, {v}, from {c}.",
                  "Speaker: {n}, {v}, from {c}", "{n}, {v}, from {c}, is joining the club."),
    "contact": ("Name: {n}\nAge: {v}", "Patient: {n}\nAge: {v}\nCity: {c}", "Full name - {n}\nAge - {v}",
                "Name: {n} | Age: {v} | City: {c}", "Client: {n}\nAge (years): {v}"),
    "reddit": ("Me ({v}F) and my partner have been together for three years.", "{v}M here, looking for advice.",
               "My sister ({v}F) keeps borrowing money and never pays it back.",
               "Throwaway account. I ({v}M) need advice.", "{v}F, first time posting here."),
    "worded": ("My grandmother {f} is {v} and still drives.", "{f}, a {v}-year-old nurse, asked about this.",
               "{n} turned {v} last week.", "My uncle {f} is {v} years old.", "Our client {n}, aged {v}, needs a plan."),
}
AGES = {"bracket": (3, 79)}    # every other layout: 18–79, as identities.age


def _person(rng: random.Random, non_ascii: bool) -> ids.Ctx:
    """A training-half identity whose full name is (or is not) plain ASCII."""
    for _ in range(2000):
        c = ids.Ctx(rng, False, "train")
        if (not f"{c.first} {c.last}".isascii()) == non_ascii:
            return c
    raise RuntimeError("no training-half name of the wanted script")


def _fill(template: str, slots: Dict[str, str]) -> tuple:
    """*template* with its slots filled, and the {v} span."""
    out, span = "", None
    for part in re.split(r"(\{[nfcv]\})", template):
        if part in ("{n}", "{f}", "{c}", "{v}"):
            if part == "{v}":
                span = [len(out), len(out) + len(slots["v"])]
            out += slots[part[1]]
        else:
            out += part
    return out, span


def _carrier(rng: random.Random) -> str:
    text, _ = data.join(data.body(rng, rng.choice(TASKS), negatives_only=True))
    return text.strip()


def _trained(rng: random.Random, layout: str) -> dict:
    n = rng.randint(18, 79)
    worded = rng.random() < WORDED
    v = {"value": ids.number_words(n) if worded else str(n), "type": "AGE", "fmt": "words" if worded else "plain"}
    block = {"sentence": data.closed_sentence, "form": data.form_block, "json": data.json_block,
             "table": data.table_block}[layout]
    pieces = block(rng, v, "US") if layout == "sentence" else block(rng, [v], "US")
    carrier = [data.Piece(_carrier(rng))]
    pieces = carrier + [data.Piece("\n\n")] + pieces if rng.random() < .5 else pieces + [data.Piece("\n\n")] + carrier
    text, spans = data.join(pieces)
    (span,) = [s[:2] for s in spans if s[2] == "AGE"]
    return {"text": text, "age": span, "value": v["value"], "worded": worded, "non_ascii": None, "pool_tokens": []}


def _new(rng: random.Random, layout: str) -> dict:
    lo, hi = AGES.get(layout, (18, 79))
    n = rng.randint(lo, hi)
    worded = layout == "worded"
    value = ids.number_words(n) if worded else str(n)
    non_ascii = rng.random() < NON_ASCII if layout != "reddit" else None
    c = _person(rng, non_ascii) if non_ascii is not None else None
    slots = {"v": value}
    if c is not None:
        slots.update(n=f"{c.first} {c.last}", f=c.first, c=ids.city(c)[0])
    if layout == "signoff":
        line, span = _fill("{n}, {v}" + rng.choice(("", "", ".")), slots)
        head = _carrier(rng) + "\n\n" + rng.choice(bank.SIGN_OFFS) + "\n"
        text, span = head + line, [span[0] + len(head), span[1] + len(head)]
    else:
        line, span = _fill(rng.choice(TEMPLATES[layout]), slots)
        carrier = _carrier(rng)
        if layout == "eol" or rng.random() < .5:              # eol: the line ends, the message goes on
            text = line + "\n" + carrier
        else:
            head = carrier + rng.choice(("\n\n", " ", "\n"))
            text, span = head + line, [span[0] + len(head), span[1] + len(head)]
    return {"text": text, "age": span, "value": value, "worded": worded, "non_ascii": non_ascii,
            "pool_tokens": sorted(c.pool_tokens) if c is not None else []}


def build(seed: int = derive_seed("age_probe", VERSION), sizes: Optional[Dict[str, int]] = None) -> List[dict]:
    """The probe's messages, in layout order, each with its age's span
    (*sizes*: messages per layout, default ``TRAINED`` and ``NEW``)."""
    rng = random.Random(seed)
    sizes = sizes or {**TRAINED, **NEW}
    out = []
    for layout in LAYOUTS:
        for _ in range(sizes.get(layout, 0)):
            m = (_trained if layout in TRAINED else _new)(rng, layout)
            s, e = m["age"]
            if m["text"][s:e] != m["value"]:
                raise RuntimeError(f"{layout}: the age's span does not hold its value")
            out.append({"id": f"ap-{len(out):04d}", "layout": layout, "trained": layout in TRAINED, **m})
    return out


def messages_sha(msgs: Sequence[dict]) -> str:
    return sha256(json.dumps([[m["id"], m["text"], m["age"]] for m in msgs], ensure_ascii=False))


# ── scoring ──────────────────────────────────────────────────────────────────

def caught(m: dict, edits: Sequence[Sequence]) -> bool:
    """The scorer's leak rule at the age's span: every letter and digit inside
    an edit, and a value of length ≥ 4 not inside an overlapping surrogate."""
    from bench import realworld as rw
    s, e = m["age"]
    ed = [(a, b, None, rep) for a, b, _t, rep in edits]
    if not rw._covered(s, e, m["text"], ed):
        return False
    return not (len(m["value"]) >= 4 and any(a < e and s < b and m["value"] in rep for a, b, _, rep in ed))


def summary(msgs: Sequence[dict], hit: Dict[str, bool]) -> dict:
    """Recall (caught / n) per layout, pooled over the trained and the new
    layouts, the sign-off layouts, and by the name's script."""
    from bench.realdata.score import rate

    def r(sel):
        sel = list(sel)
        return rate(sum(hit[m["id"]] for m in sel), len(sel))

    return {"by_layout": {lay: r(m for m in msgs if m["layout"] == lay) for lay in LAYOUTS},
            "trained": r(m for m in msgs if m["trained"]), "new": r(m for m in msgs if not m["trained"]),
            "sign_off": r(m for m in msgs if m["layout"] in SIGN_OFF),
            "by_script": {"ascii": r(m for m in msgs if m["non_ascii"] is False),
                          "non_ascii": r(m for m in msgs if m["non_ascii"])},
            "all": r(msgs)}


def tagger_part(msgs: Sequence[dict], device: str, batch: int, reuse: bool, log=print) -> dict:
    from bench.tagger import evaluate as E
    from surrogateshield.core.detection import config as C
    from surrogateshield.core.detection import pii_tagger as T
    stage = next(s for s in C.benchmark().detectors if s.name == "pii_tagger")
    local = T.local_dir(stage.model)
    if not local:
        raise SystemExit(f"the tagger {stage.model!r} is not installed")
    T.check_pin(local, stage.revision)
    units = [{"mid": m["id"], "gold": {"text": m["text"]}} for m in msgs]
    cands = E.candidates(Path(local), f"age-probe-{messages_sha(msgs)[:12]}", {"probe": units}, device, batch,
                         reuse, log)["probe"]
    out = {"model": stage.model, "weights_sha256": T.weights_sha256(local), "thresholds": {}}
    for th in THRESHOLDS:
        out["thresholds"][str(th)] = {
            "any": summary(msgs, {m["id"]: caught(m, E.edits_at(cands[m["id"]], th)) for m in msgs}),
            "as_age": summary(msgs, {m["id"]: caught(m, E.edits_at(cands[m["id"]], th, {"AGE"})) for m in msgs})}
    return out


def ss_part(msgs: Sequence[dict], variant: str, config: Optional[Path]) -> dict:
    from bench.realdata import yardstick
    from bench.tagger.evaluate import code_stamp
    src, path = BUILD / "messages.jsonl", BUILD / "ss" / f"{variant}.jsonl"
    write_jsonl(src, [{"id": m["id"], "text": m["text"]} for m in msgs], private=True)
    yardstick.run_ss(config, src, path)
    rows = {r["id"]: r for r in read_jsonl(path)}
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    hit = {m["id"]: "refused" not in rows[m["id"]] and caught(m, rows[m["id"]]["edits"]) for m in msgs}
    return {"code": code_stamp(), "detection_config_hash": meta["config"].get("detection_config_hash"),
            "config_file": meta["config"].get("detection_config_file"), "date": datetime.date.today().isoformat(),
            "refused": sum("refused" in r for r in rows.values()), **summary(msgs, hit)}


def layouts_doc(msgs: Sequence[dict]) -> dict:
    return {lay: {"n": sum(m["layout"] == lay for m in msgs), "trained": lay in TRAINED,
                  "non_ascii": sum(bool(m["non_ascii"]) for m in msgs if m["layout"] == lay),
                  "worded": sum(m["worded"] for m in msgs if m["layout"] == lay)} for lay in LAYOUTS}


def _pct(r: dict) -> str:
    return f"{r['k']}/{r['n']}"


def markdown(doc: dict) -> str:
    th = doc["tagger"]["thresholds"]
    shown = [t for t in ("0.5", "0.7", "0.9") if t in th]
    variants = list(doc["ss"])
    lines = ["# AGE probe (synthetic, diagnostic)", "",
             f"Command: `{doc['command']}` at commit `{doc['git']['commit'][:12]}`", "",
             f"{doc['n']} synthetic messages (seed {doc['seed']}, messages {doc['messages_sha256'][:12]}); names, "
             "cities and ages from the training half of the identity pools. Recall = ages caught / ages (the "
             "scorer's leak rule at the known span). Tagger alone "
             f"`{doc['tagger']['model']}`: `as_age` its AGE candidates only, `any` every candidate.", "",
             "| layout | trained | n | non-ASCII names | " + " | ".join(f"tagger as_age @{t}" for t in shown)
             + " | " + " | ".join(f"tagger any @{t}" for t in shown) + " | "
             + " | ".join(f"SS `{v}`" for v in variants) + " |",
             "|" + "---|" * (4 + 2 * len(shown) + len(variants))]
    rows = [(lay, "yes" if doc["layouts"][lay]["trained"] else "no", doc["layouts"][lay]["n"],
             doc["layouts"][lay]["non_ascii"], lambda part, lay=lay: part["by_layout"][lay]) for lay in LAYOUTS]
    rows += [("**trained**", "", sum(x["n"] for x in doc["layouts"].values() if x["trained"]), "",
              lambda part: part["trained"]),
             ("**new**", "", sum(x["n"] for x in doc["layouts"].values() if not x["trained"]), "",
              lambda part: part["new"]),
             ("**sign-off (eol + signoff)**", "", "", "", lambda part: part["sign_off"]),
             ("non-ASCII names", "", "", "", lambda part: part["by_script"]["non_ascii"]),
             ("ASCII names (new layouts)", "", "", "", lambda part: part["by_script"]["ascii"])]
    for name, tr, n, na, get in rows:
        cells = [_pct(get(th[t]["as_age"])) for t in shown] + [_pct(get(th[t]["any"])) for t in shown]
        cells += [_pct(get(doc["ss"][v])) for v in variants]
        lines.append(f"| {name} | {tr} | {n} | {na} | " + " | ".join(cells) + " |")
    lines += ["", "Tagger `as_age` recall on the sign-off layouts by threshold (D's trigger: < 0.95 after A–C): "
              + ", ".join(f"{t}: {th[t]['as_age']['sign_off']['rate']}" for t in th) + ".", "",
              "SS variants: " + "; ".join(f"`{v}` config `{str(s['detection_config_hash'])[:16]}`, code "
                                          f"`{s['code'][:12]}`, {s['date']}" for v, s in doc["ss"].items()) + "."]
    return "\n".join(lines) + "\n"


def run(out: Path, variants: Sequence[str], config: Optional[Path], device: str, batch: int, reuse: bool,
        log=print) -> dict:
    from bench.realdata.score import rel
    seed = derive_seed("age_probe", VERSION)
    msgs = build(seed)
    BUILD.mkdir(parents=True, exist_ok=True)
    write_jsonl(BUILD / "probe.jsonl", msgs, private=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    if old and old.get("messages_sha256") != messages_sha(msgs):
        raise SystemExit(f"{rel(out)} was made from other probe messages; move it away first")
    ss = dict(old.get("ss", {}))
    for v in variants:
        log(f"ss {v} ...")
        ss[v] = ss_part(msgs, v, config)
    cmd = ("PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m "
           "bench.tagger.age_probe" + "".join(f" --ss {v}" for v in variants)
           + (f" --config {rel(config)}" if config else "") + f" --out {rel(out)}")
    doc = {"command": cmd, "git": git_state(), "role": "diagnostic, synthetic", "version": VERSION, "seed": seed,
           "messages_sha256": messages_sha(msgs), "n": len(msgs), "layouts": layouts_doc(msgs),
           "tagger": tagger_part(msgs, device, batch, reuse, log), "ss": ss}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ss", action="append", default=[], help="run SS as this variant (repeatable)")
    ap.add_argument("--config", type=Path, help="a partial config file merged on the benchmark config")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reuse", action="store_true", help="reuse the tagger's cached candidates")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    doc = run(a.out, a.ss, a.config, a.device, a.batch, a.reuse)
    th = doc["tagger"]["thresholds"]
    print(f"{doc['n']} messages; tagger as_age sign-off recall "
          + ", ".join(f"{t}: {th[t]['as_age']['sign_off']['rate']}" for t in th))
    for v, s in doc["ss"].items():
        print(f"ss {v}: all {s['all']['k']}/{s['all']['n']}, new {s['new']['k']}/{s['new']['n']}, "
              f"sign-off {s['sign_off']['k']}/{s['sign_off']['n']}, non-ASCII {s['by_script']['non_ascii']['k']}/"
              f"{s['by_script']['non_ascii']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
