"""E8 on a sealed split, read letter for letter: exact recovery only, partial
recoveries set aside. Made on 2026-10-09, after test-3's pre-registered H8'''
was missed, from the attacker replies the scored run cached and the committed
result. It sends nothing to any provider.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.attacker_exact --split test3 --out bench/results/attacker_exact_test3.json

H8''' (``HYPOTHESES_TEST3.md`` §4, operationalised in ``live.py``) counts a
value as recovered when a guess equals it or a part of it (an e-mail domain, a
URL host, a name or organisation word, a birth year, ...). The authors asked
for the same replies read letter for letter: a value counts only when a guess
equals it under ``live.canon`` (case, URL scheme, punctuation and spacing
folded; nothing else). This module

* re-reads the per-value outcomes the scored run cached
  (``experiment/realdata/<split>/attacker-rows.jsonl``, no text) and refuses
  unless they reproduce every arm summary of the committed result, so the
  reading is anchored to the run that was scored once;
* reports, per group and arm, exact recovery and "known letter for letter"
  (exact recoveries plus the values the arm left in), with the partials beside;
* re-reads H8'''s three parts with exact recovery only: (a) no exact recovery
  in the single-message condition, (b) none in the conversation condition,
  (c) SS's exact recovery not above GLiNER-PII's (point estimates; the paired
  cluster bootstrap over messages beside);
* opens the cached replies behind SS's exact rows and records how each came
  about, as flags and counts only (no value, no text).

The pre-registered verdict (``verdict_<split>.json``) is copied in and is not
changed by this file: H8''' is in the GO rule and its reading is the one
pre-registered.

**The revised clause and the authors' decision.** On 2026-10-09, after test-3
was scored, the authors replaced the attacker clause for the paper: the two
zero-recovery parts of H8''' are dropped and the comparative part is read
letter for letter (``H8-rev``: SS's exact recovery, over the values it
replaced, not above GLiNER-PII's by point estimate, pooled). Their reason: a
zero-recovery clause is falsified by one hit and measures nothing against the
alternatives; a comparative clause does. This module evaluates ``H8-rev`` on
the same replies and records the authors' decision for the paper ("go for the
paper" when every other pre-registered GO-rule part holds as scored and
``H8-rev`` holds). The decision is a deviation from the pre-registration, made
after the result, and is labelled as one wherever it is reported; the
pre-registered verdict is not relabelled.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bench.realdata import live as L
from bench.realdata import score
from bench.realdata.common import DATASETS, ROOT, derive_seed, file_sha256, git_state, read_jsonl

RESULTS = ROOT / "bench" / "results"
MADE = "2026-10-09"
ATTACKED = ("exact", "partial", "not_recovered")
LEFT_IN = ("leaked", "policy")
NOT_ATTACKED = ("unavailable", "refused")
EXACT_ARMS = ("ss", "ss-conversation")
REV = "H8-rev"
REV_TEXT = ("SS's exact recovery (letter for letter, over the values it replaced) is not above GLiNER-PII's by "
            "point estimate, pooled; each dataset and every other baseline reported beside. Adopted by the authors "
            "on 2026-10-09, after test-3 was scored, in place of H8''' for the paper.")
REV_REASON = ("a zero-recovery clause is falsified by one hit and measures nothing against the alternatives; the "
              "comparative clause measures SS against the strongest baseline on the same prompts. Partial "
              "recoveries are parts the surrogates keep by design and are reported, not counted.")
DISCLOSURE = ("a deviation from the pre-registration, made after the result. The paper states the pre-registered "
              "clause, that it was missed (exact recoveries in both conditions, all URLs whose surrogate kept the "
              "path), and the clause it reports under. Nothing was re-run; no provider call was made.")
# partial classes a surrogate rule keeps on purpose (DETECTOR_REPORT §6)
KEPT_BY_RULE = {"email_domain": "identity.py: free-mail domains are kept",
                "org_word": "mimic.py: an organisation keeps its legal and industry words",
                "url_host": "mimic.py: a URL keeps its host and path layout",
                "address_part": "mimic.py: a town becomes a real town of the same country",
                "birth_year": "mimic.py: a date of birth moves by at most two years",
                "phone_area": "mimic.py: a phone keeps its country code and trunk digit"}
OPAQUE = re.compile(r"[A-Za-z0-9_-]{20,}")


def anchor(rows: Sequence[dict], committed: Dict[str, dict]) -> int:
    """Recompute every arm summary from *rows* with the scorer's own
    ``summarise_attack`` and compare it with the committed one; the number of
    summaries checked, or SystemExit on the first difference."""
    checked, bad = 0, []
    for g, res in committed.items():
        for arm, summary in res.items():
            if arm.startswith("H"):
                continue
            mine = L.summarise_attack([r for r in rows if r["arm"] == arm and L._in(g, r["dataset"])])
            checked += 1
            if mine != summary:
                bad.append(f"{g}/{arm}")
    if bad:
        raise SystemExit("the cached rows do not reproduce the committed summaries: " + ", ".join(bad))
    return checked


def summarise_exact(rows: Sequence[dict]) -> dict:
    """One arm's reading: exact recovery over the values it replaced, known
    letter for letter (exact + left in) over the values with a reply, and the
    partials set aside by class."""
    c = Counter(r["outcome"] for r in rows)
    attacked = sum(c[o] for o in ATTACKED)
    left_in = sum(c[o] for o in LEFT_IN)
    unavailable = sum(c[o] for o in NOT_ATTACKED)
    classes = Counter(r["class"] for r in rows if r["outcome"] == "partial")
    visible = Counter(r["class"] for r in rows if r["outcome"] == "partial" and r["visible"])
    return {"values": len(rows), "left_in": left_in, "unavailable": unavailable, "attacked": attacked,
            "exact": score.rate(c["exact"], attacked),
            "exact_types": dict(sorted(Counter(r["type"] for r in rows if r["outcome"] == "exact").items())),
            "known_letter_for_letter": score.rate(c["exact"] + left_in, len(rows) - unavailable),
            "partial_set_aside": {"n": c["partial"],
                                  "classes": {k: {"n": v, "visible_in_text": visible[k], "kept_by_rule": k in KEPT_BY_RULE}
                                              for k, v in sorted(classes.items())}}}


def _exact_rate(a: Optional[dict]) -> Optional[float]:
    return a["exact"]["rate"] if a and a["attacked"] else None


def h8_exact(res: dict, rows: Sequence[dict], g: str, plan: L.Plan) -> dict:
    """H8'''s three parts with exact recovery only, on group *g*."""
    ss, gl, conv = res["ss"], res.get("gliner_pii"), res.get("ss-conversation")
    per: Dict[str, Dict[str, list]] = {}
    for r in rows:
        if r["arm"] in ("ss", "gliner_pii") and L._in(g, r["dataset"]) and r["outcome"] in ATTACKED:
            x = per.setdefault(f"{r['dataset']}/{r['id']}", {"ss": [0, 0], "gliner_pii": [0, 0]})[r["arm"]]
            x[0] += r["outcome"] == "exact"
            x[1] += 1
    both = sorted(k for k, v in per.items() if v["ss"][1] and v["gliner_pii"][1])
    diff = score.bootstrap(both, [tuple(per[k]["ss"]) for k in both], [tuple(per[k]["gliner_pii"]) for k in both],
                           derive_seed(plan.tag("live-boot"), "attack-exact", g)) if both else {"diff": None, "ci95": None}
    parts = {"no_exact_single": ss["attacked"] > 0 and ss["exact"]["k"] == 0,
             "no_exact_conversation": bool(conv) and conv["attacked"] > 0 and conv["exact"]["k"] == 0,
             "exact_not_above_gliner_pii": (_exact_rate(ss) is not None and _exact_rate(gl) is not None
                                            and _exact_rate(ss) <= _exact_rate(gl))}
    return {**parts, "holds": all(parts.values()), "fails_on": sorted(k for k, v in parts.items() if not v),
            "exact": {"ss": ss["exact"], "gliner_pii": gl and gl["exact"],
                      "ss-gliner_pii": {**diff, "messages": len(both)}}}


def revised_clause(results: Dict[str, dict], plan: L.Plan) -> dict:
    """``H8-rev`` on the letter-for-letter reading: the deciding part pooled,
    each dataset and every other baseline beside."""
    key = f"{plan.h8} letter-for-letter"
    pooled = results["all"]
    ss = _exact_rate(pooled["ss"])
    baselines = {}
    for arm, a in pooled.items():
        if arm in ("ss", "ss-conversation", key):
            continue
        r = _exact_rate(a)
        baselines[arm] = {"exact": a["exact"], "ss_not_above": ss is not None and r is not None and ss <= r}
    return {"clause": REV_TEXT, "reason": REV_REASON, "adopted": MADE, "in_place_of": plan.h8,
            "holds": pooled[key]["exact_not_above_gliner_pii"],
            "pooled": {"ss": pooled["ss"]["exact"], "gliner_pii": pooled["gliner_pii"]["exact"],
                       "ss-gliner_pii": pooled[key]["exact"]["ss-gliner_pii"]},
            "per_dataset": {g: r[key]["exact_not_above_gliner_pii"] for g, r in results.items() if g != "all"},
            "beside_every_baseline": baselines,
            "ss_lowest_exact_of_all_arms": all(b["ss_not_above"] for b in baselines.values())}


def decision(verdict: dict, rev: dict, plan: L.Plan) -> dict:
    """The authors' decision for the paper: every pre-registered GO-rule part
    other than the attacker's, as scored, and ``H8-rev`` in its place."""
    others = {k: v for k, v in verdict["go_parts"].items() if k != plan.h8}
    parts = {**others, REV: rev["holds"]}
    go = all(parts.values())
    return {"date": MADE, "by": "the authors", "decision": "go for the paper" if go else "no go for the paper",
            "under": f"{REV} in place of {plan.h8}; every other GO-rule part as scored in verdict_{verdict['split']}.json",
            "go_rule_parts": parts,
            "pre_registered_verdict": f"{verdict['verdict']} on {', '.join(verdict['failed'] or [])}; unchanged",
            "reported_as": DISCLOSURE}


def url_mechanism(gold: str, surrogate: Optional[str], guess: str) -> dict:
    """How one exact URL recovery came about, as flags: what the surrogate
    changed and kept, and whether the guess equals the original letter for
    letter (before any folding)."""
    t, g = L._url(gold), L._url(guess)
    th, tp = (t.split("/", 1) + [""])[:2]
    out = {"guess_equals_gold_letter_for_letter": guess.strip() == gold.strip(), "guess_equals_gold_canon": g == t,
           "path_has_opaque_identifier": bool(OPAQUE.search(tp))}
    if surrogate is not None:
        sh, sp = (L._url(surrogate).split("/", 1) + [""])[:2]
        tl, sl = th.split("."), sh.split(".")
        out.update(surrogate_changed_host=sh != th, surrogate_kept_path=sp == tp,
                   host_labels_changed=sum(a != b for a, b in zip(tl, sl)) if len(tl) == len(sl) else None)
    return out


def open_exact_rows(rows: Sequence[dict], split: str, plan: L.Plan) -> List[dict]:
    """SS's exact rows, each opened against the split's data and the cached
    reply: the condition, the type and the mechanism flags. No value leaves."""
    import attacker
    data = L.load_test(list(DATASETS), plan.spans, split)
    cache: Dict[str, dict] = {}
    for r in read_jsonl(L.RUNS / split / "results.jsonl"):
        cache[r["key"]] = r                                   # a later row (a retry) wins, as Run reads it
    by_slot = {r["slot"]: r for r in cache.values() if r.get("stage") in ("attack", "attack-conv")}
    out = []
    for r in rows:
        if r["arm"] not in EXACT_ARMS or r["outcome"] != "exact":
            continue
        ds, ident, typ = r["dataset"], r["id"], r["type"]
        single = r["arm"] == "ss"
        mids = [ident] if single else L.turn_mids(data, ds, ident)
        gold, seen = [], set()
        for m in mids:
            for t, v in L.message_gold(data[ds]["units"][m]):
                if t == typ and v.casefold() not in seen:
                    seen.add(v.casefold())
                    gold.append(v)
        reps: Dict[str, str] = {}
        for m in mids:
            text = data[ds]["units"][m]["gold"]["text"]
            for s, e, _t, rep in (L.edits_of(data, ds, "ss", m) or []):
                for v in gold:
                    if v.casefold() in text[s:e].casefold() or text[s:e].casefold() in v.casefold():
                        reps.setdefault(v, rep)
        reply = by_slot.get(L.slot(ds, ident, "attack-ss" if single else "attack-conv"))
        parsed = attacker.parse_reply(reply["text"] or "") if reply and reply["status"] == "ok" else None
        hit = next(((gs, v) for gs in (attacker._guesses(parsed) if parsed else []) for v in gold
                    if L.recovery(typ, gs, v)[0] == "exact"), None)
        row = {"dataset": ds, "id": ident, "condition": "single" if single else "conversation", "type": typ}
        if hit is None:
            row["status"] = "no exact guess in the cached reply"
        elif typ == "URL":
            row.update(url_mechanism(hit[1], reps.get(hit[1]), hit[0]))
        else:
            row["guess_equals_gold_letter_for_letter"] = hit[0].strip() == hit[1].strip()
        out.append(row)
    return out


def mechanism_summary(opened: Sequence[dict]) -> dict:
    urls = [o for o in opened if o["type"] == "URL" and "surrogate_changed_host" in o]
    return {"exact_rows": len(opened), "url_rows": len(urls),
            "guess_equals_gold_letter_for_letter": sum(o.get("guess_equals_gold_letter_for_letter") is True for o in opened),
            "surrogate_changed_host_and_kept_path": sum(o["surrogate_changed_host"] and o["surrogate_kept_path"] for o in urls),
            "one_host_label_changed": sum(o.get("host_labels_changed") == 1 for o in urls),
            "path_has_opaque_identifier": sum(o["path_has_opaque_identifier"] for o in urls)}


def reading(rows: Sequence[dict], plan: L.Plan, datasets: Sequence[str] = DATASETS) -> Dict[str, dict]:
    out = {}
    for g in ["all", *datasets]:
        res = {arm: summarise_exact([r for r in rows if r["arm"] == arm and L._in(g, r["dataset"])])
               for arm in (*plan.attack, "ss-conversation")}
        res[f"{plan.h8} letter-for-letter"] = h8_exact(res, rows, g, plan)
        out[g] = res
    return out


def run(split: str, out: Path) -> dict:
    plan = L.PLANS[split]
    committed_path = RESULTS / f"attacker_realdata_{split}.json"
    committed = json.loads(committed_path.read_text())
    if committed.get("split") != split:
        raise SystemExit(f"{score.rel(committed_path)}: split {committed.get('split')!r}, not {split}")
    rows_path = L.RUNS / split / "attacker-rows.jsonl"
    rows = read_jsonl(rows_path)
    checked = anchor(rows, committed["results"])
    verdict_path = RESULTS / f"verdict_{split}.json"
    verdict = json.loads(verdict_path.read_text())
    pre = verdict["hypotheses"][plan.h8]
    results = reading(rows, plan)
    rev = revised_clause(results, plan)
    opened = open_exact_rows(rows, split, plan)
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python "
                      f"-m bench.realdata.attacker_exact --split {split} --out {score.rel(out)}",
           "git": git_state(), "split": split, "made": MADE,
           "reading": "letter for letter: a value counts as recovered only when a guess equals it under live.canon; "
                      "partial recoveries are set aside and reported beside",
           "decides": f"the pre-registered verdict: nothing. The authors' decision for the paper is recorded under "
                      f"'authors_decision', with the revised clause {REV} it rests on; made after the pre-registered "
                      f"{plan.h8} was missed, from the same replies, with no provider call",
           "pre_registered": {"verdict_file": score.rel(verdict_path), "sha256": file_sha256(verdict_path),
                              "verdict": verdict["verdict"], "failed": verdict.get("failed"),
                              "hypotheses": verdict["prereg"],
                              plan.h8: {k: pre[k] for k in ("holds", "no_exact_single", "no_exact_conversation",
                                                           "recovery_not_above_gliner_pii", "recovery")}},
           "inputs": {"attacker": {"file": score.rel(committed_path), "sha256": file_sha256(committed_path),
                                   "commit": (committed.get("git") or {}).get("commit")},
                      "rows": {"file": score.rel(rows_path), "sha256": file_sha256(rows_path), "n": len(rows),
                               "summaries_reproduced": checked},
                      "replies": {"file": score.rel(L.RUNS / split / "results.jsonl"), "opened": len(opened)}},
           "results": results, "exact_rows_opened": opened, "exact_mechanism": mechanism_summary(opened),
           REV: rev, "authors_decision": decision(verdict, rev, plan)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


# ── markdown ─────────────────────────────────────────────────────────────────

def _r(x: Optional[dict]) -> str:
    if not x or not x["n"]:
        return "–"
    lo, hi = x["wilson95"]
    return f"{x['k']}/{x['n']} ({x['rate'] * 100:.1f} %, {lo * 100:.1f}–{hi * 100:.1f})"


def _bs(d: dict) -> str:
    if d.get("diff") is None:
        return "–"
    lo, hi = d["ci95"]
    return f"{d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _yn(b: bool) -> str:
    return "yes" if b else "**no**"


def markdown(doc: dict) -> str:
    split, h8 = doc["split"], L.PLANS[doc["split"]].h8
    pre, key, rev, dec = doc["pre_registered"], f"{h8} letter-for-letter", doc[REV], doc["authors_decision"]
    rec = pre[h8]["recovery"]
    name = split.replace("test", "test-")
    lines = [f"# E8 on {name}, read letter for letter (made {doc['made']}, after the fact)", "",
             f"`{doc['command']}` at commit `{(doc['git'].get('commit') or '?')[:12]}`. Inputs: the committed "
             f"`{doc['inputs']['attacker']['file']}` ({doc['inputs']['attacker']['sha256'][:12]}, run at "
             f"`{str(doc['inputs']['attacker']['commit'])[:7]}`) and the per-value outcomes its run cached "
             f"({doc['inputs']['rows']['n']:,} rows, no text), which reproduce all "
             f"{doc['inputs']['rows']['summaries_reproduced']} committed arm summaries. No provider call was made.", "",
             f"**The pre-registered verdict stands: {name} is {pre['verdict']} (`{pre['verdict_file']}`), on "
             f"{', '.join(pre['failed'] or [])}.** {h8} counts exact and partial recoveries (SS {rec['ss']['k']}/"
             f"{rec['ss']['n']} vs GLiNER-PII {rec['gliner_pii']['k']}/{rec['gliner_pii']['n']}). This file reads the "
             f"same replies with exact recovery only, as the authors asked on {doc['made']}, after that result; it is "
             "reported beside the pre-registered reading and does not change the verdict.", "",
             f"**The authors' decision ({dec['date']}): {dec['decision']}**, under the revised attacker clause {REV} "
             f"in place of {h8}. {REV}: {rev['clause']} It **{'holds' if rev['holds'] else 'does not hold'}**: SS "
             f"{_r(rev['pooled']['ss'])} vs GLiNER-PII {_r(rev['pooled']['gliner_pii'])}, ss − gliner_pii "
             f"{_bs(rev['pooled']['ss-gliner_pii'])} over {rev['pooled']['ss-gliner_pii']['messages']} messages; on "
             f"each dataset: {', '.join(f'{g} {_yn(v)}' for g, v in rev['per_dataset'].items())}; SS's exact recovery "
             f"is {'the lowest of all arms' if rev['ss_lowest_exact_of_all_arms'] else 'not the lowest of all arms'} ("
             + ", ".join(f"{a} {_r(b['exact'])}" for a, b in rev["beside_every_baseline"].items()) + "). Every other "
             f"GO-rule part as scored: {', '.join(f'{k} {_yn(v)}' for k, v in dec['go_rule_parts'].items() if k != REV)}. "
             f"Reason for the revision: {rev['reason']} This is {dec['reported_as']}", "",
             "| group | arm | values | left in | unavailable | exact (of the values the arm replaced) | "
             "known letter for letter (exact + left in) | partial, set aside |", "|---|---|---|---|---|---|---|---|"]
    for g, res in doc["results"].items():
        for arm, a in res.items():
            if arm == key:
                continue
            lines.append(f"| {g} | {arm} | {a['values']} | {a['left_in']} | {a['unavailable']} | {_r(a['exact'])} "
                         f"| {_r(a['known_letter_for_letter'])} | {a['partial_set_aside']['n']} |")
    lines += ["", f"{h8} with exact recovery only (the pre-registered reading counts partials too):", ""]
    for g, res in doc["results"].items():
        h = res[key]
        lines.append(f"- {g}: **{'holds' if h['holds'] else 'does not hold'}** ((a) no exact, single-message: "
                     f"{_yn(h['no_exact_single'])}, {h['exact']['ss']['k']}; (b) no exact, conversation: "
                     f"{_yn(h['no_exact_conversation'])}, {res['ss-conversation']['exact']['k']}; (c) SS exact not above "
                     f"GLiNER-PII: {_yn(h['exact_not_above_gliner_pii'])}, {_r(h['exact']['ss'])} vs "
                     f"{_r(h['exact']['gliner_pii'])}; ss − gliner_pii {_bs(h['exact']['ss-gliner_pii'])} over "
                     f"{h['exact']['ss-gliner_pii']['messages']} messages)")
    m = doc["exact_mechanism"]
    lines += ["", f"**SS's {m['exact_rows']} exact recoveries, opened (flags only).** {m['url_rows']} are URLs. In "
              f"{m['surrogate_changed_host_and_kept_path']} of them the surrogate changed the host and kept the path "
              f"as it was ({m['one_host_label_changed']} with one host label changed), and in "
              f"{m['path_has_opaque_identifier']} the kept path carries an opaque identifier of 20 or more "
              f"characters; the attacker's guess equals the original letter for letter in "
              f"{m['guess_equals_gold_letter_for_letter']} of {m['exact_rows']}.", "",
              "| condition | dataset | id | type | surrogate changed host | kept path | host labels changed | "
              "path has opaque id | guess = original, letter for letter |", "|---|---|---|---|---|---|---|---|---|"]
    for o in doc["exact_rows_opened"]:
        lines.append(f"| {o['condition']} | {o['dataset']} | {o['id']} | {o['type']} | "
                     f"{o.get('surrogate_changed_host', '–')} | {o.get('surrogate_kept_path', '–')} | "
                     f"{o.get('host_labels_changed', '–')} | {o.get('path_has_opaque_identifier', '–')} | "
                     f"{o.get('guess_equals_gold_letter_for_letter', o.get('status', '–'))} |")
    ss = doc["results"]["all"]["ss"]["partial_set_aside"]
    lines += ["", f"**Partials set aside, SS pooled ({ss['n']}):**", "",
              "| class | n | visible in the text the attacker read | kept by a surrogate rule |", "|---|---|---|---|"]
    for k, v in ss["classes"].items():
        lines.append(f"| {k} | {v['n']} | {v['visible_in_text']} | {KEPT_BY_RULE.get(k, 'no')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=[s for s in L.PLANS if s != "test"], default="test3")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = run(a.split, a.out.resolve())
    h8 = L.PLANS[a.split].h8
    print(f"pre-registered: {doc['pre_registered']['verdict']} (unchanged); {h8} letter for letter: "
          f"{'holds' if doc['results']['all'][f'{h8} letter-for-letter']['holds'] else 'does not hold'}; "
          f"{REV}: {'holds' if doc[REV]['holds'] else 'does not hold'}; authors' decision: "
          f"{doc['authors_decision']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
