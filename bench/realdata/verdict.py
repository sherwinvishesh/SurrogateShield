"""V3 Phase 5: the test-2 verdict (GO / GO-with-caveat / NO-GO), computed from
the frozen run's result files as ``HYPOTHESES_TEST2.md`` pre-registered it.

    PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict --out bench/results/verdict_test2.json
    ... -m bench.realdata.verdict --split test3 --out bench/results/verdict_test3.json

Committed before ``FREEZE.json``, so the readings below were fixed before any
test-2 number existed. It reads result files only (no span, no text) and
estimates nothing itself: every interval is the scorer's.

Inputs (``bench/results/``) and what each decides:

* ``realdata_test2.json`` (E5): H1'', H2'', H3'', H4'', H5'', and H9'' with the next;
* ``realdata_components_test2.json`` (E9): H9'' (``h9/no_patterns`` and ``no_tagger``);
* ``perf_arms_v3.json`` (E7): H6'';
* ``utility_realdata_test2.json`` and ``multiturn_realdata_test2.json`` (E5b, E7b): H7'';
* ``attacker_realdata_test2.json`` (E8): H8'';
* ``external_nemotron_pii.json``: H10''.

A missing file leaves its hypotheses ``not run``; a verdict that needs one is
``incomplete``, never GO. The scored test-2 files must carry this
FREEZE.json's SHA-256 and the live ones split ``test2``, or nothing is written.

Readings, where the sentence leaves a choice:

* Differences are the scorers': SS − arm, paired cluster bootstrap over
  conversations (2,000 resamples), 95 % percentile interval. "Lower, CI
  excluding 0" means the interval's upper bound is below 0.
* "GLiNER-PII" is ``gliner_pii`` (GLiNER-PII at its default). H1'' and the
  six baselines of H3'' include ``gliner_pii_tuned`` as well; elsewhere the
  tuned arm is reported beside and decides nothing.
* A hypothesis that does not say "each dataset" is decided on the pooled
  group (``all``), as H4'' and H5'' say; per-dataset values are reported.
* H1'' (each dataset): against each of the six baselines, SS's injected leak
  rate is lower and the interval's upper bound is below 0.
* H2'' (each dataset): against gliner_pii, presidio_default and
  presidio_faker, the injected-slice spurious-edit rate (spurious edits /
  edits) difference has an upper bound below 0.
* H3'' (each dataset): SS's natural-slice ``negatives_untouched`` rate is
  strictly higher than each of the six baselines' (a tie fails; the
  hypothesis states no interval).
* H4'': pooled shift-slice leak difference against gliner_pii, upper bound
  below 0.
* H5'': a type fails when its pooled SS − gliner_pii leak difference
  (``differences_by_type``) is above 0.02 and the interval's lower bound is
  above 0; the hypothesis holds when no type fails.
* H6'': in ``perf_arms_v3.json``, ``ss`` with the benchmark config (the
  file records ``detection_config: null``: no partial config, the default
  ``balanced`` pipeline) has p50_ms and peak_rss_mb each at most gliner_pii's.
* H7'': live.py's ``H7''`` blocks at ``all``, as live.py fixed them: E5b
  (BERTScore difference and judge score of ss against presidio_faker and
  gliner_pii, each lower bound above −0.02) and E7b (judge grade and
  consistency of ss above each, lower bound above 0); both must hold.
* H8'': live.py's ``H8''`` at ``all``: no exact recovery in the
  single-message and the conversation condition, and SS's recovery rate
  (exact + partial) not above gliner_pii's.
* H9'': the pooled injected leak rate of ``h9/no_patterns`` (every
  PatternScan pattern off, the five types only patterns reported routed to
  the tagger, as the hypothesis's parenthesis has it) and of ``no_tagger``
  each below presidio_default's (point estimates; the hypothesis states no
  interval). ``h9/no_patterns_unrouted`` (default routing) is reported beside.
* H10'': score_external's ``H10''`` against gliner_pii (leak not above, or
  span F1 not below).

GO = H1'' and H2'' on all three datasets, H3'', H5'', H6'', H8''.
GO-with-caveat = H1'' on two datasets; on the third, SS not worse than
GLiNER-PII (the leak interval's lower bound at most 0, against gliner_pii
and gliner_pii_tuned); every other GO condition held. Anything else is
NO-GO. H4'', H7'', H9'' and H10'' are reported; the GO rule names none, and
one not run is listed under ``not_run`` without changing the verdict.

Test-3 (``--split test3``; ``HYPOTHESES_TEST3.md``, PROMPT_FOR_OPUS_V4): the
same readings and the same ``decide`` over test-3's files
(``INPUTS_TEST3``: ``realdata_test3.json``, ``realdata_components_test3.json``,
``perf_arms_v4.json``, the live ``*_realdata_test3.json`` and the external
benchmark, which is scored at test-3's freeze and must carry it). The rule is
computed under the names above and the result is written with test-3's
(H1''' ... H10'''); the live and external files name their blocks H7''',
H8''' and H10'''.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bench.realdata import score
from bench.realdata.common import COLLECTIONS, DATASETS, ROOT, file_sha256, git_state

RESULTS = ROOT / "bench" / "results"
INPUTS = {"e5": "realdata_test2.json", "components": "realdata_components_test2.json",
          "perf": "perf_arms_v3.json", "utility": "utility_realdata_test2.json",
          "multiturn": "multiturn_realdata_test2.json", "attacker": "attacker_realdata_test2.json",
          "external": "external_nemotron_pii.json"}
INPUTS_TEST3 = {"e5": "realdata_test3.json", "components": "realdata_components_test3.json",
                "perf": "perf_arms_v4.json", "utility": "utility_realdata_test3.json",
                "multiturn": "multiturn_realdata_test3.json", "attacker": "attacker_realdata_test3.json",
                "external": "external_nemotron_pii.json"}
SEALED = ("e5", "components")                    # scored on test-2: carry freeze_sha256
# per sealed collection: its inputs, the files that must carry its freeze, and its hypothesis mark
SPLITS = {"test2": {"inputs": INPUTS, "sealed": SEALED, "mark": "''"},
          "test3": {"inputs": INPUTS_TEST3, "sealed": (*SEALED, "external"), "mark": "'''"}}
LIVE = ("utility", "multiturn", "attacker")      # live on test-2: carry split
SIX = ("presidio_default", "presidio_faker", "presidio_transformers", "llm_guard", "gliner_pii", "gliner_pii_tuned")
GLINER, TUNED = "gliner_pii", "gliner_pii_tuned"
H2_AGAINST = (GLINER, "presidio_default", "presidio_faker")
H5_MARGIN = 0.02
NO_PATTERNS, NO_TAGGER, UNROUTED = "h9/no_patterns", "no_tagger", "h9/no_patterns_unrouted"
H7, H8, H10 = "H7''", "H8''", "H10''"
NOT_RUN = {"holds": None, "status": "not run"}
GO_PARTS = ("H1''", "H2''", "H3''", "H5''", "H6''", "H8''")


def load(results: Path = RESULTS, freeze: Optional[Path] = None, split: str = "test2") -> Dict[str, Optional[dict]]:
    """Every input that exists, checked: the sealed ones against the split's
    freeze, the live ones by split."""
    spec, freeze = SPLITS[split], freeze or COLLECTIONS[split].freeze
    sealed = file_sha256(freeze) if freeze.exists() else None
    docs: Dict[str, Optional[dict]] = {}
    for key, name in spec["inputs"].items():
        path = results / name
        if not path.exists():
            docs[key] = None
            continue
        doc = json.loads(path.read_text())
        where = doc.get("collection") if key == "external" else doc.get("split")
        if key in spec["sealed"] and (sealed is None or doc.get("freeze_sha256") != sealed or where != split):
            raise SystemExit(f"{name}: not a {split.replace('test', 'test-')} result of {score.rel(freeze)} "
                             f"(freeze {str(doc.get('freeze_sha256'))[:12]}, split {where})")
        if key in LIVE and doc.get("split") != split:
            raise SystemExit(f"{name}: split {doc.get('split')!r}, not {split}")
        doc["_input"] = {"file": f"bench/results/{name}", "sha256": file_sha256(path),
                         "commit": (doc.get("git") or {}).get("commit")}
        docs[key] = doc
    return docs


def _below(d: Optional[dict]) -> bool:
    return bool(d and d.get("ci95") and d["ci95"][1] < 0)


def _rate(r: Optional[dict]) -> Optional[float]:
    return r.get("rate") if r else None


def _cmp(d: Optional[dict]) -> dict:
    return {"diff": d.get("diff"), "ci95": d.get("ci95")} if d else {"diff": None, "ci95": None}


def h1(e5: dict, datasets: Sequence[str]) -> dict:
    out = {}
    for ds in datasets:
        res, dif = e5["results"][ds]["injected"], e5["differences"][ds]["injected"]
        per = {}
        for a in SIX:
            d = (dif.get(a) or {}).get("leak_rate")
            s, x = _rate(res["ss"]["leak"]), _rate((res.get(a) or {}).get("leak"))
            per[a] = {"ss": s, "arm": x, **_cmp(d), "holds": s is not None and x is not None and s < x and _below(d)}
        not_worse = {g: bool((dif.get(g) or {}).get("leak_rate", {}).get("ci95"))
                     and dif[g]["leak_rate"]["ci95"][0] <= 0 for g in (GLINER, TUNED)}
        out[ds] = {"per_baseline": per, "holds": all(p["holds"] for p in per.values()),
                   "not_worse_than_gliner": all(not_worse.values()), "not_worse": not_worse}
    return {"per_dataset": out, "holds": all(v["holds"] for v in out.values()),
            "datasets_held": sum(v["holds"] for v in out.values())}


def h2(e5: dict, datasets: Sequence[str]) -> dict:
    out = {}
    for ds in datasets:
        dif = e5["differences"][ds]["injected"]
        per = {a: {**_cmp((dif.get(a) or {}).get("spurious_rate")),
                   "holds": _below((dif.get(a) or {}).get("spurious_rate"))} for a in H2_AGAINST}
        beside = {TUNED: {**_cmp((dif.get(TUNED) or {}).get("spurious_rate")),
                          "lower": _below((dif.get(TUNED) or {}).get("spurious_rate"))}}
        out[ds] = {"per_baseline": per, "beside": beside, "holds": all(p["holds"] for p in per.values())}
    return {"per_dataset": out, "holds": all(v["holds"] for v in out.values())}


def h3(e5: dict, datasets: Sequence[str]) -> dict:
    out = {}
    for ds in datasets:
        res = e5["results"][ds]["natural"]
        s = res["ss"]["negatives_untouched"]
        per = {a: {"arm": (res.get(a) or {}).get("negatives_untouched"),
                   "holds": _rate(s) is not None and _rate((res.get(a) or {}).get("negatives_untouched")) is not None
                   and _rate(s) > _rate(res[a]["negatives_untouched"])} for a in SIX}
        out[ds] = {"ss": s, "per_baseline": per, "holds": all(p["holds"] for p in per.values())}
    return {"per_dataset": out, "holds": all(v["holds"] for v in out.values())}


def h4(e5: dict, datasets: Sequence[str]) -> dict:
    d = (e5["differences"]["all"]["shift"].get(GLINER) or {}).get("leak_rate")
    res = e5["results"]["all"]["shift"]
    return {"ss": res["ss"]["leak"], "gliner_pii": (res.get(GLINER) or {}).get("leak"), **_cmp(d), "holds": _below(d),
            "beside": {TUNED: _cmp((e5["differences"]["all"]["shift"].get(TUNED) or {}).get("leak_rate")),
                       **{ds: _cmp((e5["differences"][ds]["shift"].get(GLINER) or {}).get("leak_rate"))
                          for ds in datasets}}}


def h5(e5: dict) -> dict:
    if "differences_by_type" not in e5:
        return {**NOT_RUN, "status": "realdata_test2.json has no differences_by_type"}
    bt = e5["differences_by_type"]["groups"]["all"]
    by_type = e5["results"]["all"]["injected"]

    def fails(arm: str) -> List[str]:
        return sorted(t for t, d in (bt.get(arm) or {}).items()
                      if d.get("diff") is not None and d["diff"] > H5_MARGIN and d["ci95"][0] > 0)

    types = sorted(bt.get(GLINER) or {})
    table = {t: {"ss": by_type["ss"]["by_type"].get(t), "gliner_pii": (by_type.get(GLINER) or {}).get("by_type", {}).get(t),
                 **_cmp(bt[GLINER][t])} for t in types}
    return {"failing_types": fails(GLINER), "by_type": table, "holds": bool(types) and not fails(GLINER),
            "beside": {TUNED: {"failing_types": fails(TUNED)}}}


def h6(perf: dict) -> dict:
    rows = {r["arm"]: r for r in perf["arms"]}
    ss, gl = rows.get("ss"), rows.get(GLINER)
    if not ss or not gl:
        return {**NOT_RUN, "status": "perf_arms_v3.json lacks ss or gliner_pii"}
    keep = ("p50_ms", "p95_ms", "peak_rss_mb", "model_load_s", "cold_start_s")
    default = "detection_config" in perf and perf["detection_config"] is None
    return {"ss": {k: ss[k] for k in keep}, "gliner_pii": {k: gl[k] for k in keep}, "default_pipeline": default,
            "p50_not_above": ss["p50_ms"] <= gl["p50_ms"], "rss_not_above": ss["peak_rss_mb"] <= gl["peak_rss_mb"],
            "holds": default and ss["p50_ms"] <= gl["p50_ms"] and ss["peak_rss_mb"] <= gl["peak_rss_mb"]}


def h7(utility: Optional[dict], multiturn: Optional[dict], key: str = H7) -> dict:
    if utility is None or multiturn is None:
        return {**NOT_RUN, "status": "utility or multiturn not run"}
    u, m = utility["e5b"]["all"][key], multiturn["results"]["all"][key]
    return {"e5b": u, "e7b": m, "holds": bool(u["holds"] and m["holds"]),
            "per_dataset": {g: {"e5b": utility["e5b"][g][key]["holds"], "e7b": multiturn["results"][g][key]["holds"]}
                            for g in utility["e5b"] if g != "all" and g in multiturn["results"]}}


def h8(attacker: dict, key: str = H8) -> dict:
    h = attacker["results"]["all"][key]
    return {**h, "holds": bool(h["holds"]),
            "per_dataset": {g: r[key]["holds"] for g, r in attacker["results"].items() if g != "all"}}


def h9(e5: dict, components: dict, datasets: Sequence[str]) -> dict:
    out = {}
    for g in [*datasets, "all"]:
        res, ref = components["results"][g]["injected"], e5["results"][g]["injected"]
        pd = _rate(ref["presidio_default"]["leak"])
        row = {"presidio_default": ref["presidio_default"]["leak"],
               "same_values": res["balanced"]["protect_values"] == ref["ss"]["protect_values"],
               "balanced_equals_ss": res["balanced"]["leak"] == ref["ss"]["leak"]}
        for name, key in (("a_no_patterns", NO_PATTERNS), ("b_no_tagger", NO_TAGGER)):
            r = (res.get(key) or {}).get("leak")
            row[name] = {"leak": r, "holds": _rate(r) is not None and pd is not None and _rate(r) < pd}
        r = (res.get(UNROUTED) or {}).get("leak")
        row["beside_unrouted"] = {"leak": r, "below": _rate(r) is not None and pd is not None and _rate(r) < pd}
        out[g] = row
    a, b = out["all"]["a_no_patterns"]["holds"], out["all"]["b_no_tagger"]["holds"]
    return {"per_group": out, "a": a, "b": b, "holds": bool(a and b and out["all"]["same_values"])}


def h10(external: dict, key: str = H10) -> dict:
    h = external["hypotheses"][key]
    return {**h, "holds": bool((h.get(GLINER) or {}).get("holds"))}


def decide(h: Dict[str, dict]) -> dict:
    """The pre-registered GO rule over the hypotheses' ``holds`` (None: not run)."""
    parts = {k: h[k]["holds"] for k in GO_PARTS}
    if any(v is None for v in parts.values()):
        return {"verdict": "incomplete", "go_parts": parts, "missing": sorted(k for k, v in parts.items() if v is None)}
    if all(parts.values()):
        return {"verdict": "GO", "go_parts": parts}
    per = h["H1''"]["per_dataset"]
    failed = [ds for ds, v in per.items() if not v["holds"]]
    rest = all(parts[k] for k in GO_PARTS if k != "H1''")
    if len(per) - len(failed) == 2 and len(failed) == 1 and per[failed[0]]["not_worse_than_gliner"] and rest:
        return {"verdict": "GO-with-caveat", "go_parts": parts, "caveat": f"H1'' fails on {failed[0]}, where SS is "
                f"not worse than GLiNER-PII (default and tuned)"}
    return {"verdict": "NO-GO", "go_parts": parts, "failed": sorted(k for k, v in parts.items() if not v)}


def marked(x, mark: str):
    """*x* with every hypothesis name H<n>'' renamed H<n><mark> (keys and strings)."""
    if isinstance(x, dict):
        return {marked(k, mark): marked(v, mark) for k, v in x.items()}
    if isinstance(x, list):
        return [marked(v, mark) for v in x]
    return re.sub(r"(H\d+)''(?!')", lambda m: m.group(1) + mark, x) if isinstance(x, str) else x


def unmarked(name: str) -> str:
    return re.sub(r"(H\d+)'+$", r"\1''", name)


def verdict(docs: Dict[str, Optional[dict]], datasets: Sequence[str] = DATASETS, split: str = "test2") -> dict:
    """The rule over *docs*, under the names of ``HYPOTHESES_TEST2.md``;
    renamed with *split*'s mark after ``decide``."""
    mark = SPLITS[split]["mark"]
    e5 = docs.get("e5")
    h: Dict[str, dict] = {}
    if e5 is None:
        for k in ("H1''", "H2''", "H3''", "H4''", "H5''"):
            h[k] = dict(NOT_RUN)
    else:
        h["H1''"], h["H2''"], h["H3''"] = h1(e5, datasets), h2(e5, datasets), h3(e5, datasets)
        h["H4''"], h["H5''"] = h4(e5, datasets), h5(e5)
    h["H6''"] = h6(docs["perf"]) if docs.get("perf") else dict(NOT_RUN)
    h["H7''"] = h7(docs.get("utility"), docs.get("multiturn"), f"H7{mark}")
    h["H8''"] = h8(docs["attacker"], f"H8{mark}") if docs.get("attacker") else dict(NOT_RUN)
    h["H9''"] = h9(e5, docs["components"], datasets) if e5 and docs.get("components") else dict(NOT_RUN)
    h["H10''"] = h10(docs["external"], f"H10{mark}") if docs.get("external") else dict(NOT_RUN)
    v = {**decide(h), "not_run": [k for k, x in h.items() if x.get("holds") is None], "hypotheses": h}
    return v if mark == "''" else marked(v, mark)


def run(out: Path, results: Path = RESULTS, freeze: Optional[Path] = None, split: str = "test2") -> dict:
    coll = COLLECTIONS[split]
    freeze = freeze or coll.freeze
    docs = load(results, freeze, split)
    v = verdict(docs, split=split)
    flag = "" if split == "test2" else f" --split {split}"
    doc = {"command": f"PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.verdict{flag} "
                      f"--out {score.rel(out)}",
           "git": git_state(), "prereg": {"file": score.rel(coll.hypotheses), "sha256": file_sha256(coll.hypotheses)},
           "split": split, "freeze_file": score.rel(freeze),
           "freeze_sha256": file_sha256(freeze) if freeze.exists() else None,
           "inputs": {k: (d["_input"] if d else None) for k, d in docs.items()}, **v}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


# ── markdown ─────────────────────────────────────────────────────────────────

def _yn(x: Optional[bool]) -> str:
    return "not run" if x is None else ("holds" if x else "fails")


def _k(r: Optional[dict]) -> str:
    return "–" if not r else f"{r['k']}/{r['n']}"


def _detail(name: str, x: dict) -> str:
    name = unmarked(name)
    if x.get("holds") is None:
        return x.get("status", "not run")
    if name in ("H1''", "H2''", "H3''"):
        return "; ".join(f"{ds}: {_yn(v['holds'])}" + ("" if v["holds"] else " (" + ", ".join(
            a for a, p in v["per_baseline"].items() if not p["holds"]) + ")") for ds, v in x["per_dataset"].items())
    if name == "H4''":
        return f"shift leak SS {_k(x['ss'])}, GLiNER-PII {_k(x['gliner_pii'])}; Δ {score._diff(x)}"
    if name == "H5''":
        return "failing types: " + (", ".join(x["failing_types"]) or "none") + \
               f" (tuned, beside: {', '.join(x['beside'][TUNED]['failing_types']) or 'none'})"
    if name == "H6''":
        return (f"p50 {x['ss']['p50_ms']} vs {x['gliner_pii']['p50_ms']} ms; peak RSS {x['ss']['peak_rss_mb']} vs "
                f"{x['gliner_pii']['peak_rss_mb']} MB" + ("" if x["default_pipeline"] else "; SS not the default"))
    if name == "H7''":
        return f"E5b {_yn(x['e5b']['holds'])}, E7b {_yn(x['e7b']['holds'])}"
    if name == "H8''":
        r = x["recovery"]
        return (f"no exact (single) {x['no_exact_single']}, (conversation) {x['no_exact_conversation']}; recovery "
                f"SS {_k(r['ss'])} vs GLiNER-PII {_k(r['gliner_pii'])}")
    if name == "H9''":
        a = x["per_group"]["all"]
        return (f"Presidio-default {_k(a['presidio_default'])}; no patterns {_k(a['a_no_patterns']['leak'])} "
                f"({_yn(x['a'])}); no tagger {_k(a['b_no_tagger']['leak'])} ({_yn(x['b'])}); beside: no patterns, "
                f"default routing {_k(a['beside_unrouted']['leak'])}")
    if name == "H10''":
        g = x.get(GLINER) or {}
        return f"leak not above {g.get('leak_not_above')}, span F1 not below {g.get('span_f1_not_below')}"
    return ""


def markdown(doc: dict) -> str:
    split = doc.get("split", "test2")
    freeze = "FREEZE.json" if split == "test2" else doc["freeze_file"]
    lines = [f"# {split.replace('test', 'Test-')} verdict: {doc['verdict']}", "",
             f"`{doc['command']}` at commit `{(doc['git'].get('commit') or '?')[:12]}`; pre-registration "
             f"`{doc['prereg']['file']}` ({doc['prereg']['sha256'][:12]}), {freeze} {str(doc['freeze_sha256'])[:12]}. "
             "Readings: `bench/realdata/verdict.py`.", ""]
    if doc.get("caveat"):
        lines += [f"Caveat: {doc['caveat']}.", ""]
    lines += ["| hypothesis | in the GO rule | result | detail |", "|---|---|---|---|"]
    for name, x in doc["hypotheses"].items():
        lines.append(f"| {name} | {'yes' if unmarked(name) in GO_PARTS else 'no'} | {_yn(x.get('holds'))} "
                     f"| {_detail(name, x)} |")
    lines += ["", "Inputs:", ""]
    for k, i in doc["inputs"].items():
        lines.append(f"- {k}: " + ("not run" if not i else f"`{i['file']}` ({i['sha256'][:12]}, commit "
                                   f"{str(i['commit'])[:12]})"))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=list(SPLITS), default="test2")
    a = ap.parse_args(argv)
    doc = run(a.out.resolve(), split=a.split)
    print(doc["verdict"])
    for name, x in doc["hypotheses"].items():
        print(f"{name:7} {_yn(x.get('holds'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
