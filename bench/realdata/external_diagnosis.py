"""Why SS leaks more than GLiNER-PII on five Nemotron-PII types (H10''' held; diagnosis only).

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python \\
        -m bench.realdata.external_diagnosis --out bench/results/external_nemotron_pii_diagnosis.json

``external_nemotron_pii.json`` (H10''', run once at the test-3 freeze) has SS
leaking less than GLiNER-PII overall and on each stratum, but more on five
types: URL, AGE, ORG, HANDLE and ADDRESS. The result file has no per-type
intervals and says "not diagnosed, the detector is frozen". This module says
*how* each leaked value of those types got through, from what the run already
recorded; it changes no rule, no result and no frozen file.

It rebuilds the gold from the raw parquet (``bench.realdata.external.load``),
reads each arm's recorded spans (the private file with the replacement text
when it is present, else the committed copy with its ``copied`` flag), scores
them again with the scorer's own code path (``score.score_unit``) and stops if
any arm's per-type leaked count differs from the committed ``by_type``. Then,
per protect type and arm: how many leaked values were *untouched* (no edit of
the arm overlapped any occurrence), *repeat_untouched* (one occurrence was
replaced, another whole-word occurrence of the same value elsewhere in the
record was not), *partly covered* (an edit overlapped but a letter or digit
stayed outside every edit) or *copied* (covered, but the replacement contains
the value); which edit types of the arm overlapped them; the leak rate inside
shape classes of the gold value (scheme, field label, legal suffix, house
number first, ...); per stratum; the paired cluster bootstrap of SS − GLiNER
arm per type; each arm's edits per edit type with how many fall outside every
gold value (the spurious side GLiNER pays for its recall); and, for URL, what
the frozen URL finder itself says about each gold URL: found and personal
(surrogate), found and opaque (left as is by the rule of audit I1/I8), or not
found. Counts and flags only, no text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bench import realworld as rw
from bench.realdata import external
from bench.realdata import score as S
from bench.realdata.common import ROOT, derive_seed, file_sha256, git_state, read_jsonl

RESULT = ROOT / "bench" / "results" / f"external_{external.NAME}.json"
COMMITTED = ROOT / "bench" / "results" / "spans"
NAME = f"external-{external.NAME}"
MADE = "2026-10-09"
WORSE = ("URL", "AGE", "ORG", "HANDLE", "ADDRESS")
GLINER = ("gliner_pii", "gliner_pii_tuned")
REASONS = ("partly_covered", "copied", "untouched", "repeat_untouched")
LABEL_WORDS = {
    "URL": r"web ?site|url|link|homepage|home page|profile|page|site|portal|domain",
    "AGE": r"age|aged|years|dob",
    "ORG": r"company|employer|organi[sz]ation|business|firm|institution|school|university|agency|provider|merchant|"
           r"bank|insurer|vendor|client|customer|issuer|carrier|hospital|clinic|practice|from|at|with",
    "HANDLE": r"user ?name|username|handle|login|user|screen ?name|account|nick(?:name)?|id|alias",
    "ADDRESS": r"address|street|addr|residence|location|shipping|billing|mailing|home|office|premises",
}
LEGAL = re.compile(r"(?i)(?:^|\s)(inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|s\.a|sa|sas|sarl|"
                   r"bv|b\.v|nv|pty|llp|lp|group|holdings|partners|associates|bank|trust|foundation|institute|"
                   r"university|college|school|hospital|clinic|labs?|technologies|solutions|systems|industries|"
                   r"international|enterprises|services|consulting|logistics|insurance|capital|ventures|media|"
                   r"studios?|motors|airlines|pharma(?:ceuticals)?|health(?:care)?)\.?$")
UNIT = re.compile(r"(?i)(?:^|\s|,)(apt|apartment|suite|ste|unit|floor|fl|room|rm|building|bldg|flat|level|lvl)\b|#")
SCHEME = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://")
# the arm's edit types that answer for the five types (the rest are in the json)
EDIT_TYPES_OF_INTEREST = {
    "ss": ("url", "hostname", "age", "handle", "ORG", "address", "zip_us", "postcode_uk", "GPE", "PERSON"),
    "gliner_pii": ("url", "age", "username", "organization", "address", "location", "person"),
    "gliner_pii_tuned": ("url", "age", "username", "organization", "address", "location", "person"),
}


# ── spans ────────────────────────────────────────────────────────────────────

def load_spans(arm: str, units: Sequence[dict], input_sha: str) -> Tuple[Dict[str, dict], dict]:
    """``{mid: {"edits": [(s, e, type, replacement)], "refused"?}}`` from the
    private span file (replacement text) or the committed copy (``copied``
    flag: the replacement is taken as the original when set, else as empty)."""
    from bench.arms.run import PRIVATE
    for source, base in (("private", PRIVATE), ("committed", COMMITTED)):
        full = base / arm / f"{NAME}.jsonl"
        meta_path = Path(str(full) + ".meta.json")
        if not (full.exists() and meta_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("input_sha256") != input_sha:
            raise SystemExit(f"{arm}: {source} span file was made from another input")
        texts = {u["mid"]: u["gold"]["text"] for u in units}
        rows = {}
        for r in read_jsonl(full):
            edits = []
            for ed in r["edits"]:
                if len(ed) == 4:
                    s, e, t, rep = ed
                else:
                    s, e, t, _n, copied = ed
                    rep = texts[r["id"]][s:e] if copied else ""
                edits.append((s, e, t, rep))
            rows[r["id"]] = {"edits": edits, **({"refused": r["refused"]} if "refused" in r else {})}
        if set(rows) != set(texts):
            raise SystemExit(f"{arm}: {source} span ids differ from the input")
        return rows, {"file": S.rel(full), "source": source, "sha256": file_sha256(full),
                      "detection_config_hash": (meta.get("config") or {}).get("detection_config_hash")}
    raise SystemExit(f"{arm}: no span file for {NAME}")


def anchor(arm: str, units: Sequence[dict], rows: Dict[str, dict], committed: dict) -> dict:
    """Re-score every record with the scorer's own code path and compare the
    per-type leaked counts with the committed result."""
    leaked = Counter()
    for u in units:
        row = rows[u["mid"]]
        sc = S.score_unit(u, {"edits": [list(e) for e in row["edits"]],
                              **({"refused": row["refused"]} if "refused" in row else {})})
        leaked.update(sc["leaked"])
    want = {t: v["leaked"] for t, v in committed["by_type"].items()}
    got = {t: leaked.get(t, 0) for t in want}
    return {"reproduced": got == want, "committed_leak": committed["leak"]["k"],
            "recomputed_leak": sum(leaked.values()), "differs_on": sorted(t for t in want if got[t] != want[t])}


# ── one value ────────────────────────────────────────────────────────────────

def outcome(text: str, value: str, edits: Sequence[tuple]) -> Tuple[bool, str, Counter]:
    """(leaked, reason, overlapping edit types) of one gold value under one
    arm's edits, by the scorer's rule: an occurrence leaks when a letter or
    digit of it lies outside every edit, or when an overlapping replacement
    contains the value (4+ characters). The reason is the first of
    ``REASONS`` that applies to some occurrence; a value whose every leaking
    occurrence is untouched while another occurrence was replaced is
    ``repeat_untouched``."""
    per, types = [], Counter()
    for s, e in rw.occurrences(text, value):
        over = [ed for ed in edits if ed[0] < e and s < ed[1]]
        types.update(ed[2] for ed in over)
        if not over:
            per.append("untouched")
        elif not rw._covered(s, e, text, over):
            per.append("partly_covered")
        elif len(value) >= 4 and any(value in ed[3] for ed in over):
            per.append("copied")
        else:
            per.append("replaced")
    if "replaced" in per and "untouched" in per and not ({"partly_covered", "copied"} & set(per)):
        return True, "repeat_untouched", types
    reason = next((r for r in REASONS if r in per), "replaced")
    return reason != "replaced", reason, types


def _label_before(text: str, s: int, words: str) -> bool:
    return re.search(rf"(?is)\b(?:{words})\b[^\n]{{0,12}}?[:=\-–]\s*$", text[max(0, s - 40):s]) is not None


def _bucket(n: int, edges: Sequence[int]) -> str:
    for k in edges:
        if n <= k:
            return f"<={k}"
    return f">{edges[-1]}"


def _age_sep(text: str, s: int) -> str:
    """What stands between the nearest "age"/"aged" word before the value (at
    most 14 characters back) and the value: the cue the frozen patterns see."""
    m = re.search(r"(?i)\b(age|aged)\b(?P<gap>\W{0,12})$", text[max(0, s - 18):s])
    if not m:
        return "no age word"
    gap = m.group("gap")
    if "\n" in gap:
        return "newline"
    for ch, name in ((":", "colon"), ("=", "equals"), ("|", "pipe"), ("\t", "tab"), ("-", "dash"), ("–", "dash")):
        if ch in gap:
            return name
    return "space" if gap.strip() == "" else "other"


def shape(typ: str, text: str, value: str, stratum: str) -> Dict[str, str]:
    """Shape classes of one gold value (strings; counted, never written out)."""
    occ = rw.occurrences(text, value)
    s, e = occ[0] if occ else (0, len(value))
    f: Dict[str, object] = {"labelled_field": _label_before(text, s, LABEL_WORDS.get(typ, "")), "stratum": stratum,
                            "occurrences": _bucket(len(occ), (1,))}
    v = value
    if typ == "URL":
        rest = SCHEME.sub("", v)
        hostport = rest.split("/")[0].split("@")[-1]
        host = hostport.split(":")[0].lower()
        f.update(scheme=SCHEME.match(v) is not None, www=host.startswith("www."), port=":" in hostport,
                 ip_host=re.fullmatch(r"\d+(?:\.\d+){3}", host) is not None,
                 path="/" in rest.rstrip("/"), has_digit_in_host=any(c.isdigit() for c in host),
                 tld=("ip" if re.fullmatch(r"\d+(?:\.\d+){3}", host) else host.rsplit(".", 1)[-1] if "." in host else "-"),
                 host_labels=_bucket(host.count(".") + 1, (2, 3)))
    elif typ == "AGE":
        after = text[e:e + 14]
        gap = re.search(r"(?i)\b(?:age|aged)\b(?P<gap>\W{0,12})$", text[max(0, s - 18):s])
        f.update(digits_only=v.isdigit(), unit_inside=bool(re.search(r"(?i)year|yrs|y/o|yo|aged|old|month", v)),
                 age_word_sep=_age_sep(text, s),
                 emphasis_between=gap is not None and any(c in gap.group("gap") for c in "*_`"),
                 years_of_age_after=bool(re.match(r"(?i)\W{0,3}years?\s+of\s+age\b", after)),
                 years_after=bool(re.match(r"(?i)\W{0,3}(years?|yrs|yo|y/o|y\.o\.)\b", after)),
                 length=_bucket(len(v), (2, 3)))
    elif typ == "ORG":
        toks = v.split()
        f.update(words=_bucket(len(toks), (1, 2, 3)), legal_suffix=LEGAL.search(v) is not None,
                 ampersand="&" in v, has_digit=any(c.isdigit() for c in v),
                 all_caps_token=any(t.isupper() and len(t) >= 2 for t in toks),
                 possessive_or_apostrophe="'" in v or "’" in v)
    elif typ == "HANDLE":
        f.update(at_prefixed=s > 0 and text[s - 1] == "@", has_digit=any(c.isdigit() for c in v),
                 has_separator=any(c in v for c in "_.-"), all_lower=v == v.lower(), has_upper=any(c.isupper() for c in v),
                 length=_bucket(len(v), (6, 10)), has_space=" " in v)
    elif typ == "ADDRESS":
        f.update(has_digit=any(c.isdigit() for c in v), starts_with_digit=v[:1].isdigit(),
                 tokens=_bucket(len(v.split()), (3, 5)), unit=UNIT.search(v) is not None,
                 non_ascii=any(ord(c) > 127 for c in v), newline_inside="\n" in v, comma_inside="," in v)
    return {k: (str(x).lower() if isinstance(x, bool) else str(x)) for k, x in f.items()}


# ── the frozen URL finder's own reading ──────────────────────────────────────

def url_finder(units: Sequence[dict], ss_spans: Dict[str, dict]) -> Optional[dict]:
    """For each gold URL: what ``surrogateshield``'s frozen ``find_urls`` says
    (found and personal, found and opaque, not found) and whether SS leaked
    it. None when the library is not importable."""
    try:
        from surrogateshield.core.detection.pattern_scan import find_urls
    except ImportError:
        return None
    out = {k: {"n": 0, "leaked_by_ss": 0} for k in ("found_personal", "found_opaque", "not_found")}
    for u in units:
        text = u["gold"]["text"]
        urls = [x for x in u["gold"]["protect"] if x["type"] == "URL"]
        if not urls or "refused" in ss_spans[u["mid"]]:
            continue
        found = find_urls(text)
        for x in urls:
            occ = rw.occurrences(text, x["value"])
            if not occ:
                continue
            s, e = occ[0]
            hit = [f for f in found if f[0] < e and s < f[1]]
            kind = "not_found" if not hit else "found_personal" if any(f[2] for f in hit) else "found_opaque"
            out[kind]["n"] += 1
            out[kind]["leaked_by_ss"] += outcome(text, x["value"], ss_spans[u["mid"]]["edits"])[0]
    return {"rule": "the frozen rule (audit I1, I8): a URL gets a surrogate only when it points to a person (a "
                    "profile on a social or code host, a personal domain or TLD, a shared document, a page under "
                    "/team/ or /people/, or a site introduced as the user's own); every other URL is opaque, left "
                    "as it is, and no pattern or NER stage matches inside it", **out}


# ── the diagnosis ────────────────────────────────────────────────────────────

def diagnose(units: Sequence[dict], spans: Dict[str, Dict[str, dict]], arms: Sequence[str]) -> dict:
    by_type: Dict[str, dict] = {}
    shapes: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    strata: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    per_record: Dict[str, Dict[str, List[Tuple[int, int]]]] = {a: defaultdict(list) for a in arms}
    edit_types: Dict[str, Dict[str, Counter]] = {a: defaultdict(Counter) for a in arms}
    types = sorted({x["type"] for u in units for x in u["gold"]["protect"]})
    for t in types:
        by_type[t] = {"values": 0, "arms": {a: {"leaked": 0, "reasons": Counter(), "overlapping_edit_types": Counter(),
                                                "edit_types_on_caught": Counter()} for a in arms}}
    for u in units:
        g, text, mid = u["gold"], u["gold"]["text"], u["mid"]
        per_type_vals = Counter(x["type"] for x in g["protect"])
        leaked_here = {a: Counter() for a in arms}
        gold_spans = [sp for name in ("protect", "sensitive", "optional") for x in g[name]
                      for sp in rw.occurrences(text, x["value"])]
        for a in arms:
            row = spans[a][mid]
            if "refused" in row:
                continue
            for s, e, et, _rep in row["edits"]:
                edit_types[a][et]["edits"] += 1
                if not any(s < ge and gs < e for gs, ge in gold_spans):
                    edit_types[a][et]["spurious"] += 1
        for x in g["protect"]:
            t, v = x["type"], x["value"]
            by_type[t]["values"] += 1
            sh = shape(t, text, v, u["dataset"]) if t in WORSE else {"stratum": u["dataset"]}
            for a in arms:
                row = spans[a][mid]
                if "refused" in row:
                    leaked, reason, over = False, "refused", Counter()
                else:
                    leaked, reason, over = outcome(text, v, row["edits"])
                d = by_type[t]["arms"][a]
                if leaked:
                    leaked_here[a][t] += 1
                    d["leaked"] += 1
                    d["reasons"][reason] += 1
                    d["overlapping_edit_types"].update(over)
                else:
                    d["edit_types_on_caught"].update(over)
                for k, c in sh.items():
                    shapes[t][k][c]["n"] += 1 if a == arms[0] else 0
                    if leaked:
                        shapes[t][k][c][a] += 1
                strata[t][u["dataset"]]["n"] += 1 if a == arms[0] else 0
                if leaked:
                    strata[t][u["dataset"]][a] += 1
        for a in arms:
            for t, n in per_type_vals.items():
                per_record[a][t].append((leaked_here[a][t], n))
    clusters = [u["conv"] for u in units]
    for t in types:
        d = by_type[t]
        for a in arms:
            x = d["arms"][a]
            x["leak_rate"] = round(x["leaked"] / d["values"], 4) if d["values"] else None
            x["reasons"] = dict(sorted(x["reasons"].items()))
            for k in ("overlapping_edit_types", "edit_types_on_caught"):
                x[k] = dict(sorted(x[k].items(), key=lambda kv: (-kv[1], kv[0])))
        d["diff_ss_minus"] = {}
        for a in arms:
            if a == "ss":
                continue
            pa, pb = _records(per_record["ss"][t], units, t), _records(per_record[a][t], units, t)
            d["diff_ss_minus"][a] = S.bootstrap(clusters, pa, pb, derive_seed("external-diagnosis", external.NAME, a, t))
    out_shapes = {t: {k: {c: dict(sorted(v.items())) for c, v in sorted(classes.items())}
                      for k, classes in sorted(feats.items())} for t, feats in shapes.items() if t in WORSE}
    out_strata = {t: {s: dict(sorted(v.items())) for s, v in sorted(strata[t].items())} for t in types}
    out_edits = {a: {et: {"edits": c["edits"], "spurious": c["spurious"],
                          "spurious_rate": round(c["spurious"] / c["edits"], 4) if c["edits"] else None}
                     for et, c in sorted(edit_types[a].items(), key=lambda kv: (-kv[1]["edits"], kv[0]))} for a in arms}
    return {"types": types, "by_type": by_type, "shapes": out_shapes, "strata": out_strata, "edit_types": out_edits}


def _records(pairs: List[Tuple[int, int]], units: Sequence[dict], t: str) -> List[Tuple[int, int]]:
    """Per-record (leaked, values) of type *t*, zero for records without it,
    aligned with *units* for the cluster bootstrap."""
    it = iter(pairs)
    return [next(it) if any(x["type"] == t for x in u["gold"]["protect"]) else (0, 0) for u in units]


def run(out: Path) -> dict:
    if not RESULT.exists():
        raise SystemExit(f"{S.rel(RESULT)} is missing: the external benchmark has not been scored")
    result = json.loads(RESULT.read_text())
    arms = list(result["arms"])
    loaded = external.load()
    units, input_sha = loaded["units"], loaded["input_sha"]
    spans, files, anchors = {}, {}, {}
    for a in arms:
        spans[a], files[a] = load_spans(a, units, input_sha)
        anchors[a] = anchor(a, units, spans[a], result["results"]["all"][a])
        if not anchors[a]["reproduced"]:
            raise SystemExit(f"{a}: the recorded spans do not reproduce the committed per-type leaked counts "
                             f"(differs on {anchors[a]['differs_on']}); nothing written")
    diag = diagnose(units, spans, ["ss"] + list(GLINER))
    worse = [t for t in diag["types"]
             if diag["by_type"][t]["arms"]["ss"]["leaked"] > diag["by_type"][t]["arms"]["gliner_pii"]["leaked"]]
    doc = {"command": f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:python-library .venv/bin/python "
                      f"-m bench.realdata.external_diagnosis --out {S.rel(out)}",
           "git": git_state(), "made": MADE, "role": "diagnosis of a secondary check; changes no rule or result",
           "external": {"result_file": S.rel(RESULT), "sha256": file_sha256(RESULT), "run": result.get("git"),
                        "records": result["corpus"]["records"], "arms": arms},
           "spans": files, "anchor": anchors,
           "leak_rule": "An occurrence leaks when a letter or digit of it lies outside every edit, or when an "
                        "overlapping replacement contains the value (4+ characters). Reasons are per value: "
                        "partly_covered > copied > untouched (no edit overlapped any occurrence) > repeat_untouched "
                        "(one occurrence replaced, another whole-word occurrence of the same value left)",
           "bootstrap": {"resamples": S.RESAMPLES, "unit": "the source's record group (uid)",
                         "seed": "derive_seed('external-diagnosis', name, arm, type)", "difference": "ss − arm",
                         "ci": "percentile 2.5 / 97.5"},
           "worse_than_gliner_pii": worse, "url_finder": url_finder(units, spans["ss"]), **diag}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    return doc


# ── markdown ─────────────────────────────────────────────────────────────────

def _d(b: Optional[dict]) -> str:
    if not b or b.get("diff") is None:
        return "–"
    lo, hi = b["ci95"]
    return f"{100 * b['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]" + (" *" if b.get("excludes_0") else "")


def _cls(doc: dict, t: str, feat: str, val: str, arm: str = "ss") -> Tuple[int, int]:
    c = doc["shapes"].get(t, {}).get(feat, {}).get(val, {})
    return c.get(arm, 0), c.get("n", 0)


def _et(doc: dict, arm: str, et: str) -> dict:
    return doc["edit_types"].get(arm, {}).get(et, {"edits": 0, "spurious": 0})


def findings(doc: dict) -> List[str]:
    """What the counts say, one paragraph per type, written from the numbers."""
    bt, out = doc["by_type"], []
    if "URL" in bt:
        x, uf = bt["URL"]["arms"]["ss"], doc.get("url_finder")
        s = (f"**URL** ({x['leaked']} of {bt['URL']['values']} leaked, {x['reasons'].get('untouched', 0)} untouched): ")
        if uf:
            s += (f"the frozen URL finder saw {uf['found_personal']['n'] + uf['found_opaque']['n']} of the "
                  f"{bt['URL']['values']} gold URLs and called {uf['found_personal']['n']} of them personal (a surrogate; "
                  f"{uf['found_personal']['leaked_by_ss']} of those leaked). The other {uf['found_opaque']['n']} it "
                  f"found and left as they are: {uf['found_opaque']['leaked_by_ss']} of them leaked, which with the "
                  f"{uf['not_found']['n']} it never found ({uf['not_found']['leaked_by_ss']} leaked) is the whole SS "
                  f"count. ")
        pl, pn = _cls(doc, "URL", "path", "true"); bl, bn = _cls(doc, "URL", "path", "false")
        gpl, _ = _cls(doc, "URL", "path", "true", "gliner_pii"); gbl, _ = _cls(doc, "URL", "path", "false", "gliner_pii")
        s += (f"Nemotron-PII labels every URL as PII; SS's rule protects URLs that point to a person and treats the rest "
              f"as opaque (audit I1, I8), so this gap is a difference of taxonomy, not a pattern that fails. GLiNER-PII "
              f"tags {_et(doc, 'gliner_pii', 'url')['edits']} url spans for the {bt['URL']['values']} gold URLs and "
              f"still leaks {bt['URL']['arms']['gliner_pii']['leaked']}: {gpl} of {pn} URLs with a path "
              f"({bt['URL']['arms']['gliner_pii']['reasons'].get('partly_covered', 0)} of them partly covered, the path "
              f"cut off) against {gbl} of {bn} bare hosts; SS {pl}/{pn} and {bl}/{bn}.")
        out.append(s)
    if "AGE" in bt:
        x = bt["AGE"]["arms"]["ss"]
        rep = x["reasons"].get("repeat_untouched", 0)
        ya, yn = _cls(doc, "AGE", "years_after", "true"); na, nn = _cls(doc, "AGE", "years_after", "false")
        seps = doc["shapes"]["AGE"].get("age_word_sep", {})
        sep_txt = ", ".join(f"{k} {v.get('ss', 0)}/{v.get('n', 0)}" for k, v in seps.items())
        em, emn = _cls(doc, "AGE", "emphasis_between", "true"); yo, yon = _cls(doc, "AGE", "years_of_age_after", "true")
        out.append(f"**AGE** ({x['leaked']} of {bt['AGE']['values']} leaked): every gold age is a bare number of one or "
                   f"two digits. {rep} of the {x['leaked']} were replaced where they stand as an age and leak only "
                   f"because the same number recurs elsewhere in the record as a whole word (the scorer counts every "
                   f"occurrence). Of the remaining {x['leaked'] - rep}, {em} stand behind a Markdown-emphasised field "
                   f"label (`**Age:** 63`: {em} of the {emn} ages written that way leaked; the frozen 'age:' cue pattern "
                   f"allows spaces and a colon between the word and the number, not the closing `**`), {yo} of {yon} are "
                   f"'N years of age' (the unit patterns know 'years old', 'yo', 'y/o'), and the rest have none of the "
                   f"cues the frozen patterns need (a unit after the number: {ya}/{yn} leaked with one, {na}/{nn} "
                   f"without; 'aged'/'age:'/'turned'; a JSON or YAML key; an Age column of a pasted table; 'I'm N'; a "
                   f"name then 'is N'; '(29F)'). By the separator after the nearest 'age' word (SS leaked/n): "
                   f"{sep_txt}. GLiNER's zero-shot age label takes the field without a cue "
                   f"({bt['AGE']['arms']['gliner_pii']['leaked']} leaked, a repeat).")
    if "HANDLE" in bt:
        x = bt["HANDLE"]["arms"]["ss"]
        ll, ln = _cls(doc, "HANDLE", "labelled_field", "true"); ul, un = _cls(doc, "HANDLE", "labelled_field", "false")
        lo, lon = _cls(doc, "HANDLE", "all_lower", "true"); up, upn = _cls(doc, "HANDLE", "all_lower", "false")
        _at, no_at = _cls(doc, "HANDLE", "at_prefixed", "false")
        out.append(f"**HANDLE** ({x['leaked']} of {bt['HANDLE']['values']} leaked, {x['reasons'].get('untouched', 0)} "
                   f"untouched, {x['reasons'].get('partly_covered', 0)} partly covered): {no_at} of the "
                   f"{bt['HANDLE']['values']} gold handles have no '@' before them, so SS's '@handle' pattern never "
                   f"applies; its keyword pattern takes a bare word as a handle only behind an explicit label or "
                   f"platform cue, and a plain lowercase word only behind a 'username:'-style label. Leaked: {ll}/{ln} "
                   f"behind such a label, {ul}/{un} without one; {lo}/{lon} all-lowercase, {up}/{upn} with a capital. "
                   f"The spaCy stage read part of {x['overlapping_edit_types'].get('PERSON', 0)} of them as a PERSON and "
                   f"left the rest ({x['reasons'].get('partly_covered', 0)} partly covered). GLiNER-PII's zero-shot "
                   f"username label needs no cue ({bt['HANDLE']['arms']['gliner_pii']['leaked']} leaked; "
                   f"{_et(doc, 'gliner_pii', 'username')['edits']} username spans, "
                   f"{_et(doc, 'gliner_pii', 'username')['spurious']} outside every gold value).")
    if "ORG" in bt:
        x = bt["ORG"]["arms"]["ss"]
        w1, w1n = _cls(doc, "ORG", "words", "<=1"); w2, w2n = _cls(doc, "ORG", "words", "<=2")
        ls, lsn = _cls(doc, "ORG", "legal_suffix", "true"); nl, nln = _cls(doc, "ORG", "legal_suffix", "false")
        g_org, s_org = _et(doc, "gliner_pii", "organization"), _et(doc, "ss", "ORG")
        out.append(f"**ORG** ({x['leaked']} of {bt['ORG']['values']} leaked, {x['reasons'].get('untouched', 0)} untouched, "
                   f"{x['reasons'].get('partly_covered', 0)} partly covered, {x['reasons'].get('repeat_untouched', 0)} "
                   f"repeats): SS's ORG is the spaCy NER stage, which wants context; a synthetic one-word company name "
                   f"leaks {w1}/{w1n}, two words {w2}/{w2n}; a name without a legal suffix {nl}/{nln}, with one "
                   f"{ls}/{lsn}. SS made {s_org['edits']} ORG edits on the corpus ({s_org['spurious']} outside every "
                   f"gold value); GLiNER-PII made {g_org['edits']} organization edits for the same "
                   f"{bt['ORG']['values']} gold names, {g_org['spurious']} of them outside every gold value "
                   f"({100 * g_org['spurious'] / g_org['edits']:.0f} %): its recall on ORG is bought with spurious edits.")
    if "ADDRESS" in bt:
        x = bt["ADDRESS"]["arms"]["ss"]
        nd, ndn = _cls(doc, "ADDRESS", "starts_with_digit", "false"); d, dn = _cls(doc, "ADDRESS", "starts_with_digit", "true")
        intl = sum(v.get("ss", 0) for s, v in doc["strata"]["ADDRESS"].items() if s.startswith("intl"))
        intl_n = sum(v.get("n", 0) for s, v in doc["strata"]["ADDRESS"].items() if s.startswith("intl"))
        us = x["leaked"] - intl
        out.append(f"**ADDRESS** ({x['leaked']} of {bt['ADDRESS']['values']} leaked, {x['reasons'].get('untouched', 0)} "
                   f"untouched): the frozen street-address rule is house-number-first (number, street words, a street "
                   f"suffix, then unit, city, state, postcode). Leaked: {nd}/{ndn} addresses that do not start with a "
                   f"number against {d}/{dn} that do; {intl}/{intl_n} in the two international strata against {us} in "
                   f"the US ones. GLiNER-PII leaked {bt['ADDRESS']['arms']['gliner_pii']['leaked']} (bootstrap "
                   f"{_d(bt['ADDRESS']['diff_ss_minus'].get('gliner_pii'))}: the only one of the five whose interval "
                   f"covers zero).")
    return out


def markdown(doc: dict) -> str:
    arms = ["ss"] + list(GLINER)
    bt = doc["by_type"]
    lines = [f"# Nemotron-PII: why SS leaks more than GLiNER-PII on {len(doc['worse_than_gliner_pii'])} types "
             f"(diagnosis of `{doc['external']['result_file'].split('/')[-1]}`, made {doc['made']})", "",
             f"`{doc['command']}` at `{(doc.get('git') or {}).get('commit', '')[:12]}`. Gold rebuilt from the raw "
             f"parquet; each arm's recorded spans re-scored with the scorer's code path and anchored on the committed "
             f"per-type leaked counts (all {len(doc['external']['arms'])} arms reproduce; span files: "
             f"{', '.join(sorted({v['source'] for v in doc['spans'].values()}))}). Nothing re-run, detection as "
             f"frozen, no rule or result changed. {doc['leak_rule']}.", "",
             "## What the counts say", ""]
    for p in findings(doc):
        lines += [p, ""]
    lines += ["## Every protect type: leaked / values and the paired cluster bootstrap of SS − GLiNER arm", "",
              "| type | values | SS | GLiNER-PII | tuned | Δ vs GLiNER-PII | Δ vs tuned |", "|---|---|---|---|---|---|---|"]
    for t in doc["types"]:
        d = bt[t]
        mark = " **(SS worse)**" if t in doc["worse_than_gliner_pii"] else ""
        lines.append(f"| {t}{mark} | {d['values']} | " + " | ".join(str(d["arms"][a]["leaked"]) for a in arms)
                     + f" | {_d(d['diff_ss_minus'].get('gliner_pii'))} | {_d(d['diff_ss_minus'].get('gliner_pii_tuned'))} |")
    lines += ["", "## How the leaked values got through", "",
              "| type | arm | leaked | untouched | repeat untouched | partly covered | copied | the arm's edit types "
              "overlapping them | edit types on the values it caught |", "|---|---|---|---|---|---|---|---|---|"]
    for t in doc["worse_than_gliner_pii"]:
        for a in arms:
            x = bt[t]["arms"][a]
            r = x["reasons"]
            over = ", ".join(f"{k} {v}" for k, v in list(x["overlapping_edit_types"].items())[:6]) or "–"
            caught = ", ".join(f"{k} {v}" for k, v in list(x["edit_types_on_caught"].items())[:5]) or "–"
            lines.append(f"| {t} | {a} | {x['leaked']} | {r.get('untouched', 0)} | {r.get('repeat_untouched', 0)} "
                         f"| {r.get('partly_covered', 0)} | {r.get('copied', 0)} | {over} | {caught} |")
    if doc.get("url_finder"):
        uf = doc["url_finder"]
        lines += ["", "## The frozen URL finder on the gold URLs", "", "| the finder says | n | leaked by SS |", "|---|---|---|"]
        for k in ("found_personal", "found_opaque", "not_found"):
            lines.append(f"| {k.replace('_', ' ')} | {uf[k]['n']} | {uf[k]['leaked_by_ss']} |")
        lines += ["", uf["rule"] + "."]
    lines += ["", "## Each arm's edits by edit type (those answering for the five types; all in the json)", "",
              "| arm | edit type | edits | outside every gold value |", "|---|---|---|---|"]
    for a in arms:
        for et in EDIT_TYPES_OF_INTEREST.get(a, ()):
            c = doc["edit_types"].get(a, {}).get(et)
            if c:
                lines.append(f"| {a} | {et} | {c['edits']} | {c['spurious']} ({100 * c['spurious_rate']:.0f} %) |")
    lines += ["", "## Leak rate inside shape classes of the gold value (n; leaked by SS / GLiNER-PII / tuned)", ""]
    for t in doc["worse_than_gliner_pii"]:
        lines += [f"### {t}", "", "| class | value | n | SS | GLiNER-PII | tuned |", "|---|---|---|---|---|---|"]
        for k, classes in doc["shapes"].get(t, {}).items():
            for c, v in classes.items():
                n = v.get("n", 0)
                cells = [f"{v.get(a, 0)} ({100 * v.get(a, 0) / n:.0f} %)" if n else "–" for a in arms]
                lines.append(f"| {k} | {c} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Per stratum (n; leaked by SS / GLiNER-PII / tuned)", "", "| type | stratum | n | SS | GLiNER-PII | tuned |",
              "|---|---|---|---|---|---|"]
    for t in doc["worse_than_gliner_pii"]:
        for s, v in doc["strata"][t].items():
            lines.append(f"| {t} | {s} | {v.get('n', 0)} | " + " | ".join(str(v.get(a, 0)) for a in arms) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / f"external_{external.NAME}_diagnosis.json")
    args = ap.parse_args(argv)
    doc = run(args.out)
    bt = doc["by_type"]
    for t in doc["worse_than_gliner_pii"]:
        x = bt[t]["arms"]["ss"]
        print(f"{t:8} SS {x['leaked']}/{bt[t]['values']} vs GLiNER-PII {bt[t]['arms']['gliner_pii']['leaked']}: "
              f"{x['reasons']}; Δ {_d(bt[t]['diff_ss_minus']['gliner_pii'])}")
    if doc.get("url_finder"):
        print("URL finder:", {k: v for k, v in doc["url_finder"].items() if k != "rule"})
    print(f"-> {S.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
