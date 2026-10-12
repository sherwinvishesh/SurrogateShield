"""E1: one failure class (A-H) for every leaked natural-slice protect value.

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.natural_leaks \\
        --split dev --out bench/tagger/build/campaign/<date>/results/E1_dev.json
    ... --split devlarge [--half calib|val] --out ...

SurrogateShield's send path runs on every natural message of the split with
the cascade's trace on (as ``bench.realdata.attribute`` does for the injected
slice), and each protect value that the scorer's leak rule says leaked gets
one class:

``A`` never proposed: no stage produced an overlapping candidate and the raw
      tagger (every decoded span, no threshold) proposed nothing either;
``B`` low confidence: the raw tagger proposed an overlapping span of the
      value's type below the configured threshold;
``C`` lost in decoding: the tagger's token-level argmax covers the value's
      type on at least one token, but no decoded span of that type overlaps
      it (``_trim`` or the decoder lost it), or a decoded span covers only
      part of a value the token-level argmax covers whole;
``D`` gated: a candidate overlapping the value was removed by a later pass
      (the relation gate's rule, ``sentence_frame``, ``topical_geo``,
      ``inside_url``, ``implausible_org``, ...);
``E`` misclassified: the only candidates overlapping the value carried
      another type;
``F`` replacement failed: a final entity overlapped the value but the edits
      left part of it (partial cover, no surrogate, overlap cut), or a
      surrogate contained the value;
``G`` annotation ambiguous: not decided here (labels are never edited);
``H`` intentional policy: the scorer's ``policy`` excuse.

Counts and masked shapes only (letters -> a/A, digits -> 9): no text, no
value, no label ever reaches the output. The raw tagger's maximum token-level
probability for the value's type is reported as a bucket, and whether spaCy's
PERSON would have overlapped the value (an ensemble candidate, E9).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arms.base import msg_seed
from bench.realdata import attribute as A
from bench.realdata import score as S
from bench.realdata.common import COLLECTIONS, DATASETS, ROOT, git_state

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import realworld as rw  # noqa: E402

CLASSES = ("A", "B", "C", "D", "E", "F", "G", "H")
BUCKETS = ((0.05, "<.05"), (0.3, ".05-.3"), (0.5, ".3-.5"), (0.7, ".5-.7"), (1.01, ">=.7"))


def mask(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        elif ch in " \t":
            out.append("_")
        elif ch == "\n":
            out.append("|")
        else:
            out.append(ch)
    return "".join(out)


def case_pattern(v: str) -> str:
    pats = []
    for w in v.split():
        letters = [c for c in w if c.isalpha()]
        if not letters:
            pats.append("#")
        elif all(c.isupper() for c in letters):
            pats.append("U")
        elif letters[0].isupper():
            pats.append("T")
        else:
            pats.append("l")
    return "".join(pats)


def bucket(p: float) -> str:
    for hi, name in BUCKETS:
        if p < hi:
            return name
    return ">=.7"


class RawTagger:
    """The configured pii_tagger stage's detector, read without thresholds."""

    def __init__(self, cfg):
        from surrogateshield.core.detection import plugins
        from surrogateshield.core.detection import config as dconfig
        self.stage = cfg.stage("pii_tagger")
        self.det = plugins.get_detector(self.stage) if self.stage.enabled else None
        self.cfg = cfg
        self.public_type = dconfig.public_type
        self.plugins = plugins

    def spans(self, text: str) -> List[Tuple[int, int, str, float]]:
        if self.det is None:
            return []
        ents = self.plugins.candidates_to_entities("pii_tagger", text, self.det.detect(text, None))
        return [(e.start, e.end, self.public_type(e.type), e.score) for e in ents]

    def token_max(self, text: str, s: int, e: int, typ: str) -> Tuple[float, bool]:
        """(max P(B-typ)+P(I-typ) over the tokens overlapping text[s:e],
        whether any such token's argmax is of *typ*)."""
        if self.det is None:
            return 0.0, False
        toks = self.det.model.token_probs([text], self.det.window, self.det.stride, self.det.batch)[0]
        labels = list(self.det.model.labels)
        idx = {lab: i for i, lab in enumerate(labels)}
        tags = {lab.split("-", 1)[1] for lab in labels if "-" in lab}
        mine = [t for t in tags if self.public_type(t) == typ or t == typ]
        best, argmax_hit = 0.0, False
        for ts, te, p in toks:
            if not (ts < e and s < te):
                continue
            k = max(range(len(p)), key=p.__getitem__)
            lab = labels[k]
            if "-" in lab and lab.split("-", 1)[1] in mine:
                argmax_hit = True
            for t in mine:
                sc = p[idx.get(f"B-{t}", k)] + p[idx.get(f"I-{t}", k)] if f"B-{t}" in idx else 0.0
                best = max(best, sc)
        return best, argmax_hit


class SpacyPerson:
    def __init__(self, cfg):
        from surrogateshield.core.detection import entity_trace
        from surrogateshield.core.detection import config as dconfig
        st = cfg.stage("entity_trace")
        self.nlp = entity_trace._get_nlp(st.model or dconfig.SPACY_MODEL) if st.enabled else None

    def overlaps(self, text: str, s: int, e: int) -> bool:
        if self.nlp is None:
            return False
        doc = self.nlp(text)
        return any(x.label_ == "PERSON" and x.start_char < e and s < x.end_char for x in doc.ents)


def partial_detail(text: str, s: int, e: int, raw_spans, prepared, trace) -> dict:
    """How a partly covered value [s, e) was read, as offsets relative to
    its start (no text): the value's masked shape, the raw tagger's
    overlapping candidates, the final entities overlapping it, and every
    trace stage after which the set of overlapping candidates changed."""
    rel = lambda a, b: [a - s, b - s]
    out = {
        "value_shape": mask(text[s:e]),
        "raw_rel": sorted([*rel(a, b), t, bucket(sc)] for a, b, t, sc in raw_spans),
        "final_rel": sorted([*rel(x.start, x.end), x.type, x.source] for x in prepared.confirmed
                            if x.start < e and s < x.end),
        "edits_rel": sorted(rel(a, b) for a, b, _o, _r in prepared.edits if a < e and s < b),
    }
    stages, prev = [], None
    for name, ents in A.states(trace, prepared):
        here = sorted([*rel(x[0], x[1]), x[2], x[3], x[4]] for x in ents if x[0] < e and s < x[1])
        if here != prev:
            stages.append([name, here])
            prev = here
    out["stages_rel"] = stages
    return out


def classify(text: str, item: dict, service_query: bool, prepared, trace, raw: RawTagger,
             threshold: float) -> Tuple[str, str, dict]:
    """(class, detail, extras) for one protect value."""
    typ = item["type"]
    cause, detail = A.attribute_value(text, item, service_query, prepared, trace)
    extras: dict = {"cause": cause, "cause_detail": detail}
    if cause == "protected":
        return "protected", "", extras
    if cause == "policy":
        return "H", "policy", extras
    if cause == "refused":
        return "protected", "refused", extras
    how, occ = A.first_failure(text, item["value"], prepared.edits if prepared else [])
    s, e = occ if occ else next(iter(rw.occurrences(text, item["value"])), (0, 0))
    raw_spans = [x for x in raw.spans(text) if x[0] < e and s < x[1]]
    same = [x for x in raw_spans if x[2] == typ]
    tok_max, argmax_hit = raw.token_max(text, s, e, typ)
    extras["tagger_max_token"] = bucket(tok_max)
    extras["tagger_span_types"] = sorted({x[2] for x in raw_spans})
    extras["tagger_same_type_max_span"] = bucket(max((x[3] for x in same), default=0.0)) if same else "-"
    if cause == "surrogate":
        return "F", "surrogate_contains", extras
    if cause == "partial":
        # token argmax covers the whole value but the decoded span does not: decoding
        whole = argmax_hit and same and all(x[0] > s or x[1] < e for x in same)
        extras.update(partial_detail(text, s, e, raw_spans, prepared, trace))
        return ("C" if whole else "F"), f"partial:{detail}", extras
    if cause == "unplanned":
        return "F", f"unplanned:{detail}", extras
    if cause == "gated":
        st = A.states(trace, prepared)
        seen = [i for i, (_n, ents) in enumerate(st) if any(A.overlaps(x[0], x[1], s, e) for x in ents)]
        last = [x for x in st[seen[-1]][1] if A.overlaps(x[0], x[1], s, e)] if seen else []
        types = sorted({x[2] for x in last})
        extras["candidate_types"] = types
        if types and typ not in types and not any(raw.public_type(t) == typ for t in types):
            return "E", f"as:{','.join(types)} then {detail}", extras
        return "D", detail, extras
    # missed
    if same:
        top = max(x[3] for x in same)
        if top < threshold:
            return "B", f"span {bucket(top)}", extras
        return "D", "proposed_at_threshold_but_absent_from_trace", extras
    if raw_spans:
        return "E", f"tagger_as:{','.join(sorted({x[2] for x in raw_spans}))}", extras
    if argmax_hit:
        return "C", "token_argmax_without_span", extras
    return "A", f"token_max {bucket(tok_max)}", extras


def run(split: str, datasets: Sequence[str] = DATASETS, half: Optional[str] = None) -> dict:
    data_split, coll_name, _role = S.RUNS[split]
    coll = COLLECTIONS[coll_name]
    S.seal(coll)
    from bench.arms.ss import detection_config
    cfg = detection_config()
    threshold = cfg.min_score("pii_tagger", "PERSON")
    thresholds = {t: cfg.min_score("pii_tagger", t) for t in ("PERSON", "ORG", "LOCATION", "PHONE", "ID",
                                                               "ADDRESS", "HANDLE", "CREDENTIAL")}
    prepare = A.traced_prepare()
    raw = RawTagger(cfg)
    spacy_person = SpacyPerson(cfg)
    members = None
    if half:
        from bench.tagger import splits
        members = splits.membership(datasets)
    _hashes, loaded = S.load_split(data_split, datasets, coll.rd, coll.build, prefix=coll.prefix)
    counts: Counter = Counter()
    rows: List[dict] = []
    messages = Counter()
    for ds in datasets:
        for u in loaded[ds]["units"]:
            if "natural" not in u["slices"]:
                continue
            if members is not None and members.get(u["conv"]) != half:
                continue
            gold, text = u["gold"], u["gold"]["text"]
            messages[ds] += 1
            prepared, trace = prepare(text, msg_seed("ss", u["mid"]))
            edits = [] if prepared is None else list(prepared.edits)
            for item in gold.get("protect", []):
                typ = item["type"]
                cls, detail, extras = classify(text, item, gold.get("service_query", False), prepared, trace,
                                               raw, thresholds.get(typ, threshold))
                counts[(ds, typ, cls)] += 1
                if cls == "protected":
                    continue
                occ = list(rw.occurrences(text, item["value"]))
                s, e = occ[0]
                v = item["value"]
                rows.append({
                    "dataset": ds, "type": typ, "class": cls, "detail": detail,
                    "half": members.get(u["conv"]) if members else None,
                    "words": len(v.split()), "case": case_pattern(v),
                    "ascii": all(ord(c) < 128 for c in v), "occurrences": len(occ),
                    "position": round(s / max(1, len(text)), 2), "message_words": len(text.split()),
                    "message_edits": len(edits),
                    "any_person_edit": any(o in (x.text for x in prepared.confirmed if x.type == "PERSON")
                                           for _a, _b, o, _r in edits) if prepared else False,
                    "before": mask(text[max(0, s - 14):s]), "after": mask(text[e:e + 10]),
                    "line_shape": mask(text[text.rfind("\n", 0, s) + 1:s])[-24:],
                    "spacy_person": spacy_person.overlaps(text, s, e),
                    **extras,
                })
    by_class: Dict[str, Counter] = {}
    for (ds, typ, cls), n in counts.items():
        by_class.setdefault(typ, Counter())[cls] += n
    return {
        "split": split, "data_split": data_split, "collection": coll_name, "half": half,
        "datasets": list(datasets), "git": git_state(), "config_hash": cfg.config_hash(),
        "thresholds": thresholds, "classes": list(CLASSES),
        "natural_messages": dict(messages),
        "by_type_class": {t: dict(sorted(c.items())) for t, c in sorted(by_class.items())},
        "by_dataset_type_class": [{"dataset": ds, "type": t, "class": c, "n": n}
                                  for (ds, t, c), n in sorted(counts.items())],
        "leaked": rows,
    }


def markdown(doc: dict) -> str:
    lines = [f"# Natural-slice leak classes — `{doc['split']}`" + (f" ({doc['half']} half)" if doc["half"] else ""),
             "", f"Config `{doc['config_hash'][:16]}`, commit `{doc['git'].get('commit', '?')[:12]}`. "
             "Counts and masked shapes only.", "",
             "| type | " + " | ".join(("protected",) + CLASSES) + " |", "|---|" + "---|" * (len(CLASSES) + 1)]
    for t, c in doc["by_type_class"].items():
        lines.append(f"| {t} | " + " | ".join(str(c.get(k, 0)) for k in ("protected",) + CLASSES) + " |")
    lines += ["", "| ds | type | class | detail | half | words | case | occ | msg edits | PERSON edit | "
              "tagger tok max | tagger span types | spaCy PERSON | line shape | before | after |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["leaked"]:
        lines.append(f"| {r['dataset']} | {r['type']} | {r['class']} | {r['detail']} | {r['half'] or '-'} | "
                     f"{r['words']} | {r['case']} | {r['occurrences']} | {r['message_edits']} | "
                     f"{r['any_person_edit']} | {r.get('tagger_max_token', '-')} | "
                     f"{','.join(r.get('tagger_span_types', [])) or '-'} | {r['spacy_person']} | "
                     f"`{r['line_shape']}` | `{r['before']}` | `{r['after']}` |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=list(S.RUNS), required=True)
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--half", choices=("calib", "val"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-m", "bench.realdata.natural_leaks", *(argv or sys.argv[1:])],
                               env=env, cwd=ROOT)
    doc = run(args.split, args.datasets, args.half)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(doc))
    print(f"{args.split}{' ' + args.half if args.half else ''}: {doc['by_type_class']} -> {S.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
