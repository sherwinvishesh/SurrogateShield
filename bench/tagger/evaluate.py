"""Score PIITagger checkpoints on real development data (V3 §3.3 selection).

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.tagger.evaluate \\
        --model bench/tagger/build/models/dv3s-40k --split dev --out bench/tagger/build/eval/dv3s-40k-dev.json

The tagger is run alone, as a raw detector arm (every candidate at or above
the threshold is an edit, like ``gliner_pii``), over the split's messages, and
scored by ``score.score_unit``: injected leak by type, natural spurious edits,
negatives untouched. ``--split devlarge`` takes one half of
``bench/tagger/splits.py`` (``--half val`` for acceptance, ``calib`` to set
thresholds). Candidates and their scores are cached per model and split
(git-ignored, under ``bench/tagger/build/eval/``) so thresholds sweep without
rerunning the model. Latency is measured separately: one message at a time on
CPU, as the product runs it. Counts only: no text and no value is written to
the result.

``--ss`` also runs SurrogateShield with the tagger switched on (the
``pii_tagger`` stage, through ``$SURROGATESHIELD_DETECTION_CONFIG``, as
``bench/realdata/yardstick.py`` runs its configs) and scores it beside the
recorded ``ss``, ``gliner_pii`` and ``gliner_pii_tuned`` runs of the same
split (``bench/arms/run.PRIVATE``), on the same messages. ``--extra`` is a
partial config merged on the tagger's (other types it may emit, per-type
thresholds, gate options), named by ``--variant``; SS runs over the whole
split once per variant, and either half is scored from the same spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bench.arms import gliner_pii_tuned as tuned
from bench.arms.run import PRIVATE
from bench.realdata import gliner_sweep, score, yardstick
from bench.realdata.common import DATASETS, ROOT, file_sha256, git_state, read_jsonl, write_jsonl
from bench.tagger import splits
from surrogateshield.core.detection import pii_tagger as T

BUILD = ROOT / "bench" / "tagger" / "build" / "eval"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def model_sha(model: Path) -> str:
    """SHA-256 of the weights file (``model.safetensors``), the §6 Q3 pin."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (model / name).exists():
            return file_sha256(model / name)
    raise SystemExit(f"{model}: no weights file")


def load(split: str, datasets: Sequence[str]) -> dict:
    return score.load_split(score.RUNS[split][0], datasets)[1]


def units_for(loaded: dict, half: Optional[str]) -> Dict[str, List[dict]]:
    members = splits.membership(list(loaded)) if half else None
    return {ds: [u for u in v["units"] if not half or splits.unit_half(u, members) == half]
            for ds, v in loaded.items()}


def candidates(model: Path, tag: str, units_by_ds: Dict[str, List[dict]], device: str, batch: int,
               reuse: bool, log=print) -> Dict[str, Dict[str, list]]:
    """``{dataset: {mid: [[start, end, type, score], ...]}}``, cached."""
    out = {}
    tm = None
    for ds, units in units_by_ds.items():
        path = BUILD / model.name / f"{tag}-{ds}.jsonl"
        if reuse and path.exists():
            rows = {r["id"]: r["c"] for r in read_jsonl(path)}
            if set(rows) == {u["mid"] for u in units}:
                out[ds] = rows
                continue
        if tm is None:
            tm = T.get_model(str(model), None, device)
        texts = [u["gold"]["text"] for u in units]
        t0 = time.time()
        found = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            toks = tm.token_probs(chunk, T.DEFAULT_WINDOW, T.DEFAULT_STRIDE, batch)
            found += [T.decode(t, tk, tm.labels) for t, tk in zip(chunk, toks)]
        rows = {u["mid"]: [[c.start, c.end, c.type, c.score] for c in f] for u, f in zip(units, found)}
        write_jsonl(path, [{"id": k, "c": v} for k, v in rows.items()], private=True)
        log(f"{model.name} {tag} {ds}: {len(units)} messages in {time.time() - t0:.0f}s")
        out[ds] = rows
    return out


def edits_at(cands: Sequence[Sequence], threshold, types: Optional[set] = None) -> List[list]:
    out = []
    for s, e, t, sc in cands:
        th = threshold.get(t, threshold.get("*", 0.5)) if isinstance(threshold, dict) else threshold
        if sc >= th and (types is None or t in types):
            out.append([s, e, t, f"[{t}]"])
    return out


def evaluate(units_by_ds, cands, threshold) -> dict:
    per, pu, ps = {}, [], []
    for ds, units in units_by_ds.items():
        sc = [score.score_unit(u, {"edits": edits_at(cands[ds][u["mid"]], threshold)}) for u in units]
        per[ds] = gliner_sweep.summary(units, sc)
        pu += units
        ps += sc
    return {"all": gliner_sweep.summary(pu, ps), **per}


def score_rows(units_by_ds, rows_by_ds) -> dict:
    per, pu, ps = {}, [], []
    for ds, units in units_by_ds.items():
        sc = [score.score_unit(u, rows_by_ds[ds][u["mid"]]) for u in units]
        per[ds] = gliner_sweep.summary(units, sc)
        pu += units
        ps += sc
    return {"all": gliner_sweep.summary(pu, ps), **per}


def tuned_rows(split: str, ds: str, v: dict) -> dict:
    """``gliner_pii_tuned``: on dev from the frozen sweep (as the yardstick),
    elsewhere the arm's recorded run."""
    if split == "dev":
        sw = gliner_sweep.read_sweep(gliner_sweep.spans_path(tuned.LABEL_SET, ds), v["units"], v["input_sha"])
        return {mid: {"edits": gliner_sweep.edits_at(sp, tuned.THRESHOLD)} for mid, sp in sw.items()}
    return score.read_spans(PRIVATE / "gliner_pii_tuned" / f"{split}-{ds}.jsonl", v["units"], v["input_sha"])


def tagger_config(model: Path, device: str, thresholds: Optional[dict] = None,
                  extra: Optional[dict] = None) -> dict:
    """The partial config that switches the tagger on; *extra* is merged on
    it (its ``pii_tagger`` stage key by key, ``options`` too; any other key
    replaces)."""
    stage = {"name": "pii_tagger", "enabled": True, "model": str(model), "options": {"device": device}}
    if thresholds:
        stage["thresholds"] = thresholds
    cfg = {"detectors": [stage]}
    for k, v in (extra or {}).items():
        if k != "detectors":
            cfg[k] = v
            continue
        for st in v:
            if st.get("name") != "pii_tagger":
                cfg["detectors"].append(st)
                continue
            for sk, sv in st.items():
                stage[sk] = {**stage.get(sk, {}), **sv} if sk in ("options", "thresholds") else sv
    return cfg


CODE = ("python-library/surrogateshield", "bench/arms", "generation", "json_tester.py")


def code_stamp() -> str:
    """SHA-256 of the files an SS run executes (tracked, as on disk now): a
    spans file made under another stamp is not reused."""
    files = subprocess.run(["git", "ls-files", "--", *CODE], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()
    h = hashlib.sha256()
    for f in sorted(files):
        path = ROOT / f
        h.update(f.encode() + b"\0" + (path.read_bytes() if path.exists() else b"") + b"\0")
    return h.hexdigest()


def _fresh(path: Path, stamp: str) -> bool:
    mark = path.with_name(path.name + ".code")
    return path.exists() and mark.exists() and mark.read_text() == stamp


def _run_ss(config: Optional[Path], src: Path, path: Path, stamp: str) -> None:
    yardstick.run_ss(config, src, path)
    path.with_name(path.name + ".code").write_text(stamp)


def with_ss(model: Path, split: str, tag: str, loaded: dict, units_by_ds, device: str, reuse: bool,
            thresholds: Optional[dict] = None, log=print, variant: str = "base",
            extra: Optional[dict] = None) -> dict:
    """SS with the tagger on, beside the recorded arms of the split, on the same messages."""
    cfg = BUILD / model.name / f"ss-{split}-{variant}.config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(tagger_config(model, device, thresholds, extra), indent=1, sort_keys=True) + "\n"
    if not (cfg.exists() and cfg.read_text() == body):
        reuse_tagged = False                    # another config under this name: rerun
        cfg.write_text(body)
    else:
        reuse_tagged = reuse
    rows: Dict[str, dict] = {"ss_tagger": {}, "ss": {}, "gliner_pii": {}, "gliner_pii_tuned": {}}
    ms: List[float] = []
    stamp = code_stamp()
    for ds, v in loaded.items():
        path = BUILD / model.name / f"ss-{split}-{variant}-{ds}.jsonl"
        if not (reuse_tagged and _fresh(path, stamp)):
            t0 = time.time()
            _run_ss(cfg, v["src"], path, stamp)
            log(f"ss + {model.name} {variant} {split} {ds}: {len(v['units'])} messages in {time.time() - t0:.0f}s")
        rows["ss_tagger"][ds] = score.read_spans(path, v["units"], v["input_sha"])
        ms += [r["ms"] for u in units_by_ds[ds] if "ms" in (r := rows["ss_tagger"][ds][u["mid"]])]
        base = BUILD / "ss" / f"{split}-{ds}.jsonl"            # SS as configured, on this checkout
        if not (reuse and _fresh(base, stamp)):
            _run_ss(None, v["src"], base, stamp)
        rows["ss"][ds] = score.read_spans(base, v["units"], v["input_sha"])
        rows["gliner_pii"][ds] = score.read_spans(PRIVATE / "gliner_pii" / f"{split}-{ds}.jsonl", v["units"],
                                                  v["input_sha"])
        rows["gliner_pii_tuned"][ds] = tuned_rows(split, ds, v)
    out = {arm: brief(score_rows(units_by_ds, r)) for arm, r in rows.items()}
    if ms:
        out["ss_tagger"]["p50_ms"] = round(statistics.median(ms), 1)
    out["config"] = {"variant": variant, "file": score.rel(cfg), "sha256": file_sha256(cfg), "code": stamp,
                     "extra": extra or {}, "thresholds": thresholds or {}}
    return out


def latency(model: Path, units_by_ds, n: int = 300) -> dict:
    """One message at a time on CPU (the product's default), after a warm-up."""
    tm = T.get_model(str(model), None, "cpu")
    texts = [u["gold"]["text"] for units in units_by_ds.values() for u in units][:n]
    for t in texts[:5]:
        tm.token_probs([t])
    ms = []
    for t in texts:
        t0 = time.perf_counter()
        T.decode(t, tm.token_probs([t])[0], tm.labels)
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"messages": len(ms), "p50_ms": round(statistics.median(ms), 1), "p90_ms": round(ms[int(.9 * len(ms))], 1)}


def brief(r: dict) -> dict:
    inj, nat = r["all"]["injected"], r["all"]["natural"]
    return {"leak": inj["leak"], "macro": inj["macro_leak_rate"], "leaked_by_type": inj["leaked_by_type"],
            "values_by_type": {t: v["values"] - v["policy"] for t, v in inj["by_type"].items()},
            "natural_spurious": nat["spurious"]["k"], "natural_edits": nat["edits"],
            "negatives_untouched": nat["negatives_untouched"]}


def acceptance(arms: dict, arm: str = "ss_tagger", gliners=("gliner_pii", "gliner_pii_tuned"),
               per_type: float = 0.02, pooled: float = 0.06) -> dict:
    """V3 §3.3 on one split: every type's leak at most the better GLiNER's on
    the same rows or at most *per_type*; pooled leak at most *pooled*; natural
    spurious edits at most the current SS's. Latency and RSS are measured apart."""
    r = arms[arm]
    rate = lambda a, t: (arms[a]["leaked_by_type"].get(t, 0) / arms[a]["values_by_type"][t]
                         if arms[a]["values_by_type"].get(t) else 0.0)
    types = {}
    for t in sorted(r["values_by_type"]):
        bound = max(per_type, min(rate(g, t) for g in gliners))
        types[t] = {"rate": round(rate(arm, t), 4), "bound": round(bound, 4), "ok": rate(arm, t) <= bound}
    checks = {"types": all(v["ok"] for v in types.values()),
              "pooled": r["leak"]["rate"] <= pooled,
              "spurious": r["natural_spurious"] <= arms["ss"]["natural_spurious"]}
    return {"ok": all(checks.values()), "checks": checks, "by_type": types}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "devlarge"), default="dev")
    ap.add_argument("--half", choices=splits.HALVES)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--ss", action="store_true", help="also run SS with the tagger on, beside the recorded arms")
    ap.add_argument("--variant", default="base", help="name of the --extra config (caches SS runs by it)")
    ap.add_argument("--extra", type=Path, help="partial config JSON merged on the tagger's")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.split == "devlarge" and not a.half:
        raise SystemExit("--split devlarge needs --half (calib or val)")
    model = a.model.resolve()
    tag = a.split + (f"-{a.half}" if a.half else "")
    loaded = load(a.split, DATASETS)
    units_by_ds = units_for(loaded, a.half)
    cands = candidates(model, tag, units_by_ds, a.device, a.batch, a.reuse)
    sweep = {str(th): brief(evaluate(units_by_ds, cands, th)) for th in THRESHOLDS}
    for th, r in sweep.items():
        print(f"{model.name} {tag} @{th}: leak {r['leak']['k']}/{r['leak']['n']} = {r['leak']['rate']}  "
              f"natural spurious {r['natural_spurious']} of {r['natural_edits']}  {r['leaked_by_type']}")
    meta = json.loads((model / "train_meta.json").read_text()) if (model / "train_meta.json").exists() else {}
    doc = {"command": f"... -m bench.tagger.evaluate --model {score.rel(a.model.resolve())} --split {a.split}"
                      + (f" --half {a.half}" if a.half else "")
                      + (f" --ss --variant {a.variant}" if a.ss else "")
                      + (f" --extra {score.rel(a.extra.resolve())}" if a.extra else "")
                      + f" --out {score.rel(a.out.resolve())}",
           "git": git_state(), "model": {"path": score.rel(model), "weights_sha256": model_sha(model),
                                         "encoder": meta.get("encoder"), "revision": meta.get("revision"),
                                         "licence": meta.get("licence"), "data_sha256": meta.get("data_sha256")},
           "split": tag, "messages": {ds: len(u) for ds, u in units_by_ds.items()}, "sweep": sweep}
    if a.ss:
        extra = json.loads(a.extra.read_text()) if a.extra else None
        doc["with_ss"] = with_ss(model, a.split, tag, loaded, units_by_ds, a.device, a.reuse,
                                 variant=a.variant, extra=extra)
        for arm, r in doc["with_ss"].items():
            if arm != "config":
                print(f"{arm:18s} leak {r['leak']['k']}/{r['leak']['n']} = {r['leak']['rate']}  natural spurious "
                      f"{r['natural_spurious']} of {r['natural_edits']}  {r['leaked_by_type']}")
        doc["acceptance"] = acceptance(doc["with_ss"])
        print(f"acceptance {doc['acceptance']['ok']} {doc['acceptance']['checks']} failing types "
              f"{ {t: v for t, v in doc['acceptance']['by_type'].items() if not v['ok']} }")
    if not a.no_latency:
        doc["latency_cpu"] = latency(model, units_by_ds)
        print(f"latency (cpu, one message): {doc['latency_cpu']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
