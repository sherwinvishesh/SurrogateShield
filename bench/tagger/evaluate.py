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
split (``bench/arms/run.PRIVATE``), on the same messages.
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def tagger_config(model: Path, device: str, thresholds: Optional[dict] = None) -> dict:
    stage = {"name": "pii_tagger", "enabled": True, "model": str(model), "options": {"device": device}}
    if thresholds:
        stage["thresholds"] = thresholds
    return {"detectors": [stage]}


def with_ss(model: Path, split: str, tag: str, loaded: dict, units_by_ds, device: str, reuse: bool,
            thresholds: Optional[dict] = None, log=print) -> dict:
    """SS with the tagger on, beside the recorded arms of the split, on the same messages."""
    cfg = BUILD / model.name / f"ss-{tag}.config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(tagger_config(model, device, thresholds), indent=1) + "\n")
    rows: Dict[str, dict] = {"ss_tagger": {}, "ss": {}, "gliner_pii": {}, "gliner_pii_tuned": {}}
    ms: List[float] = []
    for ds, v in loaded.items():
        path = BUILD / model.name / f"ss-{tag}-{ds}.jsonl"
        if not (reuse and path.exists()):
            t0 = time.time()
            yardstick.run_ss(cfg, v["src"], path)
            log(f"ss + {model.name} {tag} {ds}: {len(v['units'])} messages in {time.time() - t0:.0f}s")
        rows["ss_tagger"][ds] = score.read_spans(path, v["units"], v["input_sha"])
        ms += [r["ms"] for r in rows["ss_tagger"][ds].values() if "ms" in r]
        base = BUILD / "ss" / f"{split}-{ds}.jsonl"            # SS as configured, on this checkout
        if not (reuse and base.exists()):
            yardstick.run_ss(None, v["src"], base)
        rows["ss"][ds] = score.read_spans(base, v["units"], v["input_sha"])
        rows["gliner_pii"][ds] = score.read_spans(PRIVATE / "gliner_pii" / f"{split}-{ds}.jsonl", v["units"],
                                                  v["input_sha"])
        rows["gliner_pii_tuned"][ds] = tuned_rows(split, ds, v)
    out = {arm: brief(score_rows(units_by_ds, r)) for arm, r in rows.items()}
    if ms:
        out["ss_tagger"]["p50_ms"] = round(statistics.median(ms), 1)
    out["config"] = {"file": score.rel(cfg), "sha256": file_sha256(cfg)}
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
            "natural_spurious": nat["spurious"]["k"], "natural_edits": nat["edits"],
            "negatives_untouched": nat["negatives_untouched"]}


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
                      + (f" --half {a.half}" if a.half else "") + f" --out {score.rel(a.out.resolve())}",
           "git": git_state(), "model": {"path": score.rel(model), "weights_sha256": model_sha(model),
                                         "encoder": meta.get("encoder"), "revision": meta.get("revision"),
                                         "licence": meta.get("licence"), "data_sha256": meta.get("data_sha256")},
           "split": tag, "messages": {ds: len(u) for ds, u in units_by_ds.items()}, "sweep": sweep}
    if a.ss:
        doc["with_ss"] = with_ss(model, a.split, tag, loaded, units_by_ds, a.device, a.reuse)
        for arm, r in doc["with_ss"].items():
            if arm != "config":
                print(f"{arm:18s} leak {r['leak']['k']}/{r['leak']['n']} = {r['leak']['rate']}  natural spurious "
                      f"{r['natural_spurious']} of {r['natural_edits']}  {r['leaked_by_type']}")
    if not a.no_latency:
        doc["latency_cpu"] = latency(model, units_by_ds)
        print(f"latency (cpu, one message): {doc['latency_cpu']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
