"""Fine-tune a PIITagger encoder on ``bench.tagger.data`` messages (V3 §3.3).

    PYTHONPATH=.:python-library HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.tagger.train \\
        --encoder deberta-v3-small --data bench/tagger/build/data/train-40k.jsonl \\
        --val bench/tagger/build/data/val-2k.jsonl --out bench/tagger/build/models/dv3s-40k

A plain PyTorch loop (no accelerate): AdamW, linear warm-up and decay,
gradient clipping, length-bucketed batches, a fixed seed. Labels are BIO over
``pii_tagger.TAGS``, aligned from character spans through the tokenizer's
offsets, so any fast tokenizer works. After every epoch the model is decoded
on the validation messages with the product's own decoder
(``pii_tagger.decode``) and the epoch with the best span F1 is kept.
``train_meta.json`` records the encoder (id, revision, licence), the data
hashes, every argument, per-epoch metrics and the wall time. Weights stay
under the git-ignored ``bench/tagger/build/`` (§6 Q3: local only, SHA-256
pinned by ``evaluate.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bench.realdata.common import ROOT, read_jsonl
from surrogateshield.core.detection import pii_tagger as T

# encoder name -> (hub id, pinned revision, licence); all cached in Phase 0
ENCODERS = {
    "deberta-v3-xsmall": ("microsoft/deberta-v3-xsmall", "4b419818330868dff6a60ad3e6b1c730f8b8c0c6", "MIT"),
    "deberta-v3-small": ("microsoft/deberta-v3-small", "a36c739020e01763fe789b4b85e2df55d6180012", "MIT"),
    "modernbert-base": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8", "Apache-2.0"),
    "distilbert-base-cased": ("distilbert/distilbert-base-cased", "6ea81172465e8b0ad3fddeed32b986cdcdcffcf0",
                              "Apache-2.0"),
}
LABEL2ID = {lab: i for i, lab in enumerate(T.LABELS)}
IGNORE = -100


# ── labels ───────────────────────────────────────────────────────────────────

def char_owner(text: str, spans: Sequence[Sequence]) -> List[int]:
    owner = [-1] * len(text)
    for i, (s, e, _t) in enumerate(spans):
        for c in range(s, e):
            owner[c] = i
    return owner


def token_labels(text: str, spans: Sequence[Sequence], offsets: Sequence[Tuple[int, int]],
                 owner: List[int]) -> List[int]:
    """BIO ids for one window's tokens: a token belongs to the span of its
    first non-space character; it is B when it holds the span's first character."""
    out, seen = [], set()
    for s, e in offsets:
        if e <= s:
            out.append(IGNORE)
            continue
        s0 = s
        while s < e and text[s].isspace():
            s += 1
        if s == e:                        # a whitespace token ("\n") inside a value continues it
            own = owner[s0]
            out.append(LABEL2ID[f"I-{spans[own][2]}"] if own >= 0 and s0 > spans[own][0] else LABEL2ID["O"])
            continue
        own = next((owner[c] for c in range(s, e) if owner[c] >= 0), -1)
        if own < 0:
            out.append(LABEL2ID["O"])
            continue
        ss, _se, tag = spans[own]
        first = s <= ss < e and own not in seen     # byte-level BPE can split one character over two tokens
        seen.add(own)
        out.append(LABEL2ID[f"{'B' if first else 'I'}-{tag}"])
    return out


def encode(tokenizer, rows: Sequence[dict], max_len: int, stride: int) -> List[Tuple[List[int], List[int]]]:
    out = []
    for i in range(0, len(rows), 256):
        chunk = rows[i:i + 256]
        enc = tokenizer([r["text"] for r in chunk], truncation=True, max_length=max_len, stride=stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
        for j, k in enumerate(enc["overflow_to_sample_mapping"]):
            r = chunk[k]
            labs = token_labels(r["text"], r["spans"], enc["offset_mapping"][j], char_owner(r["text"], r["spans"]))
            out.append((enc["input_ids"][j], labs))
    return out


def batches(windows: Sequence[Tuple[List[int], List[int]]], size: int, rng: random.Random, pad_id: int,
            pad_to: int = 32):
    """Length-bucketed, shuffled batches of (input_ids, attention_mask, labels),
    padded to a multiple of *pad_to* so the device sees few distinct shapes
    (MPS keeps a buffer per shape and runs out of memory otherwise)."""
    import torch
    order = sorted(range(len(windows)), key=lambda i: (len(windows[i][0]), rng.random()))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(groups)
    for g in groups:
        n = max(len(windows[i][0]) for i in g)
        n = -(-n // pad_to) * pad_to
        ids = torch.full((len(g), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(g), n), dtype=torch.long)
        labs = torch.full((len(g), n), IGNORE, dtype=torch.long)
        for r, i in enumerate(g):
            a, b = windows[i]
            ids[r, :len(a)] = torch.tensor(a)
            mask[r, :len(a)] = 1
            labs[r, :len(b)] = torch.tensor(b)
        yield ids, mask, labs


# ── evaluation (the product's decoder) ───────────────────────────────────────

def gold_candidates(row: dict) -> List[T.Candidate]:
    found = [(s, e, t, 1.0, 1) for s, e, t in row["spans"]]
    return T._join_address(row["text"], found)


def span_metrics(rows: Sequence[dict], predicted: Sequence[Sequence[T.Candidate]]) -> dict:
    """Exact (start, end, type) P/R/F1 and, per type, the share of gold values
    fully covered by some prediction of any type (the leak view)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    cov, tot = Counter(), Counter()
    for row, pred in zip(rows, predicted):
        gold = {(c.start, c.end, c.type) for c in gold_candidates(row)}
        got = {(c.start, c.end, c.type) for c in pred}
        for g in gold:
            tot[g[2]] += 1
            if any(p[0] <= g[0] and g[1] <= p[1] for p in got):
                cov[g[2]] += 1
            (tp if g in got else fn)[g[2]] += 1
        for p in got - gold:
            fp[p[2]] += 1
    def prf(a, b, c):
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        return {"p": round(p, 4), "r": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
    a, b, c = sum(tp.values()), sum(fp.values()), sum(fn.values())
    types = sorted(set(tot) | set(fp))
    return {"micro": prf(a, b, c), "coverage": round(sum(cov.values()) / max(1, sum(tot.values())), 4),
            "by_type": {t: {**prf(tp[t], fp[t], fn[t]), "coverage": round(cov[t] / tot[t], 4) if tot[t] else None,
                            "gold": tot[t]} for t in types}}


def predict(model, tokenizer, device, rows: Sequence[dict], window: int, stride: int, batch: int = 32):
    tm = T.TaggerModel.wrap(model, tokenizer, device)
    out = []
    for i in range(0, len(rows), 256):
        chunk = [r["text"] for r in rows[i:i + 256]]
        toks = tm.token_probs(chunk, window, stride, batch)
        out += [T.decode(t, tk, tm.labels) for t, tk in zip(chunk, toks)]
    return out


# ── training ─────────────────────────────────────────────────────────────────

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(a) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    hub, rev, licence = ENCODERS[a.encoder]
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    path = T._local_model(hub, rev)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(
        path, num_labels=len(T.LABELS), id2label=dict(enumerate(T.LABELS)), label2id=LABEL2ID,
        dtype=torch.float32)         # some checkpoints are stored in fp16, where DeBERTa overflows
    model.to(device)
    rows = read_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    val = read_jsonl(a.val)[:a.val_limit] if a.val else []
    t0 = time.time()
    windows = encode(tok, rows, a.max_len, a.stride)
    print(f"[train] {a.encoder} on {device}: {len(rows)} messages -> {len(windows)} windows "
          f"({time.time() - t0:.0f}s to encode)", flush=True)
    steps_per_epoch = math.ceil(len(windows) / a.batch)
    total = steps_per_epoch * a.epochs
    warm = int(total * a.warmup)
    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": a.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(a.seed)
    history, best, step = [], -1.0, 0
    a.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for ids, mask, labs in batches(windows, a.batch, rng, tok.pad_token_id or 0, a.pad_to):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labs.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += out.loss.item()
            seen += 1
            step += 1
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            if step % a.log_every == 0:
                el = time.time() - start
                print(f"[train] epoch {epoch} step {step}/{total} loss {run / seen:.4f} "
                      f"{step / el:.2f} steps/s, eta {(total - step) / (step / el) / 60:.0f} min", flush=True)
                run, seen = 0.0, 0
        rec = {"epoch": epoch, "step": step, "minutes": round((time.time() - start) / 60, 1)}
        if val:
            model.eval()
            if device.type == "mps":
                torch.mps.empty_cache()
            m = span_metrics(val, predict(model, tok, device, val, a.max_len, a.stride))
            rec["val"] = m
            print(f"[train] epoch {epoch}: val micro {m['micro']} coverage {m['coverage']}", flush=True)
            score = m["micro"]["f1"]
        else:
            score = epoch
        history.append(rec)
        if score > best:
            best = score
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            rec["saved"] = True
    meta = {"encoder": a.encoder, "hub_id": hub, "revision": rev, "licence": licence,
            "data": str(a.data.relative_to(ROOT) if a.data.is_absolute() else a.data), "data_sha256": file_sha(a.data),
            "data_meta": json.loads(Path(str(a.data) + ".meta.json").read_text()).get("data_sha256")
            if Path(str(a.data) + ".meta.json").exists() else None,
            "val": str(a.val) if a.val else None, "val_sha256": file_sha(a.val) if a.val else None,
            "messages": len(rows), "windows": len(windows),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "device": str(device), "git": git_head(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "labels": list(T.LABELS),
            "history": history, "best_val_f1": best, "minutes": round((time.time() - start) / 60, 1)}
    (a.out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", choices=list(ENCODERS), required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--pad-to", type=int, default=32)
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    meta = train(a)
    print(json.dumps({k: meta[k] for k in ("encoder", "messages", "windows", "best_val_f1", "minutes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
