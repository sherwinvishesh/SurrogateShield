"""
detection/pii_tagger.py — PIITagger

A small token-classification encoder that marks every value of the protect
types (PERSON, ORG, ADDRESS, LOCATION, AGE, DATE_OF_BIRTH, EMAIL, PHONE,
HANDLE, URL, NETWORK, ID, CREDENTIAL). It is fine-tuned by this project
(``bench/tagger/``) from a permissively licensed encoder on generated text
only, and it is configured like any stage::

    Stage("pii_tagger", model="path/or/hub-id", revision="<commit>",
          window=256, stride=64, thresholds={"PERSON": 0.6},
          options={"device": "cpu", "batch": 8})

*model* is a hub id (pinned by commit, as the other models), a local folder,
or the name of a folder under ``$SURROGATESHIELD_MODELS`` (default
``~/.cache/surrogateshield/models``). A local model's *revision* is
``sha256:<hex>`` of its weights file, checked at load, so a config names the
exact weights without naming a machine's paths.

The model's labels are BIO tags over the protect types, with ADDRESS split
into STREET / CITY / REGION / POSTCODE. Adjacent address parts are joined
back into one ADDRESS; a lone city is a LOCATION; a lone region is dropped.
A text longer than *window* tokens is read in overlapping windows and every
token keeps the prediction of the window it sits most centrally in. A
candidate's score is the mean, over its tokens, of the probability of its
type (B + I), so ``thresholds`` trade recall for precision per type.

Whether a value belongs to the user, and what to do with it, is decided
downstream (RelationGate, the resolver): the tagger only finds values.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import DetectorUnavailable
from .context_guard import _local_model
from .plugins import Candidate

logger = logging.getLogger(__name__)

TAGS = ("PERSON", "ORG", "STREET", "CITY", "REGION", "POSTCODE", "LOCATION", "AGE", "DATE_OF_BIRTH", "EMAIL",
        "PHONE", "HANDLE", "URL", "NETWORK", "ID", "CREDENTIAL")
LABELS = ("O",) + tuple(f"{p}-{t}" for t in TAGS for p in ("B", "I"))
ADDRESS_PARTS = frozenset({"STREET", "CITY", "REGION", "POSTCODE"})
DEFAULT_WINDOW = 256
DEFAULT_STRIDE = 64
# text between two address parts that still joins them ("," " - " a line break)
_ADDRESS_GAP = re.compile(r"[\s,\-–—/|·]{0,6}")
_OPENERS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSERS = {v: k for k, v in _OPENERS.items()}
_QUOTES = "\"'`“”‘’"
_WRAP = _QUOTES + "*_"
# a value that ends in an abbreviation keeps its period ("S.A.R.L.", "Ltd.", "Jr.")
_ABBREV = re.compile(r"(?i)(?:(?<![^\W\d_])[^\W\d_]\.\s?){2,}$|\b(?:inc|ltd|co|corp|jr|sr|bros|pty|ltda|cía|cia|"
                     r"st|plc|gmbh|llc|lda|ag|nv|bv|sa|sl|srl|spa|kk|com)\.$")

_models: Dict[tuple, object] = {}
_lock = threading.Lock()

# Weights that are not on a hub sit under one directory, one folder per model,
# and a stage names its folder; the revision pins the weights file by hash.
MODELS_ENV = "SURROGATESHIELD_MODELS"
DEFAULT_MODELS_DIR = os.path.join("~", ".cache", "surrogateshield", "models")
WEIGHTS_FILES = ("model.safetensors", "pytorch_model.bin")
_PIN = re.compile(r"sha256:([0-9a-f]{64})")


def models_dir() -> str:
    return os.path.expanduser(os.environ.get(MODELS_ENV) or DEFAULT_MODELS_DIR)


def local_dir(model: str) -> Optional[str]:
    """*model* as a local folder: a path to one, or a folder of that name
    under ``$SURROGATESHIELD_MODELS``; None for a hub id."""
    if os.path.isdir(model):
        return model
    if model and os.sep not in model and "/" not in model:
        path = os.path.join(models_dir(), model)
        if os.path.isdir(path):
            return path
    return None


def weights_sha256(path: str) -> str:
    for name in WEIGHTS_FILES:
        f = os.path.join(path, name)
        if os.path.isfile(f):
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            return h.hexdigest()
    raise DetectorUnavailable(f"PIITagger: {path!r} has no weights file ({', '.join(WEIGHTS_FILES)})")


def check_pin(path: str, revision: Optional[str]) -> None:
    """A local model's revision, when given, is ``sha256:<hex>`` of its
    weights file, and the file must match it."""
    if not revision:
        return
    m = _PIN.fullmatch(revision)
    if not m:
        raise DetectorUnavailable(f"PIITagger: a local model's revision is 'sha256:<64 hex>' of its weights "
                                  f"file, not {revision!r}")
    found = weights_sha256(path)
    if found != m.group(1):
        raise DetectorUnavailable(f"PIITagger: the weights in {path!r} are not the pinned ones "
                                  f"(sha256 {found[:12]}…, pinned {m.group(1)[:12]}…)")


class TaggerModel:
    """A tokenizer and encoder, loaded once per (model, revision, device)."""

    def __init__(self, model: str, revision: Optional[str] = None, device: str = "cpu"):
        local = local_dir(model)                    # found and pinned before torch is imported
        if local:
            check_pin(local, revision)
        elif "/" not in model and os.sep not in model:      # a folder name, not a hub id
            raise DetectorUnavailable(
                f"PIITagger: no model folder {model!r} under {models_dir()} (${MODELS_ENV}). Copy the "
                f"trained tagger there (python -m bench.tagger.install), point ${MODELS_ENV} at the "
                "folder that holds it, or use the 'classic' detection preset (spaCy and "
                "ContextGuard read names; no tagger).")
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise DetectorUnavailable(
                "PIITagger needs transformers and torch, which are not installed. Run: pip install "
                "transformers torch, or use the 'classic' detection preset.") from exc
        try:
            if local:
                path, hub_revision = local, None
            else:
                path = _local_model(model, revision)
                hub_revision = revision if path == model else None
            self.tokenizer = AutoTokenizer.from_pretrained(path, revision=hub_revision)
            self.model = AutoModelForTokenClassification.from_pretrained(
                path, revision=hub_revision, dtype=torch.float32)
        except (OSError, ValueError) as exc:
            raise DetectorUnavailable(
                f"PIITagger could not load {model!r} ({exc}). Download it once with network access "
                "(HF_HUB_OFFLINE unset), or switch the pii_tagger stage off.") from exc
        self._setup(torch, device, model)
        self.model.eval()
        logger.info(f"[PIITagger] loaded {model} ({len(self.labels)} labels, device={device})")

    @classmethod
    def wrap(cls, model, tokenizer, device) -> "TaggerModel":
        """A loaded model and tokenizer (training-time evaluation); the model's mode is left alone."""
        import torch
        self = cls.__new__(cls)
        self.model, self.tokenizer = model, tokenizer
        self._setup(torch, device, getattr(model, "name_or_path", "model"), move=False)
        return self

    def _setup(self, torch, device, name: str, move: bool = True) -> None:
        if not getattr(self.tokenizer, "is_fast", False):
            raise DetectorUnavailable(f"PIITagger needs a fast tokenizer (character offsets) for {name!r}")
        self.torch = torch
        self.device = torch.device(device)
        if move:
            self.model.to(self.device)
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        bad = [lab for lab in self.labels if lab != "O" and lab.split("-", 1)[-1] not in TAGS]
        if bad:
            raise DetectorUnavailable(f"PIITagger: {name!r} has labels outside the protect types: {bad[:5]}")

    def token_probs(self, texts: Sequence[str], window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE,
                    batch: int = 8) -> List[List[Tuple[int, int, List[float]]]]:
        """Per text, its tokens as ``(start, end, label probabilities)`` in
        order; a token seen by several windows keeps the most central one."""
        enc = self.tokenizer(list(texts), truncation=True, max_length=window, stride=stride,
                             return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
        owner = enc["overflow_to_sample_mapping"]
        probs = []
        torch = self.torch
        rows = list(range(len(owner)))
        with torch.inference_mode():
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                feats = [{"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j]} for j in chunk]
                padded = self.tokenizer.pad(feats, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in padded.items()}).logits
                p = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                probs.extend(p[k][:len(enc["input_ids"][j])] for k, j in enumerate(chunk))
        best: List[Dict[Tuple[int, int], Tuple[float, List[float]]]] = [dict() for _ in texts]
        for j, t in enumerate(owner):
            offs = enc["offset_mapping"][j]
            real = [k for k, (s, e) in enumerate(offs) if e > s]
            if not real:
                continue
            lo, hi = real[0], real[-1]
            for k in real:
                centre = min(k - lo, hi - k)
                key = tuple(offs[k])
                if key not in best[t] or centre > best[t][key][0]:
                    best[t][key] = (centre, probs[j][k])
        return [[(s, e, p) for (s, e), (_c, p) in sorted(b.items())] for b in best]


def get_model(model: str, revision: Optional[str] = None, device: str = "cpu") -> TaggerModel:
    key = (model, revision, device)
    with _lock:
        cached = _models.get(key)
        if cached is None:
            try:
                cached = TaggerModel(model, revision, device)
            except DetectorUnavailable as exc:
                cached = exc
            _models[key] = cached
    if isinstance(cached, DetectorUnavailable):
        raise cached
    return cached


# ── decoding ─────────────────────────────────────────────────────────────────

def _trim(text: str, s: int, e: int, tag: str) -> Tuple[int, int]:
    """Drop whitespace, sentence punctuation, unbalanced brackets and
    wrapping quotes or Markdown at the edges of [s, e)."""
    trail = ".,;:" if tag == "CREDENTIAL" else ".,;:!?"
    while e > s:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s:
            break
        first, last, inner = text[s], text[e - 1], text[s:e]
        if last in trail and not (last == "." and tag in ("ORG", "PERSON") and _ABBREV.search(text, s, e)):
            e -= 1
        elif last in _CLOSERS and inner.count(_CLOSERS[last]) < inner.count(last):
            e -= 1
        elif first in _OPENERS and inner.count(_OPENERS[first]) < inner.count(first):
            s += 1
        elif e - s > 1 and first == last and first in _WRAP:
            s, e = s + 1, e - 1
        elif first in _QUOTES and inner.count(first) % 2:
            s += 1
        elif last in _QUOTES and inner.count(last) % 2:
            e -= 1
        elif tag != "CREDENTIAL" and "*" in (first, last):
            s, e = s + (first == "*"), e - (last == "*")
        else:
            break
    return s, e


def decode(text: str, tokens: Sequence[Tuple[int, int, Sequence[float]]], labels: Sequence[str],
           merge_address: bool = True) -> List[Candidate]:
    """BIO over *tokens* (``(start, end, probabilities)``) into candidates.
    An I- after a different type (or after O) starts a new value."""
    index = {lab: i for i, lab in enumerate(labels)}
    spans: List[List] = []                 # [start, end, tag, [token scores]]
    cur: Optional[List] = None
    for s, e, p in tokens:
        k = max(range(len(p)), key=p.__getitem__)
        lab = labels[k]
        if lab == "O":
            cur = None
            continue
        bio, tag = lab.split("-", 1)
        sc = p[index.get(f"B-{tag}", k)] + p[index.get(f"I-{tag}", k)] if f"B-{tag}" in index and f"I-{tag}" in index \
            else p[k]
        if cur is not None and bio == "I" and cur[2] == tag:
            cur[1] = e
            cur[3].append(sc)
        else:
            cur = [s, e, tag, [sc]]
            spans.append(cur)
    found = []
    for s, e, tag, scores in spans:
        s, e = _trim(text, s, e, tag)
        if e > s:
            found.append((s, e, tag, sum(scores) / len(scores), len(scores)))
    return _join_address(text, found) if merge_address else \
        [Candidate(s, e, tag, round(sc, 4)) for s, e, tag, sc, _n in found if tag not in ADDRESS_PARTS]


def _join_address(text: str, found) -> List[Candidate]:
    out: List[Candidate] = []
    group: List[tuple] = []

    def flush():
        if not group:
            return
        tags = {g[2] for g in group}
        if tags <= {"CITY", "REGION"}:
            # a town ("Lyon", "Lyon, Rhône") is a place, not an address; a lone region is dropped
            out.extend(Candidate(g[0], g[1], "LOCATION", round(g[3], 4)) for g in group if g[2] == "CITY")
        else:
            n = sum(g[4] for g in group)
            score = sum(g[3] * g[4] for g in group) / n
            out.append(Candidate(group[0][0], group[-1][1], "ADDRESS", round(score, 4)))
        group.clear()

    for f in found:
        if f[2] in ADDRESS_PARTS:
            if group and not _ADDRESS_GAP.fullmatch(text[group[-1][1]:f[0]]):
                flush()
            group.append(f)
            continue
        flush()
        out.append(Candidate(f[0], f[1], f[2], round(f[3], 4)))
    flush()
    return sorted(out, key=lambda c: (c.start, c.end))


# ── the stage ────────────────────────────────────────────────────────────────

class PIITagger:
    """The ``pii_tagger`` stage's detector."""

    def __init__(self, stage):
        if not stage.model:
            raise DetectorUnavailable("the pii_tagger stage needs a model (a path or a hub id)")
        opts = dict(stage.options or {})
        device = str(opts.get("device") or ("cpu" if stage.device < 0 else f"cuda:{stage.device}"))
        self.model = get_model(stage.model, stage.revision, device)
        self.window = int(stage.window or DEFAULT_WINDOW)
        self.stride = int(stage.stride if stage.stride is not None else DEFAULT_STRIDE)
        self.batch = int(opts.get("batch", 8))
        self.merge_address = bool(opts.get("merge_address", True))

    def detect(self, text: str, view=None) -> List[Candidate]:
        if not text.strip():
            return []
        tokens = self.model.token_probs([text], self.window, self.stride, self.batch)[0]
        return decode(text, tokens, self.model.labels, self.merge_address)

    def detect_many(self, texts: Sequence[str]) -> List[List[Candidate]]:
        all_tokens = self.model.token_probs(list(texts), self.window, self.stride, self.batch)
        return [decode(t, toks, self.model.labels, self.merge_address) for t, toks in zip(texts, all_tokens)]


def factory(stage) -> PIITagger:
    return PIITagger(stage)
