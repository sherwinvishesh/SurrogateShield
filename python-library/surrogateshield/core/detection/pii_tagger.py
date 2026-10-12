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

*model* is a local folder, the name of a folder under
``$SURROGATESHIELD_MODELS`` (default ``~/.cache/surrogateshield/models``),
or a Hugging Face hub id. A *revision* ``sha256:<hex>`` pins the weights
file, checked at load, so a config names the exact weights without naming a
machine's paths; any other revision of a hub id is a commit or branch.

The default tagger, ``pii-tagger-dv3s-40k``, is such a name. When no folder
of that name is installed it is downloaded once from the Hugging Face Hub
(``config.TAGGER_HUB_SOURCES``: ``sherwinvishesh/surrogateshield-pii-tagger``
at a fixed commit) into the Hugging Face cache, and its weights are checked
against the pin. A cached download is loaded without a network request;
with ``HF_HUB_OFFLINE`` set a missing one fails closed.

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
# full-width clause punctuation; not brackets ("华为技术（深圳）有限公司" is one name)
_CJK_BREAK = re.compile(r"[，、。；：！？]")
# a value that ends in an abbreviation keeps its period ("S.A.R.L.", "Ltd.", "Jr.")
_ABBREV = re.compile(r"(?i)(?:(?<![^\W\d_])[^\W\d_]\.\s?){2,}$|\b(?:inc|ltd|co|corp|jr|sr|bros|pty|ltda|cía|cia|"
                     r"st|plc|gmbh|llc|lda|ag|nv|bv|sa|sl|srl|spa|kk|com)\.$")

_models: Dict[tuple, object] = {}
_lock = threading.Lock()

# Locally installed weights sit under one directory, one folder per model,
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
    """A pinned model's revision is ``sha256:<hex>`` of its weights file,
    and the file must match it; no revision checks nothing."""
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


def hub_source(model: str) -> Optional[Tuple[str, str]]:
    """The Hugging Face repo and commit a named tagger is downloaded from
    when no folder of that name is installed (``config.TAGGER_HUB_SOURCES``)."""
    from .config import TAGGER_HUB_SOURCES
    return TAGGER_HUB_SOURCES.get(model)


def _hub(model: str, revision: Optional[str]) -> Tuple[str, Optional[str], bool]:
    """``(repo, hub revision, weights checked)`` for a model that is not a
    local folder; a name that is neither a source nor a hub id fails."""
    source = hub_source(model)
    if source:
        return source[0], source[1], True
    if "/" not in model or model.count("/") > 1 or os.path.isabs(model) or model.startswith((".", "~")):
        raise DetectorUnavailable(
            f"PIITagger: no model folder {model!r} (looked under {models_dir()}, ${MODELS_ENV}), and it is "
            "not a Hugging Face model id. Name a folder that holds a trained tagger or a hub id, or use the "
            "'classic' detection preset (spaCy and ContextGuard read names; no tagger).")
    pinned = bool(revision and _PIN.fullmatch(revision))
    return model, None if pinned else revision, pinned


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")


def cached_path(model: str, revision: Optional[str] = None) -> Optional[str]:
    """*model*'s folder on disk, without a network request: a local folder,
    or a complete download in the Hugging Face cache; None when it is not
    there yet. The weights are not checked here (``model_path`` does)."""
    local = local_dir(model)
    if local:
        return local
    repo, rev, _checked = _hub(model, revision)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, revision=rev, local_files_only=True)
    except (ImportError, OSError, ValueError):     # not cached: LocalEntryNotFoundError
        return None
    return path if any(os.path.isfile(os.path.join(path, w)) for w in WEIGHTS_FILES) else None


def model_path(model: str, revision: Optional[str] = None, download: bool = True) -> str:
    """*model* as a folder on disk, its weights checked against a
    ``sha256:`` pin: a local folder, else the Hugging Face cache. A model
    that is not cached yet is downloaded once, unless *download* is False
    or ``HF_HUB_OFFLINE`` is set (then it fails closed)."""
    local = local_dir(model)
    if local:
        check_pin(local, revision)
        return local
    repo, rev, checked = _hub(model, revision)
    where = repo + (f"@{rev[:12]}" if rev else "")
    path = cached_path(model, revision)
    if path is None:
        if not download or _offline():
            raise DetectorUnavailable(
                f"PIITagger: {where} is not downloaded yet" + (" and HF_HUB_OFFLINE is set" if _offline() else "")
                + ". Run 'surrogateshield download' once with network access, or use the 'classic' "
                "detection preset.")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DetectorUnavailable(
                "PIITagger downloads its model with huggingface_hub, which is not installed. Run: pip install "
                "huggingface_hub, or use the 'classic' detection preset.") from exc
        logger.warning(f"[PIITagger] downloading {where} from the Hugging Face Hub (once; cached after)")
        try:
            path = snapshot_download(repo, revision=rev)
        except Exception as exc:     # noqa: BLE001 - any failed download fails closed
            raise DetectorUnavailable(
                f"PIITagger could not download {where} from the Hugging Face Hub ({exc}). Try again with "
                "network access, or use the 'classic' detection preset.") from exc
    if checked:
        check_pin(path, revision)
    return path


class TaggerModel:
    """A tokenizer and encoder, loaded once per (model, revision, device)."""

    def __init__(self, model: str, revision: Optional[str] = None, device: str = "cpu"):
        path = model_path(model, revision)          # found or downloaded, and pinned, before torch is imported
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise DetectorUnavailable(
                "PIITagger needs transformers and torch, which are not installed. Run: pip install "
                "transformers torch, or use the 'classic' detection preset.") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float32)
        except (OSError, ValueError) as exc:
            raise DetectorUnavailable(
                f"PIITagger could not load {model!r} from {path} ({exc}). The folder needs the tokenizer "
                "and config.json next to the weights; or use the 'classic' detection preset.") from exc
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
        if tag in ("PERSON", "ORG"):
            # a word piece runs over full-width punctuation ("Ilse K. Varga，34"):
            # no name holds one, so the name ends there
            cut = _CJK_BREAK.search(text, s, e)
            if cut and cut.start() > s:
                s, e = _trim(text, s, cut.start(), tag)
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

    def _probs(self, texts: Sequence[str]) -> List[List[Tuple[int, int, List[float]]]]:
        """The model's token probabilities; a model that fails on an input
        fails closed, like the other stages (audit I17): nothing about the
        text is in the message."""
        try:
            return self.model.token_probs(list(texts), self.window, self.stride, self.batch)
        except DetectorUnavailable:
            raise
        except Exception as exc:     # noqa: BLE001 - any runtime failure of the model fails closed
            raise DetectorUnavailable(
                f"PIITagger failed on this input ({type(exc).__name__}: {exc}); the text was not masked. "
                "Retry, or use the 'classic' detection preset.") from exc

    def detect(self, text: str, view=None) -> List[Candidate]:
        if not text.strip():
            return []
        tokens = self._probs([text])[0]
        return decode(text, tokens, self.model.labels, self.merge_address)

    def detect_many(self, texts: Sequence[str]) -> List[List[Candidate]]:
        all_tokens = self._probs(texts)
        return [decode(t, toks, self.model.labels, self.merge_address) for t, toks in zip(texts, all_tokens)]


def factory(stage) -> PIITagger:
    return PIITagger(stage)
