"""Arm ``gliner_pii``: ``urchade/gliner_multi_pii-v1`` (Apache-2.0), zero-shot
PII NER used for detection only, every span replaced by ``[LABEL]``. Labels
cover each ``protect`` type of ``bench/realworld/GUIDE.md`` with the model
card's own label names; threshold 0.5 (the model card's). Long messages are
read in 200-word windows overlapping by 30 words (the model reads at most 384
tokens); overlaps are resolved longest-then-score. Runs in ``.venv-baselines``.
"""

from __future__ import annotations

import re

from bench.arms.base import cli, hf_revision, resolve_overlaps, versions

ARM = "gliner_pii"
MODEL = "urchade/gliner_multi_pii-v1"
THRESHOLD = 0.5
WINDOW, OVERLAP = 200, 30
LABELS = [
    "person", "organization", "email", "phone number", "address", "location",
    "date of birth", "age", "passport number", "social security number",
    "credit card number", "bank account number", "iban", "driver's license number",
    "national id number", "tax identification number", "health insurance id number",
    "ip address", "url", "username", "password", "api key",
]


def windows(text: str, size: int = WINDOW, overlap: int = OVERLAP):
    """(start, end) character windows of at most *size* words."""
    words = [m.span() for m in re.finditer(r"\S+", text)]
    if len(words) <= size:
        return [(0, len(text))]
    out, i = [], 0
    while True:
        j = min(i + size, len(words))
        out.append((words[i][0], words[j - 1][1]))
        if j == len(words):
            return out
        i = j - overlap


def model():
    import logging
    import warnings
    from gliner import GLiNER
    logging.getLogger("transformers").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GLiNER.from_pretrained(MODEL)


def window_spans(model, text: str, labels, threshold: float):
    """``(start, end, label, score)`` of every window's decoded entities,
    before overlaps between windows are resolved."""
    spans = []
    for ws, we in windows(text):
        for ent in model.predict_entities(text[ws:we], labels, threshold=threshold):
            s, e = ws + ent["start"], ws + ent["end"]
            if text[s:e].strip():
                spans.append((s, e, ent["label"], float(ent["score"])))
    return list(set(spans))


def to_edits(spans):
    """The arm's edits: overlaps resolved longest-then-score, each span
    replaced by ``[LABEL]``."""
    return [[s, e, t, "[" + t.upper().replace(" ", "_").replace("'", "") + "]"]
            for s, e, t, _ in resolve_overlaps(spans)]


def load(threshold: float = THRESHOLD, labels=LABELS):
    m = model()

    def fn(text: str, _seed: int):
        return to_edits(window_spans(m, text, labels, threshold))
    return fn


def config() -> dict:
    return {"model": MODEL, "model_revision": hf_revision(MODEL),
            "backbone_revision": {"microsoft/mdeberta-v3-base": hf_revision("microsoft/mdeberta-v3-base")}, "labels": LABELS, "threshold": THRESHOLD, "flat_ner": True,
            "window_words": WINDOW, "overlap_words": OVERLAP,
            "replacement": "[LABEL] placeholder",
            "versions": versions("gliner", "transformers", "torch")}


if __name__ == "__main__":
    raise SystemExit(cli(ARM, load, config))
