"""Shared helpers for the real-data benchmark (bench/realdata).

Nothing here touches the network or a provider. Files that hold real
people's text are written 0600 under ``raw/`` or ``build/`` (git-ignored);
committed files never carry natural-slice text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / "bench" / "realdata"
RAW = RD / "raw"          # downloads + sampled natural text (git-ignored)
BUILD = RD / "build"      # materialised natural-slice records (git-ignored)

SEED = 20261005
DATASETS = ("oasst1", "sharegpt", "wildchat")

SOURCES = {
    "oasst1": {
        "repo": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "files": ["data/train-00000-of-00001-b42a775f407cee45.parquet",
                  "data/validation-00000-of-00001-134b8fd0c89408b6.parquet"],
        "licence": "Apache-2.0",
        "local": "oasst1",
    },
    "sharegpt": {
        "repo": "RyokoAI/ShareGPT52K",
        "revision": "6f9b78cc1dd15dbb51d3c51ccc219c558962fd77",
        "files": ["sg_90k_part1.json"],
        "licence": "CC0-1.0",
        "local": "ShareGPT52K",
    },
    "wildchat": {
        "repo": "allenai/WildChat-1M",
        "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        # two of the 14 shards, drawn with sorted(random.Random(SEED).sample(range(14), 2))
        "files": ["data/train-00004-of-00014.parquet", "data/train-00008-of-00014.parquet"],
        "licence": "ODC-BY-1.0",
        "local": "WildChat-1M",
    },
}

TASKS = ("writing", "coding", "qa", "advice", "translation", "roleplay", "business", "other")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(*parts) -> int:
    """A stable 63-bit seed from SEED and *parts* (unlike hash(), not salted per process)."""
    h = hashlib.sha256("|".join(str(p) for p in (SEED,) + parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def words(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict], private: bool = False) -> None:
    """Atomic write; *private* files are created 0600 in a 0700 directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        os.chmod(tmp, 0o600 if private else 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ── filters (pure; tested in tests/test_realdata_pull.py) ─────────────────────

_EN_STOP = frozenset("""
the be to of and a in that have i it for not on with he as you do at this but his by from
they we say her she or an will my one all would there their what so up out if about who get
which go me when make can like time no just him know take people into year your good some
could them see other than then now look only come its over think also back after use two how
our work first well way even new want because any these give day most us is are was were has
had been please help write need can't don't i'm it's
""".split())
_FENCE = re.compile(r"```.*?(```|$)", re.S)
_URL = re.compile(r"https?://\S+|www\.\S+")
_ALPHA_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_CODE_LINE = re.compile(
    r"[;{}(\[]\s*$|^\s*[)\]}]|^\s*(def |class |import |from \S+ import|#include|function |const |let |var |"
    r"public |private |return\b|if \(|for \(|while \(|elif |else:|try:|except\b|print\(|<\/?\w+[^>]*>|\$ |>>> |//|#!)"
    r"|^\s*[\w.\[\]'\"]+\s*[+\-*/]?=\s*\S|^\s*[\w.]+\(.*\)\s*$")
_JAILBREAK = re.compile(
    r"\b(do anything now|jailbr(?:eak|oken)|ignore (?:all )?(?:the )?(?:previous|prior|above) "
    r"(?:instructions|prompts?)|developer mode|DAN mode|act as DAN|you are DAN|unfiltered and "
    r"amoral|without any (?:moral|ethical) (?:restrictions|guidelines)|no (?:ethical|moral) "
    r"(?:restrictions|guidelines)|uncensored (?:ai|model|assistant)|opposite mode|evil confidant)\b",
    re.I)


def is_english(text: str) -> bool:
    """Heuristic for sources without a language field (ShareGPT): ≥ 90 % of
    letters are ASCII and ≥ 12 % of word tokens are English function words."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    toks = [w.lower() for w in _ALPHA_WORD.findall(_FENCE.sub(" ", text))]
    if len(toks) < 5:
        return False
    return sum(t in _EN_STOP for t in toks) / len(toks) >= 0.12


def prose_words(text: str) -> int:
    """Alphabetic words outside fenced code blocks, URLs and code-like lines."""
    lines = [l for l in _FENCE.sub("\n", text).splitlines() if not _CODE_LINE.search(l)]
    return len(_ALPHA_WORD.findall(_URL.sub(" ", "\n".join(lines))))


def only_code(text: str) -> bool:
    """Fewer than 8 prose words once code fences and code-like lines are removed."""
    return prose_words(text) < 8


def only_url(text: str) -> bool:
    return bool(_URL.search(text)) and len(_ALPHA_WORD.findall(_URL.sub(" ", text))) < 5


def is_jailbreak(text: str) -> bool:
    return bool(_JAILBREAK.search(text))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def first_turn_ok(text: str, english_check: bool) -> str | None:
    """Reason a first user turn is rejected, or None if it is kept."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if english_check and not is_english(text):
        return "not_english"
    n = words(text)
    if n < 15:
        return "too_short"
    if n > 400:
        return "too_long"
    if only_url(text):
        return "only_url"
    if only_code(text):
        return "only_code"
    if is_jailbreak(text):
        return "jailbreak"
    return None


def later_turn_ok(text: str, english_check: bool) -> str | None:
    """Follow-up turns may be short ("continue"); the other filters apply."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if words(text) > 400:
        return "too_long"
    if english_check and words(text) >= 5 and not is_english(text):
        return "not_english"
    if is_jailbreak(text):
        return "jailbreak"
    return None
