# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
eval_metrics.py — the single definition of every detection / protection metric.

Used by ``evaluator.py`` (scores a runner answers file), ``offline_eval.py``
(scores the cascade directly, no provider) and ``bench/``. Nothing here imports
a model or touches the network.

Definitions (TECHNICAL_AUDIT A1–A4, A6, A12):

* **Gold spans.** Every case-insensitive occurrence of every key value in the
  question, bounded by non-alphanumerics (``AZ`` does not match in ``AZURE``).
  Key values that never occur in the question are counted
  (``gold_not_in_text``) and excluded — they cannot be scored.
* **Matching.** Span overlap. A gold span is found if any predicted span
  overlaps it; a predicted span is correct if it overlaps any gold span.
  Precision = correct predictions / predictions; recall = found gold / gold.
  ``strict`` additionally reports exact-boundary matches.
* **Protection vs recognition.** Protection counts only spans that were
  replaced (or shifted) before sending. Recognition also counts spans that
  were recognised but deliberately left in place (rnr). The headline is
  protection; recognised-not-replaced values are never credited as protected.
* **Empty sets score 0.** ``prf`` returns 0.0 when a denominator is empty; no
  1.0 defaults.
* **Micro and macro.** Micro pools spans; macro averages per-type F1 over the
  types present in gold or predictions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────
# Type universes
# ─────────────────────────────────────────────

# Answer-key label → canonical gold type.
KEY_TYPE_MAP: Dict[str, str] = {
    "name": "PERSON", "person": "PERSON", "PERSON": "PERSON",
    "email": "email",
    "phone": "phone", "phone_us": "phone", "phone_uk": "phone", "phone_intl": "phone",
    "ssn": "ssn",
    "address": "address",
    "dob": "dob", "date_of_birth": "dob",
    "org": "ORG", "ORG": "ORG", "organization": "ORG",
    "gpe": "GPE", "GPE": "GPE", "location": "GPE",
    "loc": "LOC", "LOC": "LOC",
    "fac": "FAC", "FAC": "FAC",
    "credit_card": "credit_card",
    "api_key": "api_key",
    "ip_address": "ip_address", "ip": "ip_address",
    "zip": "postal_code", "zip_us": "postal_code",
    "postcode": "postal_code", "postcode_uk": "postal_code",
    "gender": "gender_indicator",
    "crypto": "crypto", "bitcoin": "crypto", "ethereum": "crypto", "wallet": "crypto",
    "bank_number": "us_bank_number", "bank_account": "us_bank_number",
    "us_bank_number": "us_bank_number", "routing_number": "us_bank_number",
    "routing": "us_bank_number",
    "driver_license": "us_driver_license", "us_driver_license": "us_driver_license",
    "drivers_license": "us_driver_license", "dl": "us_driver_license",
    "license": "us_driver_license",
}

# SurrogateShield detector type → canonical type (identity otherwise).
NORMALIZE_TYPE: Dict[str, str] = {
    "phone_us": "phone", "phone_uk": "phone", "phone_intl": "phone",
    "zip_us": "postal_code", "postcode_uk": "postal_code",
}

GEO = "GEO"
_GEO_TYPES = frozenset({"GPE", "LOC", "FAC"})

# The Presidio comparison universe: types both systems can produce under
# Presidio's shipped configuration. GPE/LOC/FAC collapse to GEO because
# Presidio's spacy.yaml maps all three to LOCATION.
SHARED_TYPES = frozenset({
    "PERSON", "email", "phone", "ssn", "credit_card", "ip_address", "crypto",
    "us_bank_number", "us_driver_license", GEO, "dob",
})

PRESIDIO_TO_SHARED: Dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "CREDIT_CARD": "credit_card",
    "IP_ADDRESS": "ip_address",
    "CRYPTO": "crypto",
    "US_BANK_NUMBER": "us_bank_number",
    "US_DRIVER_LICENSE": "us_driver_license",
    "LOCATION": GEO,
    # DATE_TIME counts as dob only when the span is date-shaped; see
    # presidio_shared_type().
}

# Gold types outside SHARED_TYPES, with the reason they are excluded from the
# head-to-head table. They are reported in an SS-only table.
SS_ONLY_REASONS: Dict[str, str] = {
    "ORG": "Presidio's shipped config ignores ORGANIZATION (spacy.yaml labels_to_ignore)",
    "address": "no Presidio default recogniser",
    "postal_code": "no Presidio default recogniser",
    "api_key": "no Presidio default recogniser",
    "gender_indicator": "no Presidio default recogniser",
}

_DATE_SHAPED = re.compile(
    r"^(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
    r"|\d{1,2} [A-Za-z]{3,9},? \d{4})$"
)


def gold_type(label: str) -> str:
    """Canonical gold type for an answer-key label ('other' if unknown)."""
    return KEY_TYPE_MAP.get(label, KEY_TYPE_MAP.get(label.lower(), "other"))


def normalize_type(t: str) -> str:
    """Canonical type for a SurrogateShield detector type."""
    return NORMALIZE_TYPE.get(t, t)


def is_date_shaped(text: str) -> bool:
    return bool(_DATE_SHAPED.match(text.strip()))


def to_shared(t: str) -> Optional[str]:
    """Map a canonical (gold or SS) type into the comparison universe, or None."""
    t = normalize_type(t)
    if t in _GEO_TYPES:
        return GEO
    return t if t in SHARED_TYPES else None


def presidio_shared_type(entity_type: str, text: str) -> Optional[str]:
    """Map a Presidio entity into the comparison universe, or None."""
    if entity_type == "DATE_TIME":
        return "dob" if is_date_shaped(text) else None
    return PRESIDIO_TO_SHARED.get(entity_type)


def ss_shared_type(t: str) -> Optional[str]:
    """Map an SS prediction into the comparison universe.

    An SS ``address`` span may cover a gold city/state, so it competes as GEO
    (Presidio's LOCATION competes for the same spans).
    """
    t = normalize_type(t)
    if t == "address":
        return GEO
    return to_shared(t)


# ─────────────────────────────────────────────
# Spans
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    value: str = ""

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


def _boundary_pattern(value: str) -> "re.Pattern[str]":
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def find_occurrences(text: str, value: str) -> List[Tuple[int, int]]:
    """All case-insensitive, alnum-bounded occurrences of *value* in *text*."""
    value = value.strip()
    if not value or not text:
        return []
    return [(m.start(), m.end()) for m in _boundary_pattern(value).finditer(text)]


def contains_value(text: Optional[str], value: str) -> bool:
    """True if *value* occurs in *text* as a whole token sequence."""
    return bool(text) and bool(find_occurrences(text, value))


def key_values(answer_key) -> List[Tuple[str, str]]:
    """Flatten an Answer-Key dict into ``[(canonical_type, value), ...]``.

    Non-string scalars are stringified (``str(v).strip()``) so a numeric key
    value is scored instead of crashing the evaluator (I32).
    """
    out: List[Tuple[str, str]] = []
    if not isinstance(answer_key, dict):
        return out
    for label, val in answer_key.items():
        t = gold_type(str(label))
        for v in (val if isinstance(val, list) else [val]):
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append((t, s))
    return out


def gold_spans(text: str, answer_key) -> Tuple[List[Span], List[Tuple[str, str]]]:
    """Gold spans for a question, plus the key values that do not occur in it."""
    spans: Dict[Tuple[int, int], Span] = {}
    missing: List[Tuple[str, str]] = []
    for t, v in key_values(answer_key):
        occ = find_occurrences(text, v)
        if not occ:
            missing.append((t, v))
        for s, e in occ:
            # The first key value wins identical offsets; distinct offsets
            # are kept even when nested (a city inside an address).
            spans.setdefault((s, e), Span(s, e, t, text[s:e]))
    return sorted(spans.values(), key=lambda sp: (sp.start, sp.end)), missing


def locate(text: str, values: Iterable[Tuple[str, str]]) -> List[Span]:
    """Spans for ``(type, value)`` predictions that have no stored offsets."""
    out: Dict[Tuple[int, int], Span] = {}
    for t, v in values:
        for s, e in find_occurrences(text, v):
            out.setdefault((s, e), Span(s, e, t, text[s:e]))
    return sorted(out.values(), key=lambda sp: (sp.start, sp.end))


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def prf(correct_pred: int, n_pred: int, found_gold: int, n_gold: int) -> Tuple[float, float, float]:
    """Precision, recall, F1. Every empty denominator scores 0.0 (A2)."""
    p = correct_pred / n_pred if n_pred else 0.0
    r = found_gold / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


Compat = Callable[[str, str], bool]


def match(
    gold: Sequence[Span],
    pred: Sequence[Span],
    compatible: Optional[Compat] = None,
) -> Tuple[List[bool], List[bool]]:
    """Overlap matching. Returns (gold_found, pred_correct) flags.

    ``compatible(gold_type, pred_type)`` restricts which pairs may match;
    None means untyped (any overlap counts).
    """
    g_hit = [False] * len(gold)
    p_hit = [False] * len(pred)
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if g.overlaps(p) and (compatible is None or compatible(g.type, p.type)):
                g_hit[i] = True
                p_hit[j] = True
    return g_hit, p_hit


@dataclass
class TypeCounts:
    gold: int = 0
    gold_found: int = 0
    pred: int = 0
    pred_correct: int = 0
    exact: int = 0

    def as_dict(self) -> dict:
        p, r, f = prf(self.pred_correct, self.pred, self.gold_found, self.gold)
        return {
            "tp": self.gold_found,
            "fn": self.gold - self.gold_found,
            "fp": self.pred - self.pred_correct,
            "gold": self.gold,
            "pred": self.pred,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "exact_boundary_matches": self.exact,
        }


@dataclass
class Tally:
    """Accumulates span counts over many questions, per type."""

    by_type: Dict[str, TypeCounts] = field(default_factory=lambda: defaultdict(TypeCounts))
    questions: int = 0
    negative_questions: int = 0
    negative_questions_with_pred: int = 0
    negative_pred_spans: int = 0

    def add(
        self,
        gold: Sequence[Span],
        pred: Sequence[Span],
        compatible: Optional[Compat] = None,
    ) -> Tuple[List[bool], List[bool]]:
        g_hit, p_hit = match(gold, pred, compatible)
        self.questions += 1
        for g, hit in zip(gold, g_hit):
            c = self.by_type[g.type]
            c.gold += 1
            c.gold_found += hit
            if any(p.start == g.start and p.end == g.end for p in pred):
                c.exact += 1
        for p, hit in zip(pred, p_hit):
            c = self.by_type[p.type]
            c.pred += 1
            c.pred_correct += hit
        if not gold:
            self.negative_questions += 1
            if pred:
                self.negative_questions_with_pred += 1
                self.negative_pred_spans += len(pred)
        return g_hit, p_hit

    def micro(self) -> dict:
        tot = TypeCounts()
        for c in self.by_type.values():
            tot.gold += c.gold
            tot.gold_found += c.gold_found
            tot.pred += c.pred
            tot.pred_correct += c.pred_correct
            tot.exact += c.exact
        return tot.as_dict()

    def macro(self) -> dict:
        rows = [c.as_dict() for c in self.by_type.values() if c.gold or c.pred]
        if not rows:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_types": 0}
        n = len(rows)
        return {
            "precision": round(sum(r["precision"] for r in rows) / n, 4),
            "recall": round(sum(r["recall"] for r in rows) / n, 4),
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "n_types": n,
        }

    def per_type(self) -> Dict[str, dict]:
        return {t: c.as_dict() for t, c in sorted(self.by_type.items()) if c.gold or c.pred}

    def negatives(self) -> dict:
        n = self.negative_questions
        return {
            "negative_questions": n,
            "negative_questions_with_any_prediction": self.negative_questions_with_pred,
            "negative_question_fp_rate": round(self.negative_questions_with_pred / n, 4) if n else 0.0,
            "fp_spans_on_negatives": self.negative_pred_spans,
        }

    def summary(self) -> dict:
        return {
            "micro": self.micro(),
            "macro": self.macro(),
            "per_type": self.per_type(),
            "negatives": self.negatives(),
            "questions": self.questions,
        }


def drop_neutral(pred: Sequence[Span], neutral_gold: Sequence[Span]) -> List[Span]:
    """Remove predictions that overlap gold of a type outside the universe.

    In the Presidio comparison a prediction that lands on, say, a gold postal
    code is neither credited nor penalised — for either system.
    """
    return [p for p in pred if not any(p.overlaps(g) for g in neutral_gold)]


# ─────────────────────────────────────────────
# Statistics (A12)
# ─────────────────────────────────────────────

def format_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    return f"{p:.3g}"


def bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean of paired differences (seeded)."""
    import random

    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_stats(a: Sequence[float], b: Sequence[float], alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison a vs b: means, sample std (ddof=1), mean difference,
    bootstrap CI, Cohen's d_z and a paired t-test when scipy is present."""
    n = len(a)
    if n != len(b):
        raise ValueError("paired_stats needs equal-length samples")
    if n < 2:
        return {"available": False, "n_paired": n, "reason": "fewer than 2 paired values"}

    def mean(xs):
        return sum(xs) / len(xs)

    def sd(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    diffs = [x - y for x, y in zip(a, b)]
    d_sd = sd(diffs)
    lo, hi = bootstrap_ci(diffs, alpha=alpha, seed=seed)
    out = {
        "available": True,
        "n_paired": n,
        "ss_mean": round(mean(a), 4),
        "ss_std": round(sd(a), 4),
        "presidio_mean": round(mean(b), 4),
        "presidio_std": round(sd(b), 4),
        "mean_diff": round(mean(diffs), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "alpha": alpha,
        "effect_size_dz": round(mean(diffs) / d_sd, 4) if d_sd else None,
        "t_statistic": None,
        "p_value": None,
        "p_display": "n/a",
        "p_significant": None,
    }
    try:
        from scipy import stats as _stats
    except ImportError:
        out["test"] = "bootstrap only (scipy not installed)"
        return out
    t, p = _stats.ttest_rel(a, b)
    out["test"] = "paired t-test (scipy.stats.ttest_rel) + bootstrap CI"
    out["t_statistic"] = None if math.isnan(t) else round(float(t), 4)
    out["p_value"] = None if math.isnan(p) else float(p)
    out["p_display"] = format_p(out["p_value"])
    out["p_significant"] = None if out["p_value"] is None else out["p_value"] < alpha
    return out
