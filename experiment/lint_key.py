#!/usr/bin/env python3
# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
lint_key.py — validate an evaluation key before any number is computed from it
(audit B1–B6, gate J11).

Generated keys (rows with ``Spans``, from experiment/make_dataset.py):
  B2  every span value is at its offsets; spans are in order and do not overlap
  B3  every span type is a known label; Answer-Key and Spans agree
  B4  every occurrence of a gold value is annotated; no e-mail / SSN / card /
      IBAN-shaped string is left unannotated unless listed in ``Keep``
  B1  no identity, name or identifying value appears in two rows
      (checked across all files given together)
  B6  ≥ 20 % PII-free rows and at least one non-English negative per file;
      with several files, no template is shared between them
  D2  no pronoun of the other gender next to a gendered identity

Legacy keys (``Question`` + ``Answer-Key`` only, e.g. experiment/test_key.json)
get the checks that need no offsets: B2 value not in question, B3 unknown or
label-like values, B4 regex-obvious PII missing from the key, B1 reuse counts.

Exit status 1 if any finding. Usage:
    python experiment/lint_key.py experiment/synth_dev_key.json experiment/synth_test_key.json
    python experiment/lint_key.py experiment/test_key.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_metrics as em  # noqa: E402

MIN_NEG_FRACTION = 0.20
# Values that identify a person: reuse across rows breaks B1.
_UNIQUE_TYPES = {"name", "email", "phone", "ssn", "credit_card", "iban", "api_key",
                 "ip_address", "address", "url", "username"}
_OBVIOUS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "ssn": re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),
    "credit_card": re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
}
_PRONOUNS = {"m": re.compile(r"\b(?:she|her|hers)\b", re.I), "f": re.compile(r"\b(?:he|him|his)\b", re.I)}
_LABEL_WORDS = {"name", "phone", "email", "ssn", "address", "dob", "zip", "org", "gpe"}


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def lint_generated(files: Dict[str, list]) -> List[str]:
    findings: List[str] = []
    seen_values: Dict[str, str] = {}
    seen_ids: Dict[int, str] = {}
    templates: Dict[str, set] = {}
    for fname, rows in files.items():
        templates[fname] = set()
        neg = neg_foreign = 0
        ids = Counter(r.get("id") for r in rows)
        findings += [f"{fname}: id {i!r} used {c} times" for i, c in ids.items() if c > 1]
        for r in rows:
            where = f"{fname}:{r.get('id')}"
            text, spans, keep = r["Question"], r["Spans"], r.get("Keep", [])
            templates[fname].add(r.get("template"))
            # B2: offsets, order, overlap
            last_end = -1
            for sp in spans:
                if text[sp["start"]:sp["end"]] != sp["value"]:
                    findings.append(f"{where}: B2 {sp['value']!r} not at [{sp['start']}:{sp['end']}]")
                if sp["start"] < last_end:
                    findings.append(f"{where}: B2 span {sp['value']!r} overlaps the previous one")
                last_end = sp["end"]
                if em.gold_type(sp["type"]) == "other":
                    findings.append(f"{where}: B3 unknown type {sp['type']!r}")
            # B3: Answer-Key ⇔ Spans
            key_pairs = set(em.key_values(r.get("Answer-Key")))
            span_pairs = {(em.gold_type(sp["type"]), sp["value"]) for sp in spans}
            if key_pairs != span_pairs:
                findings.append(f"{where}: B3 Answer-Key and Spans disagree")
            # B4: every occurrence annotated
            starts = {(sp["start"], sp["end"]) for sp in spans}
            for sp in spans:
                for s, e in em.find_occurrences(text, sp["value"]):
                    if (s, e) not in starts and not any(a <= s and e <= b for a, b in starts):
                        findings.append(f"{where}: B4 {sp['value']!r} at {s} not annotated")
            for t, pat in _OBVIOUS.items():
                for m in pat.finditer(text):
                    if any(m.start() < b and a < m.end() for a, b in starts):
                        continue
                    if any(m.group() in k for k in keep):
                        continue
                    if t == "credit_card" and not _luhn_ok(re.sub(r"\D", "", m.group())):
                        continue
                    findings.append(f"{where}: B4 unannotated {t}-shaped {m.group()!r}")
            # B1: identities and values never reused
            iid = r.get("identity_id")
            if iid is not None:
                if iid in seen_ids:
                    findings.append(f"{where}: B1 identity {iid} also in {seen_ids[iid]}")
                seen_ids[iid] = where
            for sp in spans:
                if sp["type"] in _UNIQUE_TYPES:
                    v = sp["value"].casefold()
                    if sp["type"] == "name" and " " not in v:
                        continue                  # first-only mentions repeat by design
                    if v in seen_values and not seen_values[v].startswith(where + "#"):
                        findings.append(f"{where}: B1 {sp['type']} {sp['value']!r} also in "
                                        f"{seen_values[v].split('#')[0]}")
                    seen_values.setdefault(v, where + "#")
            # D2: pronouns follow the identity's gender
            g = r.get("gender")
            if g in _PRONOUNS and spans:
                masked = text
                for sp in sorted(spans, key=lambda x: -x["start"]):
                    masked = masked[:sp["start"]] + " " * (sp["end"] - sp["start"]) + masked[sp["end"]:]
                if r.get("lang", "en") == "en" and _PRONOUNS[g].search(masked):
                    findings.append(f"{where}: D2 pronoun of the other gender for gender {g}")
            if not spans:
                neg += 1
                neg_foreign += r.get("lang", "en") != "en"
                if r.get("identity_id") is not None or r.get("Answer-Key"):
                    findings.append(f"{where}: B6 negative row carries an identity or key")
        if rows and neg / len(rows) < MIN_NEG_FRACTION:
            findings.append(f"{fname}: B6 negatives {neg}/{len(rows)} below {MIN_NEG_FRACTION:.0%}")
        if rows and not neg_foreign:
            findings.append(f"{fname}: B6 no non-English negative")
    names = list(templates)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = templates[a] & templates[b] - {None}
            if shared:
                findings.append(f"B6 templates shared by {a} and {b}: {sorted(shared)[:5]}")
    return findings


def lint_legacy(fname: str, rows: list) -> List[str]:
    findings: List[str] = []
    reuse: Dict[str, Counter] = {}
    for i, r in enumerate(rows):
        where = f"{fname}:{i}"
        q, key = r.get("Question", ""), r.get("Answer-Key")
        pairs = em.key_values(key)
        for t, v in pairs:
            if not em.find_occurrences(q, v):
                findings.append(f"{where}: B2 {t}={v!r} not in the question")
            if t == "other":
                findings.append(f"{where}: B3 unknown label for {v!r}")
            if v.lower() in _LABEL_WORDS:
                findings.append(f"{where}: B3 label word {v!r} used as a value")
            reuse.setdefault(t, Counter())[v.casefold()] += 1
        gold = [s for _t, v in pairs for s in em.find_occurrences(q, v)]
        for t, pat in _OBVIOUS.items():
            for m in pat.finditer(q):
                if any(m.start() < e and s < m.end() for s, e in gold):
                    continue
                if t == "credit_card" and not _luhn_ok(re.sub(r"\D", "", m.group())):
                    continue
                findings.append(f"{where}: B4 {t}-shaped {m.group()!r} missing from the key")
    for t, c in sorted(reuse.items()):
        repeated = sum(n for n in c.values() if n > 1)
        if t in {"PERSON", "email", "phone", "ssn", "address"} and repeated:
            findings.append(f"{fname}: B1 {t}: {repeated} of {sum(c.values())} mentions belong "
                            f"to a value used in more than one question")
    return findings


def lint_files(paths: List[Path]) -> List[str]:
    generated: Dict[str, list] = {}
    findings: List[str] = []
    for p in paths:
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            findings.append(f"{p}: not a JSON list")
        elif rows and all(isinstance(r, dict) and "Spans" in r for r in rows):
            generated[str(p)] = rows
        else:
            findings += lint_legacy(str(p), rows)
    if generated:
        findings += lint_generated(generated)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("keys", nargs="+", type=Path, help="key files (lint together for B1/B6)")
    ap.add_argument("--max", type=int, default=50, help="findings to print (default 50)")
    a = ap.parse_args()
    findings = lint_files(a.keys)
    for f in findings[:a.max]:
        print(f)
    by_code = Counter(re.search(r"\b(B\d|D\d)\b", f).group(1) for f in findings
                      if re.search(r"\b(B\d|D\d)\b", f))
    print(f"\n{len(findings)} finding(s) {dict(sorted(by_code.items()))}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
