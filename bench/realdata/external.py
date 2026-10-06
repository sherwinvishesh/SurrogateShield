"""V3 §5.4 / H10'' — the external benchmark: NVIDIA's Nemotron-PII test split.

    PYTHONPATH=.:python-library .venv/bin/python -m bench.realdata.external --build    # once: the sample, before any arm sees it
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m bench.realdata.score --external nemotron_pii \\
        --out bench/results/external_nemotron_pii.json                                # at the freeze, once

Nemotron-PII (``nvidia/Nemotron-PII`` at the revision in ``SOURCE``, CC-BY-4.0)
is fully synthetic, released after ``urchade/gliner_multi_pii-v1`` and never
seen by our tagger (trained from our generator only), so neither system was
tuned on it. Its raw parquet stays in git-ignored ``bench/realdata/raw/``.

The sample (fixed here, before any arm runs on it): records are keyed by
``(uid, locale)`` (each uid has a ``us`` and an ``intl`` text); ``PER_STRATUM``
records from each of the four strata locale × document_format, drawn with
``derive_seed("external", NAME, stratum)`` from the sorted keys. Each record is
one message, ``ext-nemotron-NNNN`` in stratum order, clustered by uid.

Gold: every span label maps through ``LABEL_MAP`` (all 55 labels; an unknown
label stops the build) onto a J2 list. A label goes to ``protect`` with the
type whose definition covers it, to ``sensitive`` for the special
categories, and to ``optional`` (neither leak nor spurious) when no type of
ours covers it: generic dates and times, coordinates, postcodes, bank-level
codes (BIC, routing number), and attributes (occupation, gender, ...). A value
is its span's text; one that is not a whole-word substring of its text
(``realworld.occurrences``) cannot be scored and is dropped, counted per label.
A value repeated under one label is one gold value (every occurrence is
checked); a value given labels of different lists or types keeps the first in
list order protect > sensitive > optional (then span order), counted. The scorer's code path is
unchanged: leak = a protect value with a letter or digit reaching the
provider at any whole-word occurrence; spurious = an edit touching no gold
value.

``--build`` writes the committed description ``bench/realdata/external/
nemotron_pii.json`` (source pin, label map, per-record ids and text hashes,
counts; no text) and the private arm input. ``load`` rebuilds the units from
the raw file and refuses any hash that moved.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bench import realworld as rw
from bench.realdata.common import BUILD, RAW, RD, derive_seed, file_sha256, sha256, write_jsonl

NAME = "nemotron_pii"
SOURCE = {"repo": "nvidia/Nemotron-PII", "revision": "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
          "file": "data/test-00000-of-00001.parquet",
          "sha256": "1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247", "licence": "CC-BY-4.0"}
PARQUET = RAW / NAME / SOURCE["file"]
FROZEN = RD / "external" / f"{NAME}.json"
INPUT = BUILD / "external" / NAME / "input.jsonl"
PER_STRATUM = 250
ID_PREFIX = "ext-nemotron"

_P, _S, _O = "protect", "sensitive", "optional"
LABEL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    **{k: (_P, "PERSON") for k in ("first_name", "last_name")},
    "company_name": (_P, "ORG"),
    "email": (_P, "EMAIL"),
    "url": (_P, "URL"),
    **{k: (_P, "PHONE") for k in ("phone_number", "fax_number")},
    "street_address": (_P, "ADDRESS"),
    **{k: (_P, "LOCATION") for k in ("city", "state", "country", "county")},
    "date_of_birth": (_P, "DATE_OF_BIRTH"),
    "age": (_P, "AGE"),
    **{k: (_P, "ID") for k in ("customer_id", "account_number", "credit_debit_card", "ssn", "national_id", "tax_id",
                               "employee_id", "medical_record_number", "health_plan_beneficiary_number",
                               "certificate_license_number", "license_plate", "vehicle_identifier",
                               "device_identifier", "unique_id", "biometric_identifier")},
    **{k: (_P, "CREDENTIAL") for k in ("password", "api_key", "pin", "cvv", "http_cookie")},
    "user_name": (_P, "HANDLE"),
    **{k: (_P, "NETWORK") for k in ("ipv4", "ipv6", "mac_address")},
    "religious_belief": (_S, "RELIGION"),
    "race_ethnicity": (_S, "ETHNICITY"),
    "sexuality": (_S, "ORIENTATION"),
    "political_view": (_S, "POLITICAL"),
    "blood_type": (_S, "HEALTH"),
    **{k: (_O, None) for k in ("date", "time", "date_time", "coordinate", "postcode", "swift_bic",
                               "bank_routing_number", "occupation", "employment_status", "education_level",
                               "language", "gender")},
}
_ORDER = {_P: 0, _S: 1, _O: 2}


def stratum(row) -> str:
    return f"{row['locale']}-{row['document_format']}"


def read_raw(parquet: Path = PARQUET, sha: Optional[str] = SOURCE["sha256"]):
    if not parquet.exists():
        raise SystemExit(f"{parquet} is missing: it was downloaded in Phase 0 ({SOURCE['repo']}@{SOURCE['revision'][:12]})")
    if sha and file_sha256(parquet) != sha:
        raise SystemExit(f"{parquet.name}: SHA-256 differs from the pinned download")
    import pandas as pd
    df = pd.read_parquet(parquet, columns=["uid", "locale", "document_format", "domain", "text", "spans"])
    return {(r["uid"], r["locale"]): r for r in df.to_dict("records")}


def gold(text: str, spans, dropped: Counter, merged: Counter) -> Dict[str, List[dict]]:
    """The J2 lists of one record (see the module docstring)."""
    chosen: Dict[str, Tuple[int, int, str, Optional[str]]] = {}
    for n, sp in enumerate(spans):
        label = sp["label"]
        if label not in LABEL_MAP:
            raise SystemExit(f"unmapped Nemotron-PII label {label!r}: add it to LABEL_MAP before building")
        lst, typ = LABEL_MAP[label]
        given, cut = sp["text"], text[sp["start"]:sp["end"]]
        value = cut if cut == str(given) or (isinstance(given, int) and cut.isdigit() and int(cut) == given) else str(given)
        if not value.strip() or not rw.occurrences(text, value):
            dropped[label] += 1
            continue
        key = (_ORDER[lst], n, lst, typ)
        if value in chosen:
            if chosen[value][2:] != (lst, typ):         # another list or type; a plain repeat is one value
                merged[label] += 1
            chosen[value] = min(chosen[value], key)
        else:
            chosen[value] = key
    out: Dict[str, List[dict]] = {name: [] for name in rw.LISTS}
    for value, (_o, _n, lst, typ) in sorted(chosen.items(), key=lambda kv: kv[1][:2]):
        out[lst].append({"value": value, "type": typ} if typ else {"value": value})
    return out


def sample(rows: Dict[tuple, dict], per_stratum: int = PER_STRATUM) -> List[tuple]:
    by: Dict[str, List[tuple]] = defaultdict(list)
    for key, r in rows.items():
        by[stratum(r)].append(key)
    keys = []
    for s in sorted(by):
        pool = sorted(by[s])
        keys += sorted(random.Random(derive_seed("external", NAME, s)).sample(pool, min(per_stratum, len(pool))))
    return keys


def units_of(rows: Dict[tuple, dict], keys: List[tuple]) -> Tuple[List[dict], Counter, Counter]:
    dropped, merged, units = Counter(), Counter(), []
    for i, key in enumerate(keys, 1):
        r = rows[key]
        spans = ast.literal_eval(r["spans"]) if isinstance(r["spans"], str) else r["spans"]
        mid = f"{ID_PREFIX}-{i:04d}"
        g = {"id": mid, "text": r["text"], "service_query": False, "category": "external", "lang": "en",
             **gold(r["text"], spans, dropped, merged)}
        units.append({"mid": mid, "conv": f"nemotron-{r['uid']}", "dataset": stratum(r), "task": r["domain"],
                      "slices": ("external",), "turn": 0, "gold": g})
    errors = rw.lint([u["gold"] for u in units], rw.id_pattern(ID_PREFIX))
    if errors:
        raise SystemExit(f"{len(errors)} lint errors in the external units, first {errors[:3]}")
    return units, dropped, merged


def write_input(units: List[dict], path: Path = INPUT) -> str:
    write_jsonl(path, [{"id": u["mid"], "text": u["gold"]["text"]} for u in units], private=True)
    return file_sha256(path)


def build(parquet: Path = PARQUET, frozen: Path = FROZEN, src: Path = INPUT, per_stratum: int = PER_STRATUM,
          sha: Optional[str] = SOURCE["sha256"]) -> dict:
    if frozen.exists():
        raise SystemExit(f"{frozen.name} exists: the external sample is drawn once")
    rows = read_raw(parquet, sha)
    keys = sample(rows, per_stratum)
    units, dropped, merged = units_of(rows, keys)
    counts = {name: dict(sorted(Counter(x.get("type") or "-" for u in units for x in u["gold"][name]).items()))
              for name in ("protect", "sensitive", "optional")}
    doc = {"name": NAME, "source": SOURCE, "per_stratum": per_stratum,
           "seed": f"derive_seed('external', '{NAME}', stratum)", "label_map": {k: list(v) for k, v in sorted(LABEL_MAP.items())},
           "strata": dict(sorted(Counter(u["dataset"] for u in units).items())),
           "records": [{"mid": u["mid"], "uid": k[0], "locale": k[1], "stratum": u["dataset"],
                        "text_sha256": sha256(u["gold"]["text"])} for u, k in zip(units, keys)],
           "values": counts, "dropped_not_whole_word": dict(sorted(dropped.items())),
           "merged_duplicate_values": dict(sorted(merged.items())), "arm_input_sha256": write_input(units, src)}
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def load(parquet: Path = PARQUET, frozen: Path = FROZEN, src: Path = INPUT,
         sha: Optional[str] = SOURCE["sha256"]) -> dict:
    """``{units, src, input_sha, corpus}`` rebuilt from the raw file, refusing
    any record or input whose hash moved from the committed description."""
    if not frozen.exists():
        raise SystemExit(f"{frozen.name} does not exist: build the sample first (--build)")
    doc = json.loads(frozen.read_text())
    if doc["label_map"] != {k: list(v) for k, v in sorted(LABEL_MAP.items())}:
        raise SystemExit("LABEL_MAP differs from the one the sample was built with")
    rows = read_raw(parquet, sha)
    keys = [(r["uid"], r["locale"]) for r in doc["records"]]
    units, _d, _m = units_of(rows, keys)
    for u, r in zip(units, doc["records"]):
        if u["mid"] != r["mid"] or sha256(u["gold"]["text"]) != r["text_sha256"]:
            raise SystemExit(f"{r['mid']}: its text differs from the committed hash")
    input_sha = write_input(units, src)
    if input_sha != doc["arm_input_sha256"]:
        raise SystemExit("the external arm input differs from the committed hash")
    return {"units": units, "src": src, "input_sha": input_sha,
            "corpus": {"records": len(units), "strata": doc["strata"], "values": doc["values"],
                       "dropped_not_whole_word": doc["dropped_not_whole_word"],
                       "merged_duplicate_values": doc["merged_duplicate_values"], "arm_input_sha256": input_sha},
            "description": doc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", required=True)
    ap.parse_args(argv)
    doc = build()
    print(f"{NAME}: {len(doc['records'])} records {doc['strata']}; protect {doc['values']['protect']}; "
          f"dropped {sum(doc['dropped_not_whole_word'].values())}, merged {sum(doc['merged_duplicate_values'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
