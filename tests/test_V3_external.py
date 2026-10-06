"""V3 §5.4 / H10'' — the external benchmark (``bench/realdata/external.py``)
and its scorer path (``score.py --external``). Model-free: a few invented
records in a parquet shaped like Nemotron-PII's, stub arms."""

import json
import re

import pytest

from bench import realworld as rw
from bench.realdata import external as E
from bench.realdata import score as S
from bench.realdata.common import file_sha256

NEMOTRON_LABELS = (
    "first_name date last_name company_name email url occupation phone_number time country customer_id state city "
    "date_of_birth account_number street_address user_name credit_debit_card date_time medical_record_number "
    "health_plan_beneficiary_number biometric_identifier county bank_routing_number employment_status employee_id "
    "language race_ethnicity education_level age postcode coordinate password ssn pin gender religious_belief "
    "fax_number ipv4 political_view http_cookie swift_bic cvv blood_type vehicle_identifier mac_address api_key "
    "license_plate sexuality ipv6 device_identifier certificate_license_number national_id unique_id tax_id").split()


def test_every_nemotron_label_maps_to_a_j2_list_and_type():
    assert len(NEMOTRON_LABELS) == 55 and set(E.LABEL_MAP) == set(NEMOTRON_LABELS)
    for label, (lst, typ) in E.LABEL_MAP.items():
        assert lst in ("protect", "sensitive", "optional"), label
        assert typ in {"protect": rw.PROTECT_TYPES, "sensitive": rw.SENSITIVE_TYPES, "optional": {None}}[lst], label
    assert {t for lst, t in E.LABEL_MAP.values() if lst == "protect"} == rw.PROTECT_TYPES   # every type is exercised


def _sp(text, value, label, given=None):
    a = text.index(value)
    return {"start": a, "end": a + len(value), "text": value if given is None else given, "label": label}


def test_gold_values_drops_and_merges():
    text = "Ada Byrne, 34, born 1990-01-02, pin 0042, Catholic; tel +1 555 0100 x Adaline."
    spans = [_sp(text, "Ada", "first_name"), _sp(text, "Byrne", "last_name"), _sp(text, "34", "age", given=34),
             _sp(text, "1990-01-02", "date"), _sp(text, "1990-01-02", "date_of_birth"),
             _sp(text, "0042", "pin", given=42), _sp(text, "Catholic", "religious_belief"),
             _sp(text, "Byrne", "last_name"),                                         # a repeat: not counted
             {"start": 0, "end": 3, "text": "Adal", "label": "first_name"},          # glued to "ine": not whole-word
             {"start": 99, "end": 120, "text": "+1 555 0100", "label": "phone_number"}]   # offsets off, text found
    dropped, merged = __import__("collections").Counter(), __import__("collections").Counter()
    g = E.gold(text, spans, dropped, merged)
    assert g["protect"] == [{"value": "Ada", "type": "PERSON"}, {"value": "Byrne", "type": "PERSON"},
                            {"value": "34", "type": "AGE"}, {"value": "1990-01-02", "type": "DATE_OF_BIRTH"},
                            {"value": "0042", "type": "CREDENTIAL"}, {"value": "+1 555 0100", "type": "PHONE"}]
    assert g["sensitive"] == [{"value": "Catholic", "type": "RELIGION"}] and g["optional"] == [] and g["keep"] == []
    assert dict(dropped) == {"first_name": 1} and dict(merged) == {"date_of_birth": 1}   # the date kept as protect
    with pytest.raises(SystemExit, match="unmapped Nemotron-PII label 'shoe_size'"):
        E.gold(text, [_sp(text, "Ada", "shoe_size")], dropped, merged)


_WORDS = ("Ada Byrne", "Lou Park", "Kai Moss", "Eve Hart")


def _parquet(path, n=6):
    """n uids × (us, intl) × the two formats' worth of invented records."""
    import pandas as pd
    rows = []
    for i in range(n):
        for loc in ("us", "intl"):
            name = _WORDS[(i + (loc == "intl")) % len(_WORDS)]
            text = f"Ticket {i}{loc}: {name} wrote from ada{i}@example.org about Python on 2026-01-0{i % 9 + 1}."
            spans = [_sp(text, name.split()[0], "first_name"), _sp(text, name.split()[1], "last_name"),
                     _sp(text, f"ada{i}@example.org", "email"), _sp(text, f"2026-01-0{i % 9 + 1}", "date")]
            rows.append({"uid": f"u{i:03d}", "locale": loc, "document_format": "structured" if i % 2 else "unstructured",
                         "domain": "Banking" if i % 3 else "Credit", "text": text, "spans": repr(spans)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def ext(tmp_path):
    pq = _parquet(tmp_path / "raw" / "test.parquet")
    kw = dict(parquet=pq, frozen=tmp_path / "external" / "x.json", src=tmp_path / "build" / "input.jsonl", sha=None)
    return kw


def test_the_sample_is_stratified_seeded_and_committed_without_text(ext):
    doc = E.build(per_stratum=2, **ext)
    assert doc["strata"] == {"intl-structured": 2, "intl-unstructured": 2, "us-structured": 2, "us-unstructured": 2}
    assert [r["mid"] for r in doc["records"]] == [f"ext-nemotron-{i:04d}" for i in range(1, 9)]
    assert doc["values"]["protect"] == {"EMAIL": 8, "PERSON": 16} and doc["values"]["optional"] == {"-": 8}
    raw = ext["frozen"].read_text()
    assert "example.org" not in raw and "Byrne" not in raw and "Ticket" not in raw
    assert doc["arm_input_sha256"] == file_sha256(ext["src"])
    again = E.sample(E.read_raw(ext["parquet"], None), 2)
    assert again == [(r["uid"], r["locale"]) for r in doc["records"]]          # the same draw from the same seed
    with pytest.raises(SystemExit, match="drawn once"):
        E.build(per_stratum=2, **ext)


def test_load_rebuilds_the_units_and_refuses_moved_data(ext, monkeypatch):
    E.build(per_stratum=2, **ext)
    got = E.load(**ext)
    assert len(got["units"]) == 8 and got["input_sha"] == got["corpus"]["arm_input_sha256"]
    u = got["units"][0]
    assert u["slices"] == ("external",) and u["conv"].startswith("nemotron-u") and u["gold"]["service_query"] is False
    monkeypatch.setitem(E.LABEL_MAP, "date", ("protect", "DATE_OF_BIRTH"))
    with pytest.raises(SystemExit, match="LABEL_MAP differs"):
        E.load(**ext)
    monkeypatch.undo()
    import pandas as pd
    df = pd.read_parquet(ext["parquet"])
    df["text"] = df["text"].str.replace("wrote", "typed")
    df.to_parquet(ext["parquet"])
    with pytest.raises(SystemExit, match="text differs from the committed hash"):
        E.load(**ext)


def _runner(spans, arms_pattern):
    def run(arm, src, name):
        rows = []
        for line in open(src):
            m = json.loads(line)
            rows.append({"id": m["id"], "ms": 1.0, "edits": [[x.start(), x.end(), "T", "Q" * 3]
                                                            for x in re.finditer(arms_pattern[arm], m["text"])]})
        out = spans / arm / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (spans / arm / f"{name}.jsonl.meta.json").write_text(json.dumps({"input_sha256": file_sha256(src)}))
    return run


def test_score_external_is_sealed_until_the_freeze_and_scores_h10(ext, tmp_path):
    E.build(per_stratum=2, **ext)
    loaded = E.load(**ext)
    hyp, freeze = tmp_path / "HYPOTHESES_TEST2.md", tmp_path / "FREEZE.json"
    hyp.write_text("H10'' ...\n")
    pats = {"ss": r"[A-Z][a-z]+ [A-Z][a-z]+|\S+@\S+\.org",           # names and emails: nothing leaks
            "gliner_pii": r"[A-Z][a-z]+ [A-Z][a-z]+|Python",           # emails leak, Python is spurious
            "gliner_pii_tuned": r"\S+@\S+\.org|Python"}
    out = tmp_path / "res" / "external_x.json"
    kw = dict(out=out, loaded=loaded, runner=_runner(tmp_path / "spans", pats), spans=tmp_path / "spans",
              log=lambda *_: None, freeze=freeze, prereg=hyp)
    with pytest.raises(SystemExit, match="only after the freeze"):
        S.score_external("x", tuple(pats), **kw)
    freeze.write_text(json.dumps({"hypotheses_sha256": file_sha256(hyp)}))
    doc = S.score_external("x", tuple(pats), **kw)
    assert doc["command"].endswith(f"--external x --out {out}")
    r = doc["results"]["all"]
    assert r["ss"]["leak"]["k"] == 0 and r["ss"]["spurious"]["k"] == 0 and r["ss"]["span_f1"] == 1.0
    assert r["gliner_pii"]["leak"]["k"] == 8 and r["gliner_pii"]["spurious"]["k"] == 8
    assert r["gliner_pii_tuned"]["leak"]["k"] == 16
    assert set(doc["results"]) == {"intl-structured", "intl-unstructured", "us-structured", "us-unstructured", "all"}
    assert doc["hypotheses"]["H10''"]["gliner_pii"] == {"leak_not_above": True, "span_f1_not_below": True, "holds": True}
    assert doc["differences"]["all"]["gliner_pii"]["leak_rate"]["diff"] < 0
    md = out.with_suffix(".md").read_text()
    assert "| gliner_pii | yes | yes | yes |" in md and "| first_name | protect | PERSON |" in md
    assert "example.org" not in md and "Byrne" not in md and "example.org" not in out.read_text()


def test_span_f1_and_h10_on_toy_aggregates():
    a = {"leak": {"rate": 0.1}, "spurious": {"rate": 0.5}}
    assert S.span_f1(a) == round(2 * 0.5 * 0.9 / 1.4, 4)
    assert S.span_f1({"leak": {"rate": None}, "spurious": {"rate": 0.5}}) is None
    res = {"ss": {"leak": {"rate": 0.2}, "span_f1": 0.7}, "gliner_pii": {"leak": {"rate": 0.1}, "span_f1": 0.6}}
    assert S.h10(res) == {"gliner_pii": {"leak_not_above": False, "span_f1_not_below": True, "holds": True}}
