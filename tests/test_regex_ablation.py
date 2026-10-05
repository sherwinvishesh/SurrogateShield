"""Model-free tests for E6 (A1, H6): bench/arms/ss_ablate.py (families,
conditions, the regex perturbation, patching PatternScan and restoring it,
span files per condition) and bench/realdata/regex_ablation.py (attribution,
H6, an end-to-end run on a synthetic benchmark with stub spans). PatternScan
is pure regex, so the real one is used; no model loads. Texts and values are
invented."""

import json
import os
import re
from pathlib import Path

import pytest

from bench.arms import ss_ablate as A
from bench.realdata import regex_ablation as E
from bench.realdata import score as S
from bench.realdata.common import file_sha256
from tests.test_realdata_score import DS, _gold, _unit, make_bench

PS = A._ps()


def _found(text):
    return sorted({e.type for e in PS.scan(text)})


# ── families and conditions ──────────────────────────────────────────────────

def test_every_regex_is_in_exactly_one_family():
    fams = A.families(PS)
    idx = sorted(i for v in fams.values() for i in v)
    assert idx == list(range(len(PS._PATTERNS)))
    assert {"phone", "postcode", "url", "address", "email"} <= set(fams)
    assert not {"phone_us", "phone_uk", "zip_us"} & set(fams)
    assert fams["url"] == fams["address"] == []


def test_conditions_are_named_once_and_draws_are_seeded():
    fams = list(A.families(PS))
    conds = A.conditions(fams, 11)
    names = [c["name"] for c in conds]
    assert len(names) == len(set(names)) == 1 + 2 * len(fams) + 2 * A.DRAWS + 3
    assert names[0] == "none" and conds[0]["drop"] == conds[0]["wrong"] == []
    d25 = [c for c in conds if c["name"].startswith("drop25-")]
    d50 = [c for c in conds if c["name"].startswith("drop50-")]
    assert {len(c["drop"]) for c in d25} == {round(0.25 * len(fams))}
    assert {len(c["drop"]) for c in d50} == {round(0.5 * len(fams))}
    assert conds == A.conditions(fams, 11) and conds != A.conditions(fams, 12)
    assert next(c for c in conds if c["name"] == "wrong-all")["wrong"] == fams


# ── the perturbation ─────────────────────────────────────────────────────────

def _m(rx, s):
    return bool(rx.search(s))


def test_lengths_make_counted_groups_one_longer():
    rx = A.perturb(re.compile(r"\b\d{3}-\d{2,4}\b"), "lengths")
    assert not _m(rx, "123-4567") and _m(rx, "1234-56789") and _m(rx, "1234-567")
    rx = A.perturb(re.compile(r"\b[a-z]+\d?\b"), "lengths")          # +, ? untouched
    assert _m(rx, "abc1") and _m(rx, "a")


def test_separators_swap_literals_and_classes_but_never_add_whitespace():
    rx = A.perturb(re.compile(r"\b\d{3}-\d{4}\b"), "separators")
    assert _m(rx, "123.4567") and not _m(rx, "123-4567")
    rx = A.perturb(re.compile(r"\b\d{3}[\s.-]\d{4}\b"), "separators")
    assert _m(rx, "123/4567") and _m(rx, "123_4567")
    assert not any(_m(rx, s) for s in ("123 4567", "123.4567", "123-4567"))
    rx = A.perturb(re.compile(r"[A-Za-z0-9._-]+@x\.org"), "separators")
    assert rx.fullmatch("a/b@x/org") and not rx.fullmatch("a.b@x.org") and not rx.fullmatch("a b@x/org")


def test_word_gaps_negated_classes_and_lookarounds_stay():
    rx = A.perturb(re.compile(r"(?<![\d.-])code\s+is\s+[^\s.-]+"), "both")
    assert _m(rx, "the code is X-1.2") and not _m(rx, "9code is X")


def test_only_the_value_group_changes():
    rx = re.compile(r"(?i)ssn[\s_-]*(?:no\.)?\s*(?P<v>\d{3}-\d{2}-\d{4})")
    q = A.perturb(rx, "separators", A.value_groups(rx, "ssn", PS))
    assert _m(q, "SSN no. 123.45.6789") and _m(q, "ssn_no. 123.45.6789") and not _m(q, "SSN 123-45-6789")
    assert A.value_groups(re.compile(r"\d+"), "ssn", PS) is None
    assert A.value_groups(re.compile(r"passport (\w+)"), "passport", PS) == {1}


def test_every_family_still_compiles_when_perturbed():
    for t, rx, _v in PS._PATTERNS:
        for how in ("lengths", "separators", "both"):
            q = A.perturb(rx, how, A.value_groups(rx, t, PS))
            assert q.groupindex == rx.groupindex
            q.search("Call 555-123-4567 or mail a.b@example.org, SSN 123-45-6789, born 12/03/1990.")


# ── patching PatternScan ─────────────────────────────────────────────────────

TEXT = "Mail ada.byrne@example.org or visit 742 Evergreen Terrace, Springfield, IL 62704."


def test_drop_removes_a_family_and_exit_restores_everything():
    before = (PS._PATTERNS, PS.find_urls, PS._URL_RE, PS.address_parser, PS._table_age_cells)
    assert {"email", "address"} <= set(_found(TEXT))
    fams = list(A.families(PS))
    cond = {"name": "x", "drop": ["email", "address", "url", "age"], "wrong": [], "how": None}
    with A.applied(cond, PS):
        found = _found(TEXT)
        assert "email" not in found and "address" not in found
        assert PS.find_urls("see https://example.org/x") == []
        assert PS.address_parser.find_addresses is not None
        assert PS.address_parser._STREET_ADDRESS_RE is before[3]._STREET_ADDRESS_RE   # forwarded
    after = (PS._PATTERNS, PS.find_urls, PS._URL_RE, PS.address_parser, PS._table_age_cells)
    assert all(a is b for a, b in zip(before, after)) and len(fams) == len(A.families(PS))
    assert {"email", "address"} <= set(_found(TEXT))


def test_dropping_address_in_patternscan_leaves_the_parser_itself():
    ap = PS.address_parser
    with A.applied({"name": "x", "drop": ["address"], "wrong": [], "how": None}, PS):
        assert ap.find_addresses(TEXT)                    # the service-query check still parses
        assert "address" not in _found(TEXT)


def test_wrong_changes_matches_and_restores_module_globals():
    text = "SSN 123-45-6789 please."
    ap = PS.address_parser
    street = ap._STREET_ADDRESS_RE
    assert "ssn" in _found(text)
    with A.applied({"name": "x", "drop": [], "wrong": ["ssn", "address"], "how": "both"}, PS):
        assert "ssn" not in _found(text)
        PS.scan(TEXT)
        assert ap._STREET_ADDRESS_RE is street            # swapped only during the call
    assert "ssn" in _found(text) and ap._STREET_ADDRESS_RE is street


def test_an_exception_inside_still_restores():
    before = PS._PATTERNS
    with pytest.raises(ValueError):
        with A.applied({"name": "x", "drop": ["email"], "wrong": ["phone"], "how": "both"}, PS):
            raise ValueError
    assert PS._PATTERNS is before


# ── span files per condition ─────────────────────────────────────────────────

def _scan_fn(text, seed):
    return [[e.start, e.end, e.type, "Q" * 3, e.source] for e in PS.scan(text)]


def test_run_writes_one_private_span_file_per_condition(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": "a", "text": TEXT}) + "\n" + json.dumps({"id": "b", "text": "hello"}) + "\n")
    out = tmp_path / "out"
    info = A.run(src, out, ["none", "drop-email"], fn=_scan_fn, seed=5)
    assert set(info["summary"]) == {"none", "drop-email"}
    rows = {c: [json.loads(l) for l in open(out / f"{c}.jsonl")] for c in ("none", "drop-email")}
    assert len(rows["none"][0]["edits"]) == len(rows["drop-email"][0]["edits"]) + 1
    assert rows["none"][0]["sources"] == ["pattern"] * len(rows["none"][0]["edits"])
    meta = json.loads((out / "none.jsonl.meta.json").read_text())
    assert meta["input_sha256"] == file_sha256(src) and meta["seed"] == 5 and meta["condition"]["name"] == "none"
    assert (os.stat(out / "none.jsonl").st_mode & 0o777) == 0o600
    with pytest.raises(SystemExit, match="unknown"):
        A.run(src, out, ["drop-nothing"], fn=_scan_fn, seed=5)


# ── the scorer ───────────────────────────────────────────────────────────────

def test_value_status_and_attribution():
    text = "Ada Byrne, 0161 496 0000."
    g = _gold("m", text, protect=[{"value": "Ada Byrne", "type": "PERSON"}, {"value": "0161 496 0000", "type": "PHONE"}])
    u = _unit(g)
    base = {"edits": [[0, 9, "T", "Lou Park"], [11, 24, "T", "0999"]], "sources": ["ner", "pattern"]}
    lost = {"edits": [[0, 9, "T", "Lou Park"]], "sources": ["ner"]}
    rec = {"edits": [[0, 9, "T", "Lou Park"], [11, 24, "T", "0999"]], "sources": ["ner", "slm"]}
    sb, sl, sr = (E.value_status(u, r) for r in (base, lost, rec))
    assert [v["by"] for v in sb] == ["ner", "pattern"] and not any(v["leaked"] for v in sb)
    assert sl[1]["leaked"] and sl[1]["by"] == "-"
    a = E.attribution([sb], [sl])
    assert a["pattern_caught_in_none"] == 1 and a["under_condition"] == {"leaked": 1} and a["lost"] == {"PHONE": 1}
    a = E.attribution([sb], [sr])
    assert a["under_condition"] == {"slm": 1} and a["recovered_by_other_stage"] == 1 and a["lost_total"] == 0
    refused = E.value_status(u, {"edits": [], "refused": "x"})
    assert [v["by"] for v in refused] == ["refused", "refused"] and not any(v["leaked"] for v in refused)


NAME = r"[A-Z][a-z]+ [A-Z][a-z]+"
DIGITS = r"\d[\d ]*\d"
SPELLED = r"ada dot \S+ at \S+ dot test"
STUB = {  # condition -> [(regex, source)]
    "none": [(NAME, "ner"), (DIGITS, "pattern"), (SPELLED, "pattern")],
    "drop-phone": [(NAME, "ner"), (SPELLED, "pattern")],
    "wrong-all": [(NAME, "ner"), (DIGITS, "slm"), (SPELLED, "ner")],
}


def _write(path, src, rows, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(str(path) + ".meta.json").write_text(json.dumps({"input_sha256": file_sha256(src), **meta}))


def _rows(src, rules):
    out = []
    for line in open(src):
        m = json.loads(line)
        hits = sorted((x.start(), x.end(), s) for rx, s in rules for x in re.finditer(rx, m["text"]))
        out.append({"id": m["id"], "edits": [[a, b, "T", "Q" * 3] for a, b, _ in hits], "sources": [s for *_, s in hits]})
    return out


def _runner(spans, pd_rules):
    def run(src, out_dir, seed):
        for c, rules in STUB.items():
            _write(out_dir / f"{c}.jsonl", src, _rows(src, rules), seed=seed, condition={"name": c})
        (out_dir / "run.json").write_text(json.dumps({"families": {"phone": 6}, "versions": {},
                                                      "conditions": [{"name": c} for c in STUB]}))
        name = out_dir.name
        _write(spans / "presidio_default" / f"{name}.jsonl", src, _rows(src, pd_rules))
        _write(spans / "ss" / f"{name}.jsonl", src, [{k: r[k] for k in ("id", "edits")} for r in _rows(src, STUB["none"])])
    return run


def _ablate(tmp_path, pd_rules=(("Python", "x"),), reuse=False):
    rd, build, spans, frozen = make_bench(tmp_path) if not (tmp_path / "rd").exists() else (
        tmp_path / "rd", tmp_path / "build", tmp_path / "spans",
        {f"{DS}/{n}": file_sha256(tmp_path / "rd" / DS / n) for n in ("dev.jsonl", "pool.jsonl", "labels.jsonl")})
    return E.ablate_split("dev", [DS], reuse, out=tmp_path / "res" / "realdata_regex_ablation_dev.json", rd=rd,
                          build=build, frozen=frozen, runner=_runner(spans, list(pd_rules)), spans=spans,
                          log=lambda *_: None)


def test_end_to_end_attribution_h6_and_byte_identical_rerun(tmp_path):
    doc = _ablate(tmp_path)
    r = doc["results"][DS]["injected"]
    assert r["none"]["leak"] == S.rate(0, 3) and r["drop-phone"]["leak"] == S.rate(1, 3)
    assert r["drop-phone"]["attribution"]["under_condition"] == {"leaked": 1, "pattern": 1}
    assert r["drop-phone"]["attribution"]["lost"] == {"PHONE": 1}
    assert r["wrong-all"]["attribution"]["under_condition"] == {"ner": 1, "slm": 1}
    assert r["wrong-all"]["leak"] == S.rate(0, 3) and r["drop-phone"]["vs_none"]["diff"] == round(1 / 3, 4)
    assert "vs_none" not in r["none"] and doc["none_vs_ss_messages_differing"] == {DS: 0}
    assert doc["reference"][DS]["injected"]["leak"] == S.rate(3, 3)
    h = doc["H6"][DS]
    assert h["H6"] and h["worst_condition"] == "drop-phone" and h["conditions_at_or_above"] == []
    assert doc["command"].split()[-1].endswith("realdata_regex_ablation_dev.json")
    out = tmp_path / "res" / "realdata_regex_ablation_dev.json"
    first = out.read_bytes()
    blob = first.decode() + out.with_suffix(".md").read_text()
    for t in ("Ada Byrne", "0161 496 0000", "Ola Nwosu", "mailbox", "bake bread"):
        assert t not in blob
    _ablate(tmp_path)
    assert out.read_bytes() == first
    _ablate(tmp_path, reuse=True)
    assert out.read_bytes() == first


def test_h6_fails_when_the_reference_leaks_less(tmp_path):
    doc = _ablate(tmp_path, pd_rules=((NAME, "x"), (DIGITS, "x"), (SPELLED, "x")))
    h = doc["H6"][DS]
    assert h["presidio_default_leak"] == 0.0 and not h["H6"]
    assert h["conditions_at_or_above"] == ["drop-phone", "none", "wrong-all"]


def test_reuse_refuses_another_draw_seed(tmp_path):
    _ablate(tmp_path)
    meta = tmp_path / "spans" / "ss_ablate" / "dev-oasst1" / "none.jsonl.meta.json"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "seed": 1}))
    with pytest.raises(SystemExit, match="seed"):
        _ablate(tmp_path, reuse=True)
