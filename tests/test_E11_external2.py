"""Test-4 option E / campaign E11: a second Nemotron-PII draw, disjoint from
the first by record group, under its own name and seed. Model-free: invented
records in a parquet shaped like Nemotron-PII's; nothing is scored."""

import json
import random
from collections import defaultdict

import pytest

from bench.realdata import external as E
from bench.realdata import score as S
from bench.realdata.common import derive_seed, file_sha256

SECOND = "nemotron_pii_2"


def _sp(text, value, label):
    a = text.index(value)
    return {"start": a, "end": a + len(value), "text": value, "label": label}


def _parquet(path, n=12):
    """n uids × (us, intl), alternating document formats; invented values."""
    import pandas as pd
    rows = []
    for i in range(n):
        for loc in ("us", "intl"):
            text = f"Ticket {i}{loc}: Lou Park wrote from lou{i}{loc}@example.org on Monday."
            spans = [_sp(text, "Lou", "first_name"), _sp(text, "Park", "last_name"),
                     _sp(text, f"lou{i}{loc}@example.org", "email")]
            rows.append({"uid": f"u{i:03d}", "locale": loc, "document_format": "structured" if i % 2 else "unstructured",
                         "domain": "Banking", "text": text, "spans": repr(spans)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def raw(tmp_path):
    return _parquet(tmp_path / "raw" / "test.parquet")


def _paths(tmp_path, name):
    return dict(frozen=tmp_path / "external" / f"{name}.json", src=tmp_path / "build" / name / "input.jsonl")


def _both(tmp_path, raw, per_stratum=2):
    first = E.build(raw, per_stratum=per_stratum, sha=None, **_paths(tmp_path, E.NAME))
    others = {E.NAME: _paths(tmp_path, E.NAME)["frozen"]}
    second = E.build(raw, per_stratum=per_stratum, sha=None, name=SECOND, others=others, **_paths(tmp_path, SECOND))
    return first, second, others


def _old_sample(rows, per_stratum):
    """The draw as it was before the name parameter existed."""
    by = defaultdict(list)
    for key, r in rows.items():
        by[E.stratum(r)].append(key)
    keys = []
    for s in sorted(by):
        pool = sorted(by[s])
        keys += sorted(random.Random(derive_seed("external", "nemotron_pii", s)).sample(pool, min(per_stratum, len(pool))))
    return keys


def test_the_first_sample_draws_exactly_as_before(raw):
    rows = E.read_raw(raw, None)
    for k in (1, 2, 3, 50):
        assert E.sample(rows, k) == _old_sample(rows, k)
        assert E.sample(rows, k, E.NAME, frozenset()) == _old_sample(rows, k)
    assert E.frozen_path() == E.FROZEN and E.input_path() == E.INPUT
    assert E.SAMPLES[E.NAME] == {"id_prefix": E.ID_PREFIX, "exclude": ()}


def test_the_second_draw_shares_no_record_group_with_the_first(tmp_path, raw):
    first, second, _ = _both(tmp_path, raw)
    a, b = {r["uid"] for r in first["records"]}, {r["uid"] for r in second["records"]}
    assert a and b and not (a & b)
    assert second["strata"] == first["strata"] == {"intl-structured": 2, "intl-unstructured": 2,
                                                    "us-structured": 2, "us-unstructured": 2}
    assert [r["mid"] for r in second["records"]] == [f"ext-nemotron2-{i:04d}" for i in range(1, 9)]
    assert second["name"] == SECOND and second["seed"] == "derive_seed('external', 'nemotron_pii_2', stratum)"
    assert second["excluded"] == {"samples": [E.NAME], "uids": len(a)} and "excluded" not in first
    text = _paths(tmp_path, SECOND)["frozen"].read_text()
    assert "example.org" not in text and "Ticket" not in text and "Park" not in text
    assert second["arm_input_sha256"] == file_sha256(_paths(tmp_path, SECOND)["src"])


def test_the_second_draw_needs_the_first_description(tmp_path, raw):
    with pytest.raises(SystemExit, match="excludes nemotron_pii, whose description"):
        E.build(raw, per_stratum=2, sha=None, name=SECOND, others={E.NAME: tmp_path / "missing.json"},
                **_paths(tmp_path, SECOND))
    assert not _paths(tmp_path, SECOND)["frozen"].exists()


def test_each_sample_is_drawn_once_and_an_unknown_name_is_refused(tmp_path, raw):
    _, _, others = _both(tmp_path, raw)
    with pytest.raises(SystemExit, match="drawn once"):
        E.build(raw, per_stratum=2, sha=None, name=SECOND, others=others, **_paths(tmp_path, SECOND))
    with pytest.raises(SystemExit, match="no external sample 'nemotron_pii_3'"):
        E.build(raw, per_stratum=2, sha=None, name="nemotron_pii_3", **_paths(tmp_path, "nemotron_pii_3"))


def test_load_rebuilds_the_second_draw_and_refuses_a_shared_group(tmp_path, raw):
    first, second, others = _both(tmp_path, raw)
    got = E.load(raw, sha=None, name=SECOND, others=others, **_paths(tmp_path, SECOND))
    assert [u["mid"] for u in got["units"]] == [r["mid"] for r in second["records"]]
    assert got["input_sha"] == second["arm_input_sha256"]
    path = _paths(tmp_path, SECOND)["frozen"]
    doc = json.loads(path.read_text())
    doc["records"][0]["uid"] = first["records"][0]["uid"]
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="1 record groups also belong to nemotron_pii"):
        E.load(raw, sha=None, name=SECOND, others=others, **_paths(tmp_path, SECOND))


def test_check_counts_what_is_left_from_the_key_columns_only(tmp_path, raw, monkeypatch):
    first = E.build(raw, per_stratum=2, sha=None, **_paths(tmp_path, E.NAME))
    others = {E.NAME: _paths(tmp_path, E.NAME)["frozen"]}
    import pandas as pd
    seen, real = [], pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, columns=None, **kw: seen.append(columns) or real(p, columns=columns, **kw))
    a = E.available(SECOND, raw, None, others, per_stratum=2)
    assert seen == [["uid", "locale", "document_format"]]                       # no text, no spans
    # 12 uids (6 per format), each in both locales; a uid the first sample drew in
    # either locale is gone from both of its strata
    gone = {fmt: {r["uid"] for r in first["records"] if r["stratum"].endswith(f"-{fmt}")}
            for fmt in ("structured", "unstructured")}
    assert a["records"] == 24 and a["excluded_uids"] == len(gone["structured"] | gone["unstructured"])
    assert a["left"] == {f"{loc}-{fmt}": 6 - len(gone[fmt]) for loc in ("intl", "us") for fmt in gone}
    assert a["enough"] is True                                                  # at least 2 left in each stratum
    assert E.available(SECOND, raw, None, others, per_stratum=5)["enough"] is False
    assert E.available(E.NAME, raw, None)["left"]["us-structured"] == 6          # the first sample excludes nothing


def test_score_external_knows_the_second_name_and_loads_it(tmp_path, monkeypatch):
    hyp, freeze = tmp_path / "HYPOTHESES_TEST3.md", tmp_path / "FREEZE.json"
    hyp.write_text("H10''' ...\n")
    freeze.write_text(json.dumps({"hypotheses_sha256": file_sha256(hyp)}))
    asked = []

    class Loaded(Exception):
        pass

    def fake_load(**kw):
        asked.append(kw)
        raise Loaded
    monkeypatch.setattr(E, "load", fake_load)
    with pytest.raises(Loaded):
        S.score_external(SECOND, ("ss",), freeze=freeze, prereg=hyp, log=lambda *_: None)
    assert asked == [{"name": SECOND}]
    with pytest.raises(SystemExit, match="no external benchmark 'nemotron_pii_9'"):
        S.score_external("nemotron_pii_9", ("ss",), freeze=freeze, prereg=hyp, log=lambda *_: None)
