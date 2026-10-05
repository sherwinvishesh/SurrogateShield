"""Model-free tests for bench/realdata/identities.py (Phase 3 fake identities):
seeded and reproducible, Latin-script, never a real provider's token shape,
valid checksums, no value inside the base text or another value, type quotas,
capacity and exclusivity respected, shifted formats only when asked."""

import json
import random
import re

import pytest

from bench.realdata import identities as I
from bench import realworld as rw

TASKS = ("coding", "writing", "qa", "advice", "business", "roleplay", "translation", "other")
SHIFT_FMTS = {"lower", "upper", "surname-first", "spelled", "dots", "words", "ordinal-of", "month-ordinal",
              "ssn-nodash", "card-spaced", "iban-spaced"}


@pytest.fixture(scope="module")
def many():
    out = []
    for i in range(240):
        task = TASKS[i % len(TASKS)]
        out.append(I.identity(random.Random(i), I.TYPES, task, shift=(i % 3 == 0)))
    return out


def test_identity_is_seeded_and_reproducible():
    a = I.identity(random.Random(5), I.TYPES, "advice", shift=True)
    b = I.identity(random.Random(5), I.TYPES, "advice", shift=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a != I.identity(random.Random(6), I.TYPES, "advice", shift=True)


def test_every_requested_type_gets_a_value_and_id_at_most_two(many):
    for d in many:
        types = [v["type"] for v in d["values"]]
        assert set(types) == set(I.TYPES)          # exclusivity is assign_types' job, not identity's
        assert types.count("ID") in (1, 2) and all(types.count(t) == 1 for t in set(types) - {"ID"})


def test_values_are_latin_never_provider_shaped_and_never_inside_each_other(many):
    for d in many:
        vals = [v["value"] for v in d["values"]]
        assert len(set(vals)) == len(vals)
        for v in vals:
            assert I.latin(v) and not I.FORBIDDEN.search(v) and v == v.strip() and "\n" not in v
            for w in vals:
                if w != v:
                    assert not rw.occurrences(w, v), (v, w)


def test_forbidden_catches_provider_prefixes_and_spares_names():
    for bad in ("xoxb-1", "shpat_1", "ghp_x", "github_pat_x", "sk-ant-x", "sk_live_x", "AKIAX", "eyJhbGci",
                "AIzaSy", "tok_1", "hf_abc", "-----BEGIN RSA", "key=sk-x"):
        assert I.FORBIDDEN.search(bad), bad
    for ok in ("Anastasia", "Saskia Brandt", "zqk_abc", "api_key=" + "a" * 32, "risk-free", "tasks_done"):
        assert not I.FORBIDDEN.search(ok), ok


def test_checksums_and_reserved_ranges(many):
    vals = [(v["fmt"], v["value"]) for d in many for v in d["values"]]
    cards = [re.sub(r"\D", "", v) for f, v in vals if f.startswith("card")]
    ibans = [v.replace(" ", "") for f, v in vals if f.startswith("iban")]
    ssns = [re.sub(r"\D", "", v) for f, v in vals if f.startswith("ssn")]
    nhs = [v.replace(" ", "") for f, v in vals if f == "nhs"]
    assert cards and ibans and ssns
    assert all(I.luhn_ok(c) and len(c) == 16 and c not in I.TEST_CARDS for c in cards)
    assert all(I.iban_ok(x) for x in ibans)
    assert all(s[:3] not in ("000", "666") and not s.startswith("9") and s[3:5] != "00" and s[5:] != "0000"
               for s in ssns)
    for n in nhs:
        r = 11 - sum(int(x) * (10 - i) for i, x in enumerate(n[:9])) % 11
        assert int(n[9]) == (0 if r == 11 else r)
    for f, v in vals:
        if f == "plain" and re.fullmatch(r"\(?\d{3}\)?[ -]\d{3}-\d{4}|\+1 \d{3} \d{3} \d{4}|\d{10}", v):
            assert re.sub(r"\D", "", v)[-7:-4] != "555", v


def test_shifted_formats_only_when_asked():
    plain = [I.identity(random.Random(i), I.TYPES, "advice", shift=False) for i in range(60)]
    shifted = [I.identity(random.Random(i), I.TYPES, "advice", shift=True) for i in range(60)]
    assert not {v["fmt"] for d in plain for v in d["values"]} & SHIFT_FMTS
    person = [v["fmt"] for d in shifted for v in d["values"] if v["type"] == "PERSON"]
    assert set(person) <= {"lower", "upper", "surname-first"} and len(set(person)) == 3
    assert all(d["shift"] for d in shifted) and not any(d["shift"] for d in plain)


def test_no_value_occurs_in_the_base_text():
    text = "Hi, I am 34 and I live in Leeds; please fix this regex for 07700 900123."
    for i in range(80):
        d = I.identity(random.Random(i), ("AGE", "LOCATION", "PERSON"), "qa", text=text)
        assert not any(rw.occurrences(text, v["value"]) for v in d["values"])
    every_age = " ".join(str(n) for n in range(18, 80))
    with pytest.raises(RuntimeError):
        I.identity(random.Random(0), ("AGE",), "qa", text=every_age)


def test_avoid_keeps_values_clear_of_the_base_keep_and_optional_values():
    avoid = ["London", "Python", "Dublin"]
    for i in range(80):
        d = I.identity(random.Random(i), ("ADDRESS", "ORG", "PERSON"), "advice", avoid=avoid)
        for v in d["values"]:
            assert not any(rw.occurrences(v["value"], a) or rw.occurrences(a, v["value"]) for a in avoid), v


def _rows(n=400, seed=3):
    r = random.Random(seed)
    return [{"task": TASKS[i % len(TASKS)], "words": r.choice((5, 12, 25, 60, 140, 400))} for i in range(n)]


def test_assign_types_meets_quota_capacity_and_exclusivity():
    rows = _rows()
    out = I.assign_types(rows, seed=11)
    assert out == I.assign_types(rows, seed=11) and len(out) == len(rows)
    counts = {t: sum(t in o for o in out) for t in I.TYPES}
    assert all(c >= I.TARGET for c in counts.values()), counts
    for r, o in zip(rows, out):
        assert 1 <= len(o) <= I.capacity(r["words"]) and len(set(o)) == len(o)
        for a, b in I.EXCLUSIVE:
            assert not (a in o and b in o)


def test_assign_types_swaps_out_person_when_short_prompts_leave_no_room():
    rows = [{"task": TASKS[i % len(TASKS)], "words": (10, 20, 30, 45)[i % 4]} for i in range(250)]
    out = I.assign_types(rows, seed=4)
    counts = {t: sum(t in o for o in out) for t in I.TYPES}
    assert all(c >= I.TARGET for c in counts.values()), counts
    drawn = sum("PERSON" in o for o in I.assign_types(rows, seed=4, target=0))
    assert 2 * I.TARGET <= counts["PERSON"] < drawn          # names gave way, down to the floor at most
    for r, o in zip(rows, out):
        assert 1 <= len(o) <= I.capacity(r["words"]) and len(set(o)) == len(o)
    for o in out:
        for a, b in I.EXCLUSIVE:
            assert not (a in o and b in o)


def test_capacity_grows_with_length_and_is_bounded():
    assert I.capacity(0) == 2 and I.capacity(40) == 4 and I.capacity(10 ** 4) == 7


def test_type_counts_count_rows_and_values():
    d = [{"values": [{"type": "ID", "value": "a"}, {"type": "ID", "value": "b"}, {"type": "PERSON", "value": "c"}]},
         {"values": [{"type": "PERSON", "value": "d"}]}]
    c = I.type_counts(d)
    assert c["rows"]["ID"] == 1 and c["values"]["ID"] == 2 and c["rows"]["PERSON"] == 2 and c["rows"]["URL"] == 0


def test_number_words():
    assert [I.number_words(n) for n in (7, 13, 40, 58)] == ["seven", "thirteen", "forty", "fifty-eight"]
