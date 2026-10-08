"""V4 §3.2: the AGE probe builds synthetic messages from the training half of
the identity pools only and scores an age by the scorer's leak rule at its
known span. Model-free: the tagger and SS parts are not run here."""

import json

import pytest

from bench.realdata import identities as ids
from bench.tagger import age_probe as A

SMALL = {lay: 4 for lay in A.LAYOUTS}


@pytest.fixture(scope="module")
def msgs():
    return A.build(sizes=SMALL)


def test_layouts_split_half_trained_half_new_and_the_full_probe_has_about_600():
    assert sum(A.TRAINED.values()) == 300 and sum(A.NEW.values()) == 302
    assert set(A.TRAINED).isdisjoint(A.NEW) and set(A.SIGN_OFF) <= set(A.NEW)
    assert set(A.TEMPLATES) | {"signoff"} == set(A.NEW)


def test_every_message_holds_its_age_at_its_span_and_is_deterministic(msgs):
    assert len(msgs) == 4 * len(A.LAYOUTS) and [m["id"] for m in msgs] == [f"ap-{i:04d}" for i in range(len(msgs))]
    for m in msgs:
        s, e = m["age"]
        assert m["text"][s:e] == m["value"] and m["trained"] == (m["layout"] in A.TRAINED)
        assert m["worded"] == (not m["value"].isdigit())
    assert all(m["worded"] for m in msgs if m["layout"] == "worded")
    assert A.messages_sha(msgs) == A.messages_sha(A.build(sizes=SMALL))
    assert A.messages_sha(msgs) != A.messages_sha(A.build(seed=1, sizes=SMALL))


def test_names_and_cities_come_from_the_training_half_only(msgs):
    toks = sorted({t for m in msgs for t in m["pool_tokens"]})
    assert toks and ids.in_pool(toks, "train") and not any(ids.in_pool([t], "eval") for t in toks)
    assert not any(ids.FORBIDDEN.search(m["text"]) for m in msgs)
    named = [m for m in msgs if m["layout"] in A.NEW and m["layout"] != "reddit"]
    assert all(m["non_ascii"] in (True, False) and m["pool_tokens"] for m in named)
    assert all(m["non_ascii"] is None for m in msgs if m["layout"] in A.TRAINED or m["layout"] == "reddit")


def test_a_name_of_the_wanted_script_is_drawn():
    import random
    rng = random.Random(3)
    for want in (True, False):
        c = A._person(rng, want)
        assert (not f"{c.first} {c.last}".isascii()) == want


def test_sign_off_ages_close_the_line_or_the_message(msgs):
    for m in msgs:
        e = m["age"][1]
        if m["layout"] == "signoff":
            assert m["text"][e:] in ("", ".")
        if m["layout"] == "eol":
            assert m["text"][e] == "\n"


def test_caught_is_the_scorers_leak_rule_at_the_span():
    m = {"text": "Thanks,\nAna Pop, 34", "age": [17, 19], "value": "34"}
    assert A.caught(m, [[17, 19, "AGE", "[AGE]"]]) and A.caught(m, [[8, 19, "PERSON", "Ion Ene, 35"]])
    assert not A.caught(m, [[8, 15, "PERSON", "Ion Ene"]]) and not A.caught(m, [[17, 18, "AGE", "3"]])
    assert A.caught(m, [[17, 19, "AGE", "34"]])                          # a 2-character value is never counted inside
    w = {"text": "She is thirty years old.", "age": [7, 13], "value": "thirty"}
    assert not A.caught(w, [[7, 13, "AGE", "thirty-one"]])              # the C bug: the original inside the surrogate
    assert A.caught(w, [[7, 13, "AGE", "twenty-eight"]])
    r = {"text": "Me (39F) and", "age": [4, 6], "value": "39"}             # glued to a letter: the span, not occurrences
    assert A.caught(r, [[4, 7, "AGE", "41F"]]) and not A.caught(r, [])


def test_summary_counts_recall_per_layout_and_pools(msgs):
    hit = {m["id"]: m["layout"] in ("signoff", "form") for m in msgs}
    s = A.summary(msgs, hit)
    assert s["by_layout"]["signoff"] == {"k": 4, "n": 4, "rate": 1.0, "wilson95": s["by_layout"]["signoff"]["wilson95"]}
    assert s["by_layout"]["eol"]["k"] == 0 and s["sign_off"]["k"] == 4 and s["sign_off"]["n"] == 8
    assert (s["trained"]["k"], s["trained"]["n"], s["new"]["k"], s["new"]["n"]) == (4, 16, 4, 32)
    named = [m for m in msgs if m["non_ascii"] is not None]
    assert s["by_script"]["ascii"]["n"] + s["by_script"]["non_ascii"]["n"] == len(named)
    assert s["all"]["k"] == 8


def test_markdown_lists_layouts_tagger_thresholds_and_ss_variants(msgs):
    hit = {m["id"]: True for m in msgs}
    part = A.summary(msgs, hit)
    doc = {"command": "c --out bench/results/age_probe_dev.json", "git": {"commit": "a" * 40}, "n": len(msgs),
           "seed": 1, "messages_sha256": "b" * 64, "layouts": A.layouts_doc(msgs),
           "tagger": {"model": "m", "thresholds": {t: {"any": part, "as_age": part} for t in ("0.5", "0.7", "0.9")}},
           "ss": {"frozen": {**part, "detection_config_hash": "c" * 64, "code": "d" * 64, "date": "2026-10-08"}}}
    md = A.markdown(doc)
    assert "| signoff | no | 4 |" in md and "SS `frozen`" in md and "tagger as_age @0.7" in md
    assert "sign-off layouts by threshold" in md and "0.5: 1.0" in md
    json.dumps(doc)
