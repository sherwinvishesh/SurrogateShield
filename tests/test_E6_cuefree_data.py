"""Campaign E6 (data VERSION 3): names without a cue in the tagger's training
messages, and fiction / essay bodies whose names are no one's. Model-free."""

import re
from collections import Counter

import pytest

from bench.realdata import identities as ids
from bench.tagger import bank
from bench.tagger import data as D

CUE_TEMPLATES = [*bank.CUE_START, *bank.CUE_POSSESSIVE, *bank.CUE_LABEL, *bank.CUE_ASIDE, *bank.CUE_BARE,
                 *bank.CUE_LIST, *bank.CUE_MDFIELD]          # the title and pair families teach a shape, cue or not
NEGATIVE_BODIES = [*bank.FICTION_BODIES, *bank.ESSAY_BODIES]


def _built(layout, n, seed):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D, "LAYOUTS", {layout: 1.0})
        return D.build(n, seed)


@pytest.fixture(scope="module")
def cuefree():
    return _built("cuefree", 400, 4245)


@pytest.fixture(scope="module")
def none_layout():
    return _built("none", 300, 4246)


def test_version_and_every_cue_form_appears(cuefree):
    assert D.VERSION == 3 and D.LAYOUTS["cuefree"] > 0
    assert {r["meta"]["cue_form"] for r in cuefree} == set(D.CUE_LAYOUTS)
    assert all(any(t == "PERSON" for _s, _e, t in r["spans"]) for r in cuefree)


def test_no_cue_template_announces_the_name():
    # the point of the layout: none of these says "my name is", "I'm", "called"
    for t in CUE_TEMPLATES:
        assert not re.search(r"(?i)\bmy name\b|\bI'?a?m \{v\}(?!')|\bcalled\b|\bname is\b", t), t


def test_spans_are_clean_train_pool_only_and_a_title_belongs_to_the_span(cuefree):
    titled = 0
    for r in cuefree:
        prev = 0
        for s, e, t in r["spans"]:
            v = r["text"][s:e]
            assert 0 <= prev <= s < e <= len(r["text"]) and t in D.TAGS and v == v.strip(), (r["id"], t)
            prev = e
            if t == "PERSON" and re.match(r"(?i)^(mr|mrs|ms|mx|dr|prof|miss|herr|mme|sr)\.? ?[^\W\d_]", v) \
                    and r["meta"]["cue_form"] in ("title", "pair"):
                titled += 1
        assert ids.in_pool(r["meta"]["pool_tokens"], "train"), r["id"]
        assert not ids.FORBIDDEN.search(r["text"]), r["id"]
    assert titled >= 20
    # the title is not left outside the span
    for r in cuefree:
        for s, e, t in r["spans"]:
            if t == "PERSON":
                assert not re.search(r"(?i)(?:^|[^\w])(?:mr|mrs|ms|dr|prof)\.? ?$", r["text"][:s]), (r["id"], r["text"][max(0, s - 6):e])


def test_a_bare_sign_off_closes_the_message_under_a_blank_line(cuefree):
    n = 0
    for r in (r for r in cuefree if r["meta"]["cue_form"] == "bare"):
        s, e, t = [sp for sp in r["spans"] if sp[2] == "PERSON"][-1]      # the sign-off's name is the last
        before = re.sub(r"[`*_]", "", r["text"][:s])
        assert re.search(r"\n+(?:- |~ |–)?$", before), (r["id"], before[-12:])
        assert not re.search(r"(?i)(regards|thanks|cheers|best|sincerely),?\s*$", before), r["id"]   # no sign-off phrase
        n += 1
    assert n >= 30


def test_a_titled_name_has_no_space_after_a_dotted_title_some_of_the_time(cuefree):
    forms = Counter()
    for r in cuefree:
        for s, e, t in r["spans"]:
            v = r["text"][s:e]
            m = re.match(r"(?i)^(mr|mrs|ms|mx|dr|prof|sr)(\.?)( ?)", v)
            if t == "PERSON" and m and r["meta"]["cue_form"] in ("title", "pair"):
                forms["nospace" if m.group(2) == "." and not m.group(3) else "space"] += 1
    assert forms["nospace"] >= 5 and forms["space"] >= 5, forms


def test_fiction_and_essay_bodies_carry_no_span_and_their_names_are_train_pool(none_layout):
    rng = __import__("random").Random(3)
    for _ in range(300):
        pieces = D.fill(rng, rng.choice(NEGATIVE_BODIES))
        assert all(p.tag is None for p in pieces)
    for slot in ("FICTION2", "SPEAKER", "SURNAME"):
        for _ in range(200):
            name = D.negative(rng, slot)
            assert ids.in_pool(ids.name_keys(name), "train"), (slot, name)
    assert sum(not r["spans"] for r in none_layout) >= .7 * len(none_layout)   # the rest: a third party's name
    kinds = Counter()
    for r in none_layout:
        kinds["screenplay"] += bool(re.search(r"\n[A-Z][A-Z' ]+\n\(|\n[A-Z][A-Z' ]+: ", r["text"]))
        kinds["citation"] += bool(re.search(r"\(\d{4}\)|et al\.|\(\w+ (?:and|&) \w+, \d{4}\)", r["text"]))
    assert kinds["screenplay"] >= 5 and kinds["citation"] >= 5, kinds


def test_the_slots_of_the_negative_bodies_are_known():
    for t in NEGATIVE_BODIES + list(bank.NEGATIVE_ONLY):
        for slot in re.findall(r"\{([A-Z0-9]+)\}", t):
            assert D.SLOT.fullmatch("{" + slot + "}"), (slot, t)


def test_build_keeps_one_copy_of_a_text():
    rows = D.build(600, 4247)
    assert len({r["text"] for r in rows}) == 600
    assert [r["id"] for r in rows] == [f"tg-{i:06d}" for i in range(600)]


def test_the_negatives_augmentation_draws_from_every_pool():
    rng = __import__("random").Random(5)
    seen = Counter()
    for _ in range(300):
        text = "".join(p.text for p in D.negative_body(rng))
        seen["fiction"] += any(text.startswith(b.split("{")[0]) and b.split("{")[0] for b in bank.FICTION_BODIES)
        seen["essay"] += any(text.startswith(b.split("{")[0]) and b.split("{")[0] for b in bank.ESSAY_BODIES)
        seen["plain"] += any(text.startswith(b.split("{")[0]) and b.split("{")[0] for b in bank.NEGATIVE_ONLY)
    assert all(seen[k] >= 20 for k in ("fiction", "essay", "plain")), seen
