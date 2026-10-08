"""V4 §3.1 D: the tagger's training data learns ages of named people.

The "age" layout (bench/tagger/data.py, bank.AGE_*) and AGE in signature
blocks, worded apart from the AGE probe (bench/tagger/age_probe.py), so the
probe goes on measuring the layout rather than these strings. Names, cities
and numbers come from the training half of the identity pool. Model-free."""

import random
import re

import pytest

from bench.realdata import identities as ids
from bench.tagger import age_probe as P
from bench.tagger import bank
from bench.tagger import data as D

D_TEMPLATES = [*bank.AGE_LINES, *bank.AGE_MID, *bank.AGE_BRACKET, *bank.AGE_FROM, *bank.AGE_REDDIT, *bank.AGE_WORDED]
PROBE_TEMPLATES = [t for ts in P.TEMPLATES.values() for t in ts]
SLOT = re.compile(r"\{\w+\}")
WORDS = {ids.number_words(n) for n in range(1, 100)}


def _bare(s):
    """*s* without the marks the markdown augmentation wraps a value in."""
    return re.sub(r"[`*_]", "", s)


def _grams(template, n=4):
    """The template's n-word runs between slots."""
    out = set()
    for seg in SLOT.split(template.lower()):
        w = re.findall(r"[a-z']+", seg)
        out |= {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    return out


def _built(layout, n, seed):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D, "LAYOUTS", {layout: 1.0})
        return D.build(n, seed)


@pytest.fixture(scope="module")
def ages():
    return _built("age", 240, 4243)


@pytest.fixture(scope="module")
def signatures():
    return _built("signature", 240, 4244)


# ── apart from the probe ─────────────────────────────────────────────────────

def test_no_template_is_a_probe_template_or_shares_four_words_with_one():
    shared = {g for t in D_TEMPLATES for g in _grams(t)} & {g for t in PROBE_TEMPLATES for g in _grams(t)}
    assert not set(D_TEMPLATES) & set(PROBE_TEMPLATES) and not shared, shared
    forms = {SLOT.sub("#", f"{who}: {{n}}\n{key}: {{v}}") for who, key in bank.AGE_FORM}
    assert not forms & {SLOT.sub("#", t) for t in PROBE_TEMPLATES}


# ── the "age" layout ─────────────────────────────────────────────────────────

def test_the_layout_is_built_deterministically_and_every_form_appears(ages):
    assert D.VERSION == 2 and D.LAYOUTS["age"] > 0
    assert D.data_hash(_built("age", 240, 4243)) == D.data_hash(ages)
    assert {r["meta"]["age_form"] for r in ages} == set(D.AGE_LAYOUTS)


def test_spans_are_clean_and_every_age_message_has_an_age(ages):
    for r in ages:
        prev = 0
        for s, e, t in r["spans"]:
            v = r["text"][s:e]
            assert 0 <= prev <= s < e <= len(r["text"]) and t in D.TAGS and v == v.strip(), (r["id"], t)
            prev = e
        got = [r["text"][s:e] for s, e, t in r["spans"] if t == "AGE"]
        assert len(got) == sum(t == "AGE" for t, _f in r["meta"]["values"]) >= 1, r["id"]
        for v in got:
            assert re.fullmatch(r"\d{1,2}", v) or v.lower() in WORDS, (r["id"], v)


def test_train_pool_only_and_no_provider_token_shape(ages):
    for r in ages:
        assert ids.in_pool(r["meta"]["pool_tokens"], "train"), r["id"]
        assert not ids.FORBIDDEN.search(r["text"]), r["id"]
        assert not any(tuple(v) in D.HELD_OUT for v in r["meta"]["values"]), r["id"]


def test_a_sign_off_puts_the_age_after_the_name_at_the_line_end(ages):
    n = 0
    for r in (r for r in ages if r["meta"]["age_form"] == "signoff"):
        spans = r["spans"]
        k = max(i for i, sp in enumerate(spans) if sp[2] == "AGE")      # the identity may hold an age too
        (ps, pe, pt), (s, e, _t) = spans[k - 1], spans[k]
        assert pt == "PERSON" and _bare(r["text"][pe:s]) == ", ", r["id"]
        assert _bare(r["text"][e:])[:1] in ("", ".", "\n"), r["id"]
        assert _bare(r["text"][:ps]).endswith("\n"), r["id"]
        n += 1
    assert n >= 30


def test_childrens_forms_hold_childrens_ages():
    rng, seen = random.Random(7), 0
    for _ in range(600):
        pieces, _toks, ages_, kind = D.age_block(rng)
        text = "".join(p.text for p in pieces)
        if text.startswith(bank.AGE_CHILDREN):
            assert all(2 <= int(a["value"]) <= 17 for a in ages_), text
            seen += 1
    assert seen >= 5


def test_a_near_miss_has_no_age():
    rng = random.Random(11)
    for _ in range(500):
        pieces, toks = D.age_near(rng)
        assert not any(p.tag == "AGE" for p in pieces)
        assert any(re.search(r"\d", p.text) and p.tag is None for p in pieces)
        assert ids.in_pool(toks, "train")


def test_near_misses_appear_in_the_layout(ages):
    near = sum(r["meta"].get("age_near", False) for r in ages)
    assert abs(near / len(ages) - D.AGE_NEAR) < .08


def test_an_age_line_closes_the_message_once(ages):
    # a sign-off or a closing line is the message's end, with no closing phrase after it
    # and nothing repeated from the message's opening
    for r in (r for r in ages if r["meta"]["age_form"] == "signoff"):
        assert "closing" not in r["meta"]["aug"], r["id"]
        (s, e, _t) = [sp for sp in r["spans"] if sp[2] == "AGE"][-1]
        assert len(_bare(r["text"][e:]).split("\n")) <= 2, r["id"]


# ── AGE in signature blocks ──────────────────────────────────────────────────

def test_a_signature_age_joins_the_name_or_has_a_key(signatures):
    n = 0
    for r in signatures:
        for k, (s, e, t) in enumerate(r["spans"]):
            if t != "AGE":
                continue
            line = r["text"][r["text"].rfind("\n", 0, s) + 1:s]
            prev = r["spans"][k - 1] if k else None
            join = _bare(r["text"][prev[1]:s]) if prev else ""
            if prev and prev[2] == "PERSON" and join in (", ", " ("):
                assert join == ", " or _bare(r["text"][e:]).startswith(")"), r["id"]
            else:
                assert _bare(line).lower().endswith(("age: ", "age ")), (r["id"], line[-8:])
            n += 1
    assert n >= 10
