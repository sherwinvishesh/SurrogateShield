"""V4 §3.1 C: an age written in words gets a surrogate that does not repeat
the original's words. "thirty" → "thirty-one" carried the value inside the
surrogate (the scorer's leak rule: a value of four or more characters inside
its replacement), a test-2 leak of cause ``surrogate``. Ages in digits keep
the ±1..3 shift. Model-free; every text is invented."""

import re

import pytest

from surrogateshield.core.detection import canonical as C
from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity, plan_substitutions
from surrogateshield.core.generation.mimic import MimicGen, _age_words_kept

SEEDS = range(25)
WORDED = [(n, C.number_words(n, j)) for n in range(18, 80) for j in ("-", " ")]


def _age_of(words: str) -> int:
    n = C.number_value(words)
    assert n is not None, words
    return n


def _allowed(n: int) -> set:
    """Where a worded age may go: a near age (±1..3) outside the original's
    tens word, else an age outside its decade (30 → 27–29 or 40–43)."""
    tens = n // 10 * 10
    near = {k for k in range(n - 3, n + 4) if k != n and k >= 1 and (n < 20 or k // 10 * 10 != tens)}
    return near or {*range(tens - 3, tens), *range(tens + 10, tens + 14)}


def _send(text, seed):
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False)
    mapping = MimicGen(seed=seed).generate_all(conf, text=text)
    out = text
    for s, e, _o, r in sorted(plan_substitutions(text, conf, mapping), reverse=True):
        out = out[:s] + r + out[e:]
    return conf, out


@pytest.mark.parametrize("n, words", WORDED)
def test_a_model_age_in_words_never_keeps_its_words(n, words):
    # a model's AGE in words: no canonical view, re-drawn on its number
    for seed in SEEDS:
        sur = MimicGen(seed=seed).generate(DetectedEntity(words, 0, len(words), "age", 0.9, "slm"))
        assert not _age_words_kept(words, sur), (words, sur)
        assert words.casefold() not in sur.casefold()
        assert _age_of(sur) in _allowed(n), (words, sur)


@pytest.mark.parametrize("n, words", WORDED[::3])
def test_a_worded_age_in_a_message_never_keeps_its_words(n, words):
    # the canonicaliser's worded view: found as "29", spelled back
    text = f"I'm {words} years old and new here"
    for seed in SEEDS:
        conf, out = _send(text, seed)
        assert [e.view for e in conf if e.type == "age"] == ["worded"], conf
        sur = re.fullmatch(r"I'm (.+) years old and new here", out).group(1)
        assert not _age_words_kept(words, sur), (words, sur)
        assert _age_of(sur) in _allowed(n), (words, sur)


def test_thirty_stays_out_of_the_thirties():
    got = {_age_of(MimicGen(seed=s).generate(DetectedEntity("thirty", 0, 6, "age", 0.9, "slm"))) for s in range(60)}
    assert got <= {27, 28, 29} and len(got) > 1


def test_mid_decade_leaves_the_decade_on_both_sides():
    got = {_age_of(MimicGen(seed=s).generate(DetectedEntity("thirty-five", 0, 11, "age", 0.9, "slm")))
           for s in range(80)}
    assert got <= {27, 28, 29, 40, 41, 42, 43}
    assert got & {27, 28, 29} and got & {40, 41, 42, 43}


@pytest.mark.parametrize("words, shape", [
    ("Thirty", r"Twenty-[A-Z]?[a-z]+"),
    ("Thirty Four", r"(Twenty|Forty) [A-Z][a-z]+|Forty"),
    ("SIXTY", r"FIFTY-[A-Z]+"),
    ("twenty nine years old", r"thirty( [a-z]+)? years old"),
])
def test_the_spelling_is_kept(words, shape):
    for seed in SEEDS:
        sur = MimicGen(seed=seed).generate(DetectedEntity(words, 0, len(words), "age", 0.9, "slm"))
        assert re.fullmatch(shape, sur), (words, sur)
        assert not _age_words_kept(words, sur)


def test_teens_and_single_words():
    for words in ("seven", "twelve", "eighteen", "nineteen"):
        for seed in SEEDS:
            sur = MimicGen(seed=seed).generate(DetectedEntity(words, 0, len(words), "age", 0.9, "slm"))
            assert words not in sur and _age_of(sur) in _allowed(_age_of(words)), (words, sur)


@pytest.mark.parametrize("age", ["29", "41", "7", "1", "34 years old", "age 63"])
def test_digit_ages_keep_the_small_shift(age):
    n = int(re.search(r"\d+", age).group())
    for seed in SEEDS:
        sur = MimicGen(seed=seed).generate(DetectedEntity(age, 0, len(age), "age"))
        m = int(re.search(r"\d+", sur).group())
        assert m != n and abs(m - n) <= 3 + (n <= 3) * 3 and m >= 1, (age, sur)
        assert re.sub(r"\d+", "#", sur) == re.sub(r"\d+", "#", age)


def test_the_draw_is_deterministic():
    a = [MimicGen(seed=s).generate(DetectedEntity("thirty-five", 0, 11, "age", 0.9, "slm")) for s in SEEDS]
    b = [MimicGen(seed=s).generate(DetectedEntity("thirty-five", 0, 11, "age", 0.9, "slm")) for s in SEEDS]
    assert a == b


@pytest.mark.parametrize("original, surrogate, kept", [
    ("thirty", "thirty-one", True),
    ("thirty", "Thirty Two", True),
    ("Thirty-Four", "thirty-one", True),
    ("thirty-four", "twenty-nine", False),
    ("thirty-four", "forty", False),
    ("nineteen", "ninety", False),
    ("ninety", "nineteen", False),
    ("eighteen", "eighteen years", True),
    ("seven", "seventeen", True),          # the original inside the surrogate
])
def test_words_kept(original, surrogate, kept):
    assert _age_words_kept(original, surrogate) is kept
