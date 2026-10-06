"""Pass E name tokens in any script: a non-ASCII letter no longer cuts a
name short ("Kovač" was "Kova"), a sentence's closing period is not part of
the name, and a capitalised initial or particle ("M.", "Ó") stays inside it.
A bare "A" or "I" is still a word. Model-free; every name is made up.
"""

import pytest

from surrogateshield.core.detection.pipeline import _detect_structural_persons


def pass_e(text):
    return sorted(e.text for e in _detect_structural_persons(text, [])[0])


@pytest.mark.parametrize("text, name", [
    ("Hi, my name is Ana Kovač and I study here.", "Ana Kovač"),
    ("her name is Đặng Thị Hoa", "Đặng Thị Hoa"),
    ("This is Liam Ó Bríain, writing about my visa.", "Liam Ó Bríain"),
    ("I'm Rita M. Solano.", "Rita M. Solano"),
    ("My name is Ada Byrne.", "Ada Byrne"),
    ("This is J Smith from accounts", "J Smith"),
    ("Jörg Lindqvist here, quick question", "Jörg Lindqvist"),
])
def test_name_in_any_script(text, name):
    assert name in pass_e(text)


@pytest.mark.parametrize("text", [
    "I'm A big fan of this",
    "This is A Test of it",
    "I am I think fine",
    "this is é",
])
def test_bare_one_letter_word_is_not_a_name(text):
    assert pass_e(text) == []
