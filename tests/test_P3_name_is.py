"""Campaign phase 3: a lower-case or word-like name said to be a name
("my name is hunter", "her name's Joy") is not dropped as a word. Model-free;
the texts are invented."""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def ent(text, span):
    i = text.index(span)
    return DetectedEntity(span, i, i + len(span), "PERSON", 0.9, "pii_tagger")


@pytest.mark.parametrize("text, span", [
    ("hello, my name is hunter and I need help with a cover letter", "hunter"),
    ("Her name's Joy, she is in year four.", "Joy"),
    ("Name: Hope", "Hope"),
    ("his name was Will", "Will"),
])
def test_a_word_said_to_be_a_name_is_a_name(text, span):
    assert not rg.is_word_name(ent(text, span), text)


def test_a_word_name_with_no_sign_of_a_person_is_still_a_word():
    text = "What rhymes with Joy in a short poem?"
    assert rg.is_word_name(ent(text, "Joy"), text)
