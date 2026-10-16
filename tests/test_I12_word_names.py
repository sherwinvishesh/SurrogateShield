"""Audit I12, gate J2 — names that are also words, and people named as the
author or subject of a public work. A lone "Joy", "Amber" or "Bill" is the
word unless something around it points to a person; "the plot of Chimamanda
Ngozi Adichie's …" is a public author. Cases from realworld dev / dev2.
Model-free: entities are built by hand, as NER would emit them.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def gate(text, *names):
    es, pos = [], 0
    for n in names:
        s = text.index(n, pos)
        es.append(DetectedEntity(n, s, s + len(n), "PERSON", 0.9, "ner"))
        pos = s + len(n)
    kept, _ = rg.gate(text, es)
    return [e.text for e in kept]


@pytest.mark.parametrize("text, names", [
    ("What rhymes with Joy, Hunter and Destiny? Writing a chorus.", ("Joy", "Hunter", "Destiny")),
    ("Is Amber or Jade softer on the Mohs scale?", ("Amber", "Jade")),
    ("Should Mark be capitalized in 'Mark the Date'? Same for Bill in 'Bill of Sale'.",
     ("Mark", "Bill")),
    ("Our team motto is 'Victory, Honor, Grace'. Turn it into Latin for a Banner?",
     ("Grace", "Banner")),
    ("Hope, Faith and Grace are the three values of our Spring Fundraiser.",
     ("Hope", "Faith", "Grace")),
    ("hey brendon, the keys are under the matt at 6 Ferncliff Rd", ("matt",)),
])
def test_I12_word_names_without_a_person_are_words(text, names):
    assert gate(text, *names) == []


@pytest.mark.parametrize("text, names", [
    ("my friends Hope and Will are coming over, and Grace said she'd bring her brother Hunter.",
     ("Hope", "Will", "Grace", "Hunter")),
    ("Hi Joy, can you send the deck?", ("Joy",)),
    ("Can you ask Mark to resend the invoice?", ("Mark",)),
    ("Faith texted me at 3am again", ("Faith",)),
    ("Thanks for everything.\n- Amber", ("Amber",)),
    ("my sister Daisy is pregnant", ("Daisy",)),
    ("Bill's wife called the clinic", ("Bill",)),
])
def test_I12_word_names_with_a_person_cue_stay_masked(text, names):
    assert gate(text, *names) == list(names)


@pytest.mark.parametrize("text, name", [
    ("Summarize the plot of Chimamanda Ngozi Adichie's Half of a Yellow Sun.",
     "Chimamanda Ngozi Adichie"),
    ("Recommend novels similar to Haruki Murakami's Kafka on the Shore.", "Haruki Murakami"),
    ("Which Shah Rukh Khan film made the most at the box office?", "Shah Rukh Khan"),
    ("What's the history of the Taj Mahal, and who was Mumtaz Mahal?", "Mumtaz Mahal"),
])
def test_I12_author_or_subject_of_a_work_is_public(text, name):
    assert gate(text, name) == []


@pytest.mark.parametrize("text, name", [
    ("Write the history of my grandfather Tomasz Wilk for the family album.", "Tomasz Wilk"),
    ("A biography of my mentor Ruth Okafor for her retirement.", "Ruth Okafor"),
    ("who was Sarah texting last night", "Sarah"),
])
def test_I12_personal_subject_is_not_public(text, name):
    assert gate(text, name) == [name]
