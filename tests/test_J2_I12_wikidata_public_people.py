"""Gates J2 / audit I12 — widely known people from a Wikidata gazetteer
(python-library/tools/build_public_people.py), so a topical message about
Paul Volcker or Christine Lagarde's successor is not rewritten, while a name
that could as well be a private person's is never in the list.
Model-free: entities are built by hand, as NER would emit them.
"""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.detection.public_names import (
    FAMOUS_SURNAMES, WIKIDATA_PEOPLE, is_listed_public_person)
from surrogateshield.core.entities import DetectedEntity


def dropped(text, *names):
    es = [DetectedEntity(n, text.index(n), text.index(n) + len(n), "PERSON", 0.9, "ner")
          for n in names]
    return sorted(e.text for e in rg.gate(text, es, ())[1])


def test_I12_gazetteer_is_loaded_and_sized():
    assert 20_000 < len(WIKIDATA_PEOPLE) < 40_000
    assert 30 < len(FAMOUS_SURNAMES) < 200


@pytest.mark.parametrize("name", [
    "Paul Volcker", "Alan Greenspan", "Mansa Musa", "Jensen Huang",
    "Nikola Jokić", "Nikola Jokic", "NIKOLA JOKIĆ", "Angela Merkel", "Churchill",
])
def test_I12_widely_known_person_is_listed(name):
    assert is_listed_public_person(name)


@pytest.mark.parametrize("name", [
    "John Smith", "David Lee", "James Brown", "Maria Garcia", "Emma Wilson",
    "Yusuf Demir",                 # a Faker tr_TR given name + surname
    "Li Na", "Wang Yi",            # every word four letters or fewer
    "Smith", "Newton", "Jobs",     # a surname alone that is common or a word
    "Priya Sharma", "Kai Nakamura",
])
def test_I12_name_that_could_be_private_is_not_listed(name):
    assert not is_listed_public_person(name)


def test_J2_topical_question_keeps_listed_people():
    text = "How did Paul Volcker's approach to inflation differ from Alan Greenspan's?"
    assert dropped(text, "Paul Volcker's", "Alan Greenspan's") == [
        "Alan Greenspan's", "Paul Volcker's"]


def test_J2_listed_name_tied_to_the_user_stays_masked():
    assert dropped("my coworker Paul Volcker is late again", "Paul Volcker") == []
