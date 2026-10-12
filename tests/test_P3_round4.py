"""Phase 3, round 4: spans whose own words say they are no name, age or
secret (``pipeline._misreads``), read after every stage, and the ensemble
fallback's single word right after a possessive ("our Mission"). Model-
free; texts and names are invented."""

import pytest

from surrogateshield.core.detection import pipeline
from surrogateshield.core.entities import DetectedEntity

KW = dict(use_context_guard=False, use_tagger=False, use_entity_trace=False)


def _ent(text, value, typ, source):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


def _rule(text, value, typ, source):
    found = pipeline._misreads(text, [_ent(text, value, typ, source)])
    return found[0][1] if found else None


# ── names ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value, typ, source, rule", [
    ("How do I port a game to Quartz Engine 5.x and keep the shaders?", "Quartz Engine", "GPE", "ner", "version"),
    ("Is Lumen Studio 12.4 worth the upgrade?", "Lumen Studio", "ORG", "pii_tagger", "version"),
    ("I'm 19 and studying Geology. Any tips?", "Geology", "ORG", "pii_tagger", "subject"),
    ("She has a degree in Botany from abroad.", "Botany", "ORG", "pii_tagger", "subject"),
    ("Suppose XYZ Corp sells 40 units at $9 each.", "XYZ", "ORG", "pattern", "placeholder"),
    ("Contoso Ltd. reported a loss this quarter.", "Contoso Ltd.", "ORG", "pii_tagger", "placeholder"),
    ("I work for a Danish shipping company in the north.", "Danish shipping", "ORG", "pattern", "kind_of_org"),
    ("Write a review. The name of the hotel is Harbor Vale and the view is great.", "Harbor Vale", "PERSON",
     "pii_tagger", "thing_name"),
    ('They sang "nova" songs and "dawn rain" tunes.', "nova", "PERSON", "structural", "lowercase_word"),
    ("Plan a trip across Braga, Faro, Evora and Sintra next spring.", "Faro", "GPE", "ner", "place_list"),
])
def test_a_misread_name(text, value, typ, source, rule):
    assert _rule(text, value, typ, source) == rule


@pytest.mark.parametrize("text, value, typ, source", [
    ("Quartz Engine is the one I use.", "Quartz Engine", "GPE", "ner"),                 # no version
    ("I took a course in Lisbon last summer.", "Lisbon", "GPE", "ner"),                  # a place, not a subject
    ("I work for the Danish shipping company in the north.", "Danish shipping", "ORG", "pattern"),  # one company
    ("I work for a Danish shipping company.", "Danish shipping", "ORG", "pii_tagger"),   # a model's: not this rule
    ("The name of the teacher is Harbor Vale.", "Harbor Vale", "PERSON", "pii_tagger"),  # a person's noun
    ("The name of the hotel owner is Harbor Vale.", "Harbor Vale", "PERSON", "pii_tagger"),
    ("My name is nova and I write songs.", "nova", "PERSON", "structural"),             # an introduction
    ("Name: nova\nRole: singer", "nova", "PERSON", "structural"),                         # a form
    ("first,last,zip\nleah,okonkwo,85251\nPing Kev about it.", "okonkwo", "PERSON", "structural"),  # a cell
    ("okonkwo;leah;85251\nPing Kev about it.", "okonkwo", "PERSON", "structural"),
    ("i met nova at the gym and we talked", "nova", "PERSON", "structural"),             # writes lower case
    ("City: Braga, Faro, Evora and Sintra", "Braga", "GPE", "ner"),                      # a form
    ("Gates Braga, Faro, Evora and B2 are open.", "Faro", "GPE", "ner"),                 # a digit in the list
    ("I live in Faro with my two cats.", "Faro", "GPE", "ner"),                          # no list
])
def test_not_a_misread_name(text, value, typ, source):
    assert _rule(text, value, typ, source) is None


# ── secrets ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value, typ, rule", [
    ('  "name": "edge-node-pool-west2",', "edge-node-pool-west2", "credential", "slug"),
    ('  "cluster_id": "web-app-v2",', "web-app-v2", "id_number", "slug"),
    ('- "Bright mornings bring patient rivers home again"', "Bright mornings bring patient rivers home again",
     "credential", "sentence_secret"),
])
def test_a_misread_secret(text, value, typ, rule):
    assert _rule(text, value, typ, "pii_tagger") == rule


@pytest.mark.parametrize("text, value, typ", [
    ('  "password": "edge-node-pool-west2",', "edge-node-pool-west2", "credential"),  # a secret's cue
    ("api_key = abc-def-123", "abc-def-123", "credential"),                            # a cue; and no g-z
    ("token: kq7-zx-91-mm", "kq7-zx-91-mm", "credential"),
    ("my seed phrase: apple river stone cloud lamp ocean", "apple river stone cloud lamp ocean", "credential"),
    ("Recovery words below\nApple river stone cloud lamp ocean", "Apple river stone cloud lamp ocean",
     "credential"),                                                                   # a cue on the line before
    ("apple river stone cloud lamp ocean", "apple river stone cloud lamp ocean", "credential"),  # a seed phrase's shape
    ("pw Hx7!kQ2 Lm9 Zt4 Rr1", "Hx7!kQ2 Lm9 Zt4 Rr1", "credential"),                   # five words
])
def test_not_a_misread_secret(text, value, typ):
    assert _rule(text, value, typ, "pii_tagger") is None


def test_a_slug_from_another_stage_stays():
    text = '  "name": "edge-node-pool-west2",'
    assert _rule(text, "edge-node-pool-west2", "credential", "pattern") is None


# ── ages ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value, rule", [
    ("i am 40 years old. my brother is 6 years older than me.", "6 years", "age_difference"),
    ("My cousin is 3 years younger than I am.", "3 years", "age_difference"),
    ("When I was 6 my sister was half my age. Now that my sister is 30, how old am I?", "30", "age_puzzle"),
    ("Which books suit children, ages 9 to 12?", "ages 9", "age_range"),
    ("Explain gravity to a five year old.", "five year old", "age_audience"),
    ("Explain it like I'm 5.", "5", "age_audience"),
])
def test_a_misread_age(text, value, rule):
    assert _rule(text, value, "age", "pattern") == rule


@pytest.mark.parametrize("text, value", [
    ("I am 34 years old and live alone.", "34 years old"),
    ("My son is 9 and loves trains.", "9"),
])
def test_not_a_misread_age(text, value):
    assert _rule(text, value, "age", "pattern") is None


def test_a_puzzle_age_stays_near_an_identifier():
    text = "Call me at 555-0142. I'm 52, how old am I in dog years?"
    age = _ent(text, "52", "age", "pattern")
    phone = _ent(text, "555-0142", "phone_us", "pattern")
    assert pipeline._misreads(text, [age, phone]) == []
    assert [r for _e, r in pipeline._misreads(text, [age])] == ["age_puzzle"]


def test_the_cascade_drops_a_misread_age_and_says_why():
    text = "i am 40 years old. my brother is 6 years older than me. how tall is he?"
    conf, _ = pipeline.run_cascade(text, **KW)
    assert [(e.type, e.text) for e in conf] == [("age", "40 years old")]
    assert "age_difference" in conf._skip_reasons.values()


# ── the ensemble fallback ────────────────────────────────────────────────────

@pytest.mark.parametrize("text, name, tied", [
    ("Thanks again for supporting our Mission to the coast.", "Mission", False),
    ("I left my Pixel on the train.", "Pixel", False),
    ("Please tell Mission that the keys are here.", "Mission", True),
    ("My sister Mira Qeli is turning thirty next week.", "Mira Qeli", True),
    ("our Mira Qeli said she would call.", "Mira Qeli", True),               # a full name after a possessive
])
def test_a_single_word_after_a_possessive_does_not_stand_in(text, name, tied):
    cands = [_ent(text, name, "PERSON", "ner")]
    assert (pipeline._tied_fallback(text, cands) == cands) is tied
