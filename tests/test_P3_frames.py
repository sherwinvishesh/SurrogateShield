"""Campaign phase 3: frames that name no one the writer knows
(``relation_gate.frame_drops``), read after every stage: an image prompt, a
footnoted text, a citation, a lower-case phrase the tagger read as a name,
and a place or an organisation in an impersonal stretch of a long English
text. Each one keeps a name with a personal cue, a tie, or an identifier
near it. Model-free; the names, places and numbers are invented."""

import pytest

from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity


def ent(text, span, kind="PERSON", source="pii_tagger", nth=0):
    i = -1
    for _ in range(nth + 1):
        i = text.index(span, i + 1)
    return DetectedEntity(span, i, i + len(span), kind, 0.9, source)


def rules(text, *ents):
    return [(e.text, r) for e, r in rg.frame_drops(text, list(ents))]


# ── image prompts ────────────────────────────────────────────────────────────

PROMPT = ("/imagine prompt: Halvard Brenn::2 standing in a misty harbour, oil painting, "
          "dramatic light --ar 16:9 --v 5")


def test_a_name_in_an_image_prompt_is_a_subject():
    assert rules(PROMPT, ent(PROMPT, "Halvard Brenn")) == [("Halvard Brenn", "image_prompt")]


def test_one_mark_is_not_a_prompt():
    text = "Tomorrow the ratio is 2::1, and Halvard Brenn will check it."
    assert rules(text, ent(text, "Halvard Brenn")) == []


def test_a_personal_cue_keeps_the_name_even_in_a_prompt():
    text = PROMPT + "\nPlease tell Ilse Varga the render is done."
    assert rules(text, ent(text, "Ilse Varga")) == []


# ── footnotes and citations ──────────────────────────────────────────────────

FOOTNOTED = ("The bridge at Kessmar was finished in 1871 [1], and its engineer later wrote "
             "about the floods that closed it [2].")


def test_a_place_in_a_footnoted_text_is_a_reference():
    # the tagger's place: one only spaCy read is ``untied_place`` first (test_P3_round2)
    assert rules(FOOTNOTED, ent(FOOTNOTED, "Kessmar", "GPE")) == [("Kessmar", "footnoted")]


@pytest.mark.parametrize("text, span", [
    ("Earlier work (Okafor & Lindqvist, 2019) found the same effect.", "Okafor"),
    ("Earlier work (Okafor & Lindqvist, 2019) found the same effect.", "Lindqvist"),
    ("As Brandt et al. showed, the effect holds in winter.", "Brandt"),
    ("The method follows Okafor (2017) closely.", "Okafor"),
])
def test_a_cited_author_is_a_reference(text, span):
    assert rules(text, ent(text, span)) == [(span, "citation")]


def test_a_person_with_a_year_in_running_text_is_not_a_citation():
    text = "Ask Okafor (2017 intake) whether the forms are in."
    assert rules(text, ent(text, "Okafor")) == []


# ── lower-case phrases ───────────────────────────────────────────────────────

def test_a_lower_case_phrase_in_mixed_case_text_is_no_name():
    text = 'The menu has dishes called "pollo verde", "mole negro" and "sopa fria". Which is spicy?'
    assert rules(text, ent(text, "mole negro")) == [("mole negro", "lowercase_phrase")]


@pytest.mark.parametrize("text, span", [
    ("Contact:\nilse varga\ncity: Kessmar", "ilse varga"),          # alone on its line, a form below
    ("Name: ilse varga, Role: buyer", "ilse varga"),                 # a field label
    ("i met ilse varga at the station today", "ilse varga"),         # all-lower-case text
    ("Hello, I'm ilse varga and I need a cover letter.", "ilse varga"),   # a self-introduction
])
def test_a_lower_case_name_in_a_form_or_an_introduction_stays(text, span):
    assert rules(text, ent(text, span)) == []


def test_only_the_taggers_lower_case_reading():
    text = 'The menu has dishes called "pollo verde", "mole negro" and "sopa fria". Which is spicy?'
    assert rules(text, ent(text, "mole negro", source="ner")) == []


# ── impersonal stretches of a long text ──────────────────────────────────────

ESSAY = ("The harbour at Kessmar grew quickly after the railway arrived. Merchants from the "
         "interior sent grain down the line, and the customs house was rebuilt in stone. "
         "By the end of the century Brenhaven Shipping owned most of the quay. "
         "Its ledgers show cargo bound for distant ports every week of the year, "
         "and the town council argued for decades about a second breakwater.")
assert len(ESSAY.split()) >= 60


def test_a_place_or_an_organisation_in_an_impersonal_stretch_is_a_subject():
    got = rules(ESSAY, ent(ESSAY, "Kessmar", "GPE"), ent(ESSAY, "Brenhaven Shipping", "ORG"))
    assert got == [("Kessmar", "impersonal_span"), ("Brenhaven Shipping", "impersonal_span")]


def test_a_first_person_sentence_next_to_it_keeps_it():
    text = ESSAY.replace("The harbour at Kessmar", "I grew up near the harbour at Kessmar", 1)
    assert rules(text, ent(text, "Kessmar", "GPE", "ner")) == []


def test_a_short_text_or_a_form_line_is_not_read_this_way():
    short = "The harbour at Kessmar grew quickly after the railway arrived."
    assert rules(short, ent(short, "Kessmar", "GPE")) == []
    form = ESSAY + "\n# name: Ilse | age: 41 | Company: Brenhaven Shipping | ref: 4471"
    assert rules(form, ent(form, "Brenhaven Shipping", "ORG", nth=1)) == []


def test_a_resume_is_not_read_this_way():
    text = ESSAY + "\nBrenhaven Shipping, deck officer, 2014 – 2019."
    assert rules(text, ent(text, "Brenhaven Shipping", "ORG")) == []


def test_an_identifier_near_keeps_it():
    text = ESSAY + " Write to the office at office@brenhaven-shipping.test about it."
    mail = ent(text, "office@brenhaven-shipping.test", "email", "pattern")
    org = ent(text, "Brenhaven Shipping", "ORG")
    assert [(e.text, r) for e, r in rg.frame_drops(text, [org], [mail])] == []


def test_other_types_are_never_dropped():
    text = PROMPT + " call 555-0142"
    phone = ent(text, "555-0142", "phone_us", "pattern")
    assert rg.frame_drops(text, [phone]) == []
