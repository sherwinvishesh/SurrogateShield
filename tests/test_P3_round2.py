"""Campaign phase 3, second step: two more frames in
``relation_gate.frame_drops``. A place only spaCy read (the tagger passed
it over), with no cue and no tie, is the writer's only in a sign-off or a
contact block (``untied_place``); an identifier elsewhere in the message is
no evidence. A name the prompt gives the assistant ("you are X, a …",
"your name will be X") is no one the writer knows (``persona``).
Model-free; the names, places and addresses are invented."""

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


# ── a place only spaCy read ──────────────────────────────────────────────────

HEADED = ("Contact: ilse.varga@example.org\n\n"
          "Write a short travel guide to the old harbour district in Kessmar, with "
          "three places to eat and one museum.")


def test_an_identifier_elsewhere_is_no_tie():
    email = ent(HEADED, "ilse.varga@example.org", "email", "pattern")
    assert rules(HEADED, ent(HEADED, "Kessmar", "GPE", "ner"), email) == [("Kessmar", "untied_place")]


@pytest.mark.parametrize("kind", ["GPE", "LOC", "FAC"])
def test_every_place_type(kind):
    text = "Write a poem about the winter light over Kessmar and the river."
    assert rules(text, ent(text, "Kessmar", kind, "ner")) == [("Kessmar", "untied_place")]


@pytest.mark.parametrize("text", [
    "We will open a second office in Kessmar.\n\nIlse Varga\nilse.varga@example.org",
    "The workshop is in Kessmar\n\n- Ilse Varga",
    "Best,\nIlse Varga, Kessmar\nilse.varga@example.org",
    "Thanks for the reply. Kessmar, ilse.varga@example.org",
])
def test_a_place_before_a_sign_off_or_a_contact_stays(text):
    ents = [ent(text, "Kessmar", "GPE", "ner")]
    if "Ilse Varga" in text:
        ents.append(ent(text, "Ilse Varga"))
    ents.append(ent(text, "ilse.varga@example.org", "email", "pattern")) if "@" in text else None
    assert [r for r in rules(text, *ents) if r[0] == "Kessmar"] == []


def test_a_short_line_under_a_contact_line_stays():
    text = "Please reply to this address:\nilse.varga@example.org\nKessmar"
    email = ent(text, "ilse.varga@example.org", "email", "pattern")
    assert rules(text, ent(text, "Kessmar", "GPE", "ner"), email) == []


def test_a_long_line_under_a_contact_line_is_not_a_sign_off():
    text = ("ilse.varga@example.org\n"
            "Write a story about two cousins who open a bakery in Kessmar during the long winter")
    email = ent(text, "ilse.varga@example.org", "email", "pattern")
    assert rules(text, ent(text, "Kessmar", "GPE", "ner"), email) == [("Kessmar", "untied_place")]


@pytest.mark.parametrize("text", [
    "I live in Kessmar and want a cheaper gym.",
    "My sister moved to Kessmar last year.",
    "I grew up near the harbour at Kessmar, so the essay is easy.",     # "I" in its sentence
    "We loved the ferry ride over to Kessmar!",                          # "we"
    "Address: Kessmar",                                                  # a form line
    "| Ilse | 41 | Kessmar |",                                           # a table
])
def test_a_cue_a_tie_a_first_person_sentence_or_a_form_keeps_it(text):
    assert rules(text, ent(text, "Kessmar", "GPE", "ner")) == []


def test_a_short_impersonal_sentence_is_enough():
    text = "The harbour at Kessmar grew quickly after the railway arrived."
    assert rules(text, ent(text, "Kessmar", "GPE", "ner")) == [("Kessmar", "untied_place")]


def test_only_spacys_reading():
    text = "Write a poem about the winter light over Kessmar and the river."
    assert rules(text, ent(text, "Kessmar", "GPE", "pii_tagger")) == []
    assert rules(text, ent(text, "Kessmar", "ORG", "ner")) == []


# ── a name the prompt gives the assistant ────────────────────────────────────

@pytest.mark.parametrize("text, span", [
    ("You are Mira Vale, a travel guide for the coastal towns. Answer in short lists.", "Mira Vale"),
    ("From now on you are now Juno the castle librarian. Stay in character.", "Juno"),
    ("For this game your name will be 'Juno'. You run the shop.", "Juno"),
])
def test_a_name_given_to_the_assistant(text, span):
    assert rules(text, ent(text, span)) == [(span, "persona")]


@pytest.mark.parametrize("text, span", [
    ("You are Mira Vale, a friend I have known since school.", "Mira Vale"),   # "I" in its sentence
    ("You are Mira Vale. Please confirm the booking.", "Mira Vale"),           # no role after
    ("Your name is Juno, right?", "Juno"),                                     # said, not given
])
def test_not_a_persona(text, span):
    assert rules(text, ent(text, span)) == []


def test_an_identifier_near_keeps_a_persona_name():
    text = "You are Mira Vale, a broker. Write to mira.vale@example.org for the files."
    email = ent(text, "mira.vale@example.org", "email", "pattern")
    assert rules(text, ent(text, "Mira Vale"), email) == []
