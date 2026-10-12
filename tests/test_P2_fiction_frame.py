"""Campaign phase 2, narrowing N1: a screenplay or a story's dialogue is not a
chat transcript — its speaker labels are characters, no one's name — and a
speaker's name standing in code is the identifier. Model-free; the names and
texts are invented."""

import pytest

from surrogateshield.core.detection import structural as st
from surrogateshield.core.entities import DetectedEntity


def _persons(text, ents=()):
    added, _removed = st.detect(text, list(ents))
    return sorted(e.text for e in added if e.type == "PERSON")


def _removed(text, ents):
    _added, removed = st.detect(text, list(ents))
    return sorted(e.text for e in removed)


def ent(text, value, typ="PERSON", source="tagger"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


SCREENPLAY = ("INT. KITCHEN - NIGHT\n"
              "MARA: (whispering) We can't stay here.\n"
              "JONAS: Then where?\n"
              "MARA: Anywhere but this house.\n"
              "JONAS: (beat) Fine.\n")
QUOTED = ("Mara: \"We can't stay here.\"\n"
          "Jonas: \"Then where?\"\n"
          "Mara: \"Anywhere.\"\n"
          "Jonas: \"Fine.\"\n")
NARRATED = ("Mara: We can't stay here.\n"
            "The wind pushed through the broken window and Mara pulled her coat tighter.\n"
            "Jonas: Then where?\n"
            "He did not answer at once; Jonas was staring at the floor, counting the boards.\n"
            "Mara: Anywhere.\n"
            "Jonas: Fine.\n")
DIRECTED = ("Mara: We can't stay here.\n"
            "(She looks at the door.)\n"
            "Jonas: Then where?\n"
            "(A long silence.)\n"
            "Mara: Anywhere.\n"
            "Jonas: Fine.\n")
CHAT = "tia: hey\nnoor: hi\ntia: where r u\nnoor: table 12\n"
STAMPED = ("[10/02/26, 8:14 PM] tia: [image omitted]\n"
           "[10/02/26, 8:15 PM] noor: omw!!\n"
           "[10/02/26, 8:15 PM] tia: [image omitted]\n"
           "[10/02/26, 8:16 PM] noor: k im at table 12\n")


# ── fiction ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [SCREENPLAY, QUOTED, NARRATED, DIRECTED])
def test_a_screenplay_or_story_names_no_speaker(text):
    assert _persons(text) == []
    assert st._fiction_labels(text) == {"mara", "jonas"}


@pytest.mark.parametrize("text", [SCREENPLAY, QUOTED, NARRATED])
def test_what_a_model_found_of_a_character_is_dropped(text):
    label = "MARA" if text is SCREENPLAY else "Mara"
    found = ent(text, label)
    assert _removed(text, [found]) == [label]
    assert _persons(text, [found]) == []


def test_a_character_s_full_name_is_not_propagated_by_the_component_rule():
    text = NARRATED.replace("Mara pulled", "Mara Lindqvist pulled")
    full = ent(text, "Mara Lindqvist")
    assert "Mara" not in _persons(text, [full])


def test_a_fiction_with_a_field_label_is_still_fiction():
    text = "Title: The House\n" + SCREENPLAY
    assert _persons(text) == []


# ── transcripts ──────────────────────────────────────────────────────────────

def test_a_two_speaker_chat_still_names_both():
    assert _persons(CHAT) == ["noor", "noor", "tia", "tia"]
    assert st._fiction_labels(CHAT) == set()


def test_a_timestamped_chat_with_attachments_is_a_transcript():
    assert _persons(STAMPED) == ["noor", "noor", "tia", "tia"]


def test_attachment_turns_are_not_stage_directions():
    text = "tia: [photo]\nnoor: nice\ntia: [photo]\nnoor: where is that\n"
    assert _persons(text) == ["noor", "noor", "tia", "tia"]


def test_a_self_label_makes_a_transcript_of_a_quoted_exchange():
    text = "me: \"are you coming?\"\nnoor: \"yes\"\nme: \"great\"\nnoor: \"see you\"\n"
    assert _persons(text) == ["noor", "noor"]


def test_capitals_need_three_letters():
    text = "AJ: hey\nBo: hi\nAJ: where r u\nBo: table 12\n"
    assert _persons(text) == ["AJ", "AJ", "Bo", "Bo"]


def test_a_long_continuation_line_is_not_narrative():
    text = ("tia: so about tomorrow\n"
            "noor: yes\n"
            "tia: I was thinking we could leave early and get there before the queue builds up\n"
            "and also bring the stuff from last week if you still have it at yours\n"
            "noor: sure\n")
    assert _persons(text) == ["noor", "noor", "tia", "tia"]


def test_three_narrative_lines_make_a_story_without_naming_anyone():
    text = ("Mara: We can't stay here.\n"
            "The wind pushed through the broken window and she pulled her coat tighter.\n"
            "Jonas: Then where?\n"
            "He did not answer at once, staring at the floor and counting the boards.\n"
            "Mara: Anywhere.\n"
            "The lamp guttered and for a moment neither of them said anything at all.\n"
            "Jonas: Fine.\n")
    assert _persons(text) == []


# ── code occurrences ─────────────────────────────────────────────────────────

def test_a_speaker_s_name_in_code_is_the_identifier():
    text = CHAT + "const noor = {};\nuser.noor.name;\n"
    assert _persons(text) == ["noor", "noor", "tia", "tia"]


def test_a_local_part_name_in_code_is_the_identifier():
    text = "ping tariq.m@example.org\nconst x = app.tariq.name;\nthanks tariq"
    email = ent(text, "tariq.m@example.org", "email", "pattern")
    assert _persons(text, [email]) == ["tariq"]
    assert text.index("thanks tariq") < text.index("tariq", text.index("thanks"))


@pytest.mark.parametrize("line, code", [
    ("const x = tia;", True),
    ("{tia}", True),
    ("tia(1)", True),
    ("obj.tia.name", True),
    ("$tia", True),
    ("  tia,", True),
    ("tia: see you there :)", False),
    ("tia, are you coming?", False),
    ("tia (the tall one)", False),
    ("ok tia = legend", False),
])
def test_code_occurrence(line, code):
    s = line.index("tia")
    assert st._code_occurrence(line, s, s + 3) is code


# ── N11: two-word and unknown characters; a name in fiction needs an identifier

from surrogateshield.core.detection import pipeline

TWO_WORD = ("LADY OKAFOR: You came back.\n"
            "OLD MAN: (coughing) I never left.\n"
            "LADY OKAFOR: Then where were you?\n"
            "OLD MAN: Here, all along.\n")
UNKNOWN = ("???: \"Who goes there?\"\n"
           "Mara: \"A friend.\"\n"
           "???: \"Friends knock.\"\n"
           "Mara: \"Then I'll knock.\"\n")


def test_a_two_word_character_label_is_fiction():
    assert st._fiction_labels(TWO_WORD) == {"lady okafor", "old man"}


def test_an_unknown_speaker_is_fiction():
    assert st._fiction_labels(UNKNOWN) == {"???", "mara"}


def test_a_chat_with_two_word_speakers_is_still_a_chat():
    text = "Ana Ruiz: are you coming?\nTariq M: yes, 5 min\nAna Ruiz: ok\nTariq M: see you\n"
    assert st._fiction_labels(text) == set()


def test_a_form_with_two_word_labels_is_not_fiction():
    text = "Full Name: Ana Ruiz\nPhone Number: 555 0100\nEmail Address: ana@example.org\nHome Address: 1 High St\n"
    assert st._fiction_labels(text) == set()


def _cascade(text):
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False)
    return sorted(e.text for e in conf if e.type == "PERSON"), conf._skip_reasons


def test_a_name_in_fiction_is_a_character_s():
    text = SCREENPLAY + "\nHi Tariq, what do you think of my scene?\n"
    persons, reasons = _cascade(text)
    assert persons == [] and "fiction" in reasons.values()


def test_a_name_in_fiction_next_to_an_identifier_is_a_person():
    text = SCREENPLAY + "\nHi Tariq, what do you think of my scene? Reply to tariq@example.org\n"
    persons, reasons = _cascade(text)
    assert persons == ["Tariq"] and "fiction" not in reasons.values()


def test_a_greeting_outside_fiction_still_names():
    persons, reasons = _cascade("Hi Tariq, what do you think of my scene?\n")
    assert persons == ["Tariq"] and "fiction" not in reasons.values()


# ── N20: a narrated story with quoted dialogue ───────────────────────────────

STORY = ('"We can\'t stay here," Mara said, pulling her coat tighter.\n'
         '"Then where?" asked Jonas, not looking up from the floor.\n'
         '"Anywhere but this house," she replied. The wind pushed through the broken window.\n'
         '"Fine," Jonas sighed, and counted the boards again.\n')
TWO_QUOTES = '"We can\'t stay here," Mara said.\n"Then where?" asked Jonas.\n'


def test_a_narrated_story_is_recognised():
    assert st._narrative(STORY) and not st._fiction_labels(STORY)


@pytest.mark.parametrize("text", [
    STORY + "I wrote this last night.",                                        # first person outside the quotes
    TWO_QUOTES,                                                                # too few quotes
    'tia: "hey"\nnoor: "hi"\ntia: "where r u"\nnoor: "table 12"\n',           # a chat: no speech verb
    'The box is marked "fragile", the label "this side up" and the sticker "do not stack".',
])
def test_what_is_not_a_narrated_story(text):
    assert not st._narrative(text)


def test_a_name_under_a_narrated_story_is_a_character_s():
    persons, reasons = _cascade(STORY + "\nHi Tariq, thoughts on this draft?\n")
    assert persons == [] and "narrative" in reasons.values()


def test_a_name_next_to_an_identifier_under_a_story_is_a_person():
    persons, reasons = _cascade(STORY + "\nHi Tariq, thoughts on this draft? Reply to tariq@example.org\n")
    assert persons == ["Tariq"] and "narrative" not in reasons.values()


def test_a_greeting_under_two_quotes_still_names():
    persons, reasons = _cascade(TWO_QUOTES + "\nHi Tariq, thoughts on this draft?\n")
    assert persons == ["Tariq"] and "narrative" not in reasons.values()


def test_a_story_in_the_first_person_still_names():
    persons, reasons = _cascade(STORY + "\nHi Tariq, I wrote this last night, thoughts?\n")
    assert persons == ["Tariq"] and "narrative" not in reasons.values()
