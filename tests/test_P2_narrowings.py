"""Campaign phase 2, narrowings N2–N5 of the frames that produced most
spurious edits on real messages: the verb frame in story prose, the
component rule on lowercase and surname-only mentions, and the age patterns
on third parties. Model-free; the names and texts are invented."""

import pytest

from surrogateshield.core.detection import pattern_scan
from surrogateshield.core.detection import structural as st
from surrogateshield.core.entities import DetectedEntity


def _persons(text, ents=()):
    added, _removed = st.detect(text, list(ents))
    return sorted(e.text for e in added if e.type == "PERSON")


def ent(text, value, typ="PERSON", source="tagger"):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), typ, 0.9, source)


# ── N3: the verb frame ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "\"Where are you?\" asked Mariam, looking up.",
    "“Not now,” told Mariam to the boy.",
    "She told Mariam to wait by the door.",
    "he asked Mariam for the keys",
    "They called Mariam twice that night.",
    "Who texted Mariam about it?",
])
def test_a_story_s_attribution_or_narration_is_not_a_verb_frame(text):
    assert _persons(text) == []


@pytest.mark.parametrize("text", [
    "I told Mariam to wait by the door.",
    "can you DM Mariam the PDF before standup?",
    "Ping Mariam about it.",
    "we asked Mariam for the keys",
    "thanks, texted Mariam already",
])
def test_the_writer_s_own_verb_frame_still_names(text):
    assert _persons(text) == ["Mariam"]


# ── N2: the parts of a name ──────────────────────────────────────────────────

def _components(text, *names):
    ents = [ent(text, n) for n in names]
    added, _removed = st._components(text, ents)
    return sorted(e.text for e in added)


def test_the_first_part_names_the_person_again():
    text = "Hannah Reyes sent the draft. Hannah, approved — thanks Hannah."
    assert _components(text, "Hannah Reyes") == ["Hannah", "Hannah"]


@pytest.mark.parametrize("text, name", [
    ("Hannah Reyes sent the draft; Reyes argued the point in court.", "Hannah Reyes"),
    ("Hannah Reyes grew up there. The Reyes family moved in 1990.", "Hannah Reyes"),
    ("Mariam Hannah Reyes signed. Hannah and Reyes are both listed.", "Mariam Hannah Reyes"),
])
def test_a_later_part_alone_is_not_the_person(text, name):
    assert "Reyes" not in _components(text, name)


def test_a_later_part_after_a_title_is_the_person():
    text = "Hannah Reyes sent the draft. Ms Reyes argued the point; Dr. Reyes agreed."
    assert _components(text, "Hannah Reyes") == ["Reyes", "Reyes"]


def test_a_middle_part_alone_is_not_the_person():
    text = "Mariam Hannah Reyes signed. Hannah and Reyes are both listed, Mariam too."
    assert _components(text, "Mariam Hannah Reyes") == ["Mariam"]


def test_a_lower_case_name_s_parts_must_be_names():
    text = "ok hannah reyes, hannah will send it; will and rose are off, reyes said"
    assert _components(text, "hannah reyes") == ["hannah"]
    assert _components("sent to will may, will replies by may", "will may") == []
    assert _components("ask anna lee, anna knows; lee too", "anna lee") == ["anna"]


def test_an_all_caps_name_still_finds_its_first_part():
    text = "JAVIER MONTOYA signed. Javier will call; Montoya is the family."
    assert _components(text, "JAVIER MONTOYA") == ["Javier"]


def test_a_title_in_the_name_is_not_a_part():
    text = "Dr Hannah Reyes signed. Hannah will call; Dr will not."
    assert _components(text, "Dr Hannah Reyes") == ["Hannah"]


# ── N5: whose age ────────────────────────────────────────────────────────────

def _ages(text):
    return [e.text for e in pattern_scan.scan(text) if e.type == "age"]


@pytest.mark.parametrize("text, age", [
    ("I'm 34 years old and tired.", "34 years old"),
    ("I'm Ana M. Ruiz, a 34-year-old vet from Boise.", "34-year-old"),
    ("My daughter is 9 years old.", "9 years old"),
    ("our son is a 12 year old with asthma", "12 year old"),
    ("you're 25 years old, act like it", "25 years old"),
    ("34 years old, female, Boise, looking for a flat", "34 years old"),
    ("Female, 34 years old, non-smoker.", "34 years old"),
    ("Age: 34\nCity: Boise", "Age: 34"),
    ("I was born in 1990, aged 34 now.", "aged 34"),
    ("me and my wife (both 41 years old)", "41 years old"),
])
def test_the_writer_s_or_their_kin_s_age_is_found(text, age):
    assert age in _ages(text)


@pytest.mark.parametrize("text", [
    "Her 9-year-old son starts school in May.",
    "The children are 9 years old.",
    "A 12-year-old student wrote this essay.",
    "She was 8 years old when it happened.",
    "lessons for 9- to 12-year-old students",
    "a class for children between 5 and 7 years old",
    "The retirement age is 65 in most countries.",
    "The legal drinking age is 21.",
    "the novel was first published in 1999, aged 23.",
])
def test_someone_else_s_or_no_one_s_age_is_not(text):
    assert _ages(text) == []


# ── R1: a title glued to a surname ───────────────────────────────────────────

from surrogateshield.core.detection.pipeline import _detect_structural_persons


def _structural(text):
    new, _superseded = _detect_structural_persons(text, [])
    return sorted(e.text for e in new if e.type == "PERSON")


@pytest.mark.parametrize("text, names", [
    ("please address the letter to Mr.Okafor and copy Dr.Ruiz", ["Okafor", "Ruiz"]),
    ("Mrs.Okafor said no; Mr.Okafor said yes.", ["Okafor", "Okafor"]),
    ("Prof.Lindqvist's seminar is on Monday.", ["Lindqvist"]),
])
def test_a_title_glued_to_a_surname_names_the_person(text, names):
    assert _structural(text) == names


@pytest.mark.parametrize("text", [
    "Prof.Dr. Lindqvist is away.",          # the glued word is another title
    "Mr.the manager said no",               # not capitalised
    "mr.okafor said no",                    # lower case is the lowercase frame's
    "Mr.A said so",                         # an initial, not a surname
])
def test_other_dotted_shapes_are_not(text):
    assert not {"A", "the", "Dr"} & set(_structural(text))


# ── N6: a title list is not a company; a company name stays on its line ──────

from surrogateshield.core.detection.pipeline import _detect_structural_orgs
from surrogateshield.core.detection import relation_gate as rg


def _orgs(text):
    return sorted(e.text for e in _detect_structural_orgs(text, []))


def test_a_quoted_title_s_words_are_not_a_company():
    text = ('"The Future of Work: Emerging Technologies and Us"\n'
            '"A Study of the Pennine Heritage Foundation and Its Critics"\n'
            '"Notes on Harrogate Capital"')
    assert _orgs(text) == []


def test_a_company_in_a_quoted_sentence_still_is():
    assert _orgs('she said "I work at Pennine Heritage Foundation now"') == ["Pennine Heritage Foundation"]
    assert _orgs('"Pennine Heritage Foundation is hiring," he wrote') == ["Pennine Heritage Foundation"]


def test_a_company_name_does_not_cross_a_line():
    assert _orgs("Pennine Heritage\nFoundation Review Board\nmeets on Monday") == []
    assert _orgs("the Pennine Heritage Foundation\nmeets on Monday") == ["Pennine Heritage Foundation"]


# ── N7: a phone number in a hex dump ─────────────────────────────────────────

def _phones(text):
    return [e.text for e in pattern_scan.scan(text) if e.type.startswith("phone")]


def test_a_phone_shaped_run_in_a_hex_dump_is_data():
    text = "00 4a1f 0161 4960 123b 0e2c\n0161 496 0123 7d00 9001 23ff\n"
    assert _phones(text) == []


def test_a_phone_number_among_words_is_a_phone():
    assert _phones("ring me on 0161 496 0123 or 07700 900123") == ["0161 496 0123", "07700 900123"]


# ── N8: labels in code are not speakers ─────────────────────────────────────

def test_labels_in_a_stylesheet_or_object_literal_are_not_speakers():
    text = ("const box = {\n  left: ['-100%', '0%'],\n  top: 0,\n};\n"
            ".box {\n  margin-left: 2em;\n  left: calc(50% - 1em);\n  top: 50%;\n}\n")
    assert _persons(text) == []


def test_a_transcript_with_a_link_line_still_has_its_speakers():
    text = "Mariam: are you coming?\nURL: https://example.test/x\nMariam: hello?\nAna: yes, 5 min"
    assert "Mariam" in _persons(text)


# ── N9, N10: products and citations are not places or people ────────────────

@pytest.mark.parametrize("text, span", [
    ("we moved from C++ to Rust", "C++"),
    ("we use Visual Studio 2.1 now", "Visual Studio"),
    ("on Ubuntu 22.04 it works", "Ubuntu"),
])
def test_a_product_or_a_versioned_name_is_not_a_place(text, span):
    assert rg.is_junk(ent(text, span, "GPE", "ner"), text)


def test_a_place_before_a_count_still_is():
    text = "we drove to Harrogate 3 times"
    assert not rg.is_junk(ent(text, "Harrogate", "GPE", "ner"), text)


@pytest.mark.parametrize("text, span, typ", [
    ("as argued before (Okafor & Ruiz, 2019, p. 4), the", "Okafor & Ruiz", "ORG"),
    ("the gap persists (see Okafor et al., 2019; Lindqvist, 2020)", "Okafor", "PERSON"),
    ("the gap persists (see Okafor et al., 2019; Lindqvist, 2020)", "Lindqvist", "PERSON"),
    ("Okafor, A. (2019). Grey skies. Journal of Weather, 3(2).", "Okafor, A", "PERSON"),
    ("Okafor and Ruiz (2019) argue that", "Okafor and Ruiz", "ORG"),
])
def test_a_citation_s_authors_are_not_pii(text, span, typ):
    assert rg.is_junk(ent(text, span, typ, "pii_tagger"), text)


@pytest.mark.parametrize("text, span", [
    ("I met Ana Ruiz in 2019, she", "Ana Ruiz"),
    ("call Ana Ruiz (2019 intake) on Monday", "Ana Ruiz"),
    ("Ana Ruiz, 2019 graduate, wrote to me", "Ana Ruiz"),
    ("Ana Ruiz. (2019 was a long year.)", "Ana Ruiz"),
])
def test_a_person_near_a_year_still_is(text, span):
    assert not rg.is_junk(ent(text, span, "PERSON", "pii_tagger"), text)


# ── N12: a field in code is not a form ───────────────────────────────────────

@pytest.mark.parametrize("text", [
    'user.name = "Mariam Okafor";',
    "self.Name = 'Tariq Mansoor'",
    'obj->name = "Hannah Reyes";',
    '$name = "Mariam Okafor";',
    'name = "Mariam Okafor"',
    "name = 'Tariq Mansoor'",
    "name = `Hannah Reyes`",
])
def test_a_name_assigned_in_code_is_not_a_form_field(text):
    assert _structural(text) == []


@pytest.mark.parametrize("text, names", [
    ("name: Tariq Mansoor\nphone: 555", ["Tariq Mansoor"]),
    ("Name = Hannah Reyes", ["Hannah Reyes"]),
    ("Full name: Mariam Okafor", ["Mariam Okafor"]),
])
def test_a_form_field_still_names(text, names):
    assert _structural(text) == names


@pytest.mark.parametrize("text", [
    'name: "Mariam Okafor"\nage: 31',
    "- name: 'Tariq Mansoor'",
    "Name = Hannah Reyes",
])
def test_a_quoted_value_after_a_colon_is_a_field(text):
    from surrogateshield.core.detection.pipeline import _code_field, _FORM_LABEL_PATTERNS
    hits = list(_FORM_LABEL_PATTERNS[0][0].finditer(text))
    assert hits and not any(_code_field(text, m) for m in hits)


# ── N13: an addressee is not a name ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Dear Hiring Manager,\nI am writing to apply",
    "Hi Team,\nquick update",
    "Dear Diary,\ntoday was long",
    "Hello Valued Customer,\nyour order",
    "Dear Committee, thank you",
    "Hey Mom, call me back",
    "Dear Future Self,",
    "Dear Sir,",
])
def test_a_greeted_addressee_is_not_a_person(text):
    assert _persons(text) == []


@pytest.mark.parametrize("text, name", [
    ("Dear Mariam,\nthanks for yesterday", "Mariam"),
    ("Hi Tariq, quick one", "Tariq"),
    ("Hello Dean, the car is ready", "Dean"),
])
def test_a_greeted_name_still_is(text, name):
    assert _persons(text) == [name]


# ── N14: a public figure's family name alone ─────────────────────────────────

def _gate(text, *specs, context=()):
    ents, pos = [], 0
    for value, typ in specs:                # each value after the previous one
        s = text.index(value, pos)
        ents.append(DetectedEntity(value, s, s + len(value), typ, 0.9, "ner"))
        pos = s + len(value)
    reasons = {}
    kept, dropped = rg.gate(text, ents, list(context), reasons=reasons)
    return sorted(e.text for e in kept), {e.text: reasons[id(e)] for e in dropped}


def test_a_public_figure_s_surname_alone_is_the_same_figure():
    text = "Barack Obama gave a speech in 2009. Obama said the crisis was global."
    kept, dropped = _gate(text, ("Barack Obama", "PERSON"), ("Obama", "PERSON"))
    assert kept == [] and dropped == {"Barack Obama": "public_person", "Obama": "public_person"}


def test_a_public_figure_s_surname_with_a_personal_cue_is_someone_else():
    text = "Barack Obama gave a speech. My neighbour Obama lent me his ladder."
    kept, dropped = _gate(text, ("Barack Obama", "PERSON"), ("Obama", "PERSON"))
    assert kept == ["Obama"] and dropped == {"Barack Obama": "public_person"}


def test_a_private_person_s_surname_is_not_public():
    text = "Mariam Okafor called. Okafor wants the report by Monday."
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"), ("Okafor", "PERSON"))
    assert kept == ["Mariam Okafor", "Okafor"]


# ── N15: a name in an impersonal text ────────────────────────────────────────

ESSAY = ("The committee met for the third time that spring and reviewed the proposals one by one. "
         "Most of them concerned the river, its bridges and the ferries that still crossed it at dawn. "
         "Mariam Okafor presented the longest of the papers, a study of the tolls collected since the war, "
         "and the room listened without interruption. The chair thanked her and moved on to the budget, "
         "which was shorter and far less interesting to everyone present. Nothing was decided before noon.")


def test_a_name_in_an_impersonal_essay_is_a_reference():
    kept, dropped = _gate(ESSAY, ("Mariam Okafor", "PERSON"))
    assert kept == [] and dropped == {"Mariam Okafor": "impersonal"}


def test_a_first_person_word_makes_the_essay_personal():
    text = ESSAY + " I was there."
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"))
    assert kept == ["Mariam Okafor"]


def test_a_second_person_word_makes_the_essay_personal():
    text = ESSAY.replace("The chair thanked her", "Your chair thanked her")
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"))
    assert kept == ["Mariam Okafor"]


def test_an_identifier_near_the_name_ties_it():
    text = ESSAY + " Contact: mariam.okafor@example.org"
    text = text.replace("Mariam Okafor presented", "Mariam Okafor (mariam.okafor@example.org) presented")
    email = ent(text, "mariam.okafor@example.org", "email", "pattern")
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"), context=[email])
    assert kept == ["Mariam Okafor"]


def test_a_byline_is_left_alone():
    text = "By Mariam Okafor\n\n" + ESSAY.replace("Mariam Okafor presented", "The secretary presented")
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"))
    assert kept == ["Mariam Okafor"]


def test_a_short_impersonal_text_is_not_an_essay():
    text = "Mariam Okafor presented the longest of the papers, a study of the tolls collected since the war."
    kept, _dropped = _gate(text, ("Mariam Okafor", "PERSON"))
    assert kept == ["Mariam Okafor"]


def test_a_text_in_another_language_needs_its_own_pronouns():
    de = ("Der Ausschuss traf sich in jenem Frühjahr zum dritten Mal und prüfte die Vorschläge der Reihe nach. "
          "Die meisten betrafen den Fluss, seine Brücken und die Fähren, die ihn noch bei Tagesanbruch überquerten. "
          "Mariam Okafor stellte die längste der Arbeiten vor, eine Studie über die seit dem Krieg erhobenen Zölle, "
          "und der Saal hörte ohne Unterbrechung zu. Der Vorsitzende dankte der Rednerin und ging zum Haushalt über, "
          "der kürzer und für alle Anwesenden weit weniger interessant war. Vor Mittag wurde nichts entschieden.")
    assert rg._impersonal(de)
    assert not rg._impersonal(de + " Ich war dabei.")


def test_a_text_of_names_alone_is_not_impersonal():
    text = " ".join(f"Okafor{i} Mansoor{i}" for i in range(40))
    assert not rg._impersonal(text)


# ── N16: a lone word the tagger is unsure of ─────────────────────────────────

def _tagged(text, value, score):
    s = text.index(value)
    return DetectedEntity(value, s, s + len(value), "PERSON", score, "pii_tagger")


@pytest.mark.parametrize("text, value", [
    ("the query SELECT returned nothing, as usual", "SELECT"),
    ("then run FLUSH on the table", "FLUSH"),
    ("The whatever of it all escaped him", "whatever"),
    ("This nagarajan field is mislabelled in the export", "nagarajan"),
])
def test_a_lone_caps_or_lower_case_word_below_point_nine_is_junk(text, value):
    assert rg.is_junk(_tagged(text, value, 0.8), text)


@pytest.mark.parametrize("text, value, score", [
    ("the query SELECT returned nothing, as usual", "SELECT", 0.95),
    ("my sister calls her NOOR", "NOOR", 0.8),
    ("Dr NOOR will see you now", "NOOR", 0.8),
    ("NOOR's bag is still here", "NOOR", 0.8),
    ("PLEASE CALL NOOR BACK TODAY, SHE IS WAITING FOR THE RESULTS", "NOOR", 0.8),
    ("ask mariam about it", "mariam", 0.8),
    ("राम ने कहा कि वह आएगा, Monday को", "राम", 0.8),
])
def test_a_sure_titled_possessive_or_shouted_word_is_not_junk(text, value, score):
    assert not rg.is_junk(_tagged(text, value, score), text)


@pytest.mark.parametrize("text, value, typ", [
    ("we switched to Okafor™ last year", "Okafor", "PERSON"),
    ("we switched to Okafor™ last year", "Okafor™", "ORG"),
    ("the Mansoor® range is sold out", "Mansoor", "PERSON"),
])
def test_a_product_mark_is_not_a_person_or_a_company(text, value, typ):
    assert rg.is_junk(ent(text, value, typ, "ner"), text)


def test_a_name_at_the_end_of_a_message_is_not_a_mark():
    text = "regards, Okafor"
    assert not rg.is_junk(ent(text, "Okafor", "PERSON", "ner"), text)


# ── N17: a place or an organisation in an impersonal text ────────────────────

from surrogateshield.core.detection import pipeline

TRUST_ESSAY = ESSAY.replace("Mariam Okafor presented", "The Okafor Bridge Trust of Okafor Bay presented")


def test_a_place_and_an_organisation_in_an_impersonal_essay_are_references():
    kept, dropped = _gate(TRUST_ESSAY, ("Okafor Bridge Trust", "ORG"), ("Okafor Bay", "GPE"))
    assert kept == [] and dropped == {"Okafor Bridge Trust": "impersonal", "Okafor Bay": "impersonal"}


def test_an_identifier_near_them_ties_them():
    text = TRUST_ESSAY.replace("of Okafor Bay presented", "of Okafor Bay (okafor.trust@example.org) presented")
    email = ent(text, "okafor.trust@example.org", "email", "pattern")
    kept, _dropped = _gate(text, ("Okafor Bridge Trust", "ORG"), ("Okafor Bay", "GPE"), context=[email])
    assert kept == ["Okafor Bay", "Okafor Bridge Trust"]


def test_a_first_person_word_makes_the_essay_personal_for_places_too():
    _kept, dropped = _gate(TRUST_ESSAY + " I was there.", ("Okafor Bridge Trust", "ORG"), ("Okafor Bay", "GPE"))
    assert dropped == {"Okafor Bridge Trust": "untied", "Okafor Bay": "untied"}     # the ordinary rule


def test_a_name_that_survived_near_them_ties_them_too():
    text = TRUST_ESSAY.replace("presented the longest", "presented the longest (Mariam Okafor, 0161 496 0123)")
    name, phone = ent(text, "Mariam Okafor", "PERSON"), ent(text, "0161 496 0123", "phone_uk", "pattern")
    kept, _dropped = _gate(text, ("Okafor Bridge Trust", "ORG"), ("Okafor Bay", "GPE"), context=[name, phone])
    assert kept == ["Okafor Bay", "Okafor Bridge Trust"]


def test_a_legal_form_does_not_rescue_a_company_in_an_impersonal_essay():
    text = TRUST_ESSAY.replace("Okafor Bridge Trust", "Okafor Bridge Ltd")
    kept, dropped = _gate(text, ("Okafor Bridge Ltd", "ORG"))
    assert kept == [] and dropped == {"Okafor Bridge Ltd": "impersonal"}
    heading = "Okafor Bridge Ltd\n\n" + ESSAY.replace("Mariam Okafor presented", "The secretary presented")
    kept, _dropped = _gate(heading, ("Okafor Bridge Ltd", "ORG"))
    assert kept == ["Okafor Bridge Ltd"]                                            # a heading is left alone


def _cascade_places(text):
    conf, _ = pipeline.run_cascade(text, use_entity_trace=False, use_context_guard=False, use_tagger=False)
    return sorted(e.text for e in conf if e.type in ("GPE", "ORG")), conf._skip_reasons


PROFILE = ESSAY.replace("Mariam Okafor presented", "The trust, located in lagos since the war, presented")


def test_a_residence_cue_in_an_impersonal_profile_is_no_one_s_place():
    places, reasons = _cascade_places(PROFILE)
    assert places == [] and "impersonal" in reasons.values()


def test_the_same_cue_with_an_identifier_near_it_is_a_place():
    text = PROFILE.replace("since the war, presented", "since the war (okafor.trust@example.org), presented")
    places, reasons = _cascade_places(text)
    assert places == ["lagos"] and "impersonal" not in reasons.values()


def test_the_same_cue_in_a_personal_message_is_a_place():
    places, _reasons = _cascade_places("I work for the trust, located in lagos since the war.")
    assert places == ["lagos"]
