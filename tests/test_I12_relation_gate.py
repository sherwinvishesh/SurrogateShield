"""Audit I12 — Pass R, the relation gate: an ORG, place or name is replaced
only when the message ties it to a person (bench/realworld/GUIDE.md). Acronyms,
code, greetings, public figures/companies and topical places stay verbatim.
Model-free: entities are built by hand, as NER would emit them.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.detection.pipeline import _detect_structural_persons
from surrogateshield.core.entities import DetectedEntity


def ents(text, *specs, source="ner"):
    out, pos = [], 0
    for value, typ in specs:
        s = text.index(value, pos if value in text[pos:] else 0)
        out.append(DetectedEntity(value, s, s + len(value), typ, 0.9, source))
        pos = s + len(value)
    return out


def dropped(text, *specs, context=()):
    _, d = rg.gate(text, ents(text, *specs), context)
    return sorted(e.text for e in d)


# ── junk ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, spec", [
    ("Can you remind me about the SLA for Q4?", ("SLA", "ORG")),
    ("Do I need a W-4 for this?", ("W-4", "ORG")),
    ("Should I elect S-corp?", ("S-corp", "ORG")),
    ("Using Python 3.12.4 on macOS", ("Python 3.12.4", "PERSON")),
    ("why does `User.objects.filter(first_name='Admin')` fail", ("User.objects.filter(first_name='Admin", "GPE")),
    ("Model HP LaserJet M404n jams", ("HP LaserJet M404n", "ORG")),
    ("Regards, see you", ("Regards", "PERSON")),
    ("Bonjour, je voudrais", ("Bonjour", "PERSON")),
])
def test_I12_junk_is_kept_verbatim(text, spec):
    assert dropped(text, spec) == [spec[0]]


def test_I12_state_code_and_caps_names_are_not_acronyms():
    assert dropped("I live in AZ", ("AZ", "GPE")) == []
    assert dropped("my sister calls her NOOR", ("NOOR", "PERSON")) == []
    assert dropped("PLEASE CONTACT JOHN, HE LIVES IN CHICAGO",
                   ("JOHN", "PERSON"), ("CHICAGO", "GPE")) == []


def test_I12_handle_shaped_person_stays_masked():
    assert dropped("username sarahm92 is locked", ("sarahm92", "PERSON")) == []


# ── people ───────────────────────────────────────────────────────────────────

def test_I12_public_figures_kept():
    text = "What did Taylor Swift say in her 2024 statement, and did Elon Musk respond?"
    assert dropped(text, ("Taylor Swift", "PERSON"), ("Elon Musk", "PERSON")) == [
        "Elon Musk", "Taylor Swift"]
    assert dropped("reforms of Emperor Meiji vs Peter the Great's",
                   ("Emperor Meiji", "PERSON"), ("Peter the Great's", "PERSON")) == [
        "Emperor Meiji", "Peter the Great's"]
    assert dropped("In Django, why", ("Django", "PERSON")) == ["Django"]


def test_I12_public_name_with_personal_cue_stays_masked():
    assert dropped("my coworker Michael Jordan is late", ("Michael Jordan", "PERSON")) == []
    assert dropped("Chase said he'd call", ("Chase", "PERSON")) == []


def test_I12_private_person_untouched():
    assert dropped("Hannah approved it", ("Hannah", "PERSON")) == []


# ── organisations and places ─────────────────────────────────────────────────

@pytest.mark.parametrize("text, spec", [
    ("I'm a nurse at Mercy General Hospital", ("Mercy General Hospital", "ORG")),
    ("I work at Microsoft", ("Microsoft", "ORG")),
    ("we live near Reno", ("Reno", "GPE")),
    ("he lives in denver now", ("denver", "GPE")),
    ("my son goes to Kyrene Middle School", ("Kyrene Middle School", "ORG")),
    ("Austin is where I live", ("Austin", "GPE")),
    ("j'habite à Nantes", ("Nantes", "GPE")),
    ("City: Flagstaff", ("Flagstaff", "GPE")),
    ("My LLC is Sunrise Pottery Studio LLC", ("Sunrise Pottery Studio LLC", "ORG")),
    ("debugging our integration at DataPulse Inc", ("DataPulse Inc", "ORG")),
    ("a resignation letter from Chinedu Okafor", ("Chinedu Okafor", "GPE")),
])
def test_I12_tied_org_or_place_masked(text, spec):
    assert dropped(text, spec) == []


@pytest.mark.parametrize("text, spec", [
    ("How far is Tempe from Phoenix?", ("Tempe", "GPE")),
    ("Compare Microsoft and Nvidia stock", ("Nvidia", "ORG")),
    ("What year did the Berlin Wall fall?", ("the Berlin Wall", "FAC")),
    ("DM me on Discord", ("Discord", "GPE")),
    ("Is Tesla stock a buy?", ("Tesla", "ORG")),
])
def test_I12_topical_or_public_dropped(text, spec):
    assert dropped(text, spec) == [spec[0]]


def test_I12_anchored_message_keeps_places_masked():
    text = "Got a ticket in Tucson. My DL is D08812345."
    dl = DetectedEntity("D08812345", text.index("D0"), text.index("D0") + 9,
                        "us_driver_license", 1.0, "pattern")
    assert dropped(text, ("Tucson", "GPE"), context=[dl]) == []
    # …but a public brand in the same anchored message is still kept
    assert dropped(text + " Amazon says", ("Amazon", "ORG"), context=[dl]) == ["Amazon"]


def test_I12_card_brand_next_to_card_number_is_public():
    # GUIDE.md: a protect ORG is an employer / school / clinic tied to a
    # person; a public company is keep, even next to the user's card number
    text = "a charge on my corporate Visa card, which is 4111111111111111."
    card = DetectedEntity("4111111111111111", text.index("4111"), text.index("4111") + 16,
                          "credit_card", 1.0, "pattern")
    assert dropped(text, ("Visa", "ORG"), context=[card]) == ["Visa"]


def test_I12_public_company_as_employer_stays_masked():
    assert dropped("I work at Microsoft and my boss is mean", ("Microsoft", "ORG")) == []
    assert dropped("I'm a nurse at Kaiser in Oakland", ("Kaiser", "ORG")) == []


def test_I12_ties_propagate_through_person_and_org():
    text = "a manager named Tariq Mansoor representing Aramco out of Flagstaff."
    assert dropped(text, ("Tariq Mansoor", "PERSON"), ("Aramco", "ORG"),
                   ("Flagstaff", "GPE")) == []


def test_I12_signature_and_header_lines_are_tied():
    text = "See you then.\n\nBest regards,\nDana Whitfield\nNorthwind Traders\nSeattle"
    assert dropped(text, ("Dana Whitfield", "PERSON"), ("Northwind Traders", "ORG"),
                   ("Seattle", "GPE")) == []


def test_I12_pattern_entities_never_gated():
    text = "SLA 85251"
    zipe = DetectedEntity("85251", 4, 9, "zip_us", 1.0, "pattern")
    kept, d = rg.gate(text, [zipe])
    assert kept == [zipe] and d == []


# ── small pattern false positives ────────────────────────────────────────────

def test_I12_handle_stop_word_one():
    text = "my psn is ryu.kobayashi07, someone hacked the psn one"
    assert [v for t, v in ((e.type, e.text) for e in ps.scan(text)) if t == "handle"] == [
        "ryu.kobayashi07"]


def test_I12_username_label_after_login():
    assert ("handle", "sarahm92") in [(e.type, e.text) for e in
                                      ps.scan("Login: username sarahm92, password x")]


def test_I12_zip_tail_of_order_code():
    text = "Order #A-77219 shows 'delivered'"
    assert [e for e in ps.scan(text) if e.type == "zip_us"] == []


def test_I12_cased_intro_needs_capitalised_name():
    new, _ = _detect_structural_persons("Should I buy VOO? I'm risk averse.", [])
    assert new == []
    new, _ = _detect_structural_persons("hi im sarah mitchell from austin", [])
    assert [e.text for e in new] == ["sarah mitchell"]


# ── evaluator plumbing ───────────────────────────────────────────────────────

def test_I12_policy_reason_registered():
    import eval_metrics
    assert "not_tied_to_person" in eval_metrics.POLICY_REASONS


@pytest.mark.parametrize("text, spec", [
    ("Combien de calories dans 250 grammes de pâtes ?", ("Combien de calories", "PERSON")),
    ("¿Cuáles son las ventajas fiscales?", ("Cuáles", "ORG")),
    ("¿Cuáles son las ventajas fiscales?", ("las ventajas", "PERSON")),
    ("O pedido ORD-562939 chegou em 12/20/2027, mas faltam 8 itens.", ("mas faltam 8 itens", "PERSON")),
    ("O pedido ORD-562939 chegou em 12/20/2027", ("ORD-562939 chegou", "PERSON")),
])
def test_I12_question_words_articles_and_digits_are_not_names(text, spec):
    assert dropped(text, spec) == [spec[0]]


def test_I12_digit_and_lowercase_rules_keep_real_names():
    assert dropped("username sarahm92 is locked", ("sarahm92", "PERSON")) == []
    assert dropped("hi im la toya from austin", ("la toya", "PERSON")) == []
    assert dropped("Quentin Dove called", ("Quentin Dove", "PERSON")) == []
