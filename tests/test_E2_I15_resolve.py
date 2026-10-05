"""Audit E2 / I15 — reconstruction rewrites whole values only.

Before: Pass 1 was a case-sensitive ``str.find`` with no word boundaries, the
fuzzy pass matched single words, and the single-token pass rewrote any
capitalised token. Each case below corrupted an answer. Runs against BOTH
trees (library ResolvePass and the config-bound root wrapper).
"""

import pytest

from surrogateshield.core.reconstruction.resolve import ResolvePass as LibRP
from reconstruction.logic import ResolvePass as RootRP


@pytest.fixture(params=["library", "root"])
def resolve(request):
    rp = LibRP() if request.param == "library" else RootRP()
    return lambda text, shadow, **kw: rp.resolve(text, shadow, **kw)


@pytest.mark.parametrize("shadow, text, expected", [
    # E2: substring of a longer word — "Lee flew to Leeds" became "Sarahds"
    ({"Sarah": "Lee"}, "Sarah flew to Leeds; Leela waved.", "Lee flew to Leeds; Leela waved."),
    # E2: a hyphen is a boundary, as in masking (which replaces "Mia Lopez"
    # inside "Mia Lopez-Garcia", privacy first) — so the compound restores
    ({"Sarah Mitchell": "Mia Lopez"}, "Sarah Mitchell-Garcia met Sarah Mitchell.",
     "Mia Lopez-Garcia met Mia Lopez."),
    # E2: a truncated surrogate inside a longer word
    ({"Tempe": "Springfield"}, "Temped in Tempe.", "Temped in Springfield."),
    # E2: digits inside a decimal, time or thousands group
    ({"8": "5"}, "Ran 8.5 km at 1:8 pace; 1,800 people; 8 laps.",
     "Ran 8.5 km at 1:8 pace; 1,800 people; 5 laps."),
    # E2: a CSV row is not a thousands group
    ({"+1-504-020-8142": "602-555-0143", "38046": "85004"},
     "Bell,+1-504-020-8142,38046", "Bell,602-555-0143,85004"),
    ({"602-555-0143": "480-555-0199"}, "a@b.com,602-555-0143,1", "a@b.com,480-555-0199,1"),
    # E2: "-based" is a suffix, not a surname
    ({"North Garyfurt": "Mumbai"}, "our North Garyfurt-based partner", "our Mumbai-based partner"),
    ({"V7713313": "998877654"}, "card DL-V7713313", "card DL-998877654"),
])
def test_E2_whole_values_only(resolve, shadow, text, expected):
    assert resolve(text, shadow) == expected


def test_I15_every_case_variant_restores_in_its_case(resolve):
    out = resolve("sarah mitchell's file and SARAH MITCHELL's badge; Sarah Mitchell.",
                  {"Sarah Mitchell": "Mia Lopez"})
    assert out == "mia lopez's file and MIA LOPEZ's badge; Mia Lopez."


def test_I15_single_word_surrogate_needs_its_case(resolve):
    # "Will" the surrogate must not rewrite the verb "will"
    assert resolve("Will said he will come.", {"Will": "Omar"}) == "Omar said he will come."
    assert resolve("WILL said so", {"Will": "Omar"}) == "OMAR said so"


def test_I15_given_name_not_before_another_surname(resolve):
    out = resolve("Tell Sarah the plan. Sarah Hamm is someone else. Dr. Mitchell agreed.",
                  {"Sarah Mitchell": "Mia Lopez"})
    assert out == "Tell Mia the plan. Sarah Hamm is someone else. Dr. Lopez agreed."


def test_I15_given_name_not_after_another_name(resolve):
    # echo turn 99: "James" (given name of a surrogate) inside "LeBron James"
    shadow = {"James Carter": "Rahul Verma"}
    assert resolve("Did LeBron James or Stephen Curry score more?", shadow) == \
        "Did LeBron James or Stephen Curry score more?"
    # sentence-initial word or title before it is fine
    assert resolve("Ask James. Then Dr. James replied.", shadow) == "Ask Rahul. Then Dr. Rahul replied."


def test_I15_name_part_the_user_typed_is_their_own_word(resolve):
    # echo turn 76: "Peter" of surrogate "Peter Hall" vs. the user's "Peter the Great"
    shadow = {"Peter Hall": "Daniel Ross"}
    sent = "Compare Peter the Great with Peter Hall's plan."
    reply = "Peter the Great reformed Russia; Peter agreed with the plan."
    assert resolve(reply, shadow) == "Daniel the Great reformed Russia; Daniel agreed with the plan."
    assert resolve(reply, shadow, sent=sent) == reply
    # typed only as part of the surrogate: the bare name part still restores
    assert resolve("Peter agreed.", shadow, sent="Ask Peter Hall.") == "Daniel agreed."


def test_I15_fuzzy_keeps_capitalisation(resolve):
    # echo turn 99: "James or" fuzzily matched surrogate "James Orr"
    assert resolve("Did LeBron James or Curry?", {"James Orr": "Rahul Verma"}) == \
        "Did LeBron James or Curry?"
    assert resolve("Ask James Ort.", {"James Orr": "Rahul Verma"}) == "Ask Rahul Verma."


def test_I15_partial_match_never_takes_text_the_user_typed(resolve):
    shadow = {"James Orr": "Rahul Verma"}
    reply = "James Ort wrote it."
    assert resolve(reply, shadow, sent="Did James Ort write it?") == reply
    assert resolve(reply, shadow) == "Rahul Verma wrote it."


def test_I15_surname_alone_needs_a_title(resolve):
    out = resolve("Mitchell Street is closed.", {"Sarah Mitchell": "Mia Lopez"})
    assert out == "Mitchell Street is closed."


def test_I15_common_word_names_are_not_rewritten(resolve):
    out = resolve("Grace was shown. May I ask?", {"Grace Hill": "Mia Lopez", "May Chen": "Ann Lee"})
    assert out == "Grace was shown. May I ask?"


def test_I15_fuzzy_needs_an_anchor_word_and_same_word_count(resolve):
    # single word surrogate: never fuzzy ("Katherine" ~ "Catherine")
    assert resolve("Catherine the Great", {"Katherine": "Priya"}) == "Catherine the Great"
    # typo in a two-word name with one word intact still restores
    assert resolve("Elias Vantre called.", {"Elias Vantree": "Jordan Mercer"}) == \
        "Jordan Mercer called."
    # no exact anchor word: left alone
    assert resolve("Elyas Vantre called.", {"Elias Vantree": "Jordan Mercer"}) == \
        "Elyas Vantre called."


def test_I5_low_entropy_restored_only_in_its_own_turn(resolve):
    shadow = {"72": "70", "female": "male", "Ann Lee": "Mia Lopez"}
    text = "He is 72, female, and Ann Lee knows."
    # sent this turn: restored
    assert resolve(text, shadow, current={"72", "female"}) == "He is 70, male, and Mia Lopez knows."
    # from an earlier turn: an ordinary "72" / "female" stays; names still restore
    assert resolve(text, shadow, current=set()) == "He is 72, female, and Mia Lopez knows."
    # no turn information: everything restores (back-compatible)
    assert resolve(text, shadow) == "He is 70, male, and Mia Lopez knows."


def test_I5_low_entropy_never_partially_matched(resolve):
    assert resolve("aged 72 years", {"72 years old": "70 years old"}) == "aged 72 years"
