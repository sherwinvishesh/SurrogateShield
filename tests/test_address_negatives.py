"""Text that must NEVER be detected as an address (false-positive gate)."""

import pytest

from surrogateshield.core.detection import address_parser as ap

NEGATIVES = [
    # counts / ratings
    "I gave it 5 stars",
    "rated 4 stars overall",
    "we got 3 points",
    "scored 99 runs today",
    "2 stars is generous",
    # durations
    "wait 3 days",
    "in 20 minutes on my cell",
    "it takes 2 hours by car",
    "roughly 6 weeks left",
    "5 min at most",
    "after 10 years there",
    # driving / idioms
    "a 3 point turn",
    "the 4 way stop",
    "made a 2 point conversion",
    "a 5 way intersection",
    # shipping ranges
    "shipping takes 3-5 business days on Oak St orders",
    "delivery in 7-10 business days",
    # versions / tech
    "version 2.4.1 shipped",
    "upgrade to 3.11 today",
    "python 3 way better",
    "see section 4.2 of the paper",
    # money
    "it costs $20",
    "a $1500 deposit",
    "save 15 percent",
    "paid 200 dollars",
    # quantities
    "2 park visits per week",
    "5 mile run this morning",
    "ran 26 miles",
    "carried 3 boxes upstairs",
    "ordered 12 units yesterday",
    "8 people showed up",
    "we planted 40 trees",
    # corrupted ground-truth value from experiment_key.json (v1 data bug)
    "1 but fail to establish a standard",
    # phone / ssn / other PII shapes (owned by other detectors)
    "SSN is 123-45-6789",
    "my number is 480-555-1234",
    "call +1-480-555-9999 today",
    "card 4111 1111 1111 1111",
    "IP is 192.168.1.1",
    # bare geographic mentions (GPE detection, not address)
    "I live in Tempe",
    "moving to Arizona next fall",
    "flights to Phoenix are cheap",
    # numbers + capitalized non-street words
    "chapter 5 way over there",
    "give me 5 us 19 examples",
    "the 3 Amigos movie",
    "room 12 opens at nine",
    "gate 22 is boarding",
    "flight 370 departed",
    "highway conditions are fine",
    "route planning is hard",
    # dates / times
    "May 5 2024 works for me",
    "at 5 pm sharp",
    "born on 12/25/1990",
    # sports
    "he shot 3 under par",
    "a 7 game series",
    # misc numerals
    "answer is 42",
    "pick a number between 1 and 100",
    "chapter 12 hill climbing algorithms",
    "6 point font is unreadable",
    "the 2 dollar bill",
    "9 to 5 job",
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_no_address_detected(text):
    found = ap.find_addresses(text)
    assert not found, f"false positive in {text!r}: {[a.full_text for a in found]}"
