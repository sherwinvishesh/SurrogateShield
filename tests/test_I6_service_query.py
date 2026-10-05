"""Audit I6 — service-query classification needs an explicit "find a place"
intent, a proximity phrase and a venue; sensitive service queries keep the
city/state and coarsen the street line instead of inventing a fake city.
Model-free.
"""

import pytest

from surrogateshield.core.detection import service_query as sq
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen


@pytest.mark.parametrize("text", [
    # the audit's bait list (exp12_semantics.py)
    "I'm nowhere near finished with the report.",
    "Where is the best place to learn Rust?",
    "My nearest relative is my aunt Maria Lopez in Boston.",
    "Is there a restaurant scene in the novel?",
    "What's around the corner for AI?",
    "Which pharmacy chain is cheapest in general?",
    # J2 dev: Tempe stayed verbatim because of this sentence
    "I'm a nurse at Mercy General Hospital in Tempe and my manager is Dr. Okafor.",
    "my best friend lives in Tempe near the hospital",
    "Hospital in Tempe is where I work",
    "My doctor in Mesa said I should rest",
    # request and location in different sentences
    "I live near Tempe and work at Intel. What's a good 401k?",
    "I'm nowhere near the end. What should I do?",
])
def test_I6_prose_is_not_a_service_query(text):
    assert sq.classify(text) == "none"
    assert sq.resolve(text, "auto") == (False, "replace")


@pytest.mark.parametrize("text", [
    "any good restaurants near 1126 E Apache Blvd, Tempe, AZ?",
    "nearest gas station close to me",
    "pizza in Tempe?",
    "coffee near me",
    # no venue word needed when the proximity object is a location
    "Can you find a notary near 40 Harbor Rd, Salem, MA?",
    "Which laundromats close to me are open late?",
])
def test_I6_real_service_queries(text):
    assert sq.classify(text) == "service"
    assert sq.resolve(text, "auto") == (True, "shift")


@pytest.mark.parametrize("text", [
    "Is there a rehab clinic near 316 Citrus Boulevard, Orlando, FL?",
    "rehab centers near my house",
    "find an immigration lawyer near me",
])
def test_I6_sensitive_service_query_is_coarse(text):
    assert sq.classify(text) == "service_coarse"
    assert sq.resolve(text, "auto") == (True, "coarse")


def test_I6_explicit_modes_and_disabled_pass_through():
    text = "Is there a rehab clinic near 316 Citrus Boulevard, Orlando, FL?"
    assert sq.resolve(text, "replace") == (True, "replace")
    assert sq.resolve(text, "shift", enabled=False) == (False, "shift")


def _addr(text):
    from surrogateshield.core.detection import address_parser
    p = address_parser.parse(text)
    e = DetectedEntity(text, 0, len(text), "address", 1.0, "pattern")
    e.parsed = p
    return e


def test_I6_coarse_keeps_city_state_and_drops_street():
    out = MimicGen(seed=1).generate(_addr("316 Citrus Boulevard, Orlando, FL"),
                                    address_mode="coarse")
    assert out == "my area, Orlando, FL"
    assert MimicGen(seed=1).generate(_addr("316 Citrus Boulevard"),
                                     address_mode="coarse") == "my area"


def test_I6_coarse_second_address_stays_one_to_one():
    m = MimicGen(seed=2).generate_all(
        [_addr("316 Citrus Boulevard"), _addr("12 Pine Street")], address_mode="coarse")
    assert len(set(m.values())) == 2 and "my area" in m.values()


def test_I6_session_masks_tempe_in_prose():
    """End to end through the library session, with the detector stubbed:
    the location in a non-service sentence is not suppressed."""
    import surrogateshield.session as s
    text = "I'm a nurse at Mercy General Hospital in Tempe."
    seen = {}

    def fake_cascade(text, skip_location_entities=False, **_):
        seen["skip"] = skip_location_entities
        return [], []
    orig = s._pipeline.run_cascade
    s._pipeline.run_cascade = fake_cascade
    try:
        s.Session()._detect(text)
    finally:
        s._pipeline.run_cascade = orig
    assert seen["skip"] is False


def test_I6_evaluator_credits_coarse_city_state_only():
    import eval_metrics as em
    q = "Find an HIV testing clinic near 4 Oak Ln, Austin, TX."
    addr = "4 Oak Ln, Austin, TX"
    sent = q.replace(addr, "my area, Austin, TX")
    spans = [{"text": addr, "start": q.index(addr), "end": q.index(addr) + len(addr),
              "type": "address", "replaced": True}]
    key = {"address": ["4 Oak Ln"], "GPE": ["Austin", "TX"]}
    out = em.classify_sent_leaks(q, key, sent, spans, "coarse", {addr: "my area, Austin, TX"})
    assert {l["value"]: l["deliberate"] for l in out["leaks"]} == {"Austin": True, "TX": True}
    assert "service_query_address_coarse" in em.POLICY_REASONS
